// x64 local register cache. Register-width NAMED locals are mirrored in callee-saved
// registers so a repeated read hits a register instead of the stack slot, and a store to a
// cached local keeps the new value in the register (write-through: the slot is ALSO written,
// so memory stays authoritative and debug info / aliasing are unaffected).
//
// Coherence — the cache is dropped whenever memory may change behind its back:
//   - a control-flow merge (X64Assembler::merge_epoch, bumped at every label bind): the
//     emission-order view of the registers doesn't hold for other inbound paths → drop ALL.
//   - a call (X64Assembler::call_epoch): the callee may write locals through an escaped
//     pointer → drop entries EXCEPT those whose local's address is provably never
//     materialized (the escape prescan below); the cache regs themselves are callee-saved.
//   - a store (x64_rc_note_store): to an exact RBP range → drop overlapping entries (slots
//     are disjoint, but a whole-aggregate store must kill cached `using`-promoted fields
//     inside it); through a pointer → drop ALL, even never-&'d locals — the backend itself
//     materializes lvalue addresses without a source-level `&` (multi-assign stores the lhs
//     through a saved R8, see x64_build_assign_stmt).
// Epoch checks are centralised in the encoder so no bind/call site can be missed.
//
// Cache registers are the callee-saved GP registers the rest of the backend never uses as
// scratch (verified: only CPUID touches RBX, with a balanced push/pop). Any that get used are
// spilled to a frame slot in the prologue and reloaded in every epilogue, with matching
// UNWIND_CODE SAVE_NONVOLs, so the Win64 preserve-nonvolatile contract still holds.

static X64Reg const x64_rc_gp[X64_RC_GP] = { X64Reg_RBX, X64Reg_R12, X64Reg_R13, X64Reg_R14, X64Reg_R15 };

// Reserved bytes for the prologue spill / each epilogue reload region: X64_RC_GP moves of
// `mov [rbp+disp32], r64` / `mov r64, [rbp+disp32]`, 7 bytes each. Patched in proc_end.
enum { X64_RC_MOV_LEN = 7, X64_RC_REGION = X64_RC_GP * X64_RC_MOV_LEN };

// ─────────────────────────────────────────────────────────────────────────────
// Arena-backed Entity* set (linear scan; N = address-taken locals per proc, small)
// ─────────────────────────────────────────────────────────────────────────────

gb_internal void x64_ent_set_init(x64EntSet *s, gbAllocator a) {
	s->alloc = a;
	s->count = 0;
	s->cap   = 16;
	s->keys  = gb_alloc_array(a, Entity *, s->cap);
}

gb_internal bool x64_ent_set_has(x64EntSet *s, Entity *e) {
	for (i32 i = 0; i < s->count; i++) {
		if (s->keys[i] == e) return true;
	}
	return false;
}

gb_internal void x64_ent_set_add(x64EntSet *s, Entity *e) {
	if (e == nullptr || x64_ent_set_has(s, e)) return;
	if (s->count == s->cap) {
		i32 new_cap = s->cap * 2;
		Entity **nk = gb_alloc_array(s->alloc, Entity *, new_cap);
		gb_memcopy(nk, s->keys, cast(usize)s->count * gb_size_of(Entity *));
		s->keys = nk;
		s->cap  = new_cap;
	}
	s->keys[s->count++] = e;
}

gb_internal void x64_rc_init(x64Procedure *p) {
	p->regcache = {};
	p->regcache.enabled = true;
	p->rc_save_region = -1;
	array_init(&p->rc_restore_regions, p->alloc, 0, 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Escape prescan: which locals may have their address materialized?
//
// Runs over the body AST before emission (flow-INsensitive on purpose: an `&x`
// after a call still reaches the call through a loop back-edge, so escape must
// be a whole-proc property). Sound by over-approximation: under an address-
// materializing construct EVERY identifier is marked, and any AST node kind the
// walker doesn't recognize sets scan_ok=false, which disables call-survival for
// the whole proc rather than risking a stale cache.
//
// Marking constructs: `&expr`, slice exprs (alias a frame array's storage),
// `using` decls/stmts (aliases), `raw_data(x)`, and args of any call with a
// #by_ptr parameter. Bodies of #force_inline-able callees are scanned too —
// they emit into THIS frame, so their `&locals` alias this frame's slots (the
// recursion predicate is a superset of x64_try_inline_call's, which only ever
// declines to inline — declined calls pass args by value and need no marks).
//
// KNOWN COUPLING: `any` boxing and variadic packing copy to temps and box the
// TEMP's address (see x64-any-box semantics) — if boxing is ever changed to
// alias the local directly, those args must be marked here too.
// ─────────────────────────────────────────────────────────────────────────────

struct x64EscapeScan {
	x64Procedure *p;
	Entity       *inline_stack[X64_MAX_INLINE_DEPTH]; // force-inline recursion guard
	int           inline_depth;
	bool          ok;
};

gb_internal void x64_escape_scan(x64EscapeScan *s, Ast *node, bool marking);

gb_internal void x64_escape_scan_slice(x64EscapeScan *s, Slice<Ast *> const &list, bool marking) {
	for_array(i, list) x64_escape_scan(s, list[i], marking);
}

gb_internal void x64_escape_scan_call(x64EscapeScan *s, AstCallExpr *ce, Ast *node, bool marking) {
	x64_escape_scan(s, ce->proc, marking);
	x64_escape_scan_slice(s, ce->args, marking);

	Entity *callee = entity_of_node(unparen_expr(ce->proc));

	// raw_data(x) yields a pointer into x's storage without a source-level `&`.
	if (callee != nullptr && callee->kind == Entity_Builtin &&
	    callee->Builtin.id == BuiltinProc_raw_data) {
		x64_escape_scan_slice(s, ce->args, true);
		return;
	}

	// A #by_ptr parameter receives the ADDRESS of its argument. Named/positional arg
	// matching isn't worth the complexity for so rare a flag: mark every arg.
	Type *ct = (ce->proc != nullptr) ? base_type(ce->proc->tav.type) : nullptr;
	if (ct != nullptr && ct->kind == Type_Proc && ct->Proc.params != nullptr) {
		TypeTuple *params = &ct->Proc.params->Tuple;
		for_array(i, params->variables) {
			if (params->variables[i]->flags & EntityFlag_ByPtr) {
				x64_escape_scan_slice(s, ce->args, true);
				break;
			}
		}
	}

	// #force_inline candidate: its body emits into this frame → scan it as our own.
	if (callee != nullptr && callee->kind == Entity_Procedure && !callee->Procedure.is_foreign) {
		DeclInfo *d  = decl_info_of_entity(callee);
		Ast      *pl = (d != nullptr) ? d->proc_lit : nullptr;
		if (pl != nullptr && pl->kind == Ast_ProcLit && pl->ProcLit.body != nullptr &&
		    (pl->ProcLit.inlining == ProcInlining_inline || ce->inlining == ProcInlining_inline)) {
			for (int i = 0; i < s->inline_depth; i++) {
				if (s->inline_stack[i] == callee) return; // already being scanned
			}
			// Depth exhausted mirrors x64_try_inline_call's guard: it won't inline either.
			if (s->inline_depth >= X64_MAX_INLINE_DEPTH) return;
			s->inline_stack[s->inline_depth++] = callee;
			x64_escape_scan(s, pl->ProcLit.body, false);
			s->inline_depth--;
		}
	}
	(void)node;
}

gb_internal void x64_escape_scan(x64EscapeScan *s, Ast *node, bool marking) {
	if (node == nullptr || !s->ok) return;

	// Type expressions are unevaluated — nothing in them materializes a runtime address.
	if (node->kind > Ast__TypeBegin && node->kind < Ast__TypeEnd) return;

	switch (node->kind) {
	case Ast_Ident:
		if (marking) {
			Entity *e = entity_of_node(node);
			if (e != nullptr && e->kind == Entity_Variable) {
				x64_ent_set_add(&s->p->regcache.escaped, e);
			}
		}
		break;

	// Leaves with no runtime addressing.
	case Ast_Implicit:
	case Ast_Uninit:
	case Ast_BasicLit:
	case Ast_BasicDirective:
	case Ast_EmptyStmt:
	case Ast_BranchStmt:
	case Ast_Label:
	case Ast_ProcGroup:
	case Ast_PackageDecl:
	case Ast_ImportDecl:
	case Ast_ForeignImportDecl:
	case Ast_ForeignBlockDecl: // foreign decls only — no body code in this frame
	case Ast_Attribute:
	case Ast_Field:
	case Ast_FieldList:
	case Ast_BitFieldField:
	case Ast_EnumFieldValue:
		break;

	// A nested proc literal cannot capture this frame's locals — its body is compiled
	// as its own procedure with its own prescan.
	case Ast_ProcLit:
		break;

	case_ast_node(ue, UnaryExpr, node);
		x64_escape_scan(s, ue->expr, marking || ue->op.kind == Token_And);
	case_end;

	case_ast_node(se2, SliceExpr, node);
		// s[a:b] aliases s's backing storage; for a frame ARRAY that's the slot itself.
		x64_escape_scan(s, se2->expr, true);
		x64_escape_scan(s, se2->low,  marking);
		x64_escape_scan(s, se2->high, marking);
	case_end;

	case_ast_node(ce, CallExpr, node);
		x64_escape_scan_call(s, ce, node, marking);
	case_end;

	case_ast_node(us, UsingStmt, node);
		x64_escape_scan_slice(s, us->list, true); // aliases its operand's fields
	case_end;

	case_ast_node(vd, ValueDecl, node);
		x64_escape_scan_slice(s, vd->values, marking);
		if (vd->is_using) x64_escape_scan_slice(s, vd->names, true);
	case_end;

	case_ast_node(rs, RangeStmt, node);
		x64_escape_scan(s, rs->init, marking);
		x64_escape_scan(s, rs->expr, marking);
		// `for &v in c` binds v to c's elements in place → c's storage is aliased.
		for_array(i, rs->vals) {
			Ast *v = rs->vals[i];
			if (v != nullptr && v->kind == Ast_UnaryExpr && v->UnaryExpr.op.kind == Token_And) {
				x64_escape_scan(s, rs->expr, true);
				break;
			}
		}
		x64_escape_scan(s, rs->body, marking);
	case_end;

	case_ast_node(urs, UnrollRangeStmt, node);
		x64_escape_scan(s, urs->init, marking);
		x64_escape_scan(s, urs->expr, marking);
		if ((urs->val0 != nullptr && urs->val0->kind == Ast_UnaryExpr && urs->val0->UnaryExpr.op.kind == Token_And) ||
		    (urs->val1 != nullptr && urs->val1->kind == Ast_UnaryExpr && urs->val1->UnaryExpr.op.kind == Token_And)) {
			x64_escape_scan(s, urs->expr, true);
		}
		x64_escape_scan(s, urs->body, marking);
	case_end;

	case Ast_InlineAsmExpr:
		s->ok = false; // opaque to the scan
		break;

	case_ast_node(el, Ellipsis, node);       x64_escape_scan(s, el->expr, marking); case_end;
	case_ast_node(te, TagExpr, node);        x64_escape_scan(s, te->expr, marking); case_end;
	case_ast_node(be, BinaryExpr, node);     x64_escape_scan(s, be->left, marking); x64_escape_scan(s, be->right, marking); case_end;
	case_ast_node(pe, ParenExpr, node);      x64_escape_scan(s, pe->expr, marking); case_end;
	case_ast_node(sel, SelectorExpr, node);  x64_escape_scan(s, sel->expr, marking); case_end;
	case Ast_ImplicitSelectorExpr:           break;
	case_ast_node(sce, SelectorCallExpr, node); x64_escape_scan(s, sce->expr, marking); x64_escape_scan(s, sce->call, marking); case_end;
	case_ast_node(ie, IndexExpr, node);      x64_escape_scan(s, ie->expr, marking); x64_escape_scan(s, ie->index, marking); case_end;
	case_ast_node(mie, MatrixIndexExpr, node); x64_escape_scan(s, mie->expr, marking); x64_escape_scan(s, mie->row_index, marking); x64_escape_scan(s, mie->column_index, marking); case_end;
	case_ast_node(de, DerefExpr, node);      x64_escape_scan(s, de->expr, marking); case_end;
	case_ast_node(cl, CompoundLit, node);    x64_escape_scan_slice(s, cl->elems, marking); case_end;
	case_ast_node(fv, FieldValue, node);     x64_escape_scan(s, fv->value, marking); case_end;
	case_ast_node(tif, TernaryIfExpr, node); x64_escape_scan(s, tif->x, marking); x64_escape_scan(s, tif->cond, marking); x64_escape_scan(s, tif->y, marking); case_end;
	case_ast_node(twh, TernaryWhenExpr, node); x64_escape_scan(s, twh->x, marking); x64_escape_scan(s, twh->cond, marking); x64_escape_scan(s, twh->y, marking); case_end;
	case_ast_node(oe, OrElseExpr, node);     x64_escape_scan(s, oe->x, marking); x64_escape_scan(s, oe->y, marking); case_end;
	case_ast_node(ore, OrReturnExpr, node);  x64_escape_scan(s, ore->expr, marking); case_end;
	case_ast_node(obe, OrBranchExpr, node);  x64_escape_scan(s, obe->expr, marking); case_end;
	case_ast_node(ta, TypeAssertion, node);  x64_escape_scan(s, ta->expr, marking); case_end;
	case_ast_node(tc, TypeCast, node);       x64_escape_scan(s, tc->expr, marking); case_end;
	case_ast_node(ac, AutoCast, node);       x64_escape_scan(s, ac->expr, marking); case_end;

	case_ast_node(es, ExprStmt, node);       x64_escape_scan(s, es->expr, marking); case_end;
	case_ast_node(as, AssignStmt, node);     x64_escape_scan_slice(s, as->lhs, marking); x64_escape_scan_slice(s, as->rhs, marking); case_end;
	case_ast_node(bs, BlockStmt, node);      x64_escape_scan_slice(s, bs->stmts, marking); case_end;
	case_ast_node(is, IfStmt, node);         x64_escape_scan(s, is->init, marking); x64_escape_scan(s, is->cond, marking); x64_escape_scan(s, is->body, marking); x64_escape_scan(s, is->else_stmt, marking); case_end;
	case_ast_node(ws, WhenStmt, node);       x64_escape_scan(s, ws->cond, marking); x64_escape_scan(s, ws->body, marking); x64_escape_scan(s, ws->else_stmt, marking); case_end;
	case_ast_node(rst, ReturnStmt, node);    x64_escape_scan_slice(s, rst->results, marking); case_end;
	case_ast_node(fs, ForStmt, node);        x64_escape_scan(s, fs->init, marking); x64_escape_scan(s, fs->cond, marking); x64_escape_scan(s, fs->post, marking); x64_escape_scan(s, fs->body, marking); case_end;
	case_ast_node(cc, CaseClause, node);     x64_escape_scan_slice(s, cc->list, marking); x64_escape_scan_slice(s, cc->stmts, marking); case_end;
	case_ast_node(ss, SwitchStmt, node);     x64_escape_scan(s, ss->init, marking); x64_escape_scan(s, ss->tag, marking); x64_escape_scan(s, ss->body, marking); case_end;
	case_ast_node(tss, TypeSwitchStmt, node); x64_escape_scan(s, tss->tag, marking); x64_escape_scan(s, tss->body, marking); case_end;
	case_ast_node(ds, DeferStmt, node);      x64_escape_scan(s, ds->stmt, marking); case_end;

	default:
		s->ok = false; // unknown construct → no call-survival for this proc (fail safe)
		break;
	}
}

gb_internal void x64_rc_escape_prescan(x64Procedure *p, Ast *body) {
	x64RegCache *rc = &p->regcache;
	if (!rc->enabled || body == nullptr) return;
	x64_ent_set_init(&rc->escaped, p->alloc);
	x64EscapeScan s = {};
	s.p  = p;
	s.ok = true;
	x64_escape_scan(&s, body, false);
	rc->scan_ok = s.ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime cache
// ─────────────────────────────────────────────────────────────────────────────

// Drop every live entry (keeps used_gp so the prologue still spills any reg touched anywhere).
gb_internal void x64_rc_invalidate_all(x64RegCache *rc) {
	for (int i = 0; i < X64_RC_GP; i++) { rc->gp[i].off = 0; rc->gp[i].type = nullptr; }
}

// Only clean-register-width SCALARS round-trip through a GP cache register. Aggregates are
// excluded even at size 8: their interior can be addressed (indexing, field selection,
// slicing, union asserts) without any `&` the escape prescan could see. Floats live in XMM
// (deferred), explicit-endian types don't match the plain load/extend below.
gb_internal bool x64_rc_cacheable(Type *t) {
	if (t == nullptr) return false;
	i64 sz = x64_type_size(t);
	if (sz != 1 && sz != 2 && sz != 4 && sz != 8) return false;
	Type *ct = core_type(t);
	switch (ct->kind) {
	case Type_Basic: {
		u32 fl = ct->Basic.flags;
		if (fl & (BasicFlag_EndianBig|BasicFlag_EndianLittle)) return false;
		if (fl & (BasicFlag_Integer|BasicFlag_Boolean|BasicFlag_Rune)) return true;
		return ct->Basic.kind == Basic_rawptr || ct->Basic.kind == Basic_typeid;
	}
	case Type_Pointer:
	case Type_MultiPointer:
	case Type_Proc:
		return true;
	case Type_BitSet: {
		Type *u = ct->BitSet.underlying;
		if (u != nullptr) {
			Type *cu = core_type(u);
			if (cu->kind == Type_Basic && (cu->Basic.flags & (BasicFlag_EndianBig|BasicFlag_EndianLittle))) return false;
		}
		return true;
	}
	default:
		return false;
	}
}

// Catch up with the invalidation clocks (lazy: checked on every read/write/store-note).
gb_internal void x64_rc_sync(x64Procedure *p) {
	x64RegCache  *rc = &p->regcache;
	X64Assembler *a  = &p->asm_;
	// A new named local can REUSE a dead local's frame offset (block-scope slot reclaim) with a
	// different escape status — entries and the write-through handshake must not carry over.
	if (rc->seq_stamp != p->named_seq || rc->merge_stamp != a->merge_epoch) {
		x64_rc_invalidate_all(rc);
		rc->pend_off    = 0;
		rc->seq_stamp   = p->named_seq;
		rc->merge_stamp = a->merge_epoch;
		rc->call_stamp  = a->call_epoch;
		return;
	}
	if (rc->call_stamp != a->call_epoch) {
		for (int i = 0; i < X64_RC_GP; i++) {
			if (rc->gp[i].type != nullptr && !rc->gp[i].survives) { rc->gp[i].off = 0; rc->gp[i].type = nullptr; }
		}
		rc->call_stamp = a->call_epoch;
	}
}

// A store is about to land at `dst` (t = stored type). Drop whatever it may touch, and
// remember an exact-slot match for x64_rc_write's write-through refill (one-shot handshake:
// x64_store_value calls note_store on entry and rc_write from its scalar-register branch).
gb_internal void x64_rc_note_store(x64Procedure *p, X64Mem dst, Type *t) {
	x64RegCache *rc = &p->regcache;
	rc->pend_off = 0;
	if (!rc->enabled) return;
	x64_rc_sync(p); // don't let a pre-merge / reused-offset entry donate its `survives` to the pend
	if (dst.rip_rel || dst.base == X64Reg_NONE) return; // globals/absolute never alias frame slots
	if (dst.base != X64Reg_RBP || dst.index != X64Reg_NONE) {
		// Through a pointer (or an indexed frame slot whose runtime target is unknown): may
		// alias ANY local — including never-&'d ones, since the backend materializes lvalue
		// addresses without a source `&` (multi-assign stores lhs through a saved R8).
		x64_rc_invalidate_all(rc);
		return;
	}
	i64 st_sz = (t != nullptr) ? x64_type_size(t) : 8;
	if (st_sz <= 0) st_sz = 8;
	for (int i = 0; i < X64_RC_GP; i++) {
		x64RegSlot *sl = &rc->gp[i];
		if (sl->type == nullptr) continue;
		i64 en_sz = x64_type_size(sl->type);
		// Range overlap, not exact-offset: a whole-aggregate store must kill cached
		// `using`-promoted fields inside it, and a field store must kill its parent.
		if ((i64)sl->off < (i64)dst.disp + st_sz && (i64)dst.disp < (i64)sl->off + en_sz) {
			if (sl->off == dst.disp && en_sz == st_sz) {
				rc->pend_off      = dst.disp;
				rc->pend_survives = sl->survives;
			}
			sl->off = 0; sl->type = nullptr;
		}
	}
}

// A bulk writer (x64_zero_mem / x64_copy_fixed) is about to write [dst, dst+size). Same drop
// rules as note_store but byte-sized and with no write-through handshake. Catches raw zero/copy
// paths that don't go through x64_store_value (named-return zero-init runs before any entry
// exists, but e.g. compound-literal zero-fills of a named local run mid-body).
gb_internal void x64_rc_note_clobber(x64Procedure *p, X64Mem dst, i64 size) {
	x64RegCache *rc = &p->regcache;
	if (!rc->enabled || size <= 0) return;
	if (dst.rip_rel || dst.base == X64Reg_NONE) return;
	if (dst.base != X64Reg_RBP || dst.index != X64Reg_NONE) {
		x64_rc_invalidate_all(rc);
		return;
	}
	for (int i = 0; i < X64_RC_GP; i++) {
		x64RegSlot *sl = &rc->gp[i];
		if (sl->type == nullptr) continue;
		i64 en_sz = x64_type_size(sl->type);
		if ((i64)sl->off < (i64)dst.disp + size && (i64)dst.disp < (i64)sl->off + en_sz) {
			sl->off = 0; sl->type = nullptr;
		}
	}
}

gb_internal int x64_rc_pick_slot(x64RegCache *rc) {
	int pick = -1;
	for (int i = 0; i < X64_RC_GP; i++) {
		if (rc->pins[i] != 0) continue; // in-flight operand — never reuse its register
		if (rc->gp[i].type == nullptr) return i;
		if (pick < 0 || rc->gp[i].age < rc->gp[pick].age) pick = i;
	}
	return pick; // LRU (eviction is a plain drop — memory is authoritative); -1 = all pinned
}

// Read the register-width named local `e` at RBP offset `off`. On a hit returns the cache
// register (no memory access). On a miss, loads it into a cache register (allocating /
// LRU-evicting one) and returns that. Returns x64v_none() if the cache is off or the type
// is not cacheable — the caller then does its normal load.
gb_internal x64Value x64_rc_read(x64Procedure *p, Entity *e, i32 off, Type *t) {
	x64RegCache *rc = &p->regcache;
	if (!rc->enabled || e == nullptr || !x64_rc_cacheable(t)) return x64v_none();
	x64_rc_sync(p);

	for (int i = 0; i < X64_RC_GP; i++) {
		if (rc->gp[i].type != nullptr && rc->gp[i].off == off && are_types_identical(rc->gp[i].type, t)) {
			rc->gp[i].age = ++rc->clock;
			return x64v_reg(t, x64_rc_gp[i]);
		}
	}

	int pick = x64_rc_pick_slot(rc);
	if (pick < 0) return x64v_none(); // every slot pinned by in-flight operands
	X64Reg        reg  = x64_rc_gp[pick];
	X64Assembler *a    = &p->asm_;
	X64OpSize     sz   = x64_op_size_of(t);
	X64Mem        mem  = x64_rbp_mem(off);
	// Fill the whole 64-bit register — sign/zero extension mirrors x64_load_addr exactly.
	Type *bt = base_type(t);
	bool signed_int = (bt->kind == Type_Basic) &&
	                  (bt->Basic.flags & BasicFlag_Integer) &&
	                  !(bt->Basic.flags & BasicFlag_Unsigned);
	if (sz == X64OpSize_64)      x64_emit_mov_rm  (a, sz, reg, mem);
	else if (signed_int)         x64_emit_movsx_rm(a, sz, reg, mem);
	else if (sz == X64OpSize_32) x64_emit_mov_rm  (a, X64OpSize_32, reg, mem);
	else                         x64_emit_movzx_rm(a, sz, reg, mem);

	x64RegSlot *sl = &rc->gp[pick];
	sl->off  = off;
	sl->type = t;
	sl->age  = ++rc->clock;
	// `using`-promoted locals alias their parent aggregate → a callee reaching the parent
	// reaches them; never let those ride across calls.
	sl->survives = rc->scan_ok &&
	               !(e->flags & EntityFlag_Using) &&
	               !x64_ent_set_has(&rc->escaped, e);
	rc->used_gp |= (u8)(1u << pick);
	return x64v_reg(t, reg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Operand pinning: a binop site holds the returned register across the OTHER
// operand's build — a later fill must not evict/reuse it. Pins protect the
// REGISTER (the in-flight x64Value), not the entry: invalidation still drops
// the entry, which is fine — the register keeps the value read at eval time,
// which is exactly left-to-right evaluation order. Pins are counted (`x + x`
// pins one slot twice) and released by the consumer after emit.
// ─────────────────────────────────────────────────────────────────────────────

gb_internal x64Value x64_rc_operand(x64Procedure *p, Entity *e, i32 off, Type *t) {
	x64Value v = x64_rc_read(p, e, off, t);
	if (v.kind == x64Value_Reg) {
		for (int i = 0; i < X64_RC_GP; i++) {
			if (x64_rc_gp[i] == v.reg) { p->regcache.pins[i]++; break; }
		}
	}
	return v;
}

gb_internal void x64_rc_unpin_value(x64Procedure *p, x64Value v) {
	if (v.kind != x64Value_Reg) return;
	for (int i = 0; i < X64_RC_GP; i++) {
		if (x64_rc_gp[i] == v.reg) {
			if (p->regcache.pins[i] > 0) p->regcache.pins[i]--;
			return;
		}
	}
}

gb_internal bool x64_rc_value_pinned(x64Procedure *p, x64Value v) {
	if (v.kind != x64Value_Reg) return false;
	for (int i = 0; i < X64_RC_GP; i++) {
		if (x64_rc_gp[i] == v.reg) return p->regcache.pins[i] > 0;
	}
	return false;
}

// Write-through: x64_store_value just stored `src` (already converted to t) to the exact
// slot that x64_rc_note_store dropped an entry for — keep the new value in a cache register
// too, so the next read hits. Memory was already written; the invariant reg==slot holds on
// every path that executes the store.
gb_internal void x64_rc_write(x64Procedure *p, X64Mem dst, Type *t, X64Reg src) {
	x64RegCache *rc   = &p->regcache;
	i32          pend = rc->pend_off;
	rc->pend_off = 0;
	if (!rc->enabled || pend == 0) return;
	if (dst.rip_rel || dst.base != X64Reg_RBP || dst.index != X64Reg_NONE || dst.disp != pend) return;
	if (!x64_rc_cacheable(t)) return;
	x64_rc_sync(p); // the conversion between note_store and here may have emitted calls/labels

	int pick = x64_rc_pick_slot(rc);
	if (pick < 0) return; // every slot pinned by in-flight operands
	X64Reg        reg  = x64_rc_gp[pick];
	X64Assembler *a    = &p->asm_;
	X64OpSize     sz   = x64_op_size_of(t);
	Type *bt = base_type(t);
	bool signed_int = (bt->kind == Type_Basic) &&
	                  (bt->Basic.flags & BasicFlag_Integer) &&
	                  !(bt->Basic.flags & BasicFlag_Unsigned);
	if (sz == X64OpSize_64)      x64_emit_mov_rr  (a, X64OpSize_64, reg, src);
	else if (signed_int)         x64_emit_movsx_rr(a, sz, reg, src);
	else if (sz == X64OpSize_32) x64_emit_mov_rr  (a, X64OpSize_32, reg, src);
	else                         x64_emit_movzx_rr(a, sz, reg, src);

	x64RegSlot *sl = &rc->gp[pick];
	sl->off      = pend;
	sl->type     = t;
	sl->age      = ++rc->clock;
	sl->survives = rc->pend_survives;
	rc->used_gp |= (u8)(1u << pick);
}

// ─────────────────────────────────────────────────────────────────────────────
// Prologue/epilogue save & restore of the used cache registers
// ─────────────────────────────────────────────────────────────────────────────

// Encode `mov [rbp+disp32], reg` (store, op=0x89) or `mov reg, [rbp+disp32]` (load, op=0x8B)
// into `out[0..6]`. Always 64-bit (REX.W) so the caller's full nonvolatile is preserved.
gb_internal void x64_rc_enc_mov(u8 *out, u8 op, X64Reg reg, i32 disp) {
	out[0] = (u8)(0x48u | (reg >= 8 ? 0x04u : 0u));     // REX.W (+ REX.R for r12..r15)
	out[1] = op;
	out[2] = (u8)(0x80u | ((reg & 7u) << 3) | 0x05u);   // ModRM mod=10 disp32, rm=101 (RBP)
	out[3] = (u8)( disp        & 0xFF);
	out[4] = (u8)((disp >>  8) & 0xFF);
	out[5] = (u8)((disp >> 16) & 0xFF);
	out[6] = (u8)((disp >> 24) & 0xFF);
}

// Reserve the prologue spill region (NOP-filled; patched with the real MOVs in proc_end).
gb_internal void x64_rc_emit_saves(x64Procedure *p) {
	if (!p->regcache.enabled) return;
	p->rc_save_region = (isize)p->asm_.code.count;
	for (int i = 0; i < X64_RC_REGION; i++) x64_enc_b(&p->asm_, 0x90u);
}

// Reserve one epilogue reload region (NOP-filled; patched in proc_end). Emitted BEFORE the
// `mov rsp,rbp; pop rbp` teardown so the reloads still read the live frame.
gb_internal void x64_rc_reserve_restore(x64Procedure *p) {
	if (!p->regcache.enabled) return;
	array_add(&p->rc_restore_regions, (isize)p->asm_.code.count);
	for (int i = 0; i < X64_RC_REGION; i++) x64_enc_b(&p->asm_, 0x90u);
}

// Assign each used cache register a frame spill slot (in proc_end, once the set is known). The
// slots must sit BELOW the deepest body temporary (frame_max), or a reclaimed body slot reusing
// that address would clobber the spilled caller register before the epilogue reloads it — so
// extend the cursor to the peak first.
gb_internal void x64_rc_ensure_save_slots(x64Procedure *p) {
	x64RegCache *rc = &p->regcache;
	if (!rc->enabled || rc->used_gp == 0) return;
	if (p->local_size < p->frame_max) p->local_size = p->frame_max;
	for (int i = 0; i < X64_RC_GP; i++) {
		if ((rc->used_gp & (1u << i)) && rc->save_off[i] == 0) rc->save_off[i] = x64_alloc_local(p, 8, 8);
	}
}

// Fill the reserved prologue/epilogue regions with the spill/reload MOVs for the used registers.
// Called from proc_end after the frame is sized.
gb_internal void x64_rc_patch(x64Procedure *p) {
	x64RegCache *rc = &p->regcache;
	if (!rc->enabled || rc->used_gp == 0) return;
	u8 *code = p->asm_.code.data;

	// Prologue: spill each used reg to its slot (0x89 = MOV [mem], reg).
	if (p->rc_save_region >= 0) {
		isize at = p->rc_save_region;
		for (int i = 0; i < X64_RC_GP; i++) {
			if (!(rc->used_gp & (1u << i))) continue;
			x64_rc_enc_mov(code + at, 0x89u, x64_rc_gp[i], rc->save_off[i]);
			at += X64_RC_MOV_LEN;
		}
	}
	// Each epilogue: reload each used reg from its slot (0x8B = MOV reg, [mem]).
	for_array(e, p->rc_restore_regions) {
		isize at = p->rc_restore_regions[e];
		for (int i = 0; i < X64_RC_GP; i++) {
			if (!(rc->used_gp & (1u << i))) continue;
			x64_rc_enc_mov(code + at, 0x8Bu, x64_rc_gp[i], rc->save_off[i]);
			at += X64_RC_MOV_LEN;
		}
	}
}

// Win64 UWOP register numbers match the X64Reg encoding (RAX=0..R15=15).
gb_internal int x64_rc_used_count(x64RegCache *rc) {
	int n = 0;
	for (int i = 0; i < X64_RC_GP; i++) if (rc->used_gp & (1u << i)) n++;
	return n;
}
