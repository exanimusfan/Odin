// x64 debug backend — statement code generation.

// Memory helpers

gb_internal void x64_zero_mem(x64Procedure *p, X64Mem dst, i64 size) {
	X64Assembler *a = &p->asm_;
	if (size <= 0) return;

	if (size <= 256) {
		i32 off = dst.disp;
		i64 rem = size;
		while (rem >= 8) {
			x64_emit_mov_mi(a, X64OpSize_64, x64_mem(dst.base, off), 0);
			off += 8; rem -= 8;
		}
		while (rem >= 4) {
			x64_emit_mov_mi(a, X64OpSize_32, x64_mem(dst.base, off), 0);
			off += 4; rem -= 4;
		}
		while (rem > 0) {
			x64_emit_mov_mi(a, X64OpSize_8, x64_mem(dst.base, off), 0);
			off += 1; rem -= 1;
		}
	} else {
		// AL=0, RDI=dst, RCX=size; REP STOSB. RDI is nonvolatile in Win64 — save/
		// restore it (dst is RBP/RAX-based, not RSP, so the push doesn't shift it).
		// Compute the address BEFORE zeroing RAX — dst.base may BE RAX (e.g. a global
		// field `app.project.panels` whose &app was loaded into RAX), and `xor eax,eax`
		// would otherwise clobber the base, leaving RDI = the bare field offset.
		x64_emit_push_r(a, X64Reg_RDI);
		x64_emit_lea(a, X64Reg_RDI, dst);
		x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, size);
		x64_enc_b(a, 0xF3); // REP prefix
		x64_enc_b(a, 0xAA); // STOSB
		x64_emit_pop_r(a, X64Reg_RDI);
	}
}

gb_internal void x64_copy_mem(x64Procedure *p, X64Mem dst, X64Mem src, i64 size) {
	if (size <= 0) return;
	X64Assembler *a = &p->asm_;
	// RSI=src, RDI=dst, RCX=size; REP MOVSB. RSI/RDI are nonvolatile in Win64 —
	// save/restore them (LLVM's memmove preserves them, so our inline copy must
	// too). src/dst are RBP/RAX-based, not RSP — the pushes don't shift them.
	x64_emit_push_r(a, X64Reg_RSI);
	x64_emit_push_r(a, X64Reg_RDI);
	x64_emit_lea(a, X64Reg_RSI, src);
	x64_emit_lea(a, X64Reg_RDI, dst);
	x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, size);
	x64_enc_b(a, 0xF3); // REP prefix
	x64_enc_b(a, 0xA4); // MOVSB
	x64_emit_pop_r(a, X64Reg_RDI);
	x64_emit_pop_r(a, X64Reg_RSI);
}

// Misc helpers

gb_internal bool x64_is_blank(Ast *ident) {
	if (ident == nullptr) return true;
	if (ident->kind != Ast_Ident) return false;
	return ident->Ident.token.string == str_lit("_");
}

// Does this lvalue root at the implicit `context` (e.g. `context.logger = …`)? Such
// assigns are SCOPED: lb_addr_store/lbAddr_Context snapshots context into a fresh local.
// Only a PURE selector chain counts — a deref/index modifies through a stored pointer,
// not the context.
gb_internal bool x64_lvalue_roots_at_context(Ast *lhs) {
	Ast *e = unparen_expr(lhs);
	while (e != nullptr && e->kind == Ast_SelectorExpr) {
		e = unparen_expr(e->SelectorExpr.expr);
	}
	return e != nullptr && e->kind == Ast_Implicit && e->Implicit.kind == Token_context;
}

gb_internal i32 x64_alloc_var(x64Procedure *p, Entity *e) {
	i64 sz = x64_type_size(e->type);
	i64 al = x64_type_align(e->type);
	if (sz <= 0) sz = 1;
	if (al <= 0) al = 1;
	i32 off = x64_alloc_local(p, sz, al);
	x64_var_set(&p->var_offsets, e, off);
	p->named_seq++; // mark: a named (scope-lived, debug-visible) local was allocated
	return off;
}

gb_internal x64Procedure::LoopInfo *x64_find_loop(x64Procedure *p, String label) {
	if (label.len == 0) {
		if (p->loops.count == 0) return nullptr;
		return &p->loops[p->loops.count - 1];
	}
	for (isize i = p->loops.count - 1; i >= 0; i--) {
		x64Procedure::LoopInfo *li = &p->loops[i];
		if (li->ast_label != nullptr && li->ast_label->kind == Ast_Label) {
			if (li->ast_label->Label.name->Ident.token.string == label) return li;
		}
	}
	return nullptr;
}

// Statement builder

// `{}` — an empty compound literal, i.e. the zero value of its type. Assigning/initializing
// with it should ZERO the destination in place; building the whole zero value into a temp and
// copying it makes a full-size stack temp (a Slot_Map-sized field ⇒ ~74MB ⇒ stack overflow:
// `app.project.panels = {}`).
gb_internal bool x64_is_empty_compound_lit(Ast *e) {
	e = unparen_expr(e);
	return e != nullptr && e->kind == Ast_CompoundLit && e->CompoundLit.elems.count == 0;
}

// Zero-in-place is only equivalent to `dst = e` when the empty literal's type has the SAME
// zero representation as dst. A variant literal into a union (`u = Variant{}`) does NOT — it must
// construct the union (set the tag), so route it through the normal build+store. Returns true only
// when zeroing `dst_type`'s bytes is the correct lowering.
gb_internal bool x64_empty_compound_lit_zeroes(Ast *e, Type *dst_type) {
	e = unparen_expr(e);
	if (e == nullptr || e->kind != Ast_CompoundLit || e->CompoundLit.elems.count != 0) return false;
	Type *lt = e->tav.type;
	if (lt == nullptr || dst_type == nullptr) return true;
	return are_types_identical(default_type(lt), default_type(dst_type));
}

// Strip a leading & prefix (for &b in slice) before entity lookup.
gb_internal Ast *x64_range_strip(Ast *v) {
	if (v && v->kind == Ast_UnaryExpr && v->UnaryExpr.op.kind == Token_And)
		return v->UnaryExpr.expr;
	return v;
}

// Mirrors lb_build_return_stmt. Builds `results` into the result ABI (RAX / XMM0 / sret /
// partial-return pointers), running defers at the LLVM-matching point. Conversion to each result
// type is handled by x64_store_value (→ x64_emit_conv).
// A `return` inside an inlined callee body: store results into the active inline frame's
// slots (with conversion), run the inlined scope's defers, and jump to its join label —
// never the real epilogue. Mirrors x64_build_return_stmt's result handling.
gb_internal void x64_build_inline_return(x64Procedure *p, Slice<Ast *> const &results) {
	x64Procedure::InlineFrame *frp = &p->inline_frames[p->inline_frames.count - 1];
	// Copy out: building a result expr may push/pop nested inline frames (array realloc).
	isize  join  = frp->join_label;
	isize  dbase = frp->defer_base;
	int    nres  = frp->nres;
	i32   *roffs = frp->result_offs;
	Type **rtyp  = frp->result_types;
	int    rc    = (int)results.count;

	if (rc == 0) {
		// bare return: named results already live in their slots
	} else if (nres == 1) {
		x64Value v = x64_build_expr(p, results[0]);
		x64_store_value(p, x64addr(x64_rbp_mem(roffs[0]), rtyp[0]), v);
	} else if (rc == 1) {
		// one expression yields the whole tuple (per-field, with implicit conversions)
		x64Value tv = x64_build_expr(p, results[0]);
		Type *src = tv.type;
		Type *sb  = (src != nullptr) ? base_type(src) : nullptr;
		if (tv.kind == x64Value_Mem && sb != nullptr && sb->kind == Type_Tuple) {
			for (int i = 0; i < nres && i < (int)sb->Tuple.variables.count; i++) {
				Type *sty = sb->Tuple.variables[i]->type;
				i32 sfoff = (i32)type_offset_of(src, i);
				x64Value fv = x64_load_addr(p, x64addr(x64_mem(tv.mem.base, tv.mem.disp + sfoff), sty));
				x64_store_value(p, x64addr(x64_rbp_mem(roffs[i]), rtyp[i]), fv);
			}
		}
	} else {
		for (int i = 0; i < nres && i < rc; i++) {
			x64Value v = x64_build_expr(p, results[i]);
			x64_store_value(p, x64addr(x64_rbp_mem(roffs[i]), rtyp[i]), v);
		}
	}
	x64_run_deferred_from(p, dbase); // emit (don't pop) the inlined scope's defers
	x64_emit_jmp(&p->asm_, join);
}

gb_internal void x64_build_return_stmt(x64Procedure *p, Slice<Ast *> const &results) {
	if (p->inline_frames.count > 0) { x64_build_inline_return(p, results); return; }

	Type *pt = p->type;
	GB_ASSERT(pt->kind == Type_Proc);

	int res_count = (int)results.count;

	Type *results_tuple = pt->Proc.results;
	int   nres = (results_tuple != nullptr) ? (int)results_tuple->Tuple.variables.count : 0;

	if (res_count == 0) {
		// Bare return: defers may modify the named-return locals — run them FIRST, then
		// write the (possibly modified) locals to the result ABI.
		x64_run_deferred(p);
		x64_emit_named_returns(p);
	} else if (nres <= 1) {
		// Build the return value BEFORE running defers (mirrors lb_build_return_stmt_internal:
		// store the result, THEN lb_emit_defer_stmts). A defer that resets the temp arena or
		// clobbers registers must NOT run before the return expression is evaluated — was the
		// os bug: `return win32_utf16_to_utf8(...)` allocated into an arena the deferred
		// TEMP_ALLOCATOR_GUARD_END had already reset, so the output aliased its input.
		x64Value v  = x64_build_expr(p, results[0]);
		Type    *rt = (nres == 1) ? results_tuple->Tuple.variables[0]->type : nullptr;
		// Store the result into its NAMED result local BEFORE running defers (mirrors LLVM: store the
		// result var, THEN lb_emit_defer_stmts) so pending defers OBSERVE — and may MODIFY — it; the
		// defer-pending marshal paths below then read THIS local. Was a non-debug gap: a single-result
		// `defer if !ok {…}` read the zero-init local (debug masked it — the old code stored here only
		// in debug, for the debugger). Guard on (defers || debug) to keep the no-defer hot path's
		// single copy. Unnamed results have no local. Was [[x64-multireturn-store-before-defers]]'s
		// single-return tail (e.g. json single-result unmarshal helpers built without -debug).
		Entity *re0   = (nres == 1) ? results_tuple->Tuple.variables[0] : nullptr;
		i32    *nloff = (re0 != nullptr && re0->kind == Entity_Variable && re0->token.string.len > 0)
		                ? x64_var_get(&p->var_offsets, re0) : nullptr;
		bool store_named = nloff != nullptr && rt != nullptr && type_size_of(rt) > 0 &&
		                   (p->deferred.count > 0 || p->module->debug_s != nullptr);
		if (store_named) x64_store_value(p, x64addr(x64_rbp_mem(*nloff), rt), v);

		if (rt == nullptr || type_size_of(rt) == 0) {
			x64_run_deferred(p); // void / zero-sized
		} else if (p->returns_by_pointer) {
			i64 rsz = type_size_of(rt);
			if (p->deferred.count == 0) {
				// No defers: copy straight into the caller's sret buffer (fast path). Spill (and
				// convert) first when v isn't already an rt-typed Mem (store_value → emit_conv).
				if (v.kind != x64Value_Mem || (v.type != nullptr && !are_types_identical(v.type, rt))) {
					i32 so = x64_alloc_local(p, rsz, type_align_of(rt));
					x64_store_value(p, x64addr(x64_rbp_mem(so), rt), v);
					v = x64v_mem(rt, x64_rbp_mem(so));
				}
				x64_emit_lea(&p->asm_, X64Reg_RAX, v.mem);
			} else {
				// Defers: source from the named local (defer-visible/mutable; already holds v) or a
				// temp; run defers; THEN copy to sret so defer mutations are reflected.
				i32 so;
				if (nloff != nullptr) { so = *nloff; }
				else { so = x64_alloc_local(p, rsz, type_align_of(rt));
				       x64_store_value(p, x64addr(x64_rbp_mem(so), rt), v); }
				x64_run_deferred(p);
				x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(so));
			}
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(x64_param_rbp_off(0)));
			x64_copy_fixed(p, x64_mem(X64Reg_RDX, 0), x64_mem(X64Reg_RAX, 0), rsz);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0))); // RAX = sret ptr
		} else if (p->deferred.count == 0) {
			// No defers: move directly — but convert to rt first if needed (store_value → emit_conv).
			if (v.type != nullptr && rt != nullptr && !are_types_identical(v.type, rt)) {
				i64 rsz = type_size_of(rt);
				i32 so = x64_alloc_local(p, rsz, type_align_of(rt));
				x64_store_value(p, x64addr(x64_rbp_mem(so), rt), v);
				v = x64v_mem(rt, x64_rbp_mem(so));
			}
			if (x64_is_float(rt)) x64_value_to_xmm(p, v, X64XmmReg_XMM0);
			else                  x64_value_to_reg(p, v, X64Reg_RAX);
		} else {
			// Defers pending: load from the named local (defers may have updated it; already holds v)
			// or, for an unnamed result, spill to a temp that survives them.
			i64 rsz = type_size_of(rt);
			i32 so;
			if (nloff != nullptr) { so = *nloff; }
			else { so = x64_alloc_local(p, rsz, type_align_of(rt));
			       x64_store_value(p, x64addr(x64_rbp_mem(so), rt), v); }
			x64_run_deferred(p);
			if (x64_is_float(rt)) {
				if (x64_is_double(rt)) x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(so));
				else                    x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(so));
			} else {
				x64_value_to_reg(p, x64v_mem(rt, x64_rbp_mem(so)), X64Reg_RAX);
			}
		}
	} else {
		// Split returns (mirror LLVM): results[0..N-2] go through hidden pointer args,
		// result[N-1] in sret/RAX. Build all N into stack temps first (building clobbers
		// RAX and the hidden-ptr regs; the ptrs survive in their homed shadow slots).
		Array<i32>   toff; array_init(&toff, temporary_allocator(), nres, nres);
		Array<Type*> ttyp; array_init(&ttyp, temporary_allocator(), nres, nres);
		for (int i = 0; i < nres; i++) ttyp[i] = results_tuple->Tuple.variables[i]->type;

		if (res_count == 1) {
			// `return multi_valued_call()` — one expression yields the whole tuple. The call's
			// result types may DIFFER from this proc's (implicit per-element conversion, e.g.
			// `(string, Allocator_Error)` → `(string, os.Error)`), so read each field at the
			// SOURCE tuple's offset/type and store it as the DEST type via x64_store_value,
			// which applies the variant→union (and other) conversions.
			x64Value tv = x64_build_expr(p, results[0]);
			Type *src_tuple = (tv.type != nullptr) ? tv.type : results_tuple;
			// Always spill the source tuple to an rbp-relative temp first: per-field reads below
			// must be stable, but tv.mem.base may be a volatile reg that store_value/copy_fixed
			// clobbers between fields.
			i32 so = x64_alloc_local(p, type_size_of(src_tuple), type_align_of(src_tuple));
			x64_store_value(p, x64addr(x64_rbp_mem(so), src_tuple), tv);
			Type *src_base = base_type(src_tuple);
			TypeTuple *stup = (src_base != nullptr && src_base->kind == Type_Tuple) ? &src_base->Tuple : nullptr;
			i64 sfoff = 0;
			for (int i = 0; i < nres; i++) {
				Type *sty = (stup != nullptr && i < (int)stup->variables.count) ? stup->variables[i]->type : ttyp[i];
				i64 ssz = type_size_of(sty);    if (ssz <= 0) ssz = 1;
				i64 sal = type_align_of(sty);   if (sal <= 0) sal = 1;
				sfoff = (sfoff + (sal - 1)) & ~(sal - 1);
				i64 dsz = type_size_of(ttyp[i]); if (dsz <= 0) dsz = 1;
				i64 dal = type_align_of(ttyp[i]); if (dal <= 0) dal = 1;
				// Store into the NAMED result local (so pending defers observe/modify it,
				// mirroring LLVM), else a fresh temp. See the per-expr branch below.
				Entity *re   = results_tuple->Tuple.variables[i];
				i32    *nloff = (re->kind == Entity_Variable && re->token.string.len > 0)
				                ? x64_var_get(&p->var_offsets, re) : nullptr;
				i32 t = (nloff != nullptr) ? *nloff : x64_alloc_local(p, dsz, dal); toff[i] = t;
				x64Value fv = x64_load_addr(p, x64addr(x64_rbp_mem((i32)(so + sfoff)), sty));
				x64_store_value(p, x64addr(x64_rbp_mem(t), ttyp[i]), fv);
				sfoff += ssz;
			}
		} else {
			// `return v0, v1, ...` — one expression per result. Store each into its NAMED result
			// local (when one exists) so pending defers OBSERVE — and may MODIFY — them before the
			// ABI write, mirroring LLVM (which stores into the result vars then lb_emit_defer_stmts).
			// Fresh temps don't reach the named locals → a `defer if !ok {…}` saw the zero-init ok
			// after `return true, nil` (THE json unmarshal_string_token bug: it freed the stored
			// string). The marshal loop below reads toff[i], so named-local writes flow through.
			for (int i = 0; i < nres && i < res_count; i++) {
				i64 fsz = type_size_of(ttyp[i]); if (fsz <= 0) fsz = 1;
				i64 fal = type_align_of(ttyp[i]); if (fal <= 0) fal = 1;
				x64Value v = x64_build_expr(p, results[i]);
				Entity *re   = results_tuple->Tuple.variables[i];
				i32    *nloff = (re->kind == Entity_Variable && re->token.string.len > 0)
				                ? x64_var_get(&p->var_offsets, re) : nullptr;
				i32 t = (nloff != nullptr) ? *nloff : x64_alloc_local(p, fsz, fal); toff[i] = t;
				x64_store_value(p, x64addr(x64_rbp_mem(t), ttyp[i]), v);
			}
		}

		x64_run_deferred(p); // defers AFTER results stored into their (named) slots, before ABI write

		for (int i = 0; i < nres; i++) {
			i64 fsz = type_size_of(ttyp[i]);
			bool is_last = (i == nres - 1);
			if (!is_last) {
				if (fsz == 0) continue;
				int s = p->first_partial_ret_slot + i;
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(s)));
				x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(toff[i]), fsz);
			} else if (fsz == 0) {
				// zero-sized last result: nothing
			} else if (p->returns_by_pointer) {
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
				x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(toff[i]), fsz);
			} else if (x64_is_float(ttyp[i])) {
				if (x64_is_double(ttyp[i])) x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(toff[i]));
				else                         x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(toff[i]));
			} else {
				x64_value_to_reg(p, x64v_mem(ttyp[i], x64_rbp_mem(toff[i])), X64Reg_RAX);
			}
		}
	}

	x64_proc_emit_epilogue(p);
	x64_emit_ret(&p->asm_);
}

// Build a list of statements, reclaiming each statement's throwaway temp slots the
// moment it ends — a "brain-dead" stack-slot reuse that keeps frames small (x64 emits
// unoptimized code, so unlike LLVM there's no later pass to color/reuse stack slots).
// A statement's temps are dead at its end UNLESS it (a) registered a defer whose
// captured arg values must outlive it (detected via p->deferred.count) or (b) allocated
// a scope-lived slot — a named local or the scoped/generated `context` (detected via
// p->named_seq). In those cases we keep the cursor so neither defer temps nor
// scope-lived slots are ever reused (debug info stays exact). frame_max already retains
// the peak for the prologue.
gb_internal void x64_build_stmt_list(x64Procedure *p, Slice<Ast *> const &stmts) {
	for_array(i, stmts) {
		i32   lmark = p->local_size;
		isize dmark = p->deferred.count;
		u32   nmark = p->named_seq;
		x64_build_stmt(p, stmts[i]);
		if (p->deferred.count == dmark && p->named_seq == nmark) {
			p->local_size = lmark;
		}
	}
}

// All Odin for-range forms (mirrors lb_build_range_stmt + its lb_build_range_* helpers): integer
// interval, Array/EnumeratedArray, Enum, Slice/DynamicArray, FixedCapacityDynamicArray, #soa struct,
// BitSet, iterator-proc (Tuple), Map, string/string16 runes, and by-reference (`for &x`).
// Shared range-loop context: the loop labels + index slot and the (already-bound) element/index range
// vars, populated by the x64_build_range_stmt dispatcher and threaded into each per-kind sub-builder
// below (mirrors how lb_build_range_stmt passes val_/idx_/loop_/done_ to its lb_build_range_* helpers).
// lbl_end + pop_loop are bound by the dispatcher AFTER the sub-builder returns; sub-builders end with
// `jmp lbl_loop` and never bind lbl_end.
struct X64RangeStmt {
	Ast    *body;
	Ast    *iter_expr;
	Type   *iter_type;
	isize   lbl_loop, lbl_post, lbl_end;
	i32     loop_idx_off;
	Entity *elem_e; i32 elem_off; bool elem_by_ref;
	Entity *ridx_e; i32 ridx_off;
	bool    reverse; // `#reverse for` — iterate the effective index high→low
};

// Integer interval `lo ..< hi` / `lo ..= hi` (mirrors lb_build_range_interval). iter_type is the
// element type, not an aggregate. The upper bound is re-evaluated each iteration (matches LLVM).
gb_internal void x64_build_range_interval(x64Procedure *p, X64RangeStmt *c) {
	ast_node(be, BinaryExpr, c->iter_expr);
	bool inclusive = (be->op.kind == Token_RangeFull || be->op.kind == Token_Ellipsis);
	// vsz/vsigned MUST come from the value var's type (when it exists) — the value var IS c->elem_e and
	// the loop loads/stores/increments it in its slot. The range expr's type can be wider (e.g. `int`)
	// than a `i32`/`u32` bound's element var; using that width made an 8-byte access of the 4-byte elem
	// slot OVERRUN into the adjacent index slot → the index increment clobbered the counter's high bits
	// → it jumped huge and the loop exited after 1 iteration. (THE blick icon-load bug: a `for i in
	// 0..<n` over an i32 length in wstring_to_utf8's NUL-trim ran once → paths kept the trailing NUL.)
	Type *vt = c->elem_e != nullptr ? x64_typed(c->elem_e->type)
	         : (c->iter_expr->tav.type ? x64_typed(c->iter_expr->tav.type) : t_int);
	X64OpSize vsz = x64_op_size_of(vt);
	bool vsigned = x64_is_signed_integer(vt);

	i32 value_off = c->elem_e ? c->elem_off : x64_alloc_local(p, 8, 8);
	i32 index_off = c->ridx_e ? c->ridx_off : c->loop_idx_off; // loop_idx_off already 0
	if (c->ridx_e) x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(index_off), 0);

	x64Value lo = x64_build_expr(p, be->left); // value = lower
	x64_value_to_reg(p, lo, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, vsz, x64_rbp_mem(value_off), X64Reg_RAX);

	x64_label_bind(&p->asm_, c->lbl_loop);
	x64Value hi = x64_build_expr(p, be->right);
	x64_value_to_reg(p, hi, X64Reg_RCX);
	x64_emit_mov_rm(&p->asm_, vsz, X64Reg_RAX, x64_rbp_mem(value_off));
	x64_emit_cmp_rr(&p->asm_, vsz, X64Reg_RAX, X64Reg_RCX);
	X64Cc cc_end = inclusive ? (vsigned ? X64Cc_G  : X64Cc_A)
	                         : (vsigned ? X64Cc_GE : X64Cc_AE);
	x64_emit_jcc(&p->asm_, cc_end, c->lbl_end);

	x64_build_stmt(p, c->body);

	x64_label_bind(&p->asm_, c->lbl_post);
	x64_emit_mov_rm(&p->asm_, vsz, X64Reg_RAX, x64_rbp_mem(value_off));
	x64_emit_inc_r(&p->asm_, vsz, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, vsz, x64_rbp_mem(value_off), X64Reg_RAX);
	if (index_off != value_off) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(index_off));
		x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(index_off), X64Reg_RAX);
	}
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// Indexed range over array / enumerated-array / slice / dynamic-array / fixed-capacity-array (mirrors
// lb_build_range_indexed). `data_off` holds a pointer to element 0; the length is either a constant
// (`len_is_const`) or read from `len_off`. `idx_min` shifts the index var for enumerated arrays
// (enum value = ordinal + min). The dispatcher does the per-kind data/len extraction, then calls this.
gb_internal void x64_build_range_indexed(x64Procedure *p, X64RangeStmt *c, i32 data_off, i32 len_off,
                                         i64 len_const, bool len_is_const, i64 elem_sz, Type *elem_t,
                                         i64 idx_min) {
	// `#reverse for`: keep loop_idx as a 0..len counter (termination), but the EFFECTIVE index used
	// for the element + index var is `len-1 - counter` (mirrors lb_build_range_indexed reverse path).
	i32 eff_off = x64_alloc_local(p, 8, 8);
	x64_label_bind(&p->asm_, c->lbl_loop);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	if (len_is_const) x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)len_const);
	else              x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
	x64_emit_jcc(&p->asm_, X64Cc_AE, c->lbl_end);

	if (c->reverse) {
		// RCX = (len-1) - counter
		if (len_is_const) x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, len_const - 1);
		else { x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(len_off)); x64_emit_dec_r(&p->asm_, X64OpSize_64, X64Reg_RCX); }
		x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(eff_off), X64Reg_RCX);
	} else {
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(eff_off), X64Reg_RAX);
	}
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(eff_off)); // RAX = effective index

	if (c->ridx_e) {
		if (idx_min != 0) {
			x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)idx_min);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->ridx_off), X64Reg_RAX);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(eff_off));
		} else {
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->ridx_off), X64Reg_RAX);
		}
	}

	if (c->elem_e) {
		x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX, (i32)elem_sz);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(data_off));
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		if (c->elem_by_ref) {
			// &x: bind the element's ADDRESS (RAX) into the pointer var.
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->elem_off), X64Reg_RAX);
		} else {
			x64Value ev = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), elem_t));
			x64_store_value(p, x64addr(x64_rbp_mem(c->elem_off), elem_t), ev);
		}
	}

	x64_build_stmt(p, c->body);

	x64_label_bind(&p->asm_, c->lbl_post);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// `for x in EnumType` (mirrors lb_build_range_enum): enums may be non-contiguous, so materialize the
// member values on the stack and index them. vals[0] = enum value, vals[1] = ordinal index.
gb_internal void x64_build_range_enum(x64Procedure *p, X64RangeStmt *c) {
	Type     *iter_type = c->iter_type;
	i64       count = iter_type->Enum.fields.count;
	Type     *bvt   = core_type(iter_type);          // integer backing type
	X64OpSize vsz   = x64_op_size_of(bvt);
	i64       esz   = type_size_of(bvt); if (esz <= 0) esz = 1;
	i32 vals_off = x64_alloc_local(p, count * esz, esz);
	for (i64 k = 0; k < count; k++) {
		Entity *f = iter_type->Enum.fields[k];
		i64 v = (f != nullptr && f->kind == Entity_Constant) ? exact_value_to_i64(f->Constant.value) : 0;
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, v);
		x64_emit_mov_mr(&p->asm_, vsz, x64_rbp_mem(vals_off + (i32)(k * esz)), X64Reg_RCX);
	}

	x64_label_bind(&p->asm_, c->lbl_loop);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)count);
	x64_emit_jcc(&p->asm_, X64Cc_AE, c->lbl_end);

	if (c->ridx_e) x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->ridx_off), X64Reg_RAX);

	if (c->elem_e) {
		// elem = values[idx]: &vals[0] + idx*esz
		x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX, (i32)esz);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(vals_off));
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64Value ev = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), iter_type));
		x64_store_value(p, x64addr(x64_rbp_mem(c->elem_off), iter_type), ev);
	}

	x64_build_stmt(p, c->body);

	x64_label_bind(&p->asm_, c->lbl_post);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// #soa range (mirrors lb_build_range_stmt_struct_soa but DIVERGES: LLVM binds `field` as a soa variable
// with selector redirection field.x → array.x[i]; we MATERIALIZE the element struct each iteration
// (elem.x = array.x[i]). Identical for the value form; only `&field` write-back would differ, which
// #soa value-range never does. Layout: fields[0..n-1] = per-member multipointers, field[n] = __$len.
gb_internal void x64_build_range_stmt_struct_soa(x64Procedure *p, X64RangeStmt *c) {
	Type *iter_type = c->iter_type;
	Type *elem_bt = base_type(iter_type->Struct.soa_elem);
	type_set_offsets(iter_type);
	if (elem_bt != nullptr) type_set_offsets(elem_bt);
	int n = (elem_bt != nullptr && elem_bt->kind == Type_Struct)
	      ? (int)elem_bt->Struct.fields.count : 0;

	// The #soa value may come from a call (sret) — build once, pin its address.
	x64Value sv = x64_build_expr(p, c->iter_expr);
	if (sv.kind != x64Value_Mem) {
		i32 tmp = x64_alloc_local(p, type_size_of(iter_type), type_align_of(iter_type));
		x64_store_value(p, x64addr(x64_rbp_mem(tmp), iter_type), sv);
		sv = x64v_mem(iter_type, x64_rbp_mem(tmp));
	}
	i32 soa_ea = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, sv.mem);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(soa_ea), X64Reg_RAX);

	// len: __$len field for slice/dynamic, soa_count for fixed (lb_soa_struct_len).
	i32 len_off = x64_alloc_local(p, 8, 8);
	if (iter_type->Struct.soa_kind == StructSoa_Fixed) {
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)iter_type->Struct.soa_count);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RAX);
	} else {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(soa_ea));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX,
		                x64_mem(X64Reg_RAX, (i32)iter_type->Struct.offsets[n]));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);
	}

	// #reverse: index runs len-1 down to 0.
	if (c->reverse) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
		x64_emit_dec_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
	}

	x64_label_bind(&p->asm_, c->lbl_loop);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	if (c->reverse) {
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
		x64_emit_jcc(&p->asm_, X64Cc_L, c->lbl_end);
	} else {
		x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
		x64_emit_jcc(&p->asm_, X64Cc_AE, c->lbl_end);
	}

	if (c->ridx_e) x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->ridx_off), X64Reg_RAX);

	if (c->elem_e && elem_bt != nullptr) {
		for (int f = 0; f < n; f++) {
			Type *ft  = elem_bt->Struct.fields[f]->type;
			i64   fsz = type_size_of(ft);
			if (fsz <= 0) continue;
			i32 dst_off = c->elem_off + (i32)elem_bt->Struct.offsets[f];
			// src = soa.field[f] (a [^]ft) + index*fsz
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(c->loop_idx_off));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(soa_ea));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX,
			                x64_mem(X64Reg_RAX, (i32)iter_type->Struct.offsets[f]));
			if (fsz != 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RDX, (i32)fsz);
			x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
			x64_copy_fixed(p, x64_rbp_mem(dst_off), x64_mem(X64Reg_RCX, 0), fsz);
		}
	}

	x64_build_stmt(p, c->body);

	x64_label_bind(&p->asm_, c->lbl_post);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	if (c->reverse) x64_emit_dec_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
	else            x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// Iterator-proc range `for v, i in f(&it)` (mirrors lb_build_range_tuple): f returns (vals…, ok: bool).
// The call is re-run each iteration to advance the iterator; the LAST tuple field is the loop condition,
// preceding fields are the values.
gb_internal void x64_build_range_tuple(x64Procedure *p, X64RangeStmt *c) {
	Type *iter_type = c->iter_type;
	int tcount = (int)iter_type->Tuple.variables.count;

	x64_label_bind(&p->asm_, c->lbl_loop);
	x64Value tup = x64_build_expr(p, c->iter_expr);   // re-call each iteration → tuple buffer
	i32 tup_off  = x64_alloc_local(p, 8, 8);          // pin base (field reads/body clobber regs)
	x64_emit_lea(&p->asm_, X64Reg_RAX, tup.mem);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(tup_off), X64Reg_RAX);

	// Condition = last field (ok: bool) — end the loop when false. The tuple buffer is reached
	// through the pinned ADDRESS in tup_off (body/field reads clobber regs), so reload it into RAX
	// before each field; x64_emit_tuple_ev then uses the canonical type_offset_of.
	Type *cond_t = iter_type->Tuple.variables[tcount-1]->type;
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(tup_off));
	x64Value cond_v = x64_emit_tuple_ev(p, x64v_mem(iter_type, x64_mem(X64Reg_RAX, 0)), tcount-1);
	x64_value_to_reg(p, cond_v, X64Reg_RCX);
	X64OpSize cond_sz = x64_op_size_of(cond_t);
	x64_emit_test_rr(&p->asm_, cond_sz, X64Reg_RCX, X64Reg_RCX);
	x64_emit_jcc(&p->asm_, X64Cc_E, c->lbl_end);

	// Bind value (field 0) and optional index (field 1).
	if (c->elem_e) {
		Type *ft = iter_type->Tuple.variables[0]->type;
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(tup_off));
		x64Value fv = x64_emit_tuple_ev(p, x64v_mem(iter_type, x64_mem(X64Reg_RAX, 0)), 0);
		x64_store_value(p, x64addr(x64_rbp_mem(c->elem_off), ft), fv);
	}
	if (c->ridx_e && tcount >= 2) {
		Type *ft = iter_type->Tuple.variables[1]->type;
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(tup_off));
		x64Value fv = x64_emit_tuple_ev(p, x64v_mem(iter_type, x64_mem(X64Reg_RAX, 0)), 1);
		x64_store_value(p, x64addr(x64_rbp_mem(c->ridx_off), ft), fv);
	}

	x64_build_stmt(p, c->body);
	x64_label_bind(&p->asm_, c->lbl_post);
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// Hash-table iteration (mirrors lb_build_range_map): scan cell slots 0..capacity, skipping
// empty/tombstone slots. For maps vals[0]=KEY, vals[1]=VALUE. cap/ks/vs/hs reloaded each iter
// (body may mutate the map).
gb_internal void x64_build_range_map(x64Procedure *p, X64RangeStmt *c) {
	Type *iter_type = c->iter_type;
	Type *kt = iter_type->Map.key;
	Type *vt = iter_type->Map.value;
	x64Value map_ptr = x64_map_addr_of(p, c->iter_expr, iter_type);

	x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), -1); // index = -1

	x64_label_bind(&p->asm_, c->lbl_loop);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX); // idx++
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);

	x64Value map_value = x64_map_load_struct(p, map_ptr, iter_type);
	x64Value cap = x64_map_cap(p, map_value);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, cap.mem);
	x64_emit_jcc(&p->asm_, X64Cc_AE, c->lbl_end); // idx >= cap → done

	x64Value ks = x64_map_data_uintptr(p, map_value);
	x64Value vs = x64_map_cell_index_static(p, kt, ks, cap); // start of value cells
	x64Value hs = x64_map_cell_index_static(p, vt, vs, cap); // start of hash cells (packed)

	// hash = *(hs + idx*8); skip slot if empty/tombstone.
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	x64_emit_shl_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 3);
	x64_emit_add_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, hs.mem);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 0));
	x64Value valid = x64_map_hash_is_valid(p, x64v_reg(t_uintptr, X64Reg_RAX));
	x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_RAX, valid.mem);
	x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	x64_emit_jcc(&p->asm_, X64Cc_E, c->lbl_loop); // invalid → next slot

	x64Value idxv = x64v_mem(t_int, x64_rbp_mem(c->loop_idx_off));
	if (c->elem_e) { // vals[0] = key
		x64Value kp = x64_map_cell_index_static(p, kt, ks, idxv);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, kp.mem);
		x64Value kv = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), kt));
		x64_store_value(p, x64addr(x64_rbp_mem(c->elem_off), kt), kv);
	}
	if (c->ridx_e) { // vals[1] = value
		x64Value vp = x64_map_cell_index_static(p, vt, vs, idxv);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, vp.mem);
		x64Value vv = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), vt));
		x64_store_value(p, x64addr(x64_rbp_mem(c->ridx_off), vt), vv);
	}

	x64_build_stmt(p, c->body);
	x64_label_bind(&p->asm_, c->lbl_post);
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// `for r, i in str` (mirrors lb_build_range_string / _string16): decode runes via
// runtime.{string,string16}_decode_rune(str[offset:]) → (rune, width). vals[0] = rune, vals[1] = BYTE
// offset of the rune's start; loop_idx_off doubles as the byte offset (advances by width, not 1).
gb_internal void x64_build_range_string(x64Procedure *p, X64RangeStmt *c) {
	Type *iter_type = c->iter_type;
	bool    is16   = is_type_string16(iter_type);
	bool    rev    = c->reverse; // `#reverse for r in str`: decode from the end via *_decode_last_rune
	String  decode = rev
	               ? (is16 ? str_lit("string16_decode_last_rune") : str_lit("string_decode_last_rune"))
	               : (is16 ? str_lit("string16_decode_rune")      : str_lit("string_decode_rune"));
	AstPackage *rt = p->module->gen->info->runtime_package;
	Entity *de = (rt != nullptr) ? scope_lookup_current(rt->scope, string_interner_insert(decode)) : nullptr;

	// A `^string` operand (`for r in ps`, ps: ^string) ranges over the pointee: the string header's
	// address is the pointer VALUE, not &pointer (mirrors x64_range_base_to_rax's via_ptr path). Was
	// core:strings.fields_iterator's `for r,offset in s` (s: ^string) reading {data,len} from &s →
	// garbage len → runaway loop + out-of-bounds field slices. THE core:image/netpbm PBM/PFM load bug.
	Type *iter_raw = c->iter_expr->tav.type ? base_type(x64_typed(c->iter_expr->tav.type)) : nullptr;
	bool via_ptr = iter_raw != nullptr && iter_raw->kind == Type_Pointer;
	i32 data_off  = x64_alloc_local(p, 8, 8);
	i32 len_off   = x64_alloc_local(p, 8, 8);
	if (via_ptr) {
		x64_value_to_reg(p, x64_build_expr(p, c->iter_expr), X64Reg_RAX); // RAX = pointer value = &string
	} else {
		x64Addr str_addr = x64_build_addr(p, c->iter_expr);
		x64_emit_lea(&p->asm_, X64Reg_RAX, str_addr.mem);
	}
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 0));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_off), X64Reg_RCX);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);

	i32 sub_off   = x64_alloc_local(p, 16, 8); // substring {ptr,len}
	i32 subp_off  = x64_alloc_local(p, 8, 8);  // &substring
	i32 rune_off  = x64_alloc_local(p, 8, 8);  // rune out (partial-return)
	i32 runep_off = x64_alloc_local(p, 8, 8);  // &rune
	i32 width_off = x64_alloc_local(p, 8, 8);  // width (survives body)

	if (de != nullptr && de->kind == Entity_Procedure &&
	    !de->Procedure.is_foreign &&
	    de->min_dep_count.load(std::memory_order_relaxed) == 0) {
		x64_enqueue_oncall(p, de);
	}

	// #reverse: the offset (loop_idx_off) starts at len and walks DOWN to 0.
	if (rev) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
	}

	x64_label_bind(&p->asm_, c->lbl_loop);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
	if (rev) {
		// while offset > 0
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
		x64_emit_jcc(&p->asm_, X64Cc_LE, c->lbl_end);
	} else {
		x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
		x64_emit_jcc(&p->asm_, X64Cc_AE, c->lbl_end);
	}

	if (rev) {
		// substring.data = data ; substring.len = offset  (str[:offset]) — RAX holds offset
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(data_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sub_off), X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sub_off + 8), X64Reg_RAX);
	} else {
		if (c->ridx_e) x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->ridx_off), X64Reg_RAX);
		// substring.data = data + offset ; substring.len = len - offset  (str[offset:])
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(data_off));
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sub_off), X64Reg_RCX);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(len_off));
		x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sub_off + 8), X64Reg_RCX);
	}
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(sub_off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(subp_off), X64Reg_RAX);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(rune_off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(runep_off), X64Reg_RAX);

	if (de != nullptr && de->kind == Entity_Procedure) {
		// args: [string param (16B → indirect ptr)] [rune partial-return ptr]; width in RAX.
		x64Value cargs[2];
		cargs[0] = x64v_mem(t_rawptr, x64_rbp_mem(subp_off));
		cargs[1] = x64v_mem(t_rawptr, x64_rbp_mem(runep_off));
		// Multi-result proc → x64_emit_call returns none, but the LAST result (width, int) is left in
		// RAX. Read RAX directly (passing the none value through value_to_reg would zero RAX → width 0
		// → infinite loop). Guard width≥1 so it can never hang.
		x64_emit_call(p, x64_get_entity_name(de), de->type, cargs, 2);
		isize w_ok = x64_label_alloc(&p->asm_);
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
		x64_emit_jcc(&p->asm_, X64Cc_NE, w_ok);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
		x64_label_bind(&p->asm_, w_ok);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(width_off), X64Reg_RAX);
	} else {
		// No decoder available (shouldn't happen): advance by 1 to avoid an infinite loop.
		x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(width_off), 1);
	}

	// #reverse: offset -= width NOW (before the body), and the index var = the new offset.
	if (rev) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(width_off));
		x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
		if (c->ridx_e) x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->ridx_off), X64Reg_RAX);
	}

	if (c->elem_e) {
		x64Value rv = x64_load_addr(p, x64addr(x64_rbp_mem(rune_off), t_rune));
		x64_store_value(p, x64addr(x64_rbp_mem(c->elem_off), t_rune), rv);
	}

	x64_build_stmt(p, c->body);

	x64_label_bind(&p->asm_, c->lbl_post);
	if (!rev) {
		// forward: offset += width
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(c->loop_idx_off));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(width_off));
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(c->loop_idx_off), X64Reg_RAX);
	}
	x64_emit_jmp(&p->asm_, c->lbl_loop);
}

// Address of a for-range source. An addressable source (a variable/field/deref) gives its address
// directly; a non-addressable R-VALUE (slice/array compound literal, a call result) is materialized to
// a temp first — else x64_build_addr returns a garbage/zero header and the loop reads len==0 (was
// `for x in []T{…}` / `for x in f()` iterating 0 times).
gb_internal x64Addr x64_range_source_addr(x64Procedure *p, Ast *iter_expr, Type *iter_type) {
	Ast *e = unparen_expr(iter_expr);
	if (e->tav.mode == Addressing_Variable) {
		return x64_build_addr(p, iter_expr);
	}
	// Non-addressable r-value (compound lit, call result): build the value. A slice/array/dynarray
	// literal builds into a stack temp returned as Mem — use that header directly (its .len is set).
	x64Value sv = x64_build_expr(p, iter_expr);
	if (sv.kind == x64Value_Mem) return x64addr(sv.mem, iter_type);
	return x64addr(x64_spill_value(p, sv, iter_type).mem, iter_type);
}

// Puts the ADDRESS of the aggregate to range over into RAX. `via_ptr`: the source expr is a POINTER
// to the aggregate (`for x in p`, p: ^[]T/^[N]T) — its VALUE is the address, so load it; otherwise
// LEA the aggregate's own storage. Mirrors LLVM lb_build_addr_ptr + the is_type_pointer(type_deref)
// load in lb_build_range_stmt.
gb_internal void x64_range_base_to_rax(x64Procedure *p, Ast *iter_expr, Type *iter_type, bool via_ptr) {
	if (via_ptr) {
		x64_value_to_reg(p, x64_build_expr(p, iter_expr), X64Reg_RAX);
		return;
	}
	x64Addr a = x64_range_source_addr(p, iter_expr, iter_type);
	x64_emit_lea(&p->asm_, X64Reg_RAX, a.mem);
}

gb_internal void x64_build_range_stmt(x64Procedure *p, Ast *stmt) {
	ast_node(rs, RangeStmt, stmt);
		// Strip parens: a `ParenExpr` node carries NO tav.type (the checker records the type on the
		// inner operand), so reading `rs->expr->tav.type` on `for x in (s)` / `for x in ([]T{…})` gave
		// nullptr → iter_type nullptr → the body was SKIPPED entirely (0 iterations, for BOTH slices and
		// arrays). The value/addr builders unparen internally; the TYPE reads below did not. Mirrors LLVM
		// lb_build_range_stmt using type_of_expr(expr) (which sees through parens).
		Ast  *iter_expr = unparen_expr(rs->expr);
		// A pointer source (`for x in p`, p: ^[]T/^[N]T/^map) ranges over the pointee: deref for the
		// kind dispatch, remember via_ptr so the base address is the pointer VALUE (mirrors LLVM's
		// base_type(type_deref(expr_type)) + is_type_pointer load).
		Type *iter_raw  = iter_expr->tav.type ? base_type(x64_typed(iter_expr->tav.type)) : nullptr;
		bool  iter_via_ptr = iter_raw != nullptr && iter_raw->kind == Type_Pointer;
		Type *iter_type = iter_expr->tav.type ? base_type(type_deref(x64_typed(iter_expr->tav.type))) : nullptr;

		isize lbl_loop = x64_label_alloc(&p->asm_);
		isize lbl_post = x64_label_alloc(&p->asm_);
		isize lbl_end  = x64_label_alloc(&p->asm_);
		x64_push_loop(p, lbl_end, lbl_post, rs->label);

		i32 loop_idx_off = x64_alloc_local(p, 8, 8);
		x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(loop_idx_off), 0);

		// Odin RangeStmt ordering: vals[0] = element value, vals[1] = index.
		Entity *elem_e = nullptr; i32 elem_off = 0; bool elem_by_ref = false;
		Entity *ridx_e = nullptr; i32 ridx_off = 0;

		if (rs->vals.count > 0) {
			Ast *v = x64_range_strip(rs->vals[0]);
			if (v && !x64_is_blank(v)) {
				elem_e = entity_of_node(v);
				if (elem_e && elem_e->kind == Entity_Variable) {
					if ((elem_e->flags & EntityFlag_Value) == 0) {
						// `for &x in …`: x is a by-reference lvalue (type is the element type, used
						// directly — `x = …`, not `x^`). Bind it as an INDIRECT local: an 8-byte slot
						// holding the element ADDRESS; x64_entity_addr derefs it. Mirrors
						// lb_store_range_stmt_val's reference path (lb_add_entity(e, ptr)).
						elem_by_ref = true;
						elem_off = x64_alloc_local(p, 8, 8);
						x64_var_set(&p->var_offsets, elem_e, elem_off);
						array_add(&p->indirect_params, elem_e);
					} else {
						elem_off = x64_alloc_var(p, elem_e);
					}
				} else {
					elem_e = nullptr;
				}
			}
		}
		if (rs->vals.count > 1) {
			Ast *v = x64_range_strip(rs->vals[1]);
			if (v && !x64_is_blank(v)) {
				ridx_e = entity_of_node(v);
				if (ridx_e && ridx_e->kind == Entity_Variable) ridx_off = x64_alloc_var(p, ridx_e);
				else ridx_e = nullptr;
			}
		}

		X64RangeStmt c = {rs->body, iter_expr, iter_type, lbl_loop, lbl_post, lbl_end,
		                  loop_idx_off, elem_e, elem_off, elem_by_ref, ridx_e, ridx_off, rs->reverse};

		// Integer interval `lo ..< hi` / `lo ..= hi`: iter_type is the element type, not an aggregate,
		// so it must be tested before the aggregate dispatch. lbl_end + pop_loop are at the shared tail.
		if (is_ast_range(iter_expr)) {
			x64_build_range_interval(p, &c);
		} else if (iter_type == nullptr) {
			// Unknown range type — skip body, bind labels (lbl_end bound by the shared tail)
			x64_label_bind(&p->asm_, lbl_loop);
			x64_label_bind(&p->asm_, lbl_post);
		} else if (iter_type->kind == Type_Array || iter_type->kind == Type_EnumeratedArray) {
			// Enumerated arrays share fixed-array storage; the index var is the ordinal plus the enum min.
			bool  is_enum = iter_type->kind == Type_EnumeratedArray;
			i64   arr_len = is_enum ? iter_type->EnumeratedArray.count : iter_type->Array.count;
			Type *elem_t  = is_enum ? iter_type->EnumeratedArray.elem  : iter_type->Array.elem;
			i64   idx_min = (is_enum && iter_type->EnumeratedArray.min_value)
			              ? exact_value_to_i64(*iter_type->EnumeratedArray.min_value) : 0;
			i64   elem_sz = type_size_of(elem_t);
			// Pin the array base to a stack slot: the address may be RAX-relative
			// (globals via LEA [RIP+sym]) and the in-loop index load clobbers RAX.
			i32 arr_ptr_off = x64_alloc_local(p, 8, 8);
			x64_range_base_to_rax(p, iter_expr, iter_type, iter_via_ptr);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(arr_ptr_off), X64Reg_RAX);
			x64_build_range_indexed(p, &c, arr_ptr_off, 0, arr_len, true, elem_sz, elem_t, idx_min);

		} else if (iter_type->kind == Type_Enum) {
			x64_build_range_enum(p, &c);

		} else if (iter_type->kind == Type_Slice || iter_type->kind == Type_DynamicArray) {
			Type *elem_t  = (iter_type->kind == Type_Slice)
			              ? iter_type->Slice.elem
			              : iter_type->DynamicArray.elem;
			i64   elem_sz = type_size_of(elem_t);

			i32 data_off = x64_alloc_local(p, 8, 8);
			i32 len_off  = x64_alloc_local(p, 8, 8);

			// Pin the slice struct addr in RAX, read .data (off 0)/.len (off 8) via RCX.
			// The base may be register-relative (e.g. `data.loggers`, `data` a ^T in RAX) —
			// reading .data straight into RAX would clobber the base before .len@+8.
			x64_range_base_to_rax(p, iter_expr, iter_type, iter_via_ptr);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 0));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_off), X64Reg_RCX);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);
			x64_build_range_indexed(p, &c, data_off, len_off, 0, false, elem_sz, elem_t, 0);

		} else if (iter_type->kind == Type_FixedCapacityDynamicArray) {
			// Inline {data: [N]E @0, len: int after the data array}: data ptr = &value, no ptr load.
			Type *elem_t  = iter_type->FixedCapacityDynamicArray.elem;
			i64   elem_sz = type_size_of(elem_t);
			i64   len_fld = x64_fca_len_offset(iter_type);

			i32 data_off = x64_alloc_local(p, 8, 8);
			i32 len_off  = x64_alloc_local(p, 8, 8);

			x64_range_base_to_rax(p, iter_expr, iter_type, iter_via_ptr);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_off), X64Reg_RAX);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)len_fld));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);
			x64_build_range_indexed(p, &c, data_off, len_off, 0, false, elem_sz, elem_t, 0);

		} else if (iter_type->kind == Type_Struct && iter_type->Struct.soa_kind != StructSoa_None) {
			x64_build_range_stmt_struct_soa(p, &c);

		} else if (iter_type->kind == Type_BitSet) {
			// Iterate set bits low→high (mirrors lb_build_range_stmt Type_BitSet): remaining =
			// set & valid-mask; each step the element = ctz(remaining)+lower, then clear the
			// lowest set bit (remaining &= remaining-1). bit_set is an integer ≤8 bytes.
			// #reverse iterates set bits high→low: reverse_bits(remaining) over the backing-int
			// width, then ctz finds the highest original bit first; element = (lower+bits-1) - ctz.
			i64 lower = iter_type->BitSet.lower;
			i64 nbits = iter_type->BitSet.upper - lower + 1; if (nbits < 0) nbits = 0;
			i64 mask_bits = 8 * type_size_of(iter_type);
			u64 all_mask = (nbits >= 64) ? ~(u64)0 : (((u64)1 << nbits) - 1);
			bool rev = c.reverse;

			i32 rem_off = x64_alloc_local(p, 8, 8);
			x64_value_to_reg(p, x64_build_expr(p, iter_expr), X64Reg_RAX);
			if (all_mask != ~(u64)0) {
				x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)all_mask);
				x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
			}
			if (rev) {
				// full 64-bit bit-reverse (swap 1s/2s/4s + bswap), then shift down to mask_bits
				struct { i64 m; u8 s; } steps[3] = {
					{ (i64)0x5555555555555555ull, 1 },
					{ (i64)0x3333333333333333ull, 2 },
					{ (i64)0x0F0F0F0F0F0F0F0Full, 4 },
				};
				for (isize i = 0; i < 3; i++) {
					x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RDX, steps[i].m);
					x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
					x64_emit_shr_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, steps[i].s);
					x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
					x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
					x64_emit_shl_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, steps[i].s);
					x64_emit_or_rr (&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
				}
				x64_emit_bswap_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
				if (mask_bits < 64) x64_emit_shr_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (u8)(64 - mask_bits));
			}
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(rem_off), X64Reg_RAX);

			x64_label_bind(&p->asm_, lbl_loop);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(rem_off));
			x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
			x64_emit_jcc(&p->asm_, X64Cc_E, lbl_end);

			// element value: forward = ctz(remaining) + lower ; reverse = (lower+bits-1) - ctz(remaining)
			x64_emit_bsf_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
			if (rev) {
				x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RDX, lower + mask_bits - 1);
				x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RCX);
				x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
			} else if (lower != 0) {
				x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i32)lower);
			}
			if (elem_e) x64_emit_mov_mr(&p->asm_, x64_op_size_of(elem_e->type), x64_rbp_mem(elem_off), X64Reg_RCX);

			// remaining &= remaining - 1 (clear lowest set bit); RAX still holds remaining
			x64_emit_lea(&p->asm_, X64Reg_RCX, x64_mem(X64Reg_RAX, -1));
			x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(rem_off), X64Reg_RAX);

			if (ridx_e) {
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ridx_off), X64Reg_RAX);
			}

			x64_build_stmt(p, rs->body);

			x64_label_bind(&p->asm_, lbl_post);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(loop_idx_off), X64Reg_RAX);
			x64_emit_jmp(&p->asm_, lbl_loop);

		} else if (iter_type->kind == Type_Tuple) {
			x64_build_range_tuple(p, &c);

		} else if (iter_type->kind == Type_Map) {
			x64_build_range_map(p, &c);

		} else if (iter_type->kind == Type_Basic && is_type_string(iter_type) && !is_type_cstring(iter_type)) {
			x64_build_range_string(p, &c);

		} else {
			x64_build_expr(p, iter_expr); // unsupported type: eval for side effects
		}

		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
}

// Value switch (mirrors lb_build_switch_stmt): build the tag once, dispatch by comparing it against
// each case's value/range list (jump to the case label), then emit the case bodies (each its own defer
// scope; implicit break — no fallthrough).
gb_internal void x64_build_switch_stmt(x64Procedure *p, Ast *node) {
	ast_node(ss, SwitchStmt, node);
		if (ss->init != nullptr) x64_build_stmt(p, ss->init);

		isize lbl_end = x64_label_alloc(&p->asm_);
		x64_push_loop(p, lbl_end, -1, ss->label);

		i32   tag_off = 0;
		Type *tag_t   = nullptr;
		bool  has_tag = (ss->tag != nullptr);
		if (has_tag) {
			tag_t = ss->tag->tav.type;
			tag_off = x64_alloc_local(p, type_size_of(tag_t), type_align_of(tag_t));
			x64Value tv = x64_build_expr(p, ss->tag);
			x64_store_value(p, x64addr(x64_rbp_mem(tag_off), tag_t), tv);
		}

		GB_ASSERT(ss->body != nullptr && ss->body->kind == Ast_BlockStmt);
		AstBlockStmt *body = &ss->body->BlockStmt;

		Array<isize> case_lbls;
		array_init(&case_lbls, temporary_allocator(), (isize)body->stmts.count, (isize)body->stmts.count);
		for (isize ci = 0; ci < body->stmts.count; ci++) {
			case_lbls[ci] = x64_label_alloc(&p->asm_);
		}

		// Dispatch: compare tag against each case's list
		isize default_idx = -1;
		for (isize ci = 0; ci < body->stmts.count; ci++) {
			Ast *cc_ast = body->stmts[ci];
			if (cc_ast == nullptr || cc_ast->kind != Ast_CaseClause) continue;
			AstCaseClause *cc = &cc_ast->CaseClause;

			if (cc->list.count == 0) { default_idx = ci; continue; }

			for_array(vi, cc->list) {
				Ast *case_expr = cc->list[vi];

				// Range case: lo..=hi or lo..<hi
				if (case_expr->kind == Ast_BinaryExpr) {
					AstBinaryExpr *rbe = &case_expr->BinaryExpr;
					if ((rbe->op.kind == Token_RangeFull || rbe->op.kind == Token_RangeHalf) && has_tag && tag_t) {
						X64OpSize sz    = x64_op_size_of(tag_t);
						bool     sign_  = x64_is_signed_integer(tag_t);
						X64Cc    lt_cc  = sign_ ? X64Cc_L  : X64Cc_B;
						X64Cc    gt_cc  = sign_ ? X64Cc_G  : X64Cc_A;
						X64Cc    ge_cc  = sign_ ? X64Cc_GE : X64Cc_AE;
						isize    skip   = x64_label_alloc(&p->asm_);

						// if tag < lo, skip to next case (spill bounds first — same clobber as above)
						x64Value lv = x64_spill_value(p, x64_build_expr(p, rbe->left), tag_t);
						x64Value tv = x64_load_addr(p, x64addr(x64_rbp_mem(tag_off), tag_t));
						x64_value_to_reg(p, tv, X64Reg_RAX);
						x64_value_to_reg(p, lv, X64Reg_RCX);
						x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
						x64_emit_jcc(&p->asm_, lt_cc, skip);

						// if tag > hi (..=) or tag >= hi (..<), skip
						x64Value hv  = x64_spill_value(p, x64_build_expr(p, rbe->right), tag_t);
						x64Value tv2 = x64_load_addr(p, x64addr(x64_rbp_mem(tag_off), tag_t));
						x64_value_to_reg(p, tv2, X64Reg_RAX);
						x64_value_to_reg(p, hv,  X64Reg_RCX);
						x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
						X64Cc hi_skip = (rbe->op.kind == Token_RangeFull) ? gt_cc : ge_cc;
						x64_emit_jcc(&p->asm_, hi_skip, skip);

						x64_emit_jmp(&p->asm_, case_lbls[ci]);
						x64_label_bind(&p->asm_, skip);
						continue;
					}
				}

				// Single value case
				x64Value v = x64_build_expr(p, case_expr);
				if (has_tag && tag_t) {
					// SPILL the case value before touching the tag: a RUNTIME case expr leaves v in
					// RAX, which a tag load would clobber. Compare via x64_emit_comp so STRING/cstring/
					// float/aggregate tags compare by CONTENT — a raw cmp_rr read only the first 8 bytes
					// (a string's DATA pointer) → matched interned literals but missed equal-content
					// substrings (THE json `false`-keyword switch → Unexpected_Token bug). Mirrors
					// lb_build_switch_stmt's lb_emit_comp(Token_CmpEq).
					x64Value vs = x64_spill_value(p, v, tag_t);
					x64Value tv = x64v_mem(tag_t, x64_rbp_mem(tag_off));
					x64Value eq = x64_emit_comp(p, Token_CmpEq, tv, vs);
					x64_value_to_reg(p, eq, X64Reg_RAX);
					x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
					x64_emit_jcc(&p->asm_, X64Cc_NE, case_lbls[ci]);
				} else {
					x64_value_to_reg(p, v, X64Reg_RAX);
					x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
					x64_emit_jcc(&p->asm_, X64Cc_NE, case_lbls[ci]);
				}
			}
		}
		x64_emit_jmp(&p->asm_, default_idx >= 0 ? case_lbls[default_idx] : lbl_end);

		isize saved_ft = p->fallthrough_lbl; // restore after this switch (nested-switch safe)
		for (isize ci = 0; ci < body->stmts.count; ci++) { // case bodies
			Ast *cc_ast = body->stmts[ci];
			if (cc_ast == nullptr || cc_ast->kind != Ast_CaseClause) continue;
			AstCaseClause *cc = &cc_ast->CaseClause;

			x64_label_bind(&p->asm_, case_lbls[ci]);
			// `fallthrough` in this case body jumps to the NEXT case clause's body (source order),
			// skipping its condition (Odin semantics). Last case → lbl_end.
			p->fallthrough_lbl = (ci + 1 < body->stmts.count) ? case_lbls[ci + 1] : lbl_end;
			isize defer_base = p->deferred.count;
			x64_build_stmt_list(p, cc->stmts);
			x64_scope_end(p, defer_base); // case-local defers run here, not at fn exit
			x64_emit_jmp(&p->asm_, lbl_end); // implicit break (Odin has no fallthrough by default)
		}
		p->fallthrough_lbl = saved_ft;

		array_free(&case_lbls);
		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
}

// Type switch `switch v in expr { case T: … }` over union/any (mirrors lb_build_type_switch_stmt):
// load the dispatch tag (union variant index or any.id), compare against each case type, then in each
// body bind the implicit entity (copy the variant data for a single concrete case; alias the operand
// for multi/default).
gb_internal void x64_build_type_switch_stmt(x64Procedure *p, Ast *node) {
	ast_node(ss, TypeSwitchStmt, node);
		if (ss->tag == nullptr || ss->tag->kind != Ast_AssignStmt) return;
		AstAssignStmt *tag_assign = &ss->tag->AssignStmt;
		if (tag_assign->rhs.count < 1) return;
		Ast *tag_expr = tag_assign->rhs[0];

		x64Value parent_v   = x64_build_expr(p, tag_expr);
		Type    *parent_raw = x64_typed(tag_expr->tav.type);
		Type    *parent_bt  = base_type(parent_raw);
		if (parent_bt == nullptr) return;

		// Auto-deref: switch v in ptr_to_union / ptr_to_any
		if (is_type_pointer(parent_bt)) {
			parent_bt = base_type(type_deref(parent_raw));
		}

		TypeSwitchKind switch_kind = check_valid_type_switch_type(parent_raw);
		if (switch_kind == TypeSwitch_Invalid) return;

		// Stabilise effective address of the source value
		i32 parent_ea_off = x64_alloc_local(p, 8, 8);
		if (is_type_pointer(base_type(parent_raw))) {
			// parent_v is already a pointer; its value IS the EA
			x64_value_to_reg(p, parent_v, X64Reg_RAX);
		} else {
			if (parent_v.kind != x64Value_Mem) {
				i64 psz = x64_type_size(parent_bt);
				i64 pal = x64_type_align(parent_bt);
				if (psz <= 0) psz = 8; if (pal <= 0) pal = 8;
				i32 poff = x64_alloc_local(p, psz, pal);
				x64_store_value(p, x64addr(x64_rbp_mem(poff), parent_bt), parent_v);
				parent_v = x64v_mem(parent_bt, x64_rbp_mem(poff));
			}
			x64_emit_lea(&p->asm_, X64Reg_RAX, parent_v.mem);
		}
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(parent_ea_off), X64Reg_RAX);

		// Load and save the dispatch tag (any.id or union variant tag)
		i32 tag_val_off = x64_alloc_local(p, 8, 8);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(parent_ea_off));
		if (switch_kind == TypeSwitch_Any) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8)); // any.id
		} else {
			x64_emit_union_tag_value(p, x64_mem(X64Reg_RAX, 0), parent_bt, X64Reg_RCX);
		}
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(tag_val_off), X64Reg_RCX);

		GB_ASSERT(ss->body != nullptr && ss->body->kind == Ast_BlockStmt);
		AstBlockStmt *body = &ss->body->BlockStmt;

		isize lbl_end = x64_label_alloc(&p->asm_);
		x64_push_loop(p, lbl_end, -1, ss->label);

		Array<isize> case_lbls;
		array_init(&case_lbls, temporary_allocator(), (isize)body->stmts.count, (isize)body->stmts.count);
		for (isize ci = 0; ci < body->stmts.count; ci++) {
			case_lbls[ci] = x64_label_alloc(&p->asm_);
		}

		// Dispatch: compare tag against each case list
		isize default_idx = -1;
		for (isize ci = 0; ci < body->stmts.count; ci++) {
			Ast *cc_ast = body->stmts[ci];
			if (cc_ast == nullptr || cc_ast->kind != Ast_CaseClause) continue;
			AstCaseClause *cc = &cc_ast->CaseClause;
			if (cc->list.count == 0) { default_idx = ci; continue; }

			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(tag_val_off));
			for_array(ti, cc->list) {
				Type *case_type = type_of_expr(cc->list[ti]);
				if (case_type == nullptr) continue;
				// `case nil:` — check the RAW type: x64_typed maps untyped_nil→rawptr, which would make
				// is_type_untyped_nil false, so the nil case fell to the union branch below and got
				// `continue`d (no comparison emitted) → a nil union/any never matched `case nil`. Was
				// core:flags set_option: `switch &e in error { case nil: register_field() }` never ran on
				// a successful (nil) parse → positional tracking bit never set → positionals misassigned.
				if (is_type_untyped_nil(case_type)) {
					x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
					x64_emit_jcc(&p->asm_, X64Cc_E, case_lbls[ci]);
					continue;
				}
				case_type = x64_typed(case_type);
				if (switch_kind == TypeSwitch_Any) {
					u64 expected_id = type_hash_canonical_type(case_type);
					x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)expected_id);
					x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
					x64_emit_jcc(&p->asm_, X64Cc_E, case_lbls[ci]);
				} else { // TypeSwitch_Union
					if (!union_is_variant_of(parent_bt, case_type)) continue;
					i64 expected_tag = union_variant_index_checked(parent_bt, case_type);
					x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)expected_tag);
					x64_emit_jcc(&p->asm_, X64Cc_E, case_lbls[ci]);
				}
			}
		}
		x64_emit_jmp(&p->asm_, default_idx >= 0 ? case_lbls[default_idx] : lbl_end);

		for (isize ci = 0; ci < body->stmts.count; ci++) { // case bodies
			Ast *cc_ast = body->stmts[ci];
			if (cc_ast == nullptr || cc_ast->kind != Ast_CaseClause) continue;
			AstCaseClause *cc = &cc_ast->CaseClause;

			x64_label_bind(&p->asm_, case_lbls[ci]);

			// Bind the implicit case entity. A single concrete-type case binds it to that
			// type (copy the variant's data); a multi/default (`case:`) case binds it to the
			// OPERAND type (union/any), aliasing the whole value. ALL cases must register it
			// — else a body reference falls to x64_global_mem and emits a bogus extern (e.g.
			// `specific_type_info` in a `#partial switch …{ … case: }`).
			Entity *case_entity = implicit_entity_of_node(cc_ast);
			if (case_entity != nullptr) {
				Type *ct    = x64_typed(case_entity->type);
				Type *ct_bt = base_type(ct);
				if (ct_bt != nullptr && !is_type_untyped_nil(ct)) {
					bool specific     = (cc->list.count == 1); // bound to one concrete type
					bool by_reference = (case_entity->flags & EntityFlag_Value) == 0;

					// RCX = &source data
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(parent_ea_off));
					if (switch_kind == TypeSwitch_Any && specific) {
						// Deref: RCX = any.data (&actual T). Single concrete type only;
						// a multi/default binding keeps the `any` itself.
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RCX, 0));
					}
					// Union: variant data (or whole union for multi/default) is at offset 0,
					// so RCX already = &operand.

					if (by_reference) {
						// `switch &dst in v`: bind dst as an INDIRECT local — an 8-byte slot
						// holding the source ADDRESS (RCX); x64_entity_addr derefs it so writes
						// (`dst = …`) reach the operand's data. Mirrors lb's by_reference path
						// (lb_add_entity(e, ptr)). Was the json assign_int/assign_bool store loss.
						i32 slot = x64_alloc_local(p, 8, 8);
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slot), X64Reg_RCX);
						x64_var_set(&p->var_offsets, case_entity, slot);
						array_add(&p->indirect_params, case_entity);
					} else {
						i64 ct_sz = x64_type_size(ct);
						if (ct_sz > 0) {
							i32 ent_off = x64_alloc_local(p, ct_sz, x64_type_align(ct));
							x64_zero_mem(p, x64_rbp_mem(ent_off), ct_sz);
							// Copy via R10 scratch to keep RCX intact.
							if (ct_sz <= 64) {
								x64_copy_fixed(p, x64_rbp_mem(ent_off), x64_mem(X64Reg_RCX, 0), ct_sz);
							} else {
								x64_copy_mem(p, x64_rbp_mem(ent_off), x64_mem(X64Reg_RCX, 0), ct_sz);
							}
							x64_var_set(&p->var_offsets, case_entity, ent_off);
						}
					}
				}
			}

			isize defer_base = p->deferred.count;
			x64_build_stmt_list(p, cc->stmts);
			x64_scope_end(p, defer_base); // case-local defers run here, not at fn exit
			x64_emit_jmp(&p->asm_, lbl_end);
		}

		array_free(&case_lbls);
		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
}

// `=`, op-assign (`+=` …), multi-assign/swap, and tuple unpack (mirrors lb_build_assign_stmt +
// lb_build_assignment). Map-lvalue forms (`m[k]=v`, `m[k]op=v`) and scoped `context.field=X` are
// intercepted before the generic x64_build_addr path.
gb_internal void x64_build_assign_stmt(x64Procedure *p, Ast *node) {
	ast_node(as, AssignStmt, node);
		TokenKind op        = as->op.kind;
		int       lhs_count = (int)as->lhs.count;
		int       rhs_count = (int)as->rhs.count;

		if (op == Token_Eq) {
			// `m[k] = v`: map insert (no x64Addr can represent a map lvalue) — mirror
			// lb_addr_store(lbAddr_Map). Intercept before the generic lvalue path.
			if (lhs_count == 1 && rhs_count == 1 && !x64_is_blank(as->lhs[0]) &&
			    as->lhs[0]->kind == Ast_IndexExpr) {
				Ast *lie = as->lhs[0];
				Type *ibt = base_type(type_deref(lie->IndexExpr.expr->tav.type));
				if (ibt != nullptr && ibt->kind == Type_Map) {
					x64_build_map_index_store(p, lie->IndexExpr.expr, lie->IndexExpr.index, as->rhs[0], node);
					return;
				}
			}
			// `context.field = X`: mirror lb_addr_store(lbAddr_Context). Eval RHS against
			// the CURRENT context, then snapshot context into a fresh local, make it active,
			// and store the field into THAT — a scoped change.
			if (lhs_count == 1 && rhs_count == 1 && !x64_is_blank(as->lhs[0]) &&
			    x64_lvalue_roots_at_context(as->lhs[0])) {
				Type *ft = as->lhs[0]->tav.type;
				i64 fsz = ft ? type_size_of(ft) : 8; if (fsz <= 0) fsz = 1;
				i64 fal = ft ? type_align_of(ft) : 8; if (fal <= 0) fal = 1;
				x64Value rv = x64_build_expr(p, as->rhs[0]); // sees the OLD context
				i32 rv_off  = x64_alloc_local(p, fsz, fal);
				x64_store_value(p, x64addr(x64_rbp_mem(rv_off), ft), rv);
				x64_push_new_context(p);                     // fresh scoped context, now active
				x64Addr lhs = x64_build_addr(p, as->lhs[0]); // field addr in the NEW context (RBP-based)
				x64_store_value(p, lhs, x64v_mem(ft, x64_rbp_mem(rv_off)));
				return;
			}
			if (lhs_count == 1 && rhs_count == 1) {
				if (x64_is_blank(as->lhs[0])) { x64_build_expr(p, as->rhs[0]); return; }
				// bit_field member write (`x.field = v`): read-modify-write the backing integer
				// (no addressable byte location). Must precede the generic lvalue path.
				{
					Type *ft = nullptr, *bk = nullptr; i64 boff = 0, bsize = 0, byoff = 0;
					if (x64_bit_field_member_info(as->lhs[0], &ft, &bk, &boff, &bsize, &byoff)) {
						x64_bit_field_store(p, as->lhs[0], as->rhs[0], ft, bk, boff, bsize, byoff);
						return;
					}
				}
				// `soa[idx] = v` whole-element write → scatter v's components into the field arrays
				// (no single lvalue address exists; x64 has no Addressing_SoaVariable).
				{
					Ast *lhs0 = unparen_expr(as->lhs[0]);
					if (lhs0->kind == Ast_IndexExpr) {
						Type *lbt = base_type(type_deref(lhs0->IndexExpr.expr->tav.type));
						if (lbt != nullptr && lbt->kind == Type_Struct && lbt->Struct.soa_kind != StructSoa_None) {
							Type *et = x64_typed(as->lhs[0]->tav.type);
							x64Value rv = x64_emit_conv(p, x64_build_expr(p, as->rhs[0]), as->rhs[0]->tav.type, et);
							x64_soa_index_element(p, lhs0, et, /*store*/true, rv);
							return;
						}
					}
				}
				// `v.xy = rhs` multi-component swizzle LVALUE: no single contiguous address — SCATTER each
				// rhs element into v[swizzle_indices[i]] (mirrors lb_addr_store lbAddr_Swizzle). Single-comp
				// `.y` (swizzle_count==0) keeps the addressable build_addr path below. Was test_issue_1730:
				// `out.yz = ll.yz` fell through to build_addr → &out (offset 0) → wrote [2,3,0,0] not [0,2,3,0].
				{
					Ast *lhs0 = unparen_expr(as->lhs[0]);
					if (lhs0->kind == Ast_SelectorExpr && lhs0->SelectorExpr.swizzle_count > 0) {
						AstSelectorExpr *se = &lhs0->SelectorExpr;
						bool  via_ptr = is_type_pointer(se->expr->tav.type);
						Type *dst_t   = base_type(via_ptr ? type_deref(se->expr->tav.type) : se->expr->tav.type);
						Type *elem    = (dst_t->kind == Type_SimdVector) ? dst_t->SimdVector.elem : dst_t->Array.elem;
						i64       esz = type_size_of(elem);
						X64OpSize osz = x64_op_size_of(elem);
						Type *rt  = x64_typed(as->lhs[0]->tav.type);
						x64Value src = x64_spill_value(p, x64_emit_conv(p, x64_build_expr(p, as->rhs[0]), as->rhs[0]->tav.type, rt), rt);
						i32 base_off = x64_alloc_local(p, 8, 8);
						if (via_ptr) { x64_value_to_reg(p, x64_build_expr(p, se->expr), X64Reg_RAX); }
						else         { x64_emit_lea(&p->asm_, X64Reg_RAX, x64_build_addr(p, se->expr).mem); }
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base_off), X64Reg_RAX);
						for (u8 i = 0; i < se->swizzle_count; i++) {
							u8 idx = (se->swizzle_indices >> (i*2)) & 3;
							X64Mem sm = src.mem; sm.disp += (i32)(i*esz);
							x64_emit_mov_rm(&p->asm_, osz, X64Reg_RAX, sm);                       // RAX = rhs[i]
							x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(base_off)); // RCX = &dst
							x64_emit_mov_mr(&p->asm_, osz, x64_mem(X64Reg_RCX, (i32)(idx*esz)), X64Reg_RAX); // dst[idx]=rhs[i]
						}
						return;
					}
				}
				x64Addr lhs_addr = x64_build_addr(p, as->lhs[0]);
				// `lhs = {}` → zero the destination in place (no full-size temp+copy). No RHS
				// build means the (possibly register-based) lhs address isn't clobbered; zero_mem
				// lea's it immediately.
				if (x64_empty_compound_lit_zeroes(as->rhs[0], as->lhs[0]->tav.type)) {
					Type *lt = as->lhs[0]->tav.type;
					i64 lsz = lt ? type_size_of(lt) : 0;
					if (lsz > 0) x64_zero_mem(p, lhs_addr.mem, lsz);
					return;
				}
				// Pin a dynamic base register (e.g. ptr.field's ptr in RAX) to a stack slot:
				// rhs eval below may clobber it.
				bool lhs_needs_pin = !lhs_addr.mem.rip_rel &&
				                      lhs_addr.mem.base != X64Reg_RBP &&
				                      lhs_addr.mem.base != X64Reg_NONE;
				i32 pin_off = 0;
				if (lhs_needs_pin) {
					pin_off = x64_alloc_local(p, 8, 8);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(pin_off), lhs_addr.mem.base);
				}
				x64Value rv = x64_build_expr(p, as->rhs[0]);
				if (lhs_needs_pin) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(pin_off));
					lhs_addr.mem.base = X64Reg_RCX;
				}
				x64_store_value(p, lhs_addr, rv);
			} else if (lhs_count == rhs_count) {
				// a, b = c, d — stash rhs to stack temps first
				Array<i32> rhs_offs; array_init(&rhs_offs, temporary_allocator(), lhs_count, lhs_count);
				for (int ri = 0; ri < rhs_count; ri++) {
					if (x64_is_blank(as->lhs[ri])) { rhs_offs[ri] = 0; continue; }
					// A `nil`/untyped RHS (e.g. `a, b = x, nil`) has an UNTYPED tav.type — sizing the
					// temp from it would feed type_size_of an untyped nil (assert). Use the LHS's
					// concrete type; x64_store_value of the None value then zeroes the slot.
					Type *t  = as->rhs[ri]->tav.type;
					if (t == nullptr || is_type_untyped(t)) t = as->lhs[ri]->tav.type;
					i64 sz   = t ? type_size_of(t) : 8;
					i64 al   = t ? type_align_of(t) : 8;
					i32 tmp  = x64_alloc_local(p, sz, al);
					rhs_offs[ri] = tmp;
					x64Value rv = x64_build_expr(p, as->rhs[ri]);
					x64_store_value(p, x64addr(x64_rbp_mem(tmp), t), rv);
				}
				for (int li = 0; li < lhs_count; li++) {
					if (x64_is_blank(as->lhs[li])) continue;
					Type    *t       = as->lhs[li]->tav.type;
					// Keep the value as a stack ref, NOT a register: x64_build_addr below loads the
					// lhs base into RAX, which would clobber a register-held value (then the store
					// would write the base pointer, not the value). store_value copies mem→mem.
					x64Value tv      = x64v_mem(t, x64_rbp_mem(rhs_offs[li]));
					x64Addr  lhs_addr = x64_build_addr(p, as->lhs[li]);
					x64_store_value(p, lhs_addr, tv);
				}
				array_free(&rhs_offs);
			} else {
				// `a, b, … = expr` where expr yields a tuple — unpack each field into its lvalue.
				x64Value result = x64_build_expr(p, as->rhs[0]);
				if (result.kind == x64Value_Mem && result.type != nullptr &&
				    is_type_tuple(result.type)) {
					Type *tuple = result.type;
					int n = gb_min(lhs_count, (int)tuple->Tuple.variables.count);

					// Stabilise the tuple base address (lhs address building below
					// clobbers RAX, which may be the tuple's base register).
					i32 tup_off = x64_alloc_local(p, 8, 8);
					x64_emit_lea(&p->asm_, X64Reg_RAX, result.mem);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(tup_off), X64Reg_RAX);

					Array<i32>   tmp_offs; array_init(&tmp_offs, temporary_allocator(), n, n);
					Array<Type*> ftypes;   array_init(&ftypes,   temporary_allocator(), n, n);
					for (int i = 0; i < n; i++) {
						Entity *tv = tuple->Tuple.variables[i];
						i64 fsz = type_size_of(tv->type); if (fsz <= 0) fsz = 1;
						i64 fal = type_align_of(tv->type); if (fal <= 0) fal = 1;
						i32 tmp = x64_alloc_local(p, fsz, fal);
						tmp_offs[i] = tmp; ftypes[i] = tv->type;
						// Reload the pinned base (lhs-addr building clobbers RAX), then read field i
						// at the canonical type_offset_of.
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(tup_off));
						x64Value fv = x64_emit_tuple_ev(p, x64v_mem(tuple, x64_mem(X64Reg_RAX, 0)), i);
						x64_store_value(p, x64addr(x64_rbp_mem(tmp), tv->type), fv);
					}
					// R8 = &lhs (untouched by the copy).
					for (int i = 0; i < n; i++) {
						if (x64_is_blank(as->lhs[i])) continue;
						x64Addr la = x64_build_addr(p, as->lhs[i]);
						x64_emit_lea(&p->asm_, X64Reg_R8, la.mem);
						x64_store_value(p, x64addr(x64_mem(X64Reg_R8, 0), la.type),
						                x64v_mem(ftypes[i], x64_rbp_mem(tmp_offs[i])));
					}
					array_free(&ftypes);
					array_free(&tmp_offs);
				}
			}
		} else {
			// `m[k] op= v` → m[k] = m[k] op v (no x64Addr for a map lvalue). The key is
			// evaluated twice (load + store) — fine for the usual var/literal key.
			if (lhs_count == 1 && rhs_count == 1 && as->lhs[0]->kind == Ast_IndexExpr) {
				Ast *lie = as->lhs[0];
				Type *ibt = base_type(type_deref(lie->IndexExpr.expr->tav.type));
				if (ibt != nullptr && ibt->kind == Type_Map) {
					Type *vt = ibt->Map.value;
					x64Value cur = x64_build_map_index_load(p, lie->IndexExpr.expr, lie->IndexExpr.index, vt);
					cur = x64_spill_value(p, cur, vt);
					x64Value rv = x64_build_expr(p, as->rhs[0]);
					rv = x64_spill_value(p, rv, rv.type ? rv.type : vt);
					TokenKind bop = cast(TokenKind)(Token_Add + (op - Token_AddEq));
					x64Value nv = x64_emit_arith(p, bop, cur, rv, vt);
					nv = x64_spill_value(p, nv, vt);
					x64Value map_ptr = x64_map_addr_of(p, lie->IndexExpr.expr, ibt);
					x64_internal_dynamic_map_set(p, map_ptr, ibt, lie->IndexExpr.index, nv, node);
					return;
				}
			}
			// Compound assignment (+=, -=, …); single lhs only.
			if (lhs_count == 1 && rhs_count == 1) {
				// `||=` / `&&=` are SHORT-CIRCUIT logical ops, not value ops: build the logical expr
				// and store (mirrors lb_build_assign_stmt's Token_CmpAnd/CmpOr branch). The generic
				// integer compound path below has NO switch case for Token_CmpOrEq/CmpAndEq → it
				// silently no-op'd (lhs unchanged). THE blick no-text bug: `updated_points ||=
				// shape_was_newly_created` never set updated_points → the glyph curve-point buffer was
				// never uploaded to the GPU → glyphs rasterized from empty/stale curves → no text.
				{
					TokenKind lbop = cast(TokenKind)(Token_Add + (op - Token_AddEq));
					if (lbop == Token_CmpAnd || lbop == Token_CmpOr) {
						Type *lty = x64_typed(as->lhs[0]->tav.type);
						x64Value nv = x64_emit_logical_binary_expr(p, lbop, as->lhs[0], as->rhs[0], lty);
						x64_store_value(p, x64_build_addr(p, as->lhs[0]), nv);
						return;
					}
				}
				x64Addr  lhs_addr = x64_build_addr(p, as->lhs[0]);
				// Use the lvalue's address type (lb_addr_type(lhs)), not tav.type, which
				// doesn't reliably resolve to bit_set here.
				Type    *t        = x64_typed(lhs_addr.type != nullptr ? lhs_addr.type : as->lhs[0]->tav.type);
				X64OpSize sz      = x64_op_size_of(t);

				// 128-bit integer / 16-byte bit_set compound assign: route through x64_emit_arith
				// (→ x64_emit_arith_i128). The single-register path below truncates/crashes, and the
				// 8-byte rhs spill it uses would overflow for a 16-byte rhs. lhs = (*lhs) OP rhs.
				{
					Type *abt0 = base_type(t);
					if (abt0 != nullptr && type_size_of(t) == 16 &&
					    (is_type_integer(t) || abt0->kind == Type_BitSet)) {
						i32 ea = x64_alloc_local(p, 8, 8);
						x64_emit_lea(&p->asm_, X64Reg_RAX, lhs_addr.mem);
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ea), X64Reg_RAX);
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ea));
						x64Value cur = x64_spill_value(p, x64v_mem(t, x64_mem(X64Reg_R8, 0)), t); // copy *lhs
						x64Value rv  = x64_build_expr(p, as->rhs[0]);
						TokenKind bop = cast(TokenKind)(Token_Add + (op - Token_AddEq));
						x64Value nv  = x64_spill_value(p, x64_emit_arith(p, bop, cur, rv, t), t);
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ea)); // reload &lhs
						x64_store_value(p, x64addr(x64_mem(X64Reg_R8, 0), t), nv);
						return;
					}
				}

				// Stabilise &lhs to a stack slot: RHS eval and the op below clobber
				// RAX/RCX/RDX, which could be the LHS base register (e.g. ptr.field += x).
				i32 ea_off = x64_alloc_local(p, 8, 8);
				x64_emit_lea(&p->asm_, X64Reg_RAX, lhs_addr.mem);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ea_off), X64Reg_RAX);

				// RHS FIRST (clobbers RAX) and spill, so the LHS load into RAX survives. Spill with the
				// RHS's OWN type — for `arr op= scalar` the rhs is a scalar, NOT the array type t (storing
				// it as t would mis-size it).
				x64Value rv = x64_build_expr(p, as->rhs[0]);
				Type *rv_type = x64_typed(rv.type ? rv.type : t);
				i64 rv_sz = type_size_of(rv_type); if (rv_sz < 8) rv_sz = 8;
				i32 rhs_off = x64_alloc_local(p, (i32)rv_sz, 8);
				x64_store_value(p, x64addr(x64_rbp_mem(rhs_off), rv_type), rv);

				// FLOAT scalar AND aggregate (array/#simd/matrix) compound assign (+=,-=,*=,/=): the integer
				// register path below does integer imul/add/div on the raw BIT PATTERNS → garbage. Route
				// through x64_emit_arith (SSE for scalar float; element-wise + scalar broadcast for
				// arrays/#simd/matrices via x64_emit_arith_array), like `lhs = lhs op rhs`. Was: scalar
				// `v.f *= magic.f` → 0 (f16→f32); AND `arr /= s` integer-divided packed float bytes →
				// denormal garbage — blick block_size_from_id's `res /= dpi_scale` → viewport size 0 →
				// scrollview/timeline virtualization collapse (panels rendered nearly empty).
				Type *tbt_ca = base_type(t);
				bool is_aggr_arith = tbt_ca != nullptr && (tbt_ca->kind == Type_Array ||
				                     tbt_ca->kind == Type_SimdVector || tbt_ca->kind == Type_Matrix);
				if (x64_is_float(t) || is_aggr_arith) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ea_off)); // R8 = &lhs
					x64Value cur = x64_spill_value(p, x64v_mem(t, x64_mem(X64Reg_R8, 0)), t); // copy *lhs
					x64Value rv2 = x64v_mem(rv_type, x64_rbp_mem(rhs_off));
					TokenKind bop = cast(TokenKind)(Token_Add + (op - Token_AddEq));
					x64Value nv = x64_spill_value(p, x64_emit_arith(p, bop, cur, rv2, t), t);
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ea_off)); // reload (arith clobbered)
					x64_store_value(p, x64addr(x64_mem(X64Reg_R8, 0), t), nv);
					return;
				}

				// RAX = *lhs, RCX = rhs, R8 = &lhs (untouched by the op).
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ea_off));
				x64_value_to_reg(p, x64v_mem(t, x64_mem(X64Reg_R8, 0)), X64Reg_RAX);
				x64_value_to_reg(p, x64v_mem(t, x64_rbp_mem(rhs_off)), X64Reg_RCX);

				// bit_set `+` is union (OR), `-` is difference (AND-NOT), NOT integer add/sub
				// (empty-set `+= {x}` matches ADD, hiding this; an already-set bit doubles
				// under ADD). Mirrors lb_emit_arith. Detect via LHS type OR the RHS operand
				// type — `s += {x}`'s RHS literal resolves to Type_BitSet when the LHS doesn't.
				Type *abt = base_type(t);
				bool is_bset = abt != nullptr && abt->kind == Type_BitSet;
				if (!is_bset && as->rhs[0] != nullptr && as->rhs[0]->tav.type != nullptr) {
					Type *rbt = base_type(as->rhs[0]->tav.type);
					is_bset = rbt != nullptr && rbt->kind == Type_BitSet;
				}
				switch (op) {
				case Token_AddEq:
					if (is_bset) x64_emit_or_rr (&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
					else         x64_emit_add_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
					break;
				case Token_SubEq:
					if (is_bset) {
						x64_emit_not_r (&p->asm_, sz, X64Reg_RCX);
						x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
					} else {
						x64_emit_sub_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
					}
					break;
				case Token_MulEq:    x64_emit_imul_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);   break;
				case Token_AndEq:    x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);    break;
				case Token_OrEq:     x64_emit_or_rr (&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);    break;
				case Token_XorEq:    x64_emit_xor_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);    break;
				case Token_ShlEq:    x64_emit_shl_rcl(&p->asm_, sz, X64Reg_RAX);                break;
				case Token_ShrEq:
					if (x64_is_signed_integer(t)) x64_emit_sar_rcl(&p->asm_, sz, X64Reg_RAX);
					else                           x64_emit_shr_rcl(&p->asm_, sz, X64Reg_RAX);
					break;
				case Token_AndNotEq:
					x64_emit_not_r(&p->asm_, sz, X64Reg_RCX);
					x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
					break;
				case Token_QuoEq:
					if (x64_is_signed_integer(t)) {
						if (sz == X64OpSize_64) x64_emit_cqo(&p->asm_); else x64_emit_cdq(&p->asm_);
						x64_emit_idiv_r(&p->asm_, sz, X64Reg_RCX);
					} else {
						x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX);
						x64_emit_div_r(&p->asm_, sz, X64Reg_RCX);
					}
					break;
				case Token_ModEq:
					if (x64_is_signed_integer(t)) {
						if (sz == X64OpSize_64) x64_emit_cqo(&p->asm_); else x64_emit_cdq(&p->asm_);
						x64_emit_idiv_r(&p->asm_, sz, X64Reg_RCX);
					} else {
						x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX);
						x64_emit_div_r(&p->asm_, sz, X64Reg_RCX);
					}
					x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
					break;
				default: break;
				}
				// Reload &lhs (op may clobber regs), store result.
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ea_off));
				x64_store_value(p, x64addr(x64_mem(X64Reg_R8, 0), t), x64v_reg(t, X64Reg_RAX));
			}
		}
}

// Mirrors lb_build_constant_value_decl + lb_build_static_variables + the mutable ValueDecl path:
// nested proc decls (built inline so reachability flows through the parent), @(static)/@(thread_local)
// locals lowered to module globals, and ordinary `x := …` / multi-return tuple unpack.
gb_internal void x64_build_value_decl(x64Procedure *p, Ast *node) {
	ast_node(vs, ValueDecl, node);
		if (!vs->is_mutable) {
			// While INLINING a callee body, its nested procs are emitted by the callee's
			// OWN standalone compilation (every referenced proc is compiled exactly once —
			// normally or on-demand). Re-emitting here, once per inline site, duplicates the
			// symbol across modules (LNK2005). The inlined body just references it by name.
			if (p->inline_frames.count > 0) return;
			// Nested proc decls: build inline (lb_build_stmt ValueDecl path) so
			// reachability flows through the parent body and dead procs aren't emitted.
			for_array(i, vs->names) {
				Ast *ident = vs->names[i];
				if (ident->kind != Ast_Ident) continue;
				Entity *e = entity_of_node(ident);
				if (e == nullptr || e->kind != Entity_Procedure) continue;
				if (i >= vs->values.count || vs->values[i] == nullptr) continue;
				Ast *value = unparen_expr(vs->values[i]);
				if (value->kind != Ast_ProcLit) continue;
				DeclInfo *decl = decl_info_of_entity(e);
				if (decl == nullptr || decl->proc_lit == nullptr) continue;
				GenProcsData *gpd = e->Procedure.gen_procs;
				if (gpd != nullptr) {
					for (Entity *ge : gpd->procs) {
						if (ge->min_dep_count.load(std::memory_order_relaxed) == 0) continue;
						DeclInfo *d = decl_info_of_entity(ge);
						if (d == nullptr || d->proc_lit == nullptr) continue;
						x64_build_nested_proc(p, d->proc_lit, ge);
					}
				} else {
					x64_build_nested_proc(p, decl->proc_lit, e);
				}
			}
			return;
		}

		// @(static)/@(thread_local) locals persist across calls → lowered to module
		// globals (mirrors lb_build_static_variables), not stack slots that would dangle
		// after return (e.g. os's @(static) files corrupting stdout).
		{
			bool is_static = false, is_tls = false;
			for_array(i, vs->names) {
				if (x64_is_blank(vs->names[i])) continue;
				Entity *se = entity_of_node(vs->names[i]);
				if (se == nullptr || se->kind != Entity_Variable) continue;
				if (se->flags & EntityFlag_Static)            is_static = true;
				if (se->Variable.thread_local_model.len != 0) is_tls    = true;
			}
			// thread_local MUST go to `.tls$` (accessed via the TLS sequence); a plain
			// global makes the TLS-addressed reads hit a bad address.
			if (is_static || is_tls) {
				for_array(i, vs->names) {
					if (x64_is_blank(vs->names[i])) continue;
					Entity *se = entity_of_node(vs->names[i]);
					if (se == nullptr || se->kind != Entity_Variable) continue;
					if (se->Variable.link_name.len == 0) {
						// Unique mangled name: <proc>-.<name>-<id> (matches LLVM).
						gbString gs = gb_string_make_length(permanent_allocator(),
						                                    p->link_name.text, p->link_name.len);
						gs = gb_string_append_fmt(gs, "-.%.*s-%llu",
						                          LIT(se->token.string), (unsigned long long)se->id);
						se->Variable.link_name = make_string((u8 const *)gs, gb_string_length(gs));
						// A @(static) with a const initializer must BAKE the value into
						// .rdata/.data (mirrors lb_build_static_variables). Routing it to
						// x64_emit_global_variable zero-fills .bss with NO runtime init, so the
						// static reads as all-zero — e.g. string_decode_rune's @(static,rodata)
						// accept_sizes/accept_ranges → every rune decoded as U+FFFD.
						Ast *init_val = (i < (isize)vs->values.count) ? vs->values[i] : nullptr;
						if (se->Variable.thread_local_model.len != 0) {
							x64_emit_global_tls(p->module, se, decl_info_of_entity(se));
						} else if (init_val != nullptr) {
							x64_emit_global_static_value(p->module, se, init_val);
						} else {
							x64_emit_global_variable(p->module, se);
						}
					}
				}
				return;
			}
		}

		int name_count  = (int)vs->names.count;
		int value_count = (int)vs->values.count;

		if (value_count == 0 || value_count == name_count) {
			for (int ni = 0; ni < name_count; ni++) {
				if (x64_is_blank(vs->names[ni])) continue;
				Entity *e = entity_of_node(vs->names[ni]);
				if (e == nullptr || e->kind != Entity_Variable) continue;

				i32 off = x64_alloc_var(p, e);

				if (value_count == 0 || x64_empty_compound_lit_zeroes(vs->values[ni], e->type)) {
					x64_zero_mem(p, x64_rbp_mem(off), x64_type_size(e->type)); // `x: T` or `x := {}`
				} else {
					// Reclaim the initializer's scratch ONLY when the declared type is
					// pointer-free — such a value can't alias the scratch. A slice/string/
					// pointer result CAN point INTO it (`.data`), so those are left alone.
					// Odin temporaries are scope-lived; this is the safe statement-level subset.
					bool reclaim = x64_type_is_pointer_free(e->type);
					i32   imark = p->local_size;
					isize idef  = p->deferred.count;
					u32   inmd  = p->named_seq;
					x64Value v = x64_build_expr(p, vs->values[ni]);
					x64_store_value(p, x64addr(x64_rbp_mem(off), e->type), v);
					if (reclaim && p->deferred.count == idef && p->named_seq == inmd) p->local_size = imark;
				}
			}
		} else {
			// value_count == 1, name_count > 1 — multi-return (TypeAssert, proc, etc.)
			for (int ni = 0; ni < name_count; ni++) {
				if (x64_is_blank(vs->names[ni])) continue;
				Entity *e = entity_of_node(vs->names[ni]);
				if (e == nullptr || e->kind != Entity_Variable) continue;
				i32 off = x64_alloc_var(p, e);
				x64_zero_mem(p, x64_rbp_mem(off), x64_type_size(e->type));
			}
			// Reclaim the result/unpack scratch only when every unpacked value is pointer-free
			// (see branch above) — a slice/string/pointer field can point INTO the scratch.
			bool reclaim = x64_type_is_pointer_free(vs->values[0]->tav.type);
			i32   imark = p->local_size;
			isize idef  = p->deferred.count;
			u32   inmd  = p->named_seq;
			x64Value result = x64_build_expr(p, vs->values[0]);
			// Unpack tuple POSITIONALLY: name[i] ⇐ field i. A blank `_` is skipped but
			// STILL consumes its field — else `_, ok := …` reads field 0 (value) into ok.
			if (result.kind == x64Value_Mem && result.type != nullptr && is_type_tuple(result.type)) {
				Type *tuple   = base_type(result.type);
				isize tcount  = tuple->Tuple.variables.count;
				for (int ni = 0; ni < name_count && (isize)ni < tcount; ni++) {
					if (x64_is_blank(vs->names[ni])) continue; // blank still "consumes" field ni (by index)
					Entity *ne = entity_of_node(vs->names[ni]);
					if (ne == nullptr || ne->kind != Entity_Variable) continue;
					i32 *ne_off = x64_var_get(&p->var_offsets, ne);
					if (ne_off == nullptr) continue;
					x64Value fv = x64_emit_tuple_ev(p, result, ni); // field ni @ type_offset_of
					x64_store_value(p, x64addr(x64_rbp_mem(*ne_off), ne->type), fv);
				}
			}
			if (reclaim && p->deferred.count == idef && p->named_seq == inmd) p->local_size = imark;
		}
}

// Branch on `cond` to true_lbl/false_lbl, recursing on &&/||/! for SHORT-CIRCUIT branching (mirrors
// lb_build_cond): `a && b` jumps to false on the first false operand; `a || b` jumps to true on the
// first true one — never materialising a bool. `true_is_fallthrough` says which target is the next
// instruction, so the leaf emits exactly one conditional jump (no redundant `jmp`); the caller binds
// that fallthrough label right after the call. `!` swaps the targets (and the fallthrough sense).
gb_internal void x64_build_cond(x64Procedure *p, Ast *cond, isize true_lbl, isize false_lbl,
                                bool true_is_fallthrough) {
	cond = unparen_expr(cond);
	if (cond->kind == Ast_UnaryExpr && cond->UnaryExpr.op.kind == Token_Not) {
		x64_build_cond(p, cond->UnaryExpr.expr, false_lbl, true_lbl, !true_is_fallthrough);
		return;
	}
	if (cond->kind == Ast_BinaryExpr) {
		TokenKind op = cond->BinaryExpr.op.kind;
		if (op == Token_CmpAnd) {
			isize mid = x64_label_alloc(&p->asm_);
			x64_build_cond(p, cond->BinaryExpr.left, mid, false_lbl, /*true_is_fallthrough*/true);
			x64_label_bind(&p->asm_, mid);
			x64_build_cond(p, cond->BinaryExpr.right, true_lbl, false_lbl, true_is_fallthrough);
			return;
		}
		if (op == Token_CmpOr) {
			isize mid = x64_label_alloc(&p->asm_);
			x64_build_cond(p, cond->BinaryExpr.left, true_lbl, mid, /*true_is_fallthrough*/false);
			x64_label_bind(&p->asm_, mid);
			x64_build_cond(p, cond->BinaryExpr.right, true_lbl, false_lbl, true_is_fallthrough);
			return;
		}
	}
	x64Value cv = x64_build_expr(p, cond);
	x64_value_to_reg(p, cv, X64Reg_RAX);
	x64_emit_test_rr(&p->asm_, x64_op_size_of(cond->tav.type), X64Reg_RAX, X64Reg_RAX);
	if (true_is_fallthrough) x64_emit_jcc(&p->asm_, X64Cc_E,  false_lbl); // false → jump; true falls through
	else                     x64_emit_jcc(&p->asm_, X64Cc_NE, true_lbl);  // true → jump; false falls through
}

// `when` (#if): the taken branch's stmts join the ENCLOSING scope, not a new block. Build the list
// DIRECTLY (mirrors lb_build_when_stmt) — routing via the BlockStmt handler would push a defer/context
// scope the checker never created.
gb_internal void x64_build_when_stmt(x64Procedure *p, Ast *node) {
	ast_node(ws, WhenStmt, node);
	if (!ws->is_cond_determined) return;
	if (ws->determined_cond) {
		if (ws->body != nullptr && ws->body->kind == Ast_BlockStmt) {
			x64_build_stmt_list(p, ws->body->BlockStmt.stmts);
		}
	} else if (ws->else_stmt != nullptr) {
		if (ws->else_stmt->kind == Ast_BlockStmt) {
			x64_build_stmt_list(p, ws->else_stmt->BlockStmt.stmts);
		} else if (ws->else_stmt->kind == Ast_WhenStmt) {
			x64_build_stmt(p, ws->else_stmt); // `else when` chains
		}
	}
}

gb_internal void x64_build_if_stmt(x64Procedure *p, Ast *node) {
	ast_node(ifs, IfStmt, node);
	// The whole if-statement is a scope (mirrors lb_build_if_stmt's lb_open_scope(is->scope) /
	// lb_close_scope at the end). Defers registered while building the init or the CONDITION must run
	// at if-end — most importantly an `@(deferred_none)` proc called as the condition: the UI idiom
	// `if parent_scope(b) { …children… }` defers parent_pop to the close of the if. Was: x64 had no
	// such scope, so the condition's deferred call leaked past the if and never ran (blick: every
	// `if ui.parent_scope(top_row){}` left its block pushed → following siblings became its children).
	isize defer_base = p->deferred.count;
	i32   ctx_save   = p->ctx_override_off;
	i32   frame_save = p->local_size;
	if (ifs->init != nullptr) x64_build_stmt(p, ifs->init);

	isize lbl_then = x64_label_alloc(&p->asm_);
	isize lbl_else = x64_label_alloc(&p->asm_);
	isize lbl_end  = x64_label_alloc(&p->asm_);

	x64_build_cond(p, ifs->cond, lbl_then, lbl_else, /*true_is_fallthrough*/true);
	x64_label_bind(&p->asm_, lbl_then); // || may jump here; the true-fallthrough also lands here
	x64_build_stmt(p, ifs->body);
	x64_emit_jmp(&p->asm_, lbl_end);

	x64_label_bind(&p->asm_, lbl_else);
	if (ifs->else_stmt != nullptr) x64_build_stmt(p, ifs->else_stmt);
	x64_label_bind(&p->asm_, lbl_end);
	// Run+pop the if-scope's defers (init/condition) on the merged path — both branches reach here.
	// A non-local exit (return/break) inside a branch already ran these via x64_run_deferred_from.
	x64_scope_end(p, defer_base);
	// Never reclaim below escape_floor: a `p = &Foo{}` in a branch (with p in an outer scope)
	// keeps its slot alive past the if (mirrors the BlockStmt reclaim). Was: this reset dropped
	// the escaped Parser in odin/parser parse_package's `if p==nil { p = &Parser{} }` → a later
	// make() reused the slot and corrupted the pointee.
	p->local_size      = gb_max(frame_save, p->escape_floor);
	p->ctx_override_off = ctx_save;
}

gb_internal void x64_build_for_stmt(x64Procedure *p, Ast *node) {
	ast_node(fors, ForStmt, node);
	if (fors->init != nullptr) x64_build_stmt(p, fors->init);

	isize lbl_loop = x64_label_alloc(&p->asm_);
	isize lbl_post = x64_label_alloc(&p->asm_);
	isize lbl_end  = x64_label_alloc(&p->asm_);

	x64_push_loop(p, lbl_end, lbl_post, fors->label);
	x64_label_bind(&p->asm_, lbl_loop);

	if (fors->cond != nullptr) {
		isize lbl_body = x64_label_alloc(&p->asm_);
		x64_build_cond(p, fors->cond, lbl_body, lbl_end, /*true_is_fallthrough*/true);
		x64_label_bind(&p->asm_, lbl_body);
	}

	x64_build_stmt(p, fors->body);

	x64_label_bind(&p->asm_, lbl_post);
	if (fors->post != nullptr) x64_build_stmt(p, fors->post);
	x64_emit_jmp(&p->asm_, lbl_loop);

	x64_label_bind(&p->asm_, lbl_end);
	x64_pop_loop(p);
}

// `#unroll for v[, i] in EXPR { body }` (mirrors lb_build_unroll_range_stmt). The plain form fully
// unrolls a COMPILE-TIME range — interval (const bounds), enum type, fixed/enumerated array, const
// string — emitting the body once per element with v/i set. Was UNHANDLED → the loop was silently
// skipped (body never ran). `#unroll(N)` (a runtime-stepped slice loop) is still a gap.
gb_internal void x64_build_unroll_range_stmt(x64Procedure *p, Ast *stmt) {
	ast_node(rs, UnrollRangeStmt, stmt);
	if (rs->init != nullptr) x64_build_stmt(p, rs->init);

	Ast *val0 = x64_range_strip(rs->val0);
	Ast *val1 = x64_range_strip(rs->val1);
	Entity *e0 = (val0 && !x64_is_blank(val0)) ? entity_of_node(val0) : nullptr;
	Entity *e1 = (val1 && !x64_is_blank(val1)) ? entity_of_node(val1) : nullptr;
	Type *t0 = nullptr, *t1 = nullptr;
	if (e0 != nullptr && e0->kind == Entity_Variable) { x64_alloc_var(p, e0); t0 = e0->type; } else e0 = nullptr;
	if (e1 != nullptr && e1->kind == Entity_Variable) { x64_alloc_var(p, e1); t1 = e1->type; } else e1 = nullptr;

	if (rs->args.count != 0) {
		GB_PANIC("x64 '#unroll(N) for' (runtime-stepped) unimplemented — mirror lb_build_unroll_range_stmt");
		return;
	}

	Ast *expr = unparen_expr(rs->expr);
	TypeAndValue tav = type_and_value_of_expr(expr);

	if (is_ast_range(expr)) {
		TokenKind op  = expr->BinaryExpr.op.kind;
		i64 lo = exact_value_to_i64(expr->BinaryExpr.left->tav.value);
		i64 hi = exact_value_to_i64(expr->BinaryExpr.right->tav.value);
		if (op != Token_RangeHalf) hi += 1; // `..=`/`..` inclusive
		i64 idx = 0;
		for (i64 v = lo; v < hi; v++, idx++) {
			if (e0) x64_store_value(p, x64_build_addr(p, val0), x64v_imm(t0, v));
			if (e1) x64_store_value(p, x64_build_addr(p, val1), x64v_imm(t1, idx));
			x64_build_stmt(p, rs->body);
		}
		return;
	}
	if (tav.mode == Addressing_Type) {
		Type *bet = base_type(type_deref(tav.type));
		GB_ASSERT(bet != nullptr && bet->kind == Type_Enum);
		for_array(i, bet->Enum.fields) {
			Entity *field = bet->Enum.fields[i];
			if (e0) x64_store_value(p, x64_build_addr(p, val0), x64v_imm(t0, exact_value_to_i64(field->Constant.value)));
			if (e1) x64_store_value(p, x64_build_addr(p, val1), x64v_imm(t1, (i64)i));
			x64_build_stmt(p, rs->body);
		}
		return;
	}

	Type *t = base_type(expr->tav.type);
	switch (t->kind) {
	case Type_Basic: { // constant string
		GB_ASSERT(is_type_string(t));
		String str = expr->tav.value.value_string;
		Rune cp = 0; isize offset = 0;
		while (offset < str.len) {
			isize w = gb_utf8_decode(str.text + offset, str.len - offset, &cp);
			if (e0) x64_store_value(p, x64_build_addr(p, val0), x64v_imm(t0, (i64)cp));
			if (e1) x64_store_value(p, x64_build_addr(p, val1), x64v_imm(t1, (i64)offset));
			x64_build_stmt(p, rs->body);
			offset += w;
		}
	} break;
	case Type_Array:
	case Type_EnumeratedArray: {
		i64   count   = (t->kind == Type_Array) ? t->Array.count : t->EnumeratedArray.count;
		Type *et      = (t->kind == Type_Array) ? t->Array.elem  : t->EnumeratedArray.elem;
		i64   esz     = type_size_of(et); if (esz <= 0) esz = 1;
		i64   idx_min = (t->kind == Type_EnumeratedArray && t->EnumeratedArray.min_value)
		                ? exact_value_to_i64(*t->EnumeratedArray.min_value) : 0;
		x64Value arr = x64_build_expr(p, expr);
		x64Value am  = (arr.kind == x64Value_Mem) ? arr : x64_spill_value(p, arr, t);
		i32 base_off = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, am.mem);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base_off), X64Reg_RAX);
		for (i64 i = 0; i < count; i++) {
			if (e0) {
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_off));
				x64Value ev = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, (i32)(i*esz)), et));
				x64_store_value(p, x64_build_addr(p, val0), ev);
			}
			if (e1) x64_store_value(p, x64_build_addr(p, val1), x64v_imm(t1, i + idx_min));
			x64_build_stmt(p, rs->body);
		}
	} break;
	default:
		GB_PANIC("x64 '#unroll for': invalid range type");
		break;
	}
}

gb_internal void x64_build_stmt(x64Procedure *p, Ast *stmt) {
	if (stmt == nullptr) return;

	// CodeView: map the code that follows to this statement's source line.
	x64_record_line(p, stmt);

	// Apply this statement's `#no_bounds_check`/`#bounds_check` (inherited by nested stmts/exprs);
	// restored at the end. Was: x64 only honoured global -no-bounds-check → e.g. the runtime's
	// `#no_bounds_check ... &raw.data[raw.len]` (fixed-cap append at capacity) trapped.
	u16 prev_sf = x64_push_state_flags(p, stmt);

	switch (stmt->kind) {

	case Ast_BadStmt:
	case Ast_EmptyStmt:
		break;

	case_ast_node(ws, WhenStmt, stmt); {
		x64_build_when_stmt(p, stmt);
	} case_end;

	case_ast_node(block, BlockStmt, stmt); {
		isize defer_base = p->deferred.count;
		i32   ctx_save   = p->ctx_override_off; // scoped `context` modifications: restore on exit
		i32   frame_save = p->local_size;       // reclaim this block's locals+temps at scope exit
		// A labeled block (`label: { … break label }`) is a break target: `break label` jumps past
		// the block. Push a loop entry (continue disabled = -1) so x64_find_loop resolves the label;
		// mirrors LLVM's lb_build_stmt BlockStmt target-list path. Was: x64 ignored block->label →
		// `break label` found no loop → silently no-op'd (fell through the block instead of exiting).
		bool  labeled  = block->label != nullptr;
		isize lbl_done = -1;
		if (labeled) {
			lbl_done = x64_label_alloc(&p->asm_);
			x64_push_loop(p, lbl_done, -1, block->label);
		}
		x64_build_stmt_list(p, block->stmts);
		x64_scope_end(p, defer_base); // run+pop defers registered in this block
		// lbl_done binds AFTER scope_end's defer-run code: `break label` emits its OWN defer-run at the
		// break site then jumps here, past scope_end (so defers run exactly once on either path).
		if (labeled) {
			x64_pop_loop(p);
			x64_label_bind(&p->asm_, lbl_done);
		}
		// Lexical scope exit: everything declared in the block is now dead (Odin scope lifetime),
		// so a sibling/later scope can reuse the slots. frame_max keeps the prologue peak.
		// EXCEPT address-escaped temporaries (`p = &Foo{}` with p in an outer scope): their storage
		// outlives the block, so never reclaim below escape_floor (mirrors LLVM entry-hoisted allocas).
		// (Debug: sibling scopes alias slots until S_BLOCK32 lexical-scope records are emitted.)
		p->local_size      = gb_max(frame_save, p->escape_floor);
		p->ctx_override_off = ctx_save; // a context.field=X inside this block doesn't leak out
	} case_end;

	case_ast_node(es, ExprStmt, stmt); {
		x64_build_expr(p, es->expr);
	} case_end;

	case_ast_node(vs, ValueDecl, stmt); {
		x64_build_value_decl(p, stmt);
	} case_end;

	case_ast_node(as, AssignStmt, stmt); {
		x64_build_assign_stmt(p, stmt);
	} case_end;

	case_ast_node(rets, ReturnStmt, stmt); {
		x64_build_return_stmt(p, rets->results);
	} case_end;

	case_ast_node(ifs, IfStmt, stmt); {
		x64_build_if_stmt(p, stmt);
	} case_end;

	case_ast_node(fors, ForStmt, stmt); {
		x64_build_for_stmt(p, stmt);
	} case_end;

	case_ast_node(rs, RangeStmt, stmt); {
		x64_build_range_stmt(p, stmt);
	} case_end;

	// `#unroll for` — fully unrolled compile-time range. Was missing → body silently skipped.
	case_ast_node(urs, UnrollRangeStmt, stmt); {
		x64_build_unroll_range_stmt(p, stmt);
	} case_end;

	case_ast_node(ds, DeferStmt, stmt); {
		x64Procedure::DeferEntry de = {};
		de.stmt = ds->stmt;
		array_add(&p->deferred, de);
	} case_end;

	case_ast_node(bs, BranchStmt, stmt); {
		String lbl_name = (bs->label != nullptr) ? bs->label->Ident.token.string : String{};

		switch (bs->token.kind) {
		case Token_break: {
			x64Procedure::LoopInfo *li = x64_find_loop(p, lbl_name);
			// Run defers registered since the target loop/switch was entered
			// (lbDeferExit_Branch) — NOT outer/fn-level defers.
			if (li) x64_run_deferred_from(p, li->defer_base);
			if (li) x64_emit_jmp(&p->asm_, li->lbl_break);
			break;
		}
		case Token_continue: {
			// Unlabeled continue skips switch entries (lbl_continue == -1) to the
			// nearest real loop.
			isize target = -1;
			isize floor  = 0;
			if (lbl_name.len == 0) {
				for (isize i = p->loops.count - 1; i >= 0; i--) {
					if (p->loops[i].lbl_continue >= 0) {
						target = p->loops[i].lbl_continue;
						floor  = p->loops[i].defer_base;
						break;
					}
				}
			} else {
				x64Procedure::LoopInfo *li = x64_find_loop(p, lbl_name);
				if (li && li->lbl_continue >= 0) { target = li->lbl_continue; floor = li->defer_base; }
			}
			x64_run_deferred_from(p, floor);
			if (target >= 0) x64_emit_jmp(&p->asm_, target);
			break;
		}
		case Token_fallthrough:
			// Jump to the next case clause's body (set per-case by x64_build_switch_stmt). Was a
			// no-op → `case '+': … fallthrough; case 'i': …` never ran the 'i' body (parse_f64 "+inf").
			if (p->fallthrough_lbl >= 0) x64_emit_jmp(&p->asm_, p->fallthrough_lbl);
			break;
		default:
			break;
		}
	} case_end;

	case_ast_node(ss, SwitchStmt, stmt); {
		x64_build_switch_stmt(p, stmt);
	} case_end;

	// Type-switch: switch v in expr { case T: ... }
	case_ast_node(ss, TypeSwitchStmt, stmt); {
		x64_build_type_switch_stmt(p, stmt);
	} case_end;

	default:
		break;
	}

	p->state_flags = prev_sf;
}
