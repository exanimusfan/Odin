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
		x64_emit_push_r(a, X64Reg_RDI);
		x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
		x64_emit_lea(a, X64Reg_RDI, dst);
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

// Strip a leading & prefix (for &b in slice) before entity lookup.
gb_internal Ast *x64_range_strip(Ast *v) {
	if (v && v->kind == Ast_UnaryExpr && v->UnaryExpr.op.kind == Token_And)
		return v->UnaryExpr.expr;
	return v;
}

gb_internal void x64_build_stmt(x64Procedure *p, Ast *stmt) {
	if (stmt == nullptr) return;

	// CodeView: map the code that follows to this statement's source line.
	x64_record_line(p, stmt);

	switch (stmt->kind) {

	case Ast_BadStmt:
	case Ast_EmptyStmt:
		break;

	case_ast_node(ws, WhenStmt, stmt); {
		// `when` (#if): the taken branch's stmts join the ENCLOSING scope, not a new
		// block. Build the list DIRECTLY (mirrors lb_build_when_stmt) — routing via the
		// BlockStmt handler would push a defer/context scope the checker never created
		// (running when-body defers early / scoping context changes to the when).
		if (!ws->is_cond_determined) break;
		if (ws->determined_cond) {
			if (ws->body != nullptr && ws->body->kind == Ast_BlockStmt) {
				for_array(i, ws->body->BlockStmt.stmts) {
					x64_build_stmt(p, ws->body->BlockStmt.stmts[i]);
				}
			}
		} else if (ws->else_stmt != nullptr) {
			if (ws->else_stmt->kind == Ast_BlockStmt) {
				for_array(i, ws->else_stmt->BlockStmt.stmts) {
					x64_build_stmt(p, ws->else_stmt->BlockStmt.stmts[i]);
				}
			} else if (ws->else_stmt->kind == Ast_WhenStmt) {
				x64_build_stmt(p, ws->else_stmt); // `else when` chains
			}
		}
	} case_end;

	case_ast_node(block, BlockStmt, stmt); {
		isize defer_base = p->deferred.count;
		i32   ctx_save   = p->ctx_override_off; // scoped `context` modifications: restore on exit
		for_array(i, block->stmts) {
			x64_build_stmt(p, block->stmts[i]);
		}
		x64_scope_end(p, defer_base); // run+pop defers registered in this block
		p->ctx_override_off = ctx_save; // a context.field=X inside this block doesn't leak out
	} case_end;

	case_ast_node(es, ExprStmt, stmt); {
		x64_build_expr(p, es->expr);
	} case_end;

	case_ast_node(vs, ValueDecl, stmt); {
		if (!vs->is_mutable) {
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
			break;
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
						if (se->Variable.thread_local_model.len != 0) {
							x64_emit_global_tls(p->module, se, decl_info_of_entity(se));
						} else {
							x64_emit_global_variable(p->module, se);
						}
					}
				}
				break;
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

				if (value_count == 0) {
					x64_zero_mem(p, x64_rbp_mem(off), x64_type_size(e->type));
				} else {
					x64Value v = x64_build_expr(p, vs->values[ni]);
					x64_store_value(p, x64addr(x64_rbp_mem(off), e->type), v);
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
			x64Value result = x64_build_expr(p, vs->values[0]);
			// Unpack tuple POSITIONALLY: name[i] ⇐ field i. A blank `_` is skipped but
			// STILL consumes its field — else `_, ok := …` reads field 0 (value) into ok.
			if (result.kind == x64Value_Mem && result.type != nullptr && is_type_tuple(result.type)) {
				Type *tuple   = result.type;
				isize tcount  = tuple->Tuple.variables.count;
				i64   foff    = 0;
				for (int ni = 0; ni < name_count && (isize)ni < tcount; ni++) {
					Entity *tv  = tuple->Tuple.variables[ni];
					i64 fsz     = type_size_of(tv->type); if (fsz <= 0) fsz = 1;
					i64 fal     = type_align_of(tv->type); if (fal <= 0) fal = 1;
					foff        = align_formula(foff, fal);
					if (!x64_is_blank(vs->names[ni])) {
						Entity *ne = entity_of_node(vs->names[ni]);
						if (ne != nullptr && ne->kind == Entity_Variable) {
							i32 *ne_off = x64_var_get(&p->var_offsets, ne);
							if (ne_off) {
								x64Value fv = x64_load_addr(p,
								    x64addr(x64_mem(result.mem.base, result.mem.disp + (i32)foff), tv->type));
								x64_store_value(p, x64addr(x64_rbp_mem(*ne_off), ne->type), fv);
							}
						}
					}
					foff += fsz;
				}
			}
		}
	} case_end;

	case_ast_node(as, AssignStmt, stmt); {
		TokenKind op        = as->op.kind;
		int       lhs_count = (int)as->lhs.count;
		int       rhs_count = (int)as->rhs.count;

		if (op == Token_Eq) {
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
				break;
			}
			if (lhs_count == 1 && rhs_count == 1) {
				if (x64_is_blank(as->lhs[0])) { x64_build_expr(p, as->rhs[0]); break; }
				x64Addr lhs_addr = x64_build_addr(p, as->lhs[0]);
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
					Type *t  = as->rhs[ri]->tav.type;
					if (!t)   t = as->lhs[ri]->tav.type;
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
					x64Value tv      = x64_load_addr(p, x64addr(x64_rbp_mem(rhs_offs[li]), t));
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
					i64 foff = 0;
					for (int i = 0; i < n; i++) {
						Entity *tv = tuple->Tuple.variables[i];
						i64 fsz = type_size_of(tv->type); if (fsz <= 0) fsz = 1;
						i64 fal = type_align_of(tv->type); if (fal <= 0) fal = 1;
						foff = align_formula(foff, fal);
						i32 tmp = x64_alloc_local(p, fsz, fal);
						tmp_offs[i] = tmp; ftypes[i] = tv->type;
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(tup_off));
						x64Value fv = x64_load_addr(p,
						    x64addr(x64_mem(X64Reg_RAX, (i32)foff), tv->type));
						x64_store_value(p, x64addr(x64_rbp_mem(tmp), tv->type), fv);
						foff += fsz;
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
			// Compound assignment (+=, -=, …); single lhs only.
			if (lhs_count == 1 && rhs_count == 1) {
				x64Addr  lhs_addr = x64_build_addr(p, as->lhs[0]);
				// Use the lvalue's address type (lb_addr_type(lhs)), not tav.type, which
				// doesn't reliably resolve to bit_set here.
				Type    *t        = x64_typed(lhs_addr.type != nullptr ? lhs_addr.type : as->lhs[0]->tav.type);
				X64OpSize sz      = x64_op_size_of(t);

				// Stabilise &lhs to a stack slot: RHS eval and the op below clobber
				// RAX/RCX/RDX, which could be the LHS base register (e.g. ptr.field += x).
				i32 ea_off = x64_alloc_local(p, 8, 8);
				x64_emit_lea(&p->asm_, X64Reg_RAX, lhs_addr.mem);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ea_off), X64Reg_RAX);

				// RHS FIRST (clobbers RAX) and spill, so the LHS load into RAX survives.
				x64Value rv = x64_build_expr(p, as->rhs[0]);
				i32 rhs_off = x64_alloc_local(p, 8, 8);
				x64_store_value(p, x64addr(x64_rbp_mem(rhs_off), t), rv);

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
	} case_end;

	case_ast_node(rets, ReturnStmt, stmt); {
		Type *pt = p->type;
		GB_ASSERT(pt->kind == Type_Proc);

		x64_run_deferred(p);

		int res_count = (int)rets->results.count;

		Type *results_tuple = pt->Proc.results;
		int   nres = (results_tuple != nullptr) ? (int)results_tuple->Tuple.variables.count : 0;

		if (res_count == 0) {
			// bare return — write named return locals back per the result ABI
			x64_emit_named_returns(p);
		} else if (nres <= 1) {
			x64Value v  = x64_build_expr(p, rets->results[0]);
			Type    *rt = (nres == 1) ? results_tuple->Tuple.variables[0]->type : nullptr;
			if (rt == nullptr || type_size_of(rt) == 0) {
				// void / zero-sized: nothing to return
			} else if (p->returns_by_pointer) {
				i64 rsz = type_size_of(rt);
				if (v.kind != x64Value_Mem) {
					i32 so = x64_alloc_local(p, rsz, type_align_of(rt));
					x64_store_value(p, x64addr(x64_rbp_mem(so), rt), v);
					v = x64v_mem(rt, x64_rbp_mem(so));
				}
				x64_emit_lea(&p->asm_, X64Reg_RAX, v.mem);          // RAX = &src
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RDX,
				                x64_rbp_mem(x64_param_rbp_off(0)));  // RDX = sret ptr
				x64_copy_fixed(p, x64_mem(X64Reg_RDX, 0), x64_mem(X64Reg_RAX, 0), rsz);
			} else if (x64_is_float(rt)) {
				x64_value_to_xmm(p, v, X64XmmReg_XMM0);
			} else {
				x64_value_to_reg(p, v, X64Reg_RAX);
			}
		} else {
			// Split returns (mirror LLVM): results[0..N-2] go through hidden pointer args,
			// result[N-1] in sret/RAX. Build all N into stack temps first (building clobbers
			// RAX and the hidden-ptr regs; the ptrs survive in their homed shadow slots).
			Array<i32>   toff; array_init(&toff, temporary_allocator(), nres, nres);
			Array<Type*> ttyp; array_init(&ttyp, temporary_allocator(), nres, nres);
			for (int i = 0; i < nres; i++) ttyp[i] = results_tuple->Tuple.variables[i]->type;

			if (res_count == 1) {
				// `return multi_valued_call()` — one expression yields the whole tuple.
				x64Value tv = x64_build_expr(p, rets->results[0]);
				i64 tsz = type_size_of(results_tuple);
				if (tv.kind != x64Value_Mem) {
					i32 so = x64_alloc_local(p, tsz, type_align_of(results_tuple));
					x64_store_value(p, x64addr(x64_rbp_mem(so), results_tuple), tv);
					tv = x64v_mem(results_tuple, x64_rbp_mem(so));
				}
				i32 base = x64_alloc_local(p, 8, 8);
				x64_emit_lea(&p->asm_, X64Reg_RAX, tv.mem);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base), X64Reg_RAX);
				i64 foff = 0;
				for (int i = 0; i < nres; i++) {
					i64 fsz = type_size_of(ttyp[i]); if (fsz <= 0) fsz = 1;
					i64 fal = type_align_of(ttyp[i]); if (fal <= 0) fal = 1;
					foff = (foff + (fal - 1)) & ~(fal - 1);
					i32 t = x64_alloc_local(p, fsz, fal); toff[i] = t;
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base));
					x64_copy_fixed(p, x64_rbp_mem(t), x64_mem(X64Reg_RAX, (i32)foff), fsz);
					foff += fsz;
				}
			} else {
				// `return v0, v1, ...` — one expression per result.
				for (int i = 0; i < nres && i < res_count; i++) {
					i64 fsz = type_size_of(ttyp[i]); if (fsz <= 0) fsz = 1;
					i64 fal = type_align_of(ttyp[i]); if (fal <= 0) fal = 1;
					x64Value v = x64_build_expr(p, rets->results[i]);
					i32 t = x64_alloc_local(p, fsz, fal); toff[i] = t;
					x64_store_value(p, x64addr(x64_rbp_mem(t), ttyp[i]), v);
				}
			}

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
					x64_emit_mov_rm(&p->asm_, x64_op_size_of(ttyp[i]), X64Reg_RAX, x64_rbp_mem(toff[i]));
				}
			}
		}

		x64_proc_emit_epilogue(p);
		x64_emit_ret(&p->asm_);
	} case_end;

	case_ast_node(ifs, IfStmt, stmt); {
		if (ifs->init != nullptr) x64_build_stmt(p, ifs->init);

		x64Value cv = x64_build_expr(p, ifs->cond);
		x64_value_to_reg(p, cv, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, x64_op_size_of(ifs->cond->tav.type), X64Reg_RAX, X64Reg_RAX);

		isize lbl_else = x64_label_alloc(&p->asm_);
		isize lbl_end  = x64_label_alloc(&p->asm_);

		x64_emit_jcc(&p->asm_, X64Cc_E, lbl_else);
		x64_build_stmt(p, ifs->body);
		x64_emit_jmp(&p->asm_, lbl_end);

		x64_label_bind(&p->asm_, lbl_else);
		if (ifs->else_stmt != nullptr) x64_build_stmt(p, ifs->else_stmt);
		x64_label_bind(&p->asm_, lbl_end);
	} case_end;

	case_ast_node(fors, ForStmt, stmt); {
		if (fors->init != nullptr) x64_build_stmt(p, fors->init);

		isize lbl_loop = x64_label_alloc(&p->asm_);
		isize lbl_post = x64_label_alloc(&p->asm_);
		isize lbl_end  = x64_label_alloc(&p->asm_);

		x64_push_loop(p, lbl_end, lbl_post, fors->label);
		x64_label_bind(&p->asm_, lbl_loop);

		if (fors->cond != nullptr) {
			x64Value cv = x64_build_expr(p, fors->cond);
			x64_value_to_reg(p, cv, X64Reg_RAX);
			x64_emit_test_rr(&p->asm_, x64_op_size_of(fors->cond->tav.type), X64Reg_RAX, X64Reg_RAX);
			x64_emit_jcc(&p->asm_, X64Cc_E, lbl_end);
		}

		x64_build_stmt(p, fors->body);

		x64_label_bind(&p->asm_, lbl_post);
		if (fors->post != nullptr) x64_build_stmt(p, fors->post);
		x64_emit_jmp(&p->asm_, lbl_loop);

		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
	} case_end;

	case_ast_node(rs, RangeStmt, stmt); {
		Ast  *iter_expr = rs->expr;
		Type *iter_type = iter_expr->tav.type ? base_type(x64_typed(iter_expr->tav.type)) : nullptr;

		isize lbl_loop = x64_label_alloc(&p->asm_);
		isize lbl_post = x64_label_alloc(&p->asm_);
		isize lbl_end  = x64_label_alloc(&p->asm_);
		x64_push_loop(p, lbl_end, lbl_post, rs->label);

		i32 loop_idx_off = x64_alloc_local(p, 8, 8);
		x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(loop_idx_off), 0);

		// Odin RangeStmt ordering: vals[0] = element value, vals[1] = index.
		Entity *elem_e = nullptr; i32 elem_off = 0;
		Entity *ridx_e = nullptr; i32 ridx_off = 0;

		if (rs->vals.count > 0) {
			Ast *v = x64_range_strip(rs->vals[0]);
			if (v && !x64_is_blank(v)) {
				elem_e = entity_of_node(v);
				if (elem_e && elem_e->kind == Entity_Variable) elem_off = x64_alloc_var(p, elem_e);
				else elem_e = nullptr;
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

		// Integer interval `lo ..< hi` / `lo ..= hi` (mirrors lb_build_range_interval).
		// iter_type is the element type, not an aggregate → handle before aggregate dispatch.
		if (is_ast_range(iter_expr)) {
			ast_node(be, BinaryExpr, iter_expr);
			bool inclusive = (be->op.kind == Token_RangeFull || be->op.kind == Token_Ellipsis);
			Type *vt = iter_expr->tav.type ? x64_typed(iter_expr->tav.type) : t_int;
			X64OpSize vsz = x64_op_size_of(vt);
			bool vsigned = x64_is_signed_integer(vt);

			i32 value_off = elem_e ? elem_off : x64_alloc_local(p, 8, 8);
			i32 index_off = ridx_e ? ridx_off : loop_idx_off; // loop_idx_off already 0
			if (ridx_e) x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(index_off), 0);

			x64Value lo = x64_build_expr(p, be->left); // value = lower
			x64_value_to_reg(p, lo, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, vsz, x64_rbp_mem(value_off), X64Reg_RAX);

			x64_label_bind(&p->asm_, lbl_loop);
			// upper re-evaluated each iteration (matches LLVM)
			x64Value hi = x64_build_expr(p, be->right);
			x64_value_to_reg(p, hi, X64Reg_RCX);
			x64_emit_mov_rm(&p->asm_, vsz, X64Reg_RAX, x64_rbp_mem(value_off));
			x64_emit_cmp_rr(&p->asm_, vsz, X64Reg_RAX, X64Reg_RCX);
			// end when value is past upper: >= (half) or > (inclusive)
			X64Cc cc_end = inclusive ? (vsigned ? X64Cc_G  : X64Cc_A)
			                         : (vsigned ? X64Cc_GE : X64Cc_AE);
			x64_emit_jcc(&p->asm_, cc_end, lbl_end);

			x64_build_stmt(p, rs->body);

			x64_label_bind(&p->asm_, lbl_post);
			x64_emit_mov_rm(&p->asm_, vsz, X64Reg_RAX, x64_rbp_mem(value_off));
			x64_emit_inc_r(&p->asm_, vsz, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, vsz, x64_rbp_mem(value_off), X64Reg_RAX);
			if (index_off != value_off) {
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(index_off));
				x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(index_off), X64Reg_RAX);
			}
			x64_emit_jmp(&p->asm_, lbl_loop);

			x64_label_bind(&p->asm_, lbl_end);
			x64_pop_loop(p);
			break;
		}

		if (iter_type == nullptr) {
			// Unknown range type — skip body, bind labels
			x64_label_bind(&p->asm_, lbl_loop);
			x64_label_bind(&p->asm_, lbl_post);
			x64_label_bind(&p->asm_, lbl_end);
			x64_pop_loop(p);
			break;
		} else if (iter_type->kind == Type_Array || iter_type->kind == Type_EnumeratedArray) {
			// Enumerated arrays share fixed-array storage; only the iteration variable
			// (the enum index) differs — it is the ordinal plus the enum's min value.
			bool  is_enum = iter_type->kind == Type_EnumeratedArray;
			i64   arr_len = is_enum ? iter_type->EnumeratedArray.count : iter_type->Array.count;
			Type *elem_t  = is_enum ? iter_type->EnumeratedArray.elem  : iter_type->Array.elem;
			i64   idx_min = (is_enum && iter_type->EnumeratedArray.min_value)
			              ? exact_value_to_i64(*iter_type->EnumeratedArray.min_value) : 0;
			i64   elem_sz = type_size_of(elem_t);
			x64Addr base_addr = x64_build_addr(p, iter_expr);
			// Pin the array base to a stack slot: base_addr.mem may be RAX-relative
			// (globals via LEA [RIP+sym]) and the in-loop index load clobbers RAX.
			i32 arr_ptr_off = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, base_addr.mem);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(arr_ptr_off), X64Reg_RAX);

			x64_label_bind(&p->asm_, lbl_loop);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)arr_len);
			x64_emit_jcc(&p->asm_, X64Cc_AE, lbl_end);

			// Index var (vals[1]); enumerated arrays: enum value = ordinal + min.
			if (ridx_e) {
				if (idx_min != 0) {
					x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)idx_min);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ridx_off), X64Reg_RAX);
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
				} else {
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ridx_off), X64Reg_RAX);
				}
			}

			if (elem_e) {
				x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX, (i32)elem_sz);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(arr_ptr_off));
				x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
				x64Value ev = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), elem_t));
				x64_store_value(p, x64addr(x64_rbp_mem(elem_off), elem_t), ev);
			}

			x64_build_stmt(p, rs->body);

			x64_label_bind(&p->asm_, lbl_post);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(loop_idx_off), X64Reg_RAX);
			x64_emit_jmp(&p->asm_, lbl_loop);

		} else if (iter_type->kind == Type_Slice || iter_type->kind == Type_DynamicArray) {
			Type *elem_t  = (iter_type->kind == Type_Slice)
			              ? iter_type->Slice.elem
			              : iter_type->DynamicArray.elem;
			i64   elem_sz = type_size_of(elem_t);
			x64Addr slice_addr = x64_build_addr(p, iter_expr);

			i32 data_off = x64_alloc_local(p, 8, 8);
			i32 len_off  = x64_alloc_local(p, 8, 8);

			// Pin the slice struct addr in RAX, read .data (off 0)/.len (off 8) via RCX.
			// slice_addr.mem may be register-relative (e.g. `data.loggers`, `data` a ^T in
			// RAX) — reading .data straight into RAX would clobber the base before .len@+8.
			x64_emit_lea(&p->asm_, X64Reg_RAX, slice_addr.mem);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 0));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_off), X64Reg_RCX);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);

			x64_label_bind(&p->asm_, lbl_loop);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
			x64_emit_jcc(&p->asm_, X64Cc_AE, lbl_end);

			if (ridx_e) x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ridx_off), X64Reg_RAX);

			if (elem_e) {
				x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX, (i32)elem_sz);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(data_off));
				x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
				x64Value ev = x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), elem_t));
				x64_store_value(p, x64addr(x64_rbp_mem(elem_off), elem_t), ev);
			}

			x64_build_stmt(p, rs->body);

			x64_label_bind(&p->asm_, lbl_post);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(loop_idx_off), X64Reg_RAX);
			x64_emit_jmp(&p->asm_, lbl_loop);

		} else if (iter_type->kind == Type_Struct && iter_type->Struct.soa_kind != StructSoa_None) {
			// #soa range. Mirrors lb_build_range_stmt_struct_soa but DIVERGES: LLVM binds
			// `field` as a soa variable (soa_values + selector redirection field.x →
			// array.x[i]); we MATERIALIZE the element struct each iteration (elem.x =
			// array.x[i]). Identical for the value form `for field in …`; only `&field`
			// write-back would differ, which #soa value-range never does.
			// Layout: fields[0..n-1] = per-member multipointers, field[n] = __$len.
			Type *elem_bt = base_type(iter_type->Struct.soa_elem);
			type_set_offsets(iter_type);
			if (elem_bt != nullptr) type_set_offsets(elem_bt);
			int n = (elem_bt != nullptr && elem_bt->kind == Type_Struct)
			      ? (int)elem_bt->Struct.fields.count : 0;

			// The #soa value may come from a call (sret) — build once, pin its address.
			x64Value sv = x64_build_expr(p, iter_expr);
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

			x64_label_bind(&p->asm_, lbl_loop);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
			x64_emit_jcc(&p->asm_, X64Cc_AE, lbl_end);

			if (ridx_e) x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ridx_off), X64Reg_RAX);

			if (elem_e && elem_bt != nullptr) {
				for (int f = 0; f < n; f++) {
					Type *ft  = elem_bt->Struct.fields[f]->type;
					i64   fsz = type_size_of(ft);
					if (fsz <= 0) continue;
					i32 dst_off = elem_off + (i32)elem_bt->Struct.offsets[f];
					// src = soa.field[f] (a [^]ft) + index*fsz
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(loop_idx_off));
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(soa_ea));
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX,
					                x64_mem(X64Reg_RAX, (i32)iter_type->Struct.offsets[f]));
					if (fsz != 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RDX, (i32)fsz);
					x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
					x64_copy_fixed(p, x64_rbp_mem(dst_off), x64_mem(X64Reg_RCX, 0), fsz);
				}
			}

			x64_build_stmt(p, rs->body);

			x64_label_bind(&p->asm_, lbl_post);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(loop_idx_off));
			x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(loop_idx_off), X64Reg_RAX);
			x64_emit_jmp(&p->asm_, lbl_loop);

		} else {
			x64_build_expr(p, iter_expr); // unsupported type: eval for side effects
		}

		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
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
		default:
			break;
		}
	} case_end;

	case_ast_node(ss, SwitchStmt, stmt); {
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

						// if tag < lo, skip to next case
						x64Value lv = x64_build_expr(p, rbe->left);
						x64Value tv = x64_load_addr(p, x64addr(x64_rbp_mem(tag_off), tag_t));
						x64_value_to_reg(p, tv, X64Reg_RAX);
						x64_value_to_reg(p, lv, X64Reg_RCX);
						x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
						x64_emit_jcc(&p->asm_, lt_cc, skip);

						// if tag > hi (..=) or tag >= hi (..<), skip
						x64Value hv  = x64_build_expr(p, rbe->right);
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
					x64Value tv = x64_load_addr(p, x64addr(x64_rbp_mem(tag_off), tag_t));
					x64_value_to_reg(p, tv, X64Reg_RAX);
					x64_value_to_reg(p, v,  X64Reg_RCX);
					x64_emit_cmp_rr(&p->asm_, x64_op_size_of(tag_t), X64Reg_RAX, X64Reg_RCX);
					x64_emit_jcc(&p->asm_, X64Cc_E, case_lbls[ci]);
				} else {
					x64_value_to_reg(p, v, X64Reg_RAX);
					x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
					x64_emit_jcc(&p->asm_, X64Cc_NE, case_lbls[ci]);
				}
			}
		}
		x64_emit_jmp(&p->asm_, default_idx >= 0 ? case_lbls[default_idx] : lbl_end);

		for (isize ci = 0; ci < body->stmts.count; ci++) { // case bodies
			Ast *cc_ast = body->stmts[ci];
			if (cc_ast == nullptr || cc_ast->kind != Ast_CaseClause) continue;
			AstCaseClause *cc = &cc_ast->CaseClause;

			x64_label_bind(&p->asm_, case_lbls[ci]);
			isize defer_base = p->deferred.count;
			for_array(si, cc->stmts) {
				x64_build_stmt(p, cc->stmts[si]);
			}
			x64_scope_end(p, defer_base); // case-local defers run here, not at fn exit
			x64_emit_jmp(&p->asm_, lbl_end); // implicit break (Odin has no fallthrough by default)
		}

		array_free(&case_lbls);
		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
	} case_end;

	// Type-switch: switch v in expr { case T: ... }
	case_ast_node(ss, TypeSwitchStmt, stmt); {
		if (ss->tag == nullptr || ss->tag->kind != Ast_AssignStmt) break;
		AstAssignStmt *tag_assign = &ss->tag->AssignStmt;
		if (tag_assign->rhs.count < 1) break;
		Ast *tag_expr = tag_assign->rhs[0];

		x64Value parent_v   = x64_build_expr(p, tag_expr);
		Type    *parent_raw = x64_typed(tag_expr->tav.type);
		Type    *parent_bt  = base_type(parent_raw);
		if (parent_bt == nullptr) break;

		// Auto-deref: switch v in ptr_to_union / ptr_to_any
		if (is_type_pointer(parent_bt)) {
			parent_bt = base_type(type_deref(parent_raw));
		}

		TypeSwitchKind switch_kind = check_valid_type_switch_type(parent_raw);
		if (switch_kind == TypeSwitch_Invalid) break;

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
			i64 tag_off_b = parent_bt->Union.variant_block_size;
			i64 tag_sz    = union_tag_size(parent_bt);
			X64OpSize opsz = tag_sz <= 1 ? X64OpSize_8 : tag_sz == 2 ? X64OpSize_16 :
			                 tag_sz == 4 ? X64OpSize_32 : X64OpSize_64;
			X64Mem tag_m = x64_mem(X64Reg_RAX, (i32)tag_off_b);
			if (opsz == X64OpSize_64) x64_emit_mov_rm  (&p->asm_, X64OpSize_64, X64Reg_RCX, tag_m);
			else                      x64_emit_movzx_rm(&p->asm_, opsz,         X64Reg_RCX, tag_m);
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
				case_type = x64_typed(case_type);
				if (is_type_untyped_nil(case_type)) {
					x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
					x64_emit_jcc(&p->asm_, X64Cc_E, case_lbls[ci]);
				} else if (switch_kind == TypeSwitch_Any) {
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
					i64 ct_sz = x64_type_size(ct);
					if (ct_sz > 0) {
						bool specific = (cc->list.count == 1); // bound to one concrete type
						i32 ent_off = x64_alloc_local(p, ct_sz, x64_type_align(ct));
						x64_zero_mem(p, x64_rbp_mem(ent_off), ct_sz);

						// RCX = &source data
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(parent_ea_off));
						if (switch_kind == TypeSwitch_Any && specific) {
							// Deref: RCX = any.data (&actual T). Single concrete type only;
							// a multi/default binding keeps the `any` itself, copied as-is.
							x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RCX, 0));
						}
						// Union: variant data (or whole union for multi/default) is at offset 0,
						// so RCX already = &operand.

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

			isize defer_base = p->deferred.count;
			for_array(si, cc->stmts) x64_build_stmt(p, cc->stmts[si]);
			x64_scope_end(p, defer_base); // case-local defers run here, not at fn exit
			x64_emit_jmp(&p->asm_, lbl_end);
		}

		array_free(&case_lbls);
		x64_label_bind(&p->asm_, lbl_end);
		x64_pop_loop(p);
	} case_end;

	default:
		break;
	}
}
