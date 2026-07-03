// x64 debug backend — expression code generation.

// ─────────────────────────────────────────────────────────────────────────────
// Float constant materialisation
// ─────────────────────────────────────────────────────────────────────────────

// Load a float constant into XMM0 via GPR + stack (avoids .rdata relocs).
gb_internal void x64_emit_float_const(x64Procedure *p, Type *t, f64 fval, f32 fval32) {
	X64Assembler *a  = &p->asm_;
	bool         is_d = x64_is_double(t);

	if (is_d) {
		u64 bits = 0;
		gb_memmove(&bits, &fval, 8);
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, (i64)bits);
		x64_emit_push_r(a, X64Reg_RAX);
		x64_emit_movsd_rm(a, X64XmmReg_XMM0, x64_mem(X64Reg_RSP, 0));
		x64_emit_pop_r(a, X64Reg_RAX);
	} else {
		u32 bits = 0;
		gb_memmove(&bits, &fval32, 4);
		x64_emit_mov_ri(a, X64OpSize_32, X64Reg_RAX, (i64)(u64)bits);
		x64_emit_push_r(a, X64Reg_RAX);
		x64_emit_movss_rm(a, X64XmmReg_XMM0, x64_mem(X64Reg_RSP, 0));
		x64_emit_pop_r(a, X64Reg_RAX);
	}
}

// Materialise a string into a {data,len} stack struct. Byte data is INTERNED
// (one .rdata global per unique string — mirrors lb_find_or_add_entity_string_ptr).
gb_internal x64Value x64_const_string(x64Procedure *p, String sv) {
	x64Module *m = p->module;
	i32 str_off = x64_alloc_local(p, 16, 8);
	if (sv.len > 0) {
		String sym = x64_const_intern_string(m, sv);
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(str_off), X64Reg_RAX);
	} else {
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 0); // empty → null data ptr
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(str_off), X64Reg_RAX);
	}
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)sv.len);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(str_off + 8), X64Reg_RAX);
	return x64v_mem(t_string, x64_rbp_mem(str_off));
}

// ─────────────────────────────────────────────────────────────────────────────
// Context (Odin's implicit `context`)
// ─────────────────────────────────────────────────────────────────────────────

// Ensure a local Context exists for a NON-Odin proc that references `context`.
// Mirrors lb_find_or_generate_context_ptr's generated-local path. Returns the
// RBP offset of the Context body. Only valid when !p->has_context.
gb_internal i32 x64_ensure_local_context(x64Procedure *p) {
	if (p->gen_context_off != 0) return p->gen_context_off;

	i64 csz = type_size_of(t_context); if (csz <= 0) csz = 8;
	i64 cal = type_align_of(t_context); if (cal <= 0) cal = 8;
	p->gen_context_off = x64_alloc_local(p, csz, cal);

	// runtime.__init_context(&ctx) — contextless, a single ^Context argument.
	AstPackage *rt_pkg = p->module->gen->info->runtime_package;
	Entity *ic = (rt_pkg != nullptr)
	    ? scope_lookup_current(rt_pkg->scope,
	          string_interner_insert(str_lit("__init_context")))
	    : nullptr;
	if (ic != nullptr && ic->kind == Entity_Procedure) {
		// Called by name (not via CallExpr), so the on-demand path won't see it —
		// enqueue min_dep==0 helpers ourselves or the symbol is unresolved at link.
		if (!ic->Procedure.is_foreign &&
		    ic->min_dep_count.load(std::memory_order_relaxed) == 0) {
			x64_enqueue_oncall(p, ic);
		}
		i32 ptr = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(p->gen_context_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
		x64Value cargs[1];
		cargs[0] = x64v_mem(t_context_ptr, x64_rbp_mem(ptr));
		x64_emit_call(p, x64_get_entity_name(ic), ic->type, cargs, 1);
	}
	return p->gen_context_off;
}

// Return a mem value holding a POINTER to the active Context, for the implicit
// context arg. Odin procs: the incoming context-ptr param slot; others: the
// generated local. Materialized into a fresh stack slot so it survives the
// caller's arg marshalling (which clobbers RAX).
gb_internal x64Value x64_context_ptr_value(x64Procedure *p) {
	// Scoped `context.field = X` installed a fresh local Context (mirrors pushed lbAddr_Context).
	if (p->ctx_override_off != 0) {
		i32 ptr = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(p->ctx_override_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
		return x64v_mem(t_context_ptr, x64_rbp_mem(ptr));
	}
	if (p->has_context && p->context_slot >= 0) {
		return x64v_mem(t_context_ptr,
		                x64_rbp_mem(x64_param_rbp_off(p->context_slot)));
	}
	i32 off = x64_ensure_local_context(p);
	i32 ptr = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
	return x64v_mem(t_context_ptr, x64_rbp_mem(ptr));
}

// Address of the active Context's BODY (the struct, not a pointer to it),
// resolved like x64_context_ptr_value. NOTE: the param case loads the pointer
// into RAX, so the returned Mem is RAX-relative — consume it before clobbering RAX.
gb_internal X64Mem x64_current_context_body(x64Procedure *p) {
	if (p->ctx_override_off != 0) {
		return x64_rbp_mem(p->ctx_override_off);
	}
	if (p->has_context && p->context_slot >= 0) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX,
		                x64_rbp_mem(x64_param_rbp_off(p->context_slot)));
		return x64_mem(X64Reg_RAX, 0);
	}
	i32 off = x64_ensure_local_context(p);
	return x64_rbp_mem(off);
}

// Snapshot the current Context into a fresh local and make it active (override),
// so `context.field = X` is a SCOPED modification — visible to the scope and
// callees but not the caller. Saved/restored around block scopes. Mirrors
// lb_addr_store's lbAddr_Context path.
gb_internal void x64_push_new_context(x64Procedure *p) {
	i64 csz = type_size_of(t_context); if (csz <= 0) csz = 8;
	i64 cal = type_align_of(t_context); if (cal <= 0) cal = 8;
	X64Mem cur = x64_current_context_body(p); // may load RAX (param case)
	i32 new_off = x64_alloc_local(p, csz, cal);
	x64_copy_fixed(p, x64_rbp_mem(new_off), cur, csz);
	p->ctx_override_off = new_off;
}

// Value for an omitted default parameter (mirrors lb_handle_param_value). For
// ParameterValue_Constant, materialise the stored ExactValue directly rather than
// re-evaluating original_ast_expr — that AST node often lacks a usable tav/entity
// in the backend, silently yielding 0 (e.g. `flush := true` arriving as false).
gb_internal x64Value x64_handle_param_value(x64Procedure *p, Type *ptype,
                                            ParameterValue const &pv) {
	Type *t  = x64_typed(ptype);
	Type *bt = t ? base_type(t) : nullptr;
	switch (pv.kind) {
	case ParameterValue_Constant: {
		ExactValue ev = pv.value;
		switch (ev.kind) {
		case ExactValue_Bool:
			return x64v_imm(t, ev.value_bool ? 1 : 0);
		case ExactValue_Integer:
			return x64v_imm(t, big_int_to_i64(&ev.value_integer));
		case ExactValue_Float:
			x64_emit_float_const(p, bt, ev.value_float, (f32)ev.value_float);
			return x64v_xmm(t, X64XmmReg_XMM0);
		case ExactValue_String:
			return x64_const_string(p, ev.value_string);
		case ExactValue_Procedure: {
			Entity *pe = entity_from_expr(ev.value_procedure);
			if (pe != nullptr && pe->kind == Entity_Procedure) {
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(pe));
				return x64v_reg(t, X64Reg_RAX);
			}
			return x64v_imm(t, 0);
		}
		default:
			return x64v_imm(t, 0);
		}
	}
	case ParameterValue_Value:
		return x64_build_expr(p, pv.ast_value);
	case ParameterValue_Nil:
		break; // zero — handled by caller (imm 0 / zero-fill)
	default:
		// Location / Expression (#caller_location, #caller_expression): the AST
		// path is the only source; let the caller's zero-fill cover it.
		if (pv.original_ast_expr != nullptr) {
			x64Value v = x64_build_expr(p, pv.original_ast_expr);
			if (v.kind != x64Value_None) return v;
		}
		break;
	}
	return x64v_none();
}

// `using`-field subtype conversion: narrow a struct value `v` to one of its `using`
// field types when that field type is expected (e.g. `Temp_Allocator{using allocator}`
// passed as `Allocator`). Mirrors lb_emit_conv's check_is_assignable_to_using_subtype
// path. Handles VALUE field paths (offset adjust); bails on pointer-deref paths
// (unneeded by arg lowering).
gb_internal x64Value x64_apply_using_subtype(x64Procedure *p, x64Value v, Type *want) {
	if (v.type == nullptr || want == nullptr) return v;
	if (are_types_identical(v.type, want)) return v;
	if (check_is_assignable_to_using_subtype(v.type, want) == 0) return v;
	Selection sel = {};
	sel.index.allocator = temporary_allocator();
	if (!lookup_subtype_polymorphic_selection(want, v.type, &sel)) return v;
	if (sel.entity == nullptr || sel.index.count == 0) return v;

	i64 off = 0;
	Type *cur = base_type(v.type);
	for_array(i, sel.index) {
		if (cur == nullptr || cur->kind != Type_Struct) return v; // raw_union / pointer-deref path: unsupported here
		type_set_offsets(cur);
		i32 idx = sel.index[i];
		if (idx < 0 || idx >= (i32)cur->Struct.fields.count) return v;
		off += cur->Struct.offsets[idx];
		cur = base_type(cur->Struct.fields[idx]->type);
	}
	if (v.kind == x64Value_Mem) {
		X64Mem m = v.mem; m.disp += (i32)off;
		return x64v_mem(want, m);
	}
	return v; // non-mem struct (rare) — leave as-is
}

// ─────────────────────────────────────────────────────────────────────────────
// `or_else` / `or_return` / `or_break` / `or_continue` shared helpers
// (mirror lb_emit_try_lhs_rhs / lb_emit_try_has_value)
// ─────────────────────────────────────────────────────────────────────────────

// Build `arg`, split into success value (`lhs`) and ok/err indicator (`rhs` = last
// tuple field). Non-tuple result: rhs = the value, lhs none. `result_type` is the
// type lhs is read as.
gb_internal void x64_try_lhs_rhs(x64Procedure *p, Ast *arg, Type *result_type,
                                 x64Value *lhs_, x64Value *rhs_) {
	x64Value rv  = x64_build_expr(p, arg);
	x64Value rhs = rv;
	x64Value lhs = x64v_none();

	Type *it = rv.type;
	if (it != nullptr && is_type_tuple(it) && rv.kind == x64Value_Mem) {
		TypeTuple *tup = &it->Tuple;
		int n = (int)tup->variables.count;
		i64 rhs_off = 0;
		for (int i = 0; i < n - 1; i++) {
			Entity *e = tup->variables[i];
			i64 al = type_align_of(e->type);
			rhs_off = (rhs_off + al - 1) & ~(al - 1);
			rhs_off += type_size_of(e->type);
		}
		Entity *last_e = tup->variables[n - 1];
		i64 al = type_align_of(last_e->type);
		rhs_off = (rhs_off + al - 1) & ~(al - 1);
		rhs = x64v_mem(last_e->type, x64_mem(rv.mem.base, rv.mem.disp + (i32)rhs_off));
		if (n >= 2) {
			Type *lt = result_type ? result_type : tup->variables[0]->type;
			lhs = x64v_mem(lt, rv.mem); // value part occupies the leading fields
		}
	}
	if (lhs_) *lhs_ = lhs;
	if (rhs_) *rhs_ = rhs;
}

// OR one chunk of `rhs` (at byte offset `b`) into RCX, zero-extending if requested.
gb_internal void x64_emit_or_chunk(x64Procedure *p, x64Value rhs, i64 b, X64OpSize wsz, bool zext) {
	X64Mem m = rhs.mem; m.disp += (i32)b;
	if (zext) x64_emit_movzx_rm(&p->asm_, wsz, X64Reg_RAX, m);
	else      x64_emit_mov_rm  (&p->asm_, wsz, X64Reg_RAX, m);
	x64_emit_or_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
}

// Leaves RAX = "has value" (non-zero ⇒ success / use lhs). Boolean rhs is its own
// truthiness; otherwise success means rhs == nil (mirrors lb_emit_try_has_value →
// lb_emit_comp_against_nil for the practical error types: enum/pointer/integer,
// and an all-zero check for larger nil-able aggregates like #shared_nil unions).
gb_internal void x64_try_has_value(x64Procedure *p, x64Value rhs) {
	Type *t = rhs.type ? rhs.type : t_bool;
	if (x64_is_bool(t)) {
		x64_value_to_reg(p, rhs, X64Reg_RAX); // truthy = success
		return;
	}
	i64 sz = type_size_of(t);
	if (sz <= 8) {
		x64_value_to_reg(p, rhs, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, x64_op_size_of(t), X64Reg_RAX, X64Reg_RAX);
		x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX); // AL = (rhs == nil)
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return;
	}
	// Larger nil-able value: OR every byte (8/4/2/1 chunks, zero-extended so stale
	// high bits don't pollute the test); success when every byte is zero.
	if (rhs.kind == x64Value_Mem) {
		x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RCX, X64Reg_RCX);
		i64 b = 0;
		while (sz - b >= 8) { x64_emit_or_chunk(p, rhs, b, X64OpSize_64, false); b += 8; }
		if (sz - b >= 4)    { x64_emit_or_chunk(p, rhs, b, X64OpSize_32, false); b += 4; } // 32-bit mov zero-extends
		if (sz - b >= 2)    { x64_emit_or_chunk(p, rhs, b, X64OpSize_16, true);  b += 2; }
		if (sz - b >= 1)    { x64_emit_or_chunk(p, rhs, b, X64OpSize_8,  true);          }
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX);
		x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return;
	}
	x64_value_to_reg(p, rhs, X64Reg_RAX);
}

// Two-operand scalar (int/float) min/max, returned as a stack value of type `rt`.
// Shared by min/max/clamp (mirrors lb_emit_min/max reused by lb_emit_clamp).
gb_internal x64Value x64_emit_minmax2(x64Procedure *p, Type *rt,
                                      x64Value a, x64Value b, bool is_min) {
	i64 sz = type_size_of(rt); if (sz <= 0) sz = 8;
	i64 al = type_align_of(rt); if (al <= 0) al = 8;
	i32 a_off = x64_alloc_local(p, sz, al);
	x64_store_value(p, x64addr(x64_rbp_mem(a_off), rt), a);
	i32 b_off = x64_alloc_local(p, sz, al);
	x64_store_value(p, x64addr(x64_rbp_mem(b_off), rt), b);
	i32 r_off = x64_alloc_local(p, sz, al);

	if (x64_is_float(rt)) {
		bool dbl = x64_is_double(rt);
		if (dbl) {
			x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(a_off));
			x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM1, x64_rbp_mem(b_off));
			x64_emit_ucomisd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
		} else {
			x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(a_off));
			x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM1, x64_rbp_mem(b_off));
			x64_emit_ucomiss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
		}
		// Keep a unless it must be replaced by b: min→a>b, max→a<b.
		isize keep = x64_label_alloc(&p->asm_);
		x64_emit_jcc(&p->asm_, is_min ? X64Cc_BE : X64Cc_AE, keep);
		if (dbl) x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(b_off));
		else     x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(b_off));
		x64_label_bind(&p->asm_, keep);
		if (dbl) x64_emit_movsd_mr(&p->asm_, x64_rbp_mem(r_off), X64XmmReg_XMM0);
		else     x64_emit_movss_mr(&p->asm_, x64_rbp_mem(r_off), X64XmmReg_XMM0);
		return x64v_mem(rt, x64_rbp_mem(r_off));
	}

	X64OpSize osz = x64_op_size_of(rt);
	bool sgn = x64_is_signed_integer(rt);
	// Replace a with b when: min→a>b, max→a<b.
	X64Cc cc = is_min ? (sgn ? X64Cc_G : X64Cc_A) : (sgn ? X64Cc_L : X64Cc_B);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(a_off));
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(b_off));
	x64_emit_cmp_rr(&p->asm_, osz, X64Reg_RAX, X64Reg_RCX);
	x64_emit_cmov_rr(&p->asm_, cc, osz, X64Reg_RAX, X64Reg_RCX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(r_off), X64Reg_RAX);
	return x64v_mem(rt, x64_rbp_mem(r_off));
}

// Store an array-like CompoundLit's elements into a contiguous area at `base_off`,
// element type `et`/size `esz`. Handles positional (`{a,b}`), indexed (`{[2]=x}`),
// and ranges (`{[lo..=hi]=x}`). `index_offset` is subtracted from explicit indices
// (enumerated-array min; 0 otherwise). Matrices map row-major indices through
// matrix_row_major_index_to_offset (column-major storage). Mirrors LLVM's
// lb_build_addr_compound_lit_populate + _assign_array, collapsed to one store pass
// (x64 has no const backing to overwrite).
gb_internal void x64_store_compound_elems(x64Procedure *p, Slice<Ast *> const &elems,
                                          Type *bt, Type *et, i64 esz, i32 base_off, i64 index_offset) {
	bool is_matrix = (bt->kind == Type_Matrix);
	isize elem_index = 0;
	for_array(ei, elems) {
		Ast *elem = elems[ei];
		if (elem == nullptr) { elem_index++; continue; }

		if (elem->kind == Ast_FieldValue) {
			AstFieldValue *fv = &elem->FieldValue;
			if (is_ast_range(fv->field)) {
				ast_node(ie, BinaryExpr, fv->field);
				i64 lo = exact_value_to_i64(ie->left->tav.value);
				i64 hi = exact_value_to_i64(ie->right->tav.value);
				if (ie->op.kind != Token_RangeHalf) hi += 1;
				x64Value ev = x64_build_expr(p, fv->value);
				for (i64 k = lo; k < hi; k++) {
					i64 slot = is_matrix ? matrix_row_major_index_to_offset(bt, k) : (k - index_offset);
					x64_store_value(p, x64addr(x64_mem(X64Reg_RBP, base_off + (i32)(slot * esz)), et), ev);
				}
			} else {
				i64 index = exact_value_to_i64(fv->field->tav.value);
				i64 slot = is_matrix ? matrix_row_major_index_to_offset(bt, index) : (index - index_offset);
				x64Value ev = x64_build_expr(p, fv->value);
				x64_store_value(p, x64addr(x64_mem(X64Reg_RBP, base_off + (i32)(slot * esz)), et), ev);
			}
		} else {
			// Positional: i-th element → slot i (enumerated arrays fill from the first element).
			i64 slot = is_matrix ? matrix_row_major_index_to_offset(bt, elem_index) : elem_index;
			x64Value ev = x64_build_expr(p, elem);
			x64_store_value(p, x64addr(x64_mem(X64Reg_RBP, base_off + (i32)(slot * esz)), et), ev);
			elem_index++;
		}
	}
}

// Compute the bit_set membership mask `1 << (elem - lower)` into RAX. The element value
// must already be in RCX (which is also clobbered by the `- lower`); `sz` is the backing
// integer size. Shared by the bit_set literal builder and `in`/`not_in` — keeping these in
// lockstep is what prevents the literal-vs-membership divergence class of bugs.
gb_internal void x64_emit_bit_set_mask(x64Procedure *p, X64OpSize sz, i64 lower) {
	if (lower != 0) x64_emit_sub_ri(&p->asm_, sz, X64Reg_RCX, (i32)lower);
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
	x64_emit_shl_rcl(&p->asm_, sz, X64Reg_RAX);
}

// Spill a value to a fresh stack local and return it as an RBP-relative mem, so it survives
// a later register-clobbering sub-build (the recurring clobber-bug guard). Sized by `t` (the
// value's type). For VALUE spills only — address spills (lea+mov) and fixed-width slots stay
// inline where their exact size/semantics matter.
gb_internal x64Value x64_spill_value(x64Procedure *p, x64Value v, Type *t) {
	if (t == nullptr) t = v.type ? v.type : t_int;
	i64 sz = x64_type_size(t);  if (sz <= 0) sz = 8;
	i64 al = x64_type_align(t); if (al <= 0) al = 8;
	i32 off = x64_alloc_local(p, sz, al);
	x64_store_value(p, x64addr(x64_rbp_mem(off), t), v);
	return x64v_mem(t, x64_rbp_mem(off));
}

// Materialise a `CompoundLit` AST into a zeroed stack slot. Shared by the CompoundLit
// expr case and the constant path (an aggregate constant is ExactValue_Compound over the
// same AST) — mirrors lb_const_value / lb_build_addr_compound_lit. Covers every non-dynamic
// kind (struct, [N]T/[E]T/matrix/#simd arrays, slice, bit_set, bit_field, any); map /
// [dynamic]T are feature-gated and left as their zero value.
gb_internal x64Value x64_build_compound_lit(x64Procedure *p, Ast *lit, Type *type) {
	ast_node(cl, CompoundLit, lit);
	Type *bt = base_type(type);
	i64   sz = x64_type_size(type);
	i64   al = x64_type_align(type);
	if (sz <= 0) sz = 8;
	if (al <= 0) al = 8;
	i32    res_off = x64_alloc_local(p, sz, al);
	X64Mem res_m   = x64_rbp_mem(res_off);
	x64_zero_mem(p, res_m, sz);

	if (cl->elems.count == 0) {
		return x64v_mem(type, res_m);
	}

	if (bt->kind == Type_Struct) {
		type_set_offsets(bt);
		bool named = (cl->elems[0] != nullptr && cl->elems[0]->kind == Ast_FieldValue);
		if (named) {
			for_array(ei, cl->elems) {
				Ast *elem = cl->elems[ei];
				if (elem == nullptr || elem->kind != Ast_FieldValue) continue;
				AstFieldValue *fv = &elem->FieldValue;
				if (fv->field == nullptr || fv->field->kind != Ast_Ident) continue;
				String fname = fv->field->Ident.token.string;
				i64   foff  = 0;
				Type *ftype = t_int;
				for_array(fi, bt->Struct.fields) {
					Entity *fe = bt->Struct.fields[fi];
					if (fe->token.string == fname) {
						foff  = bt->Struct.offsets[fi];
						ftype = fe->type;
						break;
					}
				}
				x64Value fval = x64_build_expr(p, fv->value);
				x64_store_value(p, x64addr(x64_mem(X64Reg_RBP, res_off + (i32)foff), ftype), fval);
			}
		} else {
			isize n = gb_min((isize)cl->elems.count, (isize)bt->Struct.fields.count);
			for (isize ei = 0; ei < n; ei++) {
				if (cl->elems[ei] == nullptr) continue;
				Entity *fe   = bt->Struct.fields[ei];
				i64    foff  = bt->Struct.offsets[ei];
				x64Value fval = x64_build_expr(p, cl->elems[ei]);
				x64_store_value(p, x64addr(x64_mem(X64Reg_RBP, res_off + (i32)foff), fe->type), fval);
			}
		}
	} else if (bt->kind == Type_Array) {
		Type *et  = bt->Array.elem;
		i64   esz = type_size_of(et); if (esz <= 0) esz = 1;
		x64_store_compound_elems(p, cl->elems, bt, et, esz, res_off, 0);
	} else if (bt->kind == Type_EnumeratedArray) {
		Type *et  = bt->EnumeratedArray.elem;
		i64   esz = type_size_of(et); if (esz <= 0) esz = 1;
		i64   off = bt->EnumeratedArray.min_value ? exact_value_to_i64(*bt->EnumeratedArray.min_value) : 0;
		x64_store_compound_elems(p, cl->elems, bt, et, esz, res_off, off);
	} else if (bt->kind == Type_SimdVector) {
		Type *et  = bt->SimdVector.elem;
		i64   esz = type_size_of(et); if (esz <= 0) esz = 1;
		x64_store_compound_elems(p, cl->elems, bt, et, esz, res_off, 0);
	} else if (bt->kind == Type_Matrix) {
		Type *et  = bt->Matrix.elem;
		i64   esz = type_size_of(et); if (esz <= 0) esz = 1;
		x64_store_compound_elems(p, cl->elems, bt, et, esz, res_off, 0);
	} else if (bt->kind == Type_Slice) {
		// `{a, b, …}` typed as []T: build a backing array on the stack, store elements, then
		// fill the header {data=&array, len=N} (mirrors lb_build_addr's CompoundLit Type_Slice).
		// Without this the header was just zeroed → callee saw a nil slice. len = max(elem
		// count, highest index+1) so indexed elements past the positional count fit (cl->max_count).
		Type *et  = bt->Slice.elem;
		i64   esz = type_size_of(et);  if (esz <= 0) esz = 1;
		i64   eal = type_align_of(et); if (eal <= 0) eal = 1;
		i64   n   = gb_max((i64)cl->elems.count, (i64)cl->max_count);
		i32 arr_off = x64_alloc_local(p, n * esz, eal);
		x64_zero_mem(p, x64_rbp_mem(arr_off), n * esz);
		x64_store_compound_elems(p, cl->elems, bt, et, esz, arr_off, 0);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(arr_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);     // .data
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, n);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off + 8), X64Reg_RAX); // .len
	} else if (bt->kind == Type_BitSet) {
		// `bit_set[E]{.A, .B}`: OR `1 << (elem - lower)` into the zeroed integer backing per
		// member (mirrors LLVM's integer-backed path; array-backed is a TODO there too).
		Type *backing = bit_set_to_int(bt);
		if (!is_type_array(backing) && x64_type_size(type) <= 8) {
			Type     *it  = bit_set_to_int(bt);
			X64OpSize osz = x64_op_size_of(it);
			i64       lower = bt->BitSet.lower;
			for_array(ei, cl->elems) {
				Ast *elem = cl->elems[ei];
				if (elem == nullptr) continue;
				x64Value ev = x64_build_expr(p, elem);
				x64_value_to_reg(p, ev, X64Reg_RCX);                        // RCX = elem
				x64_emit_bit_set_mask(p, osz, lower);                       // RAX = 1 << (elem-lower)
				x64_emit_or_mr(&p->asm_, osz, res_m, X64Reg_RAX);           // backing |= bit
			}
		}
	} else if (bt->kind == Type_BitField) {
		// `T{field = v, …}`: per field, mask the value to its bit width, shift to the field's
		// bit offset, OR into the zeroed integer backing (mirrors LLVM's single-integer path;
		// array-of-integer backing is a rare TODO, left zeroed).
		Type *backing = core_type(bt->BitField.backing_type);
		if (is_type_integer(backing) && x64_type_size(type) <= 8) {
			X64OpSize osz = x64_op_size_of(backing);
			for_array(ei, cl->elems) {
				Ast *elem = cl->elems[ei];
				if (elem == nullptr || elem->kind != Ast_FieldValue) continue;
				AstFieldValue *fv = &elem->FieldValue;
				if (fv->field == nullptr || fv->field->kind != Ast_Ident) continue;
				String fname = fv->field->Ident.token.string;
				i64 bit_offset = 0, bit_size = 0; bool found = false;
				for_array(fi, bt->BitField.fields) {
					if (bt->BitField.fields[fi]->token.string == fname) {
						bit_offset = bt->BitField.bit_offsets[fi];
						bit_size   = (i64)bt->BitField.bit_sizes[fi];
						found = true; break;
					}
				}
				if (!found || bit_size <= 0) continue;
				x64Value ev = x64_build_expr(p, fv->value);
				x64_value_to_reg(p, ev, X64Reg_RAX);                        // RAX = value
				if (bit_size < 64) {
					u64 mask = (1ull << (u64)bit_size) - 1;
					x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)mask);
					x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
				}
				if (bit_offset != 0) x64_emit_shl_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (u8)bit_offset);
				x64_emit_or_mr(&p->asm_, osz, res_m, X64Reg_RAX);           // backing |= field
			}
		}
	} else if (bt->kind == Type_Basic && is_type_any(bt)) {
		// `any{data, id}`: data: rawptr @0, id: typeid @ptr_size. Positional or named.
		i64 ptr_sz = type_size_of(t_rawptr);
		for_array(ei, cl->elems) {
			Ast *elem = cl->elems[ei];
			if (elem == nullptr) continue;
			isize index = ei;
			Ast  *val   = elem;
			if (elem->kind == Ast_FieldValue) {
				AstFieldValue *fv = &elem->FieldValue;
				index = (fv->field != nullptr && fv->field->kind == Ast_Ident &&
				         fv->field->Ident.token.string == str_lit("data")) ? 0 : 1;
				val = fv->value;
			}
			Type *ft   = (index == 0) ? t_rawptr : t_typeid;
			i64   foff = (index == 0) ? 0 : ptr_sz;
			x64Value ev = x64_build_expr(p, val);
			x64_store_value(p, x64addr(x64_mem(X64Reg_RBP, res_off + (i32)foff), ft), ev);
		}
	}
	// Type_Map / Type_DynamicArray / Type_FixedCapacityDynamicArray literals need runtime
	// calls (__dynamic_map_reserve/_set, __dynamic_array_reserve/_append) + a Source_Code_Location
	// global; feature-gated, left as the zero value (a valid empty container) for now.

	return x64v_mem(type, res_m);
}

// ─────────────────────────────────────────────────────────────────────────────
// Address building (lvalue → X64Mem)
// ─────────────────────────────────────────────────────────────────────────────

// Load the base for `agg[i]` into RAX. Array-like: its ADDRESS; slice/dynarray/
// multi-pointer/string: the `.data` pointer (`data_ptr`). `via_ptr`: aggregate reached
// through a pointer (`p[i]`), whose VALUE is its address.
gb_internal void x64_index_base_to_rax(x64Procedure *p, Ast *agg, bool via_ptr, bool data_ptr) {
	if (via_ptr) {
		x64_value_to_reg(p, x64_build_expr(p, agg), X64Reg_RAX);                       // RAX = &aggregate
		if (data_ptr) x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 0)); // → .data
	} else if (data_ptr) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_build_addr(p, agg).mem); // RAX = .data (first qword)
	} else {
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_build_addr(p, agg).mem);                  // RAX = &array
	}
}

// With the base address already in RAX, compute the element address `base + (index - sub)*esz`
// and return it. `idx_mem` holds the index (RBP-relative, stable across this). `sub` is the
// enumerated-array min (0 otherwise); `esz <= 1` skips the multiply (byte/string indexing).
gb_internal x64Addr x64_index_elem_addr(x64Procedure *p, x64Value idx_mem, i64 esz, i64 sub, Type *elem_t) {
	x64_value_to_reg(p, idx_mem, X64Reg_RCX);
	if (sub != 0)  x64_emit_sub_ri (&p->asm_, X64OpSize_64, X64Reg_RCX, (i32)sub);
	if (esz > 1)   x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX, (i32)esz);
	x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
	return x64addr(x64_mem(X64Reg_RAX, 0), elem_t);
}

gb_internal x64Addr x64_build_addr(x64Procedure *p, Ast *expr) {
	expr = unparen_expr(expr);
	Type *t = expr->tav.type;

	// A constant has no storage or linker symbol. AGGREGATE constants → emitted once into
	// .rdata and referenced by address (mirrors lb_add_global_generated_from_procedure for
	// &CONST). SCALAR constants → a stack local. Without this, &named/compound constant fell
	// through to the Ident global path → LEA [RIP+name] → unresolved.
	if (expr->tav.mode == Addressing_Constant) {
		Type *ct = t ? x64_typed(t) : t_int;
		i64 sz = x64_type_size(ct); if (sz <= 0) sz = 8;
		i64 al = x64_type_align(ct); if (al <= 0) al = 8;

		Type *bt = base_type(ct);
		bool scalar = is_type_boolean(bt) || is_type_integer(bt) || is_type_float(bt) ||
		              is_type_pointer(bt) || is_type_enum(bt);
		if (!scalar && expr->tav.value.kind != ExactValue_Invalid &&
		    x64_const_type_supported(ct)) {
			x64Module *m = p->module;
			u32 goff = x64_const_reserve(m->rdata, al, sz);
			x64_const_value(m, m->rdata, goff, ct, expr->tav.value, expr->tav.type);
			String sym = x64_const_new_name(m, "__$ccg$"); // compiler const global
			x64_const_define_symbol(m, sym, 2 /*.rdata*/, goff);
			x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
			return x64addr(x64_mem(X64Reg_RAX, 0), ct);
		}

		i32 off = x64_alloc_local(p, sz, al);
		x64_zero_mem(p, x64_rbp_mem(off), sz); // covers nil / partial-scalar constants
		x64Value v = x64_build_expr(p, expr);
		if (v.kind != x64Value_None) x64_store_value(p, x64addr(x64_rbp_mem(off), ct), v);
		return x64addr(x64_rbp_mem(off), ct);
	}

	switch (expr->kind) {

	case_ast_node(ident, Ident, expr); {
		Entity *e = ident->entity.load(std::memory_order_relaxed);
		if (e == nullptr) {
			// Synthetic / #optional_allocator_error body: entity not linked. Fall back
			// to name-based lookup through results then params.
			String name = ident->token.string;
			Type  *pt   = p->type;
			if (pt->Proc.results != nullptr) {
				for_array(ri, pt->Proc.results->Tuple.variables) {
					Entity *rv = pt->Proc.results->Tuple.variables[ri];
					if (rv->token.string == name) { e = rv; break; }
				}
			}
			if (e == nullptr && pt->Proc.params != nullptr) {
				for_array(ri, pt->Proc.params->Tuple.variables) {
					Entity *pv = pt->Proc.params->Tuple.variables[ri];
					if (pv->token.string == name) { e = pv; break; }
				}
			}
		}
		if (e == nullptr) {
			// Still unresolved — spill a zero temp so compilation continues.
			i64 sz  = t ? x64_type_size(t)  : 8;
			i64 al  = t ? x64_type_align(t) : 8;
			i32 off = x64_alloc_local(p, sz, al);
			return x64addr(x64_rbp_mem(off), t ? t : t_int);
		}
		// Non-addressable entity kinds: imports, libraries, types.
		if (e->kind == Entity_ImportName || e->kind == Entity_LibraryName ||
		    e->kind == Entity_TypeName) {
			i64 sz  = t ? x64_type_size(t)  : 8;
			i64 al  = t ? x64_type_align(t) : 8;
			i32 off = x64_alloc_local(p, sz, al);
			return x64addr(x64_rbp_mem(off), t ? t : t_int);
		}
		// Variables neither local nor file-scope are struct fields etc. — accessed via
		// SelectorExpr, not global symbols; return a dummy not LEA [RIP+name]. EXCEPTION:
		// @(static)/@(thread_local) locals are lowered to real module globals (see ValueDecl),
		// so they must use the global path (LEA / TLS seq), not this dummy-slot fallback.
		if (e->kind == Entity_Variable &&
		    !(e->flags & EntityFlag_Static) &&
		    e->Variable.thread_local_model.len == 0 &&
		    (e->scope == nullptr || (e->scope->flags & ScopeFlag_File) == 0)) {
			if (!x64_entity_is_local(p, e)) {
				i64 sz  = t ? x64_type_size(t)  : 8;
				i64 al  = t ? x64_type_align(t) : 8;
				i32 off = x64_alloc_local(p, sz, al);
				return x64addr(x64_rbp_mem(off), t ? t : t_int);
			}
		}
		if (x64_entity_is_local(p, e)) {
			return x64_entity_addr(p, e);
		}
		// File-scope global VARIABLE → a real symbol: LEA RAX, [RIP+sym].
		if (e->kind == Entity_Variable) {
			x64_load_global_addr(p, e);
			return x64addr(x64_mem(X64Reg_RAX, 0), e->type);
		}
		// Anything else (constants not marked Addressing_Constant, nil, proc values, …)
		// has no storage symbol — materialise into a local (avoids LEA [RIP+name] for a
		// non-global; the `nil` unresolved external).
		{
			Type *ct = t ? x64_typed(t) : t_int;
			i64 sz = x64_type_size(ct); if (sz <= 0) sz = 8;
			i64 al = x64_type_align(ct); if (al <= 0) al = 8;
			i32 off = x64_alloc_local(p, sz, al);
			x64_zero_mem(p, x64_rbp_mem(off), sz);
			x64Value v = x64_build_expr(p, expr);
			if (v.kind != x64Value_None) x64_store_value(p, x64addr(x64_rbp_mem(off), ct), v);
			return x64addr(x64_rbp_mem(off), ct);
		}
	} case_end;

	case_ast_node(deref, DerefExpr, expr); {
		x64Value pv = x64_build_expr(p, deref->expr);
		x64_value_to_reg(p, pv, X64Reg_RAX);
		return x64addr(x64_mem(X64Reg_RAX, 0), t);
	} case_end;

	case_ast_node(sel, SelectorExpr, expr); {
		// Package-qualified selector (e.g. `runtime.args__`): base is an import name with no
		// value (Addressing_Invalid), so build the SELECTOR directly — it resolves to the
		// global/proc entity (mirrors lb_build_addr). Without this the field-offset path below
		// would build off the package "value" (garbage).
		if (unparen_expr(sel->selector)->kind == Ast_Ident &&
		    sel->expr->tav.mode == Addressing_Invalid) {
			return x64_build_addr(p, unparen_expr(sel->selector));
		}
		Type  *base_type_raw = sel->expr->tav.type;
		x64Addr base;

		if (is_type_pointer(base_type_raw)) {
			x64Value pv = x64_build_expr(p, sel->expr);
			x64_value_to_reg(p, pv, X64Reg_RAX);
			base = x64addr(x64_mem(X64Reg_RAX, 0), type_deref(base_type_raw));
		} else {
			base = x64_build_addr(p, sel->expr);
		}

		Entity *field = entity_of_node(sel->selector);

		Type *st       = base_type(base.type);
		i64   field_off = 0;
		Type *field_type = x64_typed(t);

		// `any` is Basic (not Type_Struct): {data: rawptr @0, id: typeid @8}. The
		// struct-offset path would leave field_off=0, so handle it explicitly (else
		// `v.id` reads the data pointer).
		if (is_type_any(st) && sel->selector->kind == Ast_Ident) {
			String fn = sel->selector->Ident.token.string;
			i64    ao = (fn == "id") ? 8 : 0;
			Type  *at = (fn == "id") ? t_typeid : t_rawptr;
			X64Mem am = base.mem;
			am.disp += (i32)ao;
			return x64addr(am, at);
		}

		// Single-component array swizzle: v.x/.y/.z/.w → &v[i]; lookup_field maps the
		// component to the element index (same Selection lb_build_addr walks). Without this
		// .x/.y/.z all resolved to offset 0 (the {24,0,0} bug). swizzle_count > 0 (multi-
		// component) builds a separate value and falls through.
		if ((st->kind == Type_Array || st->kind == Type_EnumeratedArray) &&
		    sel->swizzle_count == 0 && sel->selector->kind == Ast_Ident) {
			Selection s = lookup_field(st, sel->selector->Ident.interned, false);
			if (s.index.count >= 1) {
				Type *elem = (st->kind == Type_Array) ? st->Array.elem : st->EnumeratedArray.elem;
				i64   esz  = type_size_of(elem);
				X64Mem am  = base.mem;
				am.disp += (i32)((i64)s.index[0] * esz);
				return x64addr(am, field_type ? field_type : elem);
			}
		}

		if (st->kind == Type_Struct) {
			// Prefer entity-pointer match, fall back to name match (entity_of_node can be
			// null/mismatched for imported-struct fields, e.g. io.Stream.data). Do NOT fall
			// through to the slice pseudo-field layout — struct offsets differ.
			type_set_offsets(st);
			bool found = false;
			if (field != nullptr) {
				for_array(fi, st->Struct.fields) {
					if (st->Struct.fields[fi] == field) {
						field_off  = st->Struct.offsets[fi];
						field_type = field->type;
						found = true;
						break;
					}
				}
			}
			if (!found && sel->selector->kind == Ast_Ident) {
				String fn = sel->selector->Ident.token.string;
				for_array(fi, st->Struct.fields) {
					Entity *sf = st->Struct.fields[fi];
					if (sf->token.string == fn) {
						field_off  = st->Struct.offsets[fi];
						field_type = sf->type;
						found = true;
						break;
					}
				}
			}
			// PROMOTED field via `using`: the loops above only see DIRECT fields, so a promoted
			// field falls through with field_off=0 and aliases offset 0. Walk the Selection
			// index path instead — summing each level's offset and dereferencing intermediate
			// pointer-embedded `using` members (mirrors lb_emit_deep_field_gep). Was the thread
			// bug: Thread.win32_thread_id (promoted, off 8) resolved to 0 and overwrote the handle.
			if (!found && sel->selector->kind == Ast_Ident) {
				Selection s = lookup_field(st, sel->selector->Ident.interned, false);
				if (s.entity != nullptr && s.index.count > 0) {
					X64Mem m   = base.mem;
					Type  *cur = st;
					for_array(i, s.index) {
						Type *cb = base_type(cur);
						if (cb == nullptr || cb->kind != Type_Struct) break;
						type_set_offsets(cb);
						i32 fi = s.index[i];
						if (fi < 0 || fi >= (i32)cb->Struct.fields.count) break;
						Entity *fe = cb->Struct.fields[fi];
						m.disp += (i32)cb->Struct.offsets[fi];
						cur = fe->type;
						// Intermediate `using p: ^T`: load the pointer, continue from it.
						if (i + 1 < s.index.count && is_type_pointer(base_type(cur))) {
							x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, m);
							m   = x64_mem(X64Reg_RAX, 0);
							cur = type_deref(cur);
						}
					}
					return x64addr(m, x64_typed(s.entity->type));
				}
			}
		} else if (field != nullptr) {
			field_type = field->type;
		} else {
			// Built-in pseudo-field on slice / string / dynamic array / map.
			// Layout (64-bit): data@0(8), len@8(8), cap@16(8), allocator@24(16+).
			String fname = sel->selector->Ident.token.string;
			if      (fname == "data")      { field_off =  0; }
			else if (fname == "len")       { field_off =  8; }
			else if (fname == "cap")       { field_off = 16; }
			else if (fname == "allocator") { field_off = 24; }
			// field_type stays as the tav.type of the full selector expression
		}

		X64Mem m = base.mem;
		m.disp += (i32)field_off;
		return x64addr(m, field_type ? field_type : t_int);
	} case_end;

	case_ast_node(idx, IndexExpr, expr); {
		Type *arr_t = base_type(idx->expr->tav.type);
		// Auto-deref an indexed pointer: `p[i]` (p: ^[N]T / ^[]T) indexes THROUGH the pointer
		// — like C's p->[i] (mirrors lb_build_addr). Without this a pointer base matches no
		// branch below and falls to a garbage temp.
		bool via_ptr = is_type_pointer(arr_t);
		if (via_ptr) arr_t = base_type(type_deref(idx->expr->tav.type));
		x64Value index_v = x64_build_expr(p, idx->index);
		// Spill the index BEFORE building the base: the base load reuses RAX/RCX and would
		// clobber a register-held index (was the _INTEGER_DIGITS_VAR[u%b] crash). Reloaded
		// from memory after the base is in RAX.
		Type *it = x64_typed(index_v.type ? index_v.type : t_int);
		x64Value idx_mem = x64_spill_value(p, index_v, it);
		i64 esz = type_size_of(t);

		if (arr_t->kind == Type_Array) {
			x64_index_base_to_rax(p, idx->expr, via_ptr, /*data_ptr*/false); // RAX = &array
			return x64_index_elem_addr(p, idx_mem, esz, /*sub*/0, t);
		}

		if (arr_t->kind == Type_EnumeratedArray) {
			// arr[enum_val]: element index = ORDINAL = value - min_value (dense storage).
			i64 min_v = arr_t->EnumeratedArray.min_value
			          ? exact_value_to_i64(*arr_t->EnumeratedArray.min_value) : 0;
			x64_index_base_to_rax(p, idx->expr, via_ptr, /*data_ptr*/false);
			return x64_index_elem_addr(p, idx_mem, esz, min_v, t);
		}

		if (arr_t->kind == Type_Slice || arr_t->kind == Type_DynamicArray || arr_t->kind == Type_MultiPointer) {
			// Slice/DA: index off the .data pointer (first qword). MultiPointer: the pointer IS .data.
			x64_index_base_to_rax(p, idx->expr, via_ptr, /*data_ptr*/true);
			return x64_index_elem_addr(p, idx_mem, esz, /*sub*/0, t);
		}

		// String indexing: str[i] → data_ptr + i (byte; esz=1 skips the multiply).
		if (arr_t->kind == Type_Basic && (arr_t->Basic.flags & BasicFlag_String)) {
			x64_index_base_to_rax(p, idx->expr, via_ptr, /*data_ptr*/true);
			return x64_index_elem_addr(p, idx_mem, /*esz*/1, /*sub*/0, t);
		}
		// Fallback: materialise to a temp
		{
			i64 fsz = t ? x64_type_size(t) : 8;
			i64 fal = t ? x64_type_align(t) : 8;
			i32 off = x64_alloc_local(p, fsz, fal);
			return x64addr(x64_rbp_mem(off), t ? t : t_u8);
		}
	} case_end;

	case_ast_node(implicit, Implicit, expr); {
		// 'context' keyword. Odin procs receive a context ptr in the last param slot;
		// others get a generated local Context (mirrors lb_find_or_generate_context_ptr).
		if (implicit->kind == Token_context) {
			// Scoped `context.field = X` installed a fresh local Context.
			if (p->ctx_override_off != 0) {
				return x64addr(x64_rbp_mem(p->ctx_override_off), t_context);
			}
			if (p->has_context && p->context_slot >= 0) {
				X64Mem ctx_ptr_mem = x64_rbp_mem(x64_param_rbp_off(p->context_slot));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, ctx_ptr_mem);
				return x64addr(x64_mem(X64Reg_RAX, 0), t_context);
			}
			i32 off = x64_ensure_local_context(p);
			return x64addr(x64_rbp_mem(off), t_context);
		}
		// Fallback for other implicit tokens (should not occur in practice)
		i64 sz2 = t ? x64_type_size(t) : 8;
		i64 al2 = t ? x64_type_align(t) : 8;
		if (sz2 <= 0) sz2 = 8;
		if (al2 <= 0) al2 = 8;
		i32 off2 = x64_alloc_local(p, sz2, al2);
		return x64addr(x64_rbp_mem(off2), t ? t : t_int);
	} case_end;

	default: {
		// Evaluate, spill to a stack local, return its address. Covers BasicLit, CompoundLit, etc.
		Type *t = expr->tav.type ? x64_typed(expr->tav.type) : t_int;
		i64 sz = x64_type_size(t);
		i64 al = x64_type_align(t);
		if (sz <= 0) sz = 8;
		if (al <= 0) al = 8;
		i32 off = x64_alloc_local(p, sz, al);
		x64Value v = x64_build_expr(p, expr);
		x64_store_value(p, x64addr(x64_rbp_mem(off), t), v);
		return x64addr(x64_rbp_mem(off), t);
	}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic intrinsics (intrinsics.atomic_*)
// ─────────────────────────────────────────────────────────────────────────────
// x86-64 strong memory model: aligned ≤8-byte load/store is already atomic, so atomic_load
// = plain MOV, atomic_store = LOCK XCHG (atomic + full barrier, valid for seq_cst). RMW:
// LOCK XADD (add/sub) / LOCK XCHG (exchange) return the OLD value; or/and/xor/nand use a
// LOCK CMPXCHG retry loop; compare_exchange = one LOCK CMPXCHG → (old, ok). `_explicit`
// ordering args are ignored (LOCK ops are seq_cst; x86 loads/stores need no fence). Without
// these, atomics silently no-op'd (sync.Mutex/Sema, thread flags) → threads hung. Mirrors
// LLVM lb_build_builtin_proc's atomic cases.
gb_internal bool x64_is_atomic_builtin(i32 id) {
	switch (id) {
	case BuiltinProc_atomic_thread_fence: case BuiltinProc_atomic_signal_fence:
	case BuiltinProc_atomic_load:         case BuiltinProc_atomic_load_explicit:
	case BuiltinProc_atomic_store:        case BuiltinProc_atomic_store_explicit:
	case BuiltinProc_atomic_exchange:     case BuiltinProc_atomic_exchange_explicit:
	case BuiltinProc_atomic_add:          case BuiltinProc_atomic_add_explicit:
	case BuiltinProc_atomic_sub:          case BuiltinProc_atomic_sub_explicit:
	case BuiltinProc_atomic_and:          case BuiltinProc_atomic_and_explicit:
	case BuiltinProc_atomic_nand:         case BuiltinProc_atomic_nand_explicit:
	case BuiltinProc_atomic_or:           case BuiltinProc_atomic_or_explicit:
	case BuiltinProc_atomic_xor:          case BuiltinProc_atomic_xor_explicit:
	case BuiltinProc_atomic_compare_exchange_strong: case BuiltinProc_atomic_compare_exchange_strong_explicit:
	case BuiltinProc_atomic_compare_exchange_weak:   case BuiltinProc_atomic_compare_exchange_weak_explicit:
		return true;
	}
	return false;
}

gb_internal x64Value x64_build_atomic(x64Procedure *p, Ast *expr, i32 id) {
	ast_node(ce, CallExpr, expr);
	X64Assembler *a = &p->asm_;

	if (id == BuiltinProc_atomic_thread_fence) { x64_emit_mfence(a); return x64v_none(); }
	if (id == BuiltinProc_atomic_signal_fence) { return x64v_none(); } // compiler-only on x86

	Type     *pt = base_type(x64_typed(ce->args[0]->tav.type)); // ^T
	Type     *T  = (pt != nullptr && is_type_pointer(pt)) ? x64_typed(type_deref(pt)) : t_int;
	X64OpSize sz = x64_op_size_of(T);

	// Pin the pointer (a later x64_build_expr may clobber RAX).
	x64Value pv = x64_build_expr(p, ce->args[0]);
	i32 poff = x64_alloc_local(p, 8, 8);
	x64_value_to_reg(p, pv, X64Reg_RAX);
	x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(poff), X64Reg_RAX);

	// atomic_load — a plain (atomic) load.
	if (id == BuiltinProc_atomic_load || id == BuiltinProc_atomic_load_explicit) {
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(poff));
		return x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), T));
	}

	// All remaining ops take a value in args[1]; pin it too.
	x64Value vv = x64_build_expr(p, ce->args[1]);
	i32 voff = x64_alloc_local(p, 8, 8);
	x64_value_to_reg(p, vv, X64Reg_RCX);
	x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(voff), X64Reg_RCX);

	// atomic_store — LOCK XCHG (atomic + full barrier), discard the old value.
	if (id == BuiltinProc_atomic_store || id == BuiltinProc_atomic_store_explicit) {
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8,  x64_rbp_mem(poff));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(voff));
		x64_emit_lock_xchg_mr(a, sz, x64_mem(X64Reg_R8, 0), X64Reg_RCX);
		return x64v_none();
	}

	// compare_exchange(ptr, expected, new) -> (old, ok). args[1]=expected, args[2]=new.
	if (id == BuiltinProc_atomic_compare_exchange_strong || id == BuiltinProc_atomic_compare_exchange_weak ||
	    id == BuiltinProc_atomic_compare_exchange_strong_explicit || id == BuiltinProc_atomic_compare_exchange_weak_explicit) {
		x64Value nv = x64_build_expr(p, ce->args[2]);
		x64_value_to_reg(p, nv, X64Reg_RCX);                              // RCX = new
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(voff));  // RAX = expected (CMPXCHG accumulator)
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8,  x64_rbp_mem(poff));  // R8  = ptr
		x64_emit_lock_cmpxchg_mr(a, sz, x64_mem(X64Reg_R8, 0), X64Reg_RCX); // RAX <- old; ZF = success
		x64_emit_setcc_r(a, X64Cc_E, X64Reg_RDX);                        // DL = ok
		Type *tuple = x64_typed(expr->tav.type);                         // (T, bool)
		i32 tup = x64_alloc_local(p, type_size_of(tuple), type_align_of(tuple));
		x64_emit_mov_mr(a, sz, x64_rbp_mem(tup), X64Reg_RAX);            // field 0 = old value
		i64 vsz  = type_size_of(T);            if (vsz  <= 0) vsz  = 1;
		i64 b_al = type_align_of(t_bool);      if (b_al <= 0) b_al = 1;
		i64 off1 = (vsz + (b_al - 1)) & ~(b_al - 1);
		x64_emit_mov_mr(a, X64OpSize_8, x64_rbp_mem(tup + (i32)off1), X64Reg_RDX); // field 1 = ok
		return x64v_mem(tuple, x64_rbp_mem(tup));
	}

	// RMW: ptr -> R8, value -> RCX. add/sub/exchange return the OLD value directly.
	x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8,  x64_rbp_mem(poff));
	x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(voff));

	switch (id) {
	case BuiltinProc_atomic_exchange: case BuiltinProc_atomic_exchange_explicit:
		x64_emit_lock_xchg_mr(a, sz, x64_mem(X64Reg_R8, 0), X64Reg_RCX); // RCX <- old
		return x64v_reg(T, X64Reg_RCX);
	case BuiltinProc_atomic_add: case BuiltinProc_atomic_add_explicit:
		x64_emit_lock_xadd_mr(a, sz, x64_mem(X64Reg_R8, 0), X64Reg_RCX); // RCX <- old
		return x64v_reg(T, X64Reg_RCX);
	case BuiltinProc_atomic_sub: case BuiltinProc_atomic_sub_explicit:
		x64_emit_neg_r(a, sz, X64Reg_RCX);
		x64_emit_lock_xadd_mr(a, sz, x64_mem(X64Reg_R8, 0), X64Reg_RCX); // RCX <- old
		return x64v_reg(T, X64Reg_RCX);
	default: break;
	}

	// or/and/xor/nand: LOCK CMPXCHG retry loop returning the OLD value.
	//   retry: RAX=[ptr]; RDX=RAX op RCX; lock cmpxchg [ptr],RDX; jnz retry  -> RAX = old
	isize retry = x64_label_alloc(a);
	x64_label_bind(a, retry);
	x64_emit_mov_rm(a, sz, X64Reg_RAX, x64_mem(X64Reg_R8, 0));
	x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_RAX);
	switch (id) {
	case BuiltinProc_atomic_or:  case BuiltinProc_atomic_or_explicit:  x64_emit_or_rr (a, sz, X64Reg_RDX, X64Reg_RCX); break;
	case BuiltinProc_atomic_and: case BuiltinProc_atomic_and_explicit: x64_emit_and_rr(a, sz, X64Reg_RDX, X64Reg_RCX); break;
	case BuiltinProc_atomic_xor: case BuiltinProc_atomic_xor_explicit: x64_emit_xor_rr(a, sz, X64Reg_RDX, X64Reg_RCX); break;
	case BuiltinProc_atomic_nand: case BuiltinProc_atomic_nand_explicit:
		x64_emit_and_rr(a, sz, X64Reg_RDX, X64Reg_RCX); x64_emit_not_r(a, sz, X64Reg_RDX); break;
	}
	x64_emit_lock_cmpxchg_mr(a, sz, x64_mem(X64Reg_R8, 0), X64Reg_RDX);
	x64_emit_jcc(a, X64Cc_NE, retry);
	return x64v_reg(T, X64Reg_RAX); // old value
}

// cstring → string (mirrors runtime.cstring_to_string): nil → "" ({0,0}), else
// {data=ptr, len=strlen(ptr)}, inlined as a NUL byte-scan (SIMD-widen later). Without
// this, `string(cstr)` kept only the ptr and left len=0 (os.args[i] came out empty).
gb_internal x64Value x64_cstring_to_string(x64Procedure *p, x64Value src) {
	x64_value_to_reg(p, src, X64Reg_RAX); // RAX = cstring data ptr
	i32 off = x64_alloc_local(p, 16, 8);  // string result: data@0, len@8

	isize l_nil  = x64_label_alloc(&p->asm_);
	isize l_loop = x64_label_alloc(&p->asm_);
	isize l_done = x64_label_alloc(&p->asm_);
	isize l_end  = x64_label_alloc(&p->asm_);

	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
	x64_emit_jcc(&p->asm_, X64Cc_E, l_nil); // nil cstring → ""

	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX); // data = ptr
	x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);       // cursor = ptr
	x64_label_bind(&p->asm_, l_loop);
	x64_emit_cmp_mi(&p->asm_, X64OpSize_8, x64_mem(X64Reg_RCX, 0), 0);     // [cursor] == 0?
	x64_emit_jcc(&p->asm_, X64Cc_E, l_done);
	x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RCX);
	x64_emit_jmp(&p->asm_, l_loop);
	x64_label_bind(&p->asm_, l_done);
	x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);       // len = cursor - ptr
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_RCX);
	x64_emit_jmp(&p->asm_, l_end);

	x64_label_bind(&p->asm_, l_nil);
	x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(off),     0); // data = 0
	x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), 0); // len  = 0

	x64_label_bind(&p->asm_, l_end);
	return x64v_mem(t_string, x64_rbp_mem(off));
}

// soa_zip(f0=slice0, ...) → #soa[]T (mirrors lb_soa_zip): each slice's .data (off 0)
// becomes the matching [^] field; len = signed-min of all .len (off 8). The #soa type
// is a plain struct — fields[0..n-1] are the per-member multipointers, field[n] is __$len.
gb_internal x64Value x64_soa_zip(x64Procedure *p, Ast *expr) {
	ast_node(ce, CallExpr, expr);
	Type *res_t  = x64_typed(expr->tav.type);
	Type *res_bt = base_type(res_t);
	if (res_bt == nullptr || res_bt->kind != Type_Struct) return x64v_none();
	type_set_offsets(res_bt);

	int  n        = (int)ce->args.count;
	i32 *data_off = gb_alloc_array(temporary_allocator(), i32, n);
	i32  len_off  = x64_alloc_local(p, 8, 8);

	for (int i = 0; i < n; i++) {
		Ast *arg = ce->args[i];
		if (arg->kind == Ast_FieldValue) arg = arg->FieldValue.value; // f = slice
		x64Value sv = x64_build_expr(p, arg); // []T: {data@0, len@8}
		if (sv.kind != x64Value_Mem) {
			i32 tmp = x64_alloc_local(p, 16, 8);
			x64_store_value(p, x64addr(x64_rbp_mem(tmp), sv.type ? sv.type : t_rawptr), sv);
			sv = x64v_mem(sv.type, x64_rbp_mem(tmp));
		}
		i32 d = x64_alloc_local(p, 8, 8);
		data_off[i] = d;
		x64_emit_lea(&p->asm_, X64Reg_RAX, sv.mem);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 0)); // .data
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(d), X64Reg_RCX);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8)); // .len
		if (i == 0) {
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);
		} else {
			// len = min(len, this_len) — signed, matching lb_emit_min for t_int
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
			x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
			isize skip = x64_label_alloc(&p->asm_);
			x64_emit_jcc(&p->asm_, X64Cc_GE, skip);                              // RCX >= RAX → keep RAX
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RCX);
			x64_label_bind(&p->asm_, skip);
		}
	}

	i32 res = x64_alloc_local(p, type_size_of(res_bt), type_align_of(res_bt));
	for (int i = 0; i < n; i++) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(data_off[i]));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64,
		                x64_rbp_mem(res + (i32)res_bt->Struct.offsets[i]), X64Reg_RAX);
	}
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(len_off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64,
	                x64_rbp_mem(res + (i32)res_bt->Struct.offsets[n]), X64Reg_RAX);

	return x64v_mem(res_t, x64_rbp_mem(res));
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression value building
// ─────────────────────────────────────────────────────────────────────────────

// Call a zero-argument procedure (an @(init) proc or the entry point), supplying a
// context pointer only for Odin-CC callees.
gb_internal void x64_call_no_arg(x64Procedure *p, Entity *callee) {
	if (callee == nullptr || callee->kind != Entity_Procedure) return;
	Type *cct = base_type(callee->type);
	bool nctx = cct != nullptr && cct->kind == Type_Proc &&
	            cct->Proc.calling_convention == ProcCC_Odin;
	x64Value ca[1];
	int cs = 0;
	if (nctx) {
		ca[cs++] = x64_context_ptr_value(p);
	}
	x64_emit_call(p, x64_get_entity_name(callee), callee->type, ca, cs);
}

// Stabilise a freshly-built argument value to memory/imm so a later arg evaluation
// (which reuses RAX) can't clobber it.
gb_internal x64Value x64_stabilize_value(x64Procedure *p, x64Value v) {
	if ((v.kind == x64Value_Reg || v.kind == x64Value_XmmReg) && v.type != nullptr) {
		i64 sz = type_size_of(v.type); if (sz <= 0) sz = 8;
		i64 al = type_align_of(v.type); if (al <= 0) al = 8;
		i32 off = x64_alloc_local(p, sz, al);
		x64_store_value(p, x64addr(x64_rbp_mem(off), v.type), v);
		return x64v_mem(v.type, x64_rbp_mem(off));
	}
	// A mem value based on a TRANSIENT register (e.g. `fi.writer` via the loaded `fi`
	// pointer in RAX) is NOT stable: a later arg build clobbers that base, so the eventual
	// lea/load reads the wrong address. Copy it NOW into an RBP-relative local. RBP/RIP-based
	// mems and imm/none are already stable.
	if (v.kind == x64Value_Mem && v.type != nullptr && !v.mem.rip_rel &&
	    v.mem.base != X64Reg_RBP && v.mem.base != X64Reg_NONE) {
		i64 sz = type_size_of(v.type); if (sz <= 0) sz = 8;
		i64 al = type_align_of(v.type); if (al <= 0) al = 8;
		i32 off = x64_alloc_local(p, sz, al);
		x64_copy_fixed(p, x64_rbp_mem(off), v.mem, sz);
		return x64v_mem(v.type, x64_rbp_mem(off));
	}
	return v;
}

// Reconstruct the call's result. For split returns the value is spread across the
// partial-return locals (results[0..N-2]) and the real return (sret local or RAX/XMM0);
// reassemble the whole tuple so multi-assign / unpack reads it uniformly. `reg_rv` is the
// register result for single-value returns.
gb_internal x64Value x64_reconstruct_call_result(x64Procedure *p, Type *ct, int npartial,
                                                 Type *last_rt, bool needs_rbp, i32 ret_local_off,
                                                 i32 *partial_off, x64Value reg_rv) {
	int N = (ct->Proc.results != nullptr) ? (int)ct->Proc.results->Tuple.variables.count : 0;
	if (npartial == 0) {
		if (N == 0) return x64v_none();
		if (type_size_of(last_rt) == 0) return x64v_none();
		if (needs_rbp) return x64v_mem(last_rt, x64_rbp_mem(ret_local_off));
		return reg_rv;
	}
	// Multi-result: capture the last result, then assemble the full tuple.
	i64 lsz = type_size_of(last_rt);
	i32 last_off;
	if (needs_rbp) {
		last_off = ret_local_off; // already written by the callee via sret
	} else {
		last_off = x64_alloc_local(p, lsz > 0 ? lsz : 1, type_align_of(last_rt));
		if (lsz > 0) {
			if (x64_is_float(last_rt)) {
				if (x64_is_double(last_rt)) x64_emit_movsd_mr(&p->asm_, x64_rbp_mem(last_off), X64XmmReg_XMM0);
				else                         x64_emit_movss_mr(&p->asm_, x64_rbp_mem(last_off), X64XmmReg_XMM0);
			} else {
				X64OpSize os = (lsz >= 8) ? X64OpSize_64 : (lsz >= 4) ? X64OpSize_32
				             : (lsz >= 2) ? X64OpSize_16 : X64OpSize_8;
				x64_emit_mov_mr(&p->asm_, os, x64_rbp_mem(last_off), X64Reg_RAX);
			}
		}
	}
	Type *tuple = ct->Proc.results;
	i32 tup = x64_alloc_local(p, type_size_of(tuple), type_align_of(tuple));
	i64 foff = 0;
	for (int i = 0; i < N; i++) {
		Type *ft = tuple->Tuple.variables[i]->type;
		i64 fsz = type_size_of(ft); if (fsz <= 0) fsz = 1;
		i64 fal = type_align_of(ft); if (fal <= 0) fal = 1;
		foff = (foff + (fal - 1)) & ~(fal - 1);
		i32 src = (i < N - 1) ? partial_off[i] : last_off;
		x64_copy_fixed(p, x64_rbp_mem(tup + (i32)foff), x64_rbp_mem(src), fsz);
		foff += fsz;
	}
	return x64v_mem(tuple, x64_rbp_mem(tup));
}

gb_internal x64Value x64_build_expr(x64Procedure *p, Ast *expr) {
	expr = unparen_expr(expr);
	GB_ASSERT(expr != nullptr);

	TypeAndValue tav = expr->tav;

	// `nil` → a typed zero VALUE (mirrors lb_const_nil / LLVMConstNull): not a memory
	// zero-fill. nil is Addressing_Value, so it would otherwise hit the Ident `default` →
	// x64v_none(). A real 0 value flows through normal stores/args/compares: SCALAR →
	// `mov [dst], 0`; aggregate (nil slice/dynarray/union) → a zeroed stack local.
	if (expr->kind == Ast_Ident && expr->Ident.token.string == str_lit("nil")) {
		Type *nt  = x64_typed(tav.type ? tav.type : t_rawptr);
		i64   nsz = nt ? type_size_of(nt) : 8; if (nsz <= 0) nsz = 8;
		if (nsz <= 8 && !x64_is_float(nt)) {
			return x64v_imm(nt, 0);
		}
		i64 nal = nt ? type_align_of(nt) : 8; if (nal <= 0) nal = 8;
		i32 off = x64_alloc_local(p, nsz, nal);
		x64_zero_mem(p, x64_rbp_mem(off), nsz);
		return x64v_mem(nt, x64_rbp_mem(off));
	}

	// A type used as a value → its typeid (mirrors lb_typeid). Canonical hash, matching
	// typeid_of and the emitted Type_Info.id.
	if (tav.mode == Addressing_Type && tav.type != nullptr) {
		return x64v_imm(t_typeid, (i64)type_hash_canonical_type(default_type(tav.type)));
	}

	// ── Compile-time constants ──────────────────────────────────────────
	if (tav.mode == Addressing_Constant && tav.type != nullptr) {
		Type    *ct = base_type(tav.type);
		ExactValue ev = tav.value;

		// Aggregate constant (array/struct literal, or a named constant of one):
		// materialise from its CompoundLit AST (mirrors lb_const_value(ExactValue_Compound);
		// shares the builder with the non-constant CompoundLit case).
		if (ev.kind == ExactValue_Compound) {
			if (ev.value_compound == nullptr) return x64v_none();
			return x64_build_compound_lit(p, ev.value_compound, x64_typed(tav.type));
		}

		// >8-byte scalar int constants (i128/u128) can't fit an x64v_imm (i64): emit the
		// full-precision bytes onto the stack as 64-bit chunks and return a mem value.
		if (ev.kind == ExactValue_Integer) {
			i64 isz = x64_type_size(x64_typed(tav.type));
			if (isz > 8) {
				u8 buf[32] = {};
				x64_const_int_bytes(x64_typed(tav.type), &ev.value_integer, buf, isz);
				i64 ial = x64_type_align(x64_typed(tav.type)); if (ial <= 0) ial = 8;
				i32 off = x64_alloc_local(p, isz, ial);
				for (i64 b = 0; b + 8 <= isz; b += 8) {
					u64 word = 0; gb_memmove(&word, buf + b, 8);
					x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)word);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + (i32)b), X64Reg_RAX);
				}
				return x64v_mem(x64_typed(tav.type), x64_rbp_mem(off));
			}
		}

		if (x64_is_float(ct)) {
			f64 fv = (ev.kind == ExactValue_Float)   ? ev.value_float :
			         (ev.kind == ExactValue_Integer)  ? (f64)big_int_to_i64(&ev.value_integer) : 0.0;
			x64_emit_float_const(p, ct, fv, (f32)fv);
			return x64v_xmm(x64_typed(tav.type), X64XmmReg_XMM0);
		}

		if (ev.kind == ExactValue_String) {
			return x64_const_string(p, ev.value_string);
		}

		// Constant typeid (type in a typeid context, e.g. poly param to struct_fields_zipped(T)).
		// Must be the canonical hash — same as typeid_of / Type_Info.id, or type_info_of(it)
		// misses the table. Without this it fell through to 0 → nil.
		if (ev.kind == ExactValue_Typeid) {
			Type *tt = ev.value_typeid ? ev.value_typeid : tav.type;
			return x64v_imm(t_typeid, (i64)type_hash_canonical_type(default_type(tt)));
		}

		// Proc value used as data (e.g. stored into a struct field). Top-level procs are
		// Addressing_Constant w/ ExactValue_Procedure, so they land here, not the Ident
		// Entity_Procedure case — emit lea_sym.
		if (ev.kind == ExactValue_Procedure) {
			Entity *proc_e = entity_of_node(expr);
			while (proc_e != nullptr && proc_e->kind == Entity_Constant &&
			       proc_e->Constant.value.kind == ExactValue_Procedure) {
				Ast *ast = unparen_expr(proc_e->Constant.value.value_procedure);
				if (ast == nullptr || ast->kind == Ast_ProcLit) break;
				Entity *pe = entity_from_expr(ast);
				if (pe == nullptr || pe == proc_e) break;
				proc_e = pe;
			}
			if (proc_e != nullptr && proc_e->kind == Entity_Procedure) {
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(proc_e));
				return x64v_reg(x64_typed(tav.type), X64Reg_RAX);
			}
			return x64v_imm(x64_typed(tav.type), 0);
		}

		i64 iv = 0;
		switch (ev.kind) {
		case ExactValue_Bool:    iv = ev.value_bool ? 1 : 0; break;
		case ExactValue_Integer: iv = big_int_to_i64(&ev.value_integer); break;
		case ExactValue_Float:   iv = (i64)ev.value_float; break;
		case ExactValue_Pointer: iv = ev.value_pointer; break;
		default: break;
		}
		return x64v_imm(x64_typed(tav.type), iv);
	}

	switch (expr->kind) {

	// ── Identifier ────────────────────────────────────────────────────────
	case_ast_node(ident, Ident, expr); {
		Entity *e = ident->entity.load(std::memory_order_relaxed);
		if (e == nullptr) {
			String name = ident->token.string;
			Type  *pt   = p->type;
			if (pt->Proc.results != nullptr) {
				for_array(ri, pt->Proc.results->Tuple.variables) {
					Entity *rv = pt->Proc.results->Tuple.variables[ri];
					if (rv->token.string == name) { e = rv; break; }
				}
			}
			if (e == nullptr && pt->Proc.params != nullptr) {
				for_array(ri, pt->Proc.params->Tuple.variables) {
					Entity *pv = pt->Proc.params->Tuple.variables[ri];
					if (pv->token.string == name) { e = pv; break; }
				}
			}
			if (e == nullptr) return x64v_none();
		}

		switch (e->kind) {
		case Entity_Procedure: {
			x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(e));
			return x64v_reg(e->type, X64Reg_RAX);
		}
		case Entity_Constant: {
			ExactValue ev = e->Constant.value;
			Type      *ct = base_type(e->type);
			if (x64_is_float(ct)) {
				f64 fv = (ev.kind == ExactValue_Float)   ? ev.value_float :
				         (ev.kind == ExactValue_Integer)  ? (f64)big_int_to_i64(&ev.value_integer) : 0.0;
				x64_emit_float_const(p, ct, fv, (f32)fv);
				return x64v_xmm(e->type, X64XmmReg_XMM0);
			}
			if (ev.kind == ExactValue_Procedure) {
				// Unwrap constant proc aliases: foo :: some_proc (possibly chained)
				ExactValue pv = ev;
				Entity *proc_e = nullptr;
				for (;;) {
					Ast *ast = unparen_expr(pv.value_procedure);
					if (ast == nullptr) break;
					if (ast->kind == Ast_ProcLit) break;
					Entity *pe = entity_from_expr(ast);
					if (pe == nullptr) break;
					if (pe->kind != Entity_Constant) { proc_e = pe; break; }
					if (pe->Constant.value.kind != ExactValue_Procedure) break;
					pv = pe->Constant.value;
				}
				if (proc_e != nullptr && proc_e->kind == Entity_Procedure) {
					x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(proc_e));
					return x64v_reg(e->type, X64Reg_RAX);
				}
				return x64v_imm(e->type, 0);
			}
			i64 iv = (ev.kind == ExactValue_Integer) ? big_int_to_i64(&ev.value_integer) :
			         (ev.kind == ExactValue_Bool)     ? (ev.value_bool ? 1 : 0) : 0;
			return x64v_imm(e->type, iv);
		}
		case Entity_Variable: {
			x64Addr addr = x64_entity_is_local(p, e)
			             ? x64_entity_addr(p, e)
			             : x64addr(x64_global_mem(p, e), e->type);
			return x64_load_addr(p, addr);
		}
		case Entity_TypeName:
			// A type name used as a VALUE is its typeid (e.g. poly param `T` to a `typeid`
			// arg). e->type is the bound type in an instantiation. Canonical hash.
			return x64v_imm(t_typeid, (i64)type_hash_canonical_type(default_type(e->type)));
		default:
			return x64v_none();
		}
	} case_end;

	// ── Implicit ('context') ─────────────────────────────────────────────
	case_ast_node(implicit, Implicit, expr); {
		x64Addr addr = x64_build_addr(p, expr);
		return x64_load_addr(p, addr);
	} case_end;

	// ── Unary expression ─────────────────────────────────────────────────
	case_ast_node(ue, UnaryExpr, expr); {
		TokenKind op = ue->op.kind;

		if (op == Token_And) {
			// &CompoundLit in the startup runtime: the address escapes (e.g. `INT_ZERO := &Int{}`),
			// so back it with a STATIC global, not a stack local (mirrors lb_build_expr's
			// p->is_startup &CompoundLit path).
			Ast *sub = unparen_expr(ue->expr);
			if (p->is_startup && sub != nullptr && sub->kind == Ast_CompoundLit) {
				Type   *ct  = x64_typed(sub->tav.type);
				String  sym = x64_add_global_generated(p->module, ct);
				i64 csz = type_size_of(ct); if (csz <= 0) csz = 1;
				i64 cal = type_align_of(ct); if (cal <= 0) cal = 1;
				x64Value cv = x64_build_expr(p, sub);
				i32 slv = x64_alloc_local(p, csz, cal);
				x64_store_value(p, x64addr(x64_rbp_mem(slv), ct), cv);
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
				x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(slv), csz);
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym); // return &static
				return x64v_reg(alloc_type_pointer(ct), X64Reg_RAX);
			}
			x64Addr addr = x64_build_addr(p, ue->expr);
			x64_emit_lea(&p->asm_, X64Reg_RAX, addr.mem);
			return x64v_reg(alloc_type_pointer(addr.type), X64Reg_RAX);
		}

		x64Value val = x64_build_expr(p, ue->expr);
		Type    *t   = val.type ? val.type : tav.type;

		switch (op) {
		case Token_Add:
			return val; // unary + is a no-op
		case Token_Sub: {
			if (x64_is_float(t)) {
				x64_value_to_xmm(p, val, X64XmmReg_XMM0);
				// XOR with sign bit to negate
				u64 sign_mask = x64_is_double(t) ? 0x8000000000000000ULL : 0x80000000ULL;
				x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)sign_mask);
				x64_emit_push_r(&p->asm_, X64Reg_RAX);
				if (x64_is_double(t)) {
					x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM1, x64_mem(X64Reg_RSP, 0));
					x64_emit_xorpd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				} else {
					x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM1, x64_mem(X64Reg_RSP, 0));
					x64_emit_xorps(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				}
				x64_emit_pop_r(&p->asm_, X64Reg_RAX);
				return x64v_xmm(t, X64XmmReg_XMM0);
			}
			x64_value_to_reg(p, val, X64Reg_RAX);
			x64_emit_neg_r(&p->asm_, x64_op_size_of(t), X64Reg_RAX);
			return x64v_reg(t, X64Reg_RAX);
		}
		case Token_Xor: {
			x64_value_to_reg(p, val, X64Reg_RAX);
			x64_emit_not_r(&p->asm_, x64_op_size_of(t), X64Reg_RAX);
			return x64v_reg(t, X64Reg_RAX);
		}
		case Token_Not: {
			x64_value_to_reg(p, val, X64Reg_RAX);
			x64_emit_test_rr(&p->asm_, x64_op_size_of(t), X64Reg_RAX, X64Reg_RAX);
			x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);
			x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
			return x64v_reg(t_bool, X64Reg_RAX);
		}
		default:
			GB_PANIC("x64_build_expr UnaryExpr: unhandled op %d", op);
		}
	} case_end;

	// ── Binary expression ────────────────────────────────────────────────
	case_ast_node(be, BinaryExpr, expr); {
		TokenKind op = be->op.kind;
		Type *ltype       = x64_typed(be->left->tav.type);
		Type *result_type = x64_typed(tav.type ? tav.type : ltype);
		if (ltype == nullptr || result_type == nullptr) return x64v_none();

		// Short-circuit logical operators
		if (op == Token_CmpAnd || op == Token_CmpOr) {
			x64Value lv = x64_build_expr(p, be->left);
			x64_value_to_reg(p, lv, X64Reg_RAX);
			x64_emit_test_rr(&p->asm_, x64_op_size_of(ltype), X64Reg_RAX, X64Reg_RAX);

			isize lbl_end    = x64_label_alloc(&p->asm_);
			isize lbl_result = x64_label_alloc(&p->asm_);

			if (op == Token_CmpAnd) {
				x64_emit_jcc(&p->asm_, X64Cc_E, lbl_result);  // false → skip right
			} else {
				x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_result); // true  → skip right
			}

			x64Value rv = x64_build_expr(p, be->right);
			x64_value_to_reg(p, rv, X64Reg_RAX);
			x64_emit_test_rr(&p->asm_, x64_op_size_of(be->right->tav.type), X64Reg_RAX, X64Reg_RAX);
			x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RAX);
			x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
			x64_emit_jmp(&p->asm_, lbl_end);

			x64_label_bind(&p->asm_, lbl_result);
			if (op == Token_CmpAnd) {
				x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
			} else {
				x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
			}
			x64_label_bind(&p->asm_, lbl_end);
			return x64v_reg(t_bool, X64Reg_RAX);
		}

		// bit_set membership: elem in set / elem not_in set
		if (op == Token_in || op == Token_not_in) {
			Type *rt = base_type(x64_typed(be->right->tav.type));
			if (rt != nullptr && rt->kind == Type_BitSet) {
				Type     *it = bit_set_to_int(rt);
				X64OpSize sz = x64_op_size_of(it);
				i64 lower = rt->BitSet.lower;

				// SPILL both operands first. The mask computation clobbers RAX/RCX — keeping
				// the set in a register made `value_to_reg(rv, RAX)` a no-op (RAX still held
				// the mask), degenerating the test to `mask & mask` → every elem read as "in".
				x64Value lv = x64_build_expr(p, be->left);
				Type *et   = lv.type ? lv.type : t_int;
				x64Value elem_m = x64_spill_value(p, lv, et);
				x64Value set_m  = x64_spill_value(p, x64_build_expr(p, be->right), rt);

				// RCX = elem; RAX = 1 << (elem - lower) (the bit mask)
				x64_value_to_reg(p, elem_m, X64Reg_RCX);
				x64_emit_bit_set_mask(p, sz, lower);

				// RCX = bit_set value (mask stays in RAX); RAX = mask & set
				x64_value_to_reg(p, set_m, X64Reg_RCX);
				x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
				x64_emit_test_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RAX);
				X64Cc cc = (op == Token_in) ? X64Cc_NE : X64Cc_E;
				x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX);
				x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
				return x64v_reg(t_bool, X64Reg_RAX);
			}
			// Map or unknown type — not yet implemented; return false
			(void)x64_build_expr(p, be->left);
			x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
			return x64v_reg(t_bool, X64Reg_RAX);
		}

		// Aggregate equality (arrays / simple comparable structs). The scalar path below
		// compares only one register's worth, so `[3]u32` etc. compared wrong (the
		// BINDINGS_VERSION mismatch). Compare both operands' memory (mirrors lb_emit_comp's
		// array path). Gated on is_type_simple_compare so a flat byte/word compare is valid.
		if (op == Token_CmpEq || op == Token_NotEq) {
			Type *lbt = base_type(ltype);
			bool is_agg = lbt != nullptr &&
			    (lbt->kind == Type_Array || lbt->kind == Type_EnumeratedArray ||
			     lbt->kind == Type_Struct || lbt->kind == Type_Union);
			if (is_agg && is_type_simple_compare(ltype)) {
				// &lhs (spill before building &rhs, which reuses RAX), then &rhs.
				x64Addr la = x64_build_addr(p, be->left);
				x64_emit_lea(&p->asm_, X64Reg_RAX, la.mem);
				i32 lp_off = x64_alloc_local(p, 8, 8);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(lp_off), X64Reg_RAX);

				x64Addr ra = x64_build_addr(p, be->right);
				x64_emit_lea(&p->asm_, X64Reg_RAX, ra.mem);
				i32 rp_off = x64_alloc_local(p, 8, 8);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(rp_off), X64Reg_RAX);

				// RAX = &lhs, RCX = &rhs; compare `size` bytes in 8/4/2/1 chunks via
				// R8/R9, branching to lbl_ne on the first mismatch.
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(lp_off));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(rp_off));

				isize lbl_ne   = x64_label_alloc(&p->asm_);
				isize lbl_done = x64_label_alloc(&p->asm_);
				i64 total = type_size_of(ltype);
				i64 off = 0;
				while (total - off >= 8) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_mem(X64Reg_RAX, (i32)off));
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R9, x64_mem(X64Reg_RCX, (i32)off));
					x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_R8, X64Reg_R9);
					x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_ne);
					off += 8;
				}
				while (total - off >= 4) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_32, X64Reg_R8, x64_mem(X64Reg_RAX, (i32)off));
					x64_emit_mov_rm(&p->asm_, X64OpSize_32, X64Reg_R9, x64_mem(X64Reg_RCX, (i32)off));
					x64_emit_cmp_rr(&p->asm_, X64OpSize_32, X64Reg_R8, X64Reg_R9);
					x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_ne);
					off += 4;
				}
				while (total - off >= 2) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_16, X64Reg_R8, x64_mem(X64Reg_RAX, (i32)off));
					x64_emit_mov_rm(&p->asm_, X64OpSize_16, X64Reg_R9, x64_mem(X64Reg_RCX, (i32)off));
					x64_emit_cmp_rr(&p->asm_, X64OpSize_16, X64Reg_R8, X64Reg_R9);
					x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_ne);
					off += 2;
				}
				while (total - off >= 1) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_R8, x64_mem(X64Reg_RAX, (i32)off));
					x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_R9, x64_mem(X64Reg_RCX, (i32)off));
					x64_emit_cmp_rr(&p->asm_, X64OpSize_8, X64Reg_R8, X64Reg_R9);
					x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_ne);
					off += 1;
				}
				// All chunks equal.
				x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RAX, (op == Token_CmpEq) ? 1 : 0);
				x64_emit_jmp(&p->asm_, lbl_done);
				x64_label_bind(&p->asm_, lbl_ne);
				x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RAX, (op == Token_CmpEq) ? 0 : 1);
				x64_label_bind(&p->asm_, lbl_done);
				return x64v_reg(t_bool, X64Reg_RAX);
			}
		}

		x64Value lhs = x64_build_expr(p, be->left);

		// Float binary ops
		if (x64_is_float(ltype)) {
			x64_value_to_xmm(p, lhs, X64XmmReg_XMM0);
			// Save XMM0 (LHS) to an RBP-relative local — NOT [rsp]: if the RHS
			// contains a call, the callee's shadow-space writes would clobber it.
			i32 lhs_off = x64_alloc_local(p, 8, 8);
			if (x64_is_double(ltype)) x64_emit_movsd_mr(&p->asm_, x64_rbp_mem(lhs_off), X64XmmReg_XMM0);
			else                       x64_emit_movss_mr(&p->asm_, x64_rbp_mem(lhs_off), X64XmmReg_XMM0);

			x64Value rhs = x64_build_expr(p, be->right);
			x64_value_to_xmm(p, rhs, X64XmmReg_XMM1);

			// Restore LHS into XMM0
			if (x64_is_double(ltype)) x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(lhs_off));
			else                       x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(lhs_off));

			bool is_d = x64_is_double(ltype);
			switch (op) {
			case Token_Add:
				if (is_d) x64_emit_addsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				else       x64_emit_addss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				return x64v_xmm(result_type, X64XmmReg_XMM0);
			case Token_Sub:
				if (is_d) x64_emit_subsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				else       x64_emit_subss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				return x64v_xmm(result_type, X64XmmReg_XMM0);
			case Token_Mul:
				if (is_d) x64_emit_mulsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				else       x64_emit_mulss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				return x64v_xmm(result_type, X64XmmReg_XMM0);
			case Token_Quo:
				if (is_d) x64_emit_divsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				else       x64_emit_divss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				return x64v_xmm(result_type, X64XmmReg_XMM0);
			default: {
				// Float comparison
				if (is_d) x64_emit_ucomisd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				else       x64_emit_ucomiss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
				X64Cc cc = X64Cc_E;
				switch (op) {
				case Token_CmpEq: cc = X64Cc_E;  break;
				case Token_NotEq: cc = X64Cc_NE; break;
				case Token_Lt:    cc = X64Cc_B;  break;
				case Token_LtEq:  cc = X64Cc_BE; break;
				case Token_Gt:    cc = X64Cc_A;  break;
				case Token_GtEq:  cc = X64Cc_AE; break;
				default: break;
				}
				x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX);
				x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
				return x64v_reg(t_bool, X64Reg_RAX);
			}
			}
		}

		// Integer binary ops.
		// Spill the LHS to an RBP-relative local (NOT push/[rsp]): if the RHS
		// contains a call, the callee homes its params into the caller's shadow
		// space, which overlaps a pushed value and would corrupt the LHS.
		i32 lhs_off = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, lhs, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(lhs_off), X64Reg_RAX);

		x64Value rhs = x64_build_expr(p, be->right);
		x64_value_to_reg(p, rhs, X64Reg_RCX);

		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(lhs_off));

		X64OpSize sz = x64_op_size_of(ltype);
		bool signed_ = x64_is_signed_integer(ltype);

		// bit_set semantics (integer-backed): `+`=union (OR), `-`=difference (AND-NOT);
		// `<=`/`>=`=subset/superset, `<`/`>` their strict forms; `==`/`!=`/`&`/`|`/`~` already
		// match integer eq/ne/and/or/xor. Mirrors lb_emit_arith (Add→Or, Sub→AndNot) and
		// lb_emit_comp ((a&b)==a etc.). RAX=lhs, RCX=rhs here.
		{
			Type *lbt2 = base_type(ltype);
			if (lbt2 != nullptr && lbt2->kind == Type_BitSet) {
				switch (op) {
				case Token_Add: op = Token_Or;     break;
				case Token_Sub: op = Token_AndNot; break;
				case Token_Lt: case Token_LtEq: case Token_Gt: case Token_GtEq: {
					bool subset = (op == Token_Lt || op == Token_LtEq); // ⊆ vs ⊇
					bool strict = (op == Token_Lt || op == Token_Gt);   // < / >  (proper subset/superset)
					x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RAX);
					x64_emit_and_rr(&p->asm_, sz, X64Reg_RDX, X64Reg_RCX);            // RDX = a & b
					x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RDX, subset ? X64Reg_RAX : X64Reg_RCX);
					x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_R8);                  // (a&b)==(a|b) → ⊆/⊇
					if (strict) {                                                    // && a != b
						x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
						x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_R9);
						x64_emit_and_rr(&p->asm_, X64OpSize_8, X64Reg_R8, X64Reg_R9);
					}
					x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_R8);
					return x64v_reg(t_bool, X64Reg_RAX);
				}
				}
			}
		}

#define EMIT_CMP_CC(cc) \
	x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); \
	x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX); \
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX); \
	return x64v_reg(t_bool, X64Reg_RAX)

		switch (op) {
		case Token_Add:  x64_emit_add_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
		case Token_Sub:  x64_emit_sub_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
		case Token_Mul:
			if (signed_) x64_emit_imul_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
			else         x64_emit_mul_r  (&p->asm_, sz, X64Reg_RCX);
			return x64v_reg(result_type, X64Reg_RAX);
		case Token_Quo:
			if (signed_) { if (sz==X64OpSize_64) x64_emit_cqo(&p->asm_); else x64_emit_cdq(&p->asm_); x64_emit_idiv_r(&p->asm_, sz, X64Reg_RCX); }
			else         { x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); x64_emit_div_r(&p->asm_, sz, X64Reg_RCX); }
			return x64v_reg(result_type, X64Reg_RAX);
		case Token_Mod: case Token_ModMod:
			if (signed_) { if (sz==X64OpSize_64) x64_emit_cqo(&p->asm_); else x64_emit_cdq(&p->asm_); x64_emit_idiv_r(&p->asm_, sz, X64Reg_RCX); }
			else         { x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); x64_emit_div_r(&p->asm_, sz, X64Reg_RCX); }
			x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
			return x64v_reg(result_type, X64Reg_RAX);
		case Token_And:    x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
		case Token_Or:     x64_emit_or_rr (&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
		case Token_Xor:    x64_emit_xor_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
		case Token_AndNot: x64_emit_not_r(&p->asm_, sz, X64Reg_RCX); x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
		case Token_Shl:  x64_emit_shl_rcl(&p->asm_, sz, X64Reg_RAX);              return x64v_reg(result_type, X64Reg_RAX);
		case Token_Shr:
			if (signed_) x64_emit_sar_rcl(&p->asm_, sz, X64Reg_RAX);
			else         x64_emit_shr_rcl(&p->asm_, sz, X64Reg_RAX);
			return x64v_reg(result_type, X64Reg_RAX);
		case Token_CmpEq: { EMIT_CMP_CC(X64Cc_E);  }
		case Token_NotEq: { EMIT_CMP_CC(X64Cc_NE); }
		case Token_Lt:    { EMIT_CMP_CC(signed_ ? X64Cc_L  : X64Cc_B);  }
		case Token_LtEq:  { EMIT_CMP_CC(signed_ ? X64Cc_LE : X64Cc_BE); }
		case Token_Gt:    { EMIT_CMP_CC(signed_ ? X64Cc_G  : X64Cc_A);  }
		case Token_GtEq:  { EMIT_CMP_CC(signed_ ? X64Cc_GE : X64Cc_AE); }
		default:
			GB_PANIC("x64 integer binary op %d not supported", op);
		}
#undef EMIT_CMP_CC
	} case_end;

	// ── Type cast ─────────────────────────────────────────────────────────
	case_ast_node(tc, TypeCast, expr); {
		x64Value src = x64_build_expr(p, tc->expr);
		Type *dst_base = base_type(tav.type);
		Type *src_base = base_type(tc->expr->tav.type);
		X64OpSize src_sz = x64_op_size_of(src_base);

		// cstring → string: representation change (ptr → {ptr, strlen}). The generic
		// int/ptr path below would keep only the ptr and leave len = 0.
		if (is_type_cstring(src_base) && is_type_string(dst_base)) {
			return x64_cstring_to_string(p, src);
		}

		// Same-size reinterpret of a value larger than a register (e.g.
		// transmute([]byte)string, transmute(T)struct): keep all bytes in place
		// and just retype. The register-based path below would truncate to 8 bytes.
		i64 cast_dsz = type_size_of(x64_typed(tav.type));
		i64 cast_ssz = type_size_of(x64_typed(tc->expr->tav.type));
		if (cast_dsz > 8 && cast_dsz == cast_ssz) {
			src.type = x64_typed(tav.type);
			return src;
		}

		if (x64_is_float(dst_base) && x64_is_float(src_base)) {
			x64_value_to_xmm(p, src, X64XmmReg_XMM0);
			if (x64_is_double(dst_base) && !x64_is_double(src_base))
				x64_emit_cvtss2sd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM0);
			else if (!x64_is_double(dst_base) && x64_is_double(src_base))
				x64_emit_cvtsd2ss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM0);
			return x64v_xmm(x64_typed(tav.type), X64XmmReg_XMM0);
		}
		if (x64_is_integer(dst_base) && x64_is_float(src_base)) {
			x64_value_to_xmm(p, src, X64XmmReg_XMM0);
			if (x64_is_double(src_base)) x64_emit_cvttsd2si(&p->asm_, X64Reg_RAX, X64OpSize_64, X64XmmReg_XMM0);
			else                          x64_emit_cvttss2si(&p->asm_, X64Reg_RAX, X64OpSize_64, X64XmmReg_XMM0);
			return x64v_reg(x64_typed(tav.type), X64Reg_RAX);
		}
		if (x64_is_float(dst_base) && x64_is_integer(src_base)) {
			x64_value_to_reg(p, src, X64Reg_RAX);
			if (x64_is_double(dst_base)) x64_emit_cvtsi2sd(&p->asm_, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64);
			else                          x64_emit_cvtsi2ss(&p->asm_, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64);
			return x64v_xmm(x64_typed(tav.type), X64XmmReg_XMM0);
		}
		// Int/ptr → int/ptr (widen or truncate)
		x64_value_to_reg(p, src, X64Reg_RAX);
		X64OpSize dst_sz = x64_op_size_of(dst_base);
		if ((int)dst_sz > (int)src_sz && x64_is_signed_integer(src_base) && src_sz < X64OpSize_64) {
			x64_emit_movsx_rr(&p->asm_, src_sz, X64Reg_RAX, X64Reg_RAX);
		} else if ((int)dst_sz > (int)src_sz && src_sz <= X64OpSize_16) {
			x64_emit_movzx_rr(&p->asm_, src_sz, X64Reg_RAX, X64Reg_RAX);
		}
		return x64v_reg(x64_typed(tav.type), X64Reg_RAX);
	} case_end;

	case_ast_node(ac, AutoCast, expr); {
		return x64_build_expr(p, ac->expr);
	} case_end;

	// ── DerefExpr (*ptr) ─────────────────────────────────────────────────
	case_ast_node(de, DerefExpr, expr); {
		x64Addr addr = x64_build_addr(p, expr);
		return x64_load_addr(p, addr);
	} case_end;

	// ── SelectorExpr ──────────────────────────────────────────────────────
	case_ast_node(se, SelectorExpr, expr); {
		// If the base expression is a package import (e.g. utf8.encode_rune),
		// dispatch directly on the resolved field entity so that x64_build_addr
		// never tries to load the import-name as a global symbol.
		// Do NOT do this for struct field access (f.field): those fall through to
		// x64_build_addr which computes the correct RBP+offset.
		Entity *base_ent = entity_of_node(se->expr);
		if (base_ent != nullptr && base_ent->kind == Entity_ImportName) {
			Entity *ent = entity_of_node(se->selector);
			if (ent == nullptr) return x64v_none();
			switch (ent->kind) {
			case Entity_Procedure:
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(ent));
				return x64v_reg(ent->type, X64Reg_RAX);
			case Entity_Variable: {
				x64Addr addr = x64_entity_is_local(p, ent)
				             ? x64_entity_addr(p, ent)
				             : x64addr(x64_global_mem(p, ent), ent->type);
				return x64_load_addr(p, addr);
			}
			default:
				return x64v_none(); // constant (caught above), type, etc.
			}
		}
		// Struct/union field access or other non-import selector
		x64Addr addr = x64_build_addr(p, expr);
		return x64_load_addr(p, addr);
	} case_end;

	// ── IndexExpr (arr[i]) ────────────────────────────────────────────────
	case_ast_node(ie, IndexExpr, expr); {
		x64Addr addr = x64_build_addr(p, expr);
		return x64_load_addr(p, addr);
	} case_end;

	// ── CallExpr ─────────────────────────────────────────────────────────
	case_ast_node(ce, CallExpr, expr); {
		// Skip type-conversion call form (int(x), f64(x), etc.)
		if (ce->proc->tav.mode == Addressing_Type) {
			if (ce->args.count >= 1) {
				x64Value cv = x64_build_expr(p, ce->args[0]);
				// cstring → string changes representation (8B ptr → {ptr,len}); the
				// rest are same-representation reinterprets handled by the cv as-is.
				if (is_type_cstring(base_type(ce->args[0]->tav.type)) &&
				    is_type_string(base_type(tav.type))) {
					return x64_cstring_to_string(p, cv);
				}
				return cv;
			}
			return x64v_none();
		}
		// Builtin calls (len, cap, copy, etc.) — mostly unimplemented, but a few
		// matter because LLVM lowers them to real runtime calls whose results are
		// consumed.
		if (ce->proc->tav.mode == Addressing_Builtin) {
			Entity *be = entity_of_node(unparen_expr(ce->proc));
			// Atomic intrinsics (intrinsics.atomic_*) — LOCK-prefixed ops / MOV / CMPXCHG.
			// Without these all atomics silently no-op (sync.Mutex/Sema, thread flags) → hangs.
			if (be != nullptr && be->kind == Entity_Builtin && x64_is_atomic_builtin(be->Builtin.id)) {
				return x64_build_atomic(p, expr, be->Builtin.id);
			}
			// soa_zip(...) → #soa[]T (mirrors lb_soa_zip). reflect.struct_fields_zipped
			// builds its result this way; without it the #soa return is garbage.
			if (be != nullptr && be->kind == Entity_Builtin && be->Builtin.id == BuiltinProc_soa_zip) {
				return x64_soa_zip(p, expr);
			}
			// intrinsics.__entry_point() -> run @(init) procs (mirrors LLVM's
			// _startup_runtime body) then call info->entry_point (user main).
			if (be != nullptr && be->kind == Entity_Builtin &&
			    be->Builtin.id == BuiltinProc___entry_point) {
				CheckerInfo *cinfo = p->module->gen->info;

				// Global var init is NOT done here: it runs in __$startup_runtime (called
				// before __entry_point, emitted AFTER the on-demand closure so it sees every
				// global's storage and inits all in dependency order, mirroring LLVM). By
				// now the globals are already set.
				for (Entity *ie : cinfo->init_procedures) x64_call_no_arg(p, ie);
				x64_call_no_arg(p, cinfo->entry_point);
				return x64v_none();
			}
			if (be != nullptr && be->kind == Entity_Builtin &&
			    be->Builtin.id == BuiltinProc_type_info_of && ce->args.count >= 1) {
				Ast *arg0 = ce->args[0];
				// Compile-time `type_info_of(T)` (a pointer into the type table) not yet
				// supported. Runtime `type_info_of(id)` → runtime.__type_info_of(id) (mirrors LLVM).
				if (arg0->tav.mode != Addressing_Type) {
					AstPackage *rt_pkg = p->module->gen->info->runtime_package;
					Entity *tio = (rt_pkg != nullptr)
					    ? scope_lookup_current(rt_pkg->scope,
					          string_interner_insert(str_lit("__type_info_of")))
					    : nullptr;
					if (tio != nullptr && tio->kind == Entity_Procedure) {
						// Evaluate the typeid arg and spill it before the call setup.
						x64Value idv = x64_build_expr(p, arg0);
						i32 id_off = x64_alloc_local(p, 8, 8);
						x64_value_to_reg(p, idv, X64Reg_RAX);
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(id_off), X64Reg_RAX);
						x64Value cargs[1];
						cargs[0] = x64v_mem(t_typeid, x64_rbp_mem(id_off));
						return x64_emit_call(p, x64_get_entity_name(tio), tio->type, cargs, 1);
					}
				}
			}
			// len/cap: slice/string {data@0, len@8}; dynamic array adds cap@16.
			if (be != nullptr && be->kind == Entity_Builtin &&
			    (be->Builtin.id == BuiltinProc_len || be->Builtin.id == BuiltinProc_cap) &&
			    ce->args.count >= 1) {
				Ast  *arg0 = ce->args[0];
				Type *at   = arg0->tav.type ? base_type(arg0->tav.type) : nullptr;
				x64Value av = x64_build_expr(p, arg0);
				if (at != nullptr && is_type_pointer(at)) {
					x64_value_to_reg(p, av, X64Reg_RAX);
					Type *deref = type_deref(at);
					av = x64v_mem(deref, x64_mem(X64Reg_RAX, 0));
					at = base_type(deref);
				}
				if (av.kind == x64Value_Mem && at != nullptr) {
					i32 off = -1;
					if (is_type_slice(at) || is_type_string(at)) {
						off = 8; // cap(slice) == len
					} else if (is_type_dynamic_array(at)) {
						off = (be->Builtin.id == BuiltinProc_cap) ? 16 : 8;
					}
					if (off >= 0) {
						X64Mem m = av.mem;
						m.disp += off;
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, m);
						return x64v_reg(t_int, X64Reg_RAX);
					}
				}
				return x64v_none();
			}
			// min / max — fold over args (mirrors lb_emit_min / lb_emit_max).
			if (be != nullptr && be->kind == Entity_Builtin &&
			    (be->Builtin.id == BuiltinProc_min || be->Builtin.id == BuiltinProc_max) &&
			    ce->args.count >= 1) {
				bool is_min = be->Builtin.id == BuiltinProc_min;
				Type *rt = tav.type ? x64_typed(tav.type) : x64_typed(ce->args[0]->tav.type);
				if (rt == nullptr) return x64v_none();

				if (x64_is_float(rt)) {
					bool dbl = x64_is_double(rt);
					i32 acc = x64_alloc_local(p, 8, 8);
					x64_store_value(p, x64addr(x64_rbp_mem(acc), rt), x64_build_expr(p, ce->args[0]));
					for (isize i = 1; i < ce->args.count; i++) {
						i32 boff = x64_alloc_local(p, 8, 8);
						x64_store_value(p, x64addr(x64_rbp_mem(boff), rt), x64_build_expr(p, ce->args[i]));
						if (dbl) { x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(acc));
						           x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM1, x64_rbp_mem(boff));
						           x64_emit_ucomisd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); }
						else     { x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(acc));
						           x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM1, x64_rbp_mem(boff));
						           x64_emit_ucomiss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); }
						// min: keep acc if acc<bi (below); max: keep if acc>bi (above).
						isize keep = x64_label_alloc(&p->asm_);
						x64_emit_jcc(&p->asm_, is_min ? X64Cc_B : X64Cc_A, keep);
						if (dbl) { x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(boff));
						           x64_emit_movsd_mr(&p->asm_, x64_rbp_mem(acc), X64XmmReg_XMM0); }
						else     { x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(boff));
						           x64_emit_movss_mr(&p->asm_, x64_rbp_mem(acc), X64XmmReg_XMM0); }
						x64_label_bind(&p->asm_, keep);
					}
					return x64v_mem(rt, x64_rbp_mem(acc));
				}

				// Integer / pointer / enum: cmov-based fold.
				X64OpSize osz = x64_op_size_of(rt);
				bool sgn = x64_is_signed_integer(rt);
				// Replace acc with b when: min → acc>b ; max → acc<b.
				X64Cc cc = is_min ? (sgn ? X64Cc_G : X64Cc_A)
				                  : (sgn ? X64Cc_L : X64Cc_B);
				i32 acc = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[0]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(acc), X64Reg_RAX);
				for (isize i = 1; i < ce->args.count; i++) {
					x64_value_to_reg(p, x64_build_expr(p, ce->args[i]), X64Reg_RCX);
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(acc));
					x64_emit_cmp_rr(&p->asm_, osz, X64Reg_RAX, X64Reg_RCX);
					x64_emit_cmov_rr(&p->asm_, cc, osz, X64Reg_RAX, X64Reg_RCX);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(acc), X64Reg_RAX);
				}
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(acc));
				return x64v_reg(rt, X64Reg_RAX);
			}
			// clamp(x, lo, hi) == min(max(x, lo), hi) (mirrors lb_emit_clamp).
			if (be != nullptr && be->kind == Entity_Builtin &&
			    be->Builtin.id == BuiltinProc_clamp && ce->args.count >= 3) {
				Type *rt = tav.type ? x64_typed(tav.type) : x64_typed(ce->args[0]->tav.type);
				if (rt == nullptr) return x64v_none();
				i64 sz = type_size_of(rt); if (sz <= 0) sz = 8;
				i64 al = type_align_of(rt); if (al <= 0) al = 8;
				// Spill each arg as it's built (later builds clobber registers).
				i32 x_off  = x64_alloc_local(p, sz, al);
				x64_store_value(p, x64addr(x64_rbp_mem(x_off),  rt), x64_build_expr(p, ce->args[0]));
				i32 lo_off = x64_alloc_local(p, sz, al);
				x64_store_value(p, x64addr(x64_rbp_mem(lo_off), rt), x64_build_expr(p, ce->args[1]));
				i32 hi_off = x64_alloc_local(p, sz, al);
				x64_store_value(p, x64addr(x64_rbp_mem(hi_off), rt), x64_build_expr(p, ce->args[2]));
				x64Value m = x64_emit_minmax2(p, rt, x64v_mem(rt, x64_rbp_mem(x_off)),
				                              x64v_mem(rt, x64_rbp_mem(lo_off)), false); // max(x, lo)
				return x64_emit_minmax2(p, rt, m,
				                        x64v_mem(rt, x64_rbp_mem(hi_off)), true);       // min(., hi)
			}
			// abs(x): unsigned → x; float → clear sign bit; signed → (x<0)?-x:x.
			if (be != nullptr && be->kind == Entity_Builtin &&
			    be->Builtin.id == BuiltinProc_abs && ce->args.count >= 1) {
				Type *rt = tav.type ? x64_typed(tav.type) : x64_typed(ce->args[0]->tav.type);
				x64Value xv = x64_build_expr(p, ce->args[0]);
				if (rt == nullptr) return xv;
				if (x64_is_integer(rt) && !x64_is_signed_integer(rt)) return xv; // unsigned
				if (x64_is_float(rt)) {
					i64 fsz = type_size_of(rt); if (fsz <= 0) fsz = 8;
					i32 off = x64_alloc_local(p, fsz, fsz);
					x64_store_value(p, x64addr(x64_rbp_mem(off), rt), xv);
					if (fsz == 8) {
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(off));
						x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)0x7FFFFFFFFFFFFFFFll);
						x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
					} else {
						x64_emit_mov_rm(&p->asm_, X64OpSize_32, X64Reg_RAX, x64_rbp_mem(off));
						x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RCX, (i64)0x7FFFFFFFll);
						x64_emit_and_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RCX);
						x64_emit_mov_mr(&p->asm_, X64OpSize_32, x64_rbp_mem(off), X64Reg_RAX);
					}
					return x64v_mem(rt, x64_rbp_mem(off));
				}
				X64OpSize osz = x64_op_size_of(rt);
				x64_value_to_reg(p, xv, X64Reg_RAX);
				x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
				x64_emit_neg_r(&p->asm_, osz, X64Reg_RCX);          // RCX = -x
				x64_emit_test_rr(&p->asm_, osz, X64Reg_RAX, X64Reg_RAX);
				x64_emit_cmov_rr(&p->asm_, X64Cc_S, osz, X64Reg_RAX, X64Reg_RCX); // x<0 → -x
				return x64v_reg(rt, X64Reg_RAX);
			}
			// typeid_of(T): compile-time type → its typeid (canonical type hash).
			if (be != nullptr && be->kind == Entity_Builtin &&
			    be->Builtin.id == BuiltinProc_typeid_of && ce->args.count >= 1) {
				Type *at = ce->args[0]->tav.type;
				if (at == nullptr) at = type_of_expr(ce->args[0]);
				Type *dt = default_type(at);
				return x64v_imm(t_typeid, (i64)type_hash_canonical_type(dt));
			}
			// ptr_offset(ptr, n): ptr + n*size_of(elem). ptr_sub(p0,p1): (p0-p1)/size.
			if (be != nullptr && be->kind == Entity_Builtin &&
			    (be->Builtin.id == BuiltinProc_ptr_offset || be->Builtin.id == BuiltinProc_ptr_sub) &&
			    ce->args.count >= 2) {
				Type *pt   = x64_typed(ce->args[0]->tav.type);
				Type *pbt  = base_type(pt);
				Type *elem = pbt && pbt->kind == Type_MultiPointer ? pbt->MultiPointer.elem :
				             pbt && pbt->kind == Type_Pointer       ? pbt->Pointer.elem : nullptr;
				i64 esz = elem ? type_size_of(elem) : 1; if (esz <= 0) esz = 1;

				i32 a0 = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[0]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(a0), X64Reg_RAX);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[1]), X64Reg_RCX);

				if (be->Builtin.id == BuiltinProc_ptr_offset) {
					if (esz > 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX, (i32)esz);
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(a0));
					x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
					return x64v_reg(pt, X64Reg_RAX);
				}
				// ptr_sub: byte difference / element size (signed).
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(a0));
				x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
				if (esz > 1) {
					x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)esz);
					x64_emit_cqo(&p->asm_);
					x64_emit_idiv_r(&p->asm_, X64OpSize_64, X64Reg_RCX);
				}
				return x64v_reg(t_int, X64Reg_RAX);
			}
			// intrinsics.mem_copy / mem_copy_non_overlapping (dst, src, len).
			// LLVM lowers these to llvm.memmove / llvm.memcpy. len is runtime, so
			// we emit a REP MOVSB (with backward direction for the overlapping
			// memmove case when dst > src).
			if (be != nullptr && be->kind == Entity_Builtin &&
			    (be->Builtin.id == BuiltinProc_mem_copy ||
			     be->Builtin.id == BuiltinProc_mem_copy_non_overlapping) &&
			    ce->args.count >= 3) {
				bool overlapping = be->Builtin.id == BuiltinProc_mem_copy;
				// Evaluate and spill each arg (later evals clobber RAX).
				i32 dst_off = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[0]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(dst_off), X64Reg_RAX);
				i32 src_off = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[1]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(src_off), X64Reg_RAX);
				i32 len_off = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[2]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RAX);

				// RSI/RDI are nonvolatile in Win64 — save/restore them (LLVM's
				// memmove preserves them; our inline REP MOVSB must too). The arg
				// reloads below are RBP-based, unaffected by the pushes.
				x64_emit_push_r(&p->asm_, X64Reg_RSI);
				x64_emit_push_r(&p->asm_, X64Reg_RDI);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RDI, x64_rbp_mem(dst_off));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RSI, x64_rbp_mem(src_off));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(len_off));

				isize lbl_fwd = x64_label_alloc(&p->asm_);
				isize lbl_end = x64_label_alloc(&p->asm_);
				if (overlapping) {
					// if dst <= src → forward is safe; else copy backward.
					x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_RDI, X64Reg_RSI);
					x64_emit_jcc(&p->asm_, X64Cc_BE, lbl_fwd);
					// backward: point RSI/RDI at last byte, set DF, rep movsb, clear DF.
					x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RSI, X64Reg_RCX);
					x64_emit_sub_ri(&p->asm_, X64OpSize_64, X64Reg_RSI, 1);
					x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RDI, X64Reg_RCX);
					x64_emit_sub_ri(&p->asm_, X64OpSize_64, X64Reg_RDI, 1);
					x64_enc_b(&p->asm_, 0xFD); // STD
					x64_enc_b(&p->asm_, 0xF3); // REP
					x64_enc_b(&p->asm_, 0xA4); // MOVSB
					x64_enc_b(&p->asm_, 0xFC); // CLD
					x64_emit_jmp(&p->asm_, lbl_end);
				}
				x64_label_bind(&p->asm_, lbl_fwd);
				x64_enc_b(&p->asm_, 0xF3); // REP
				x64_enc_b(&p->asm_, 0xA4); // MOVSB
				x64_label_bind(&p->asm_, lbl_end);
				x64_emit_pop_r(&p->asm_, X64Reg_RDI);
				x64_emit_pop_r(&p->asm_, X64Reg_RSI);
				return x64v_none();
			}
			// intrinsics.mem_zero / mem_zero_volatile (ptr, len). len is runtime.
			if (be != nullptr && be->kind == Entity_Builtin &&
			    (be->Builtin.id == BuiltinProc_mem_zero ||
			     be->Builtin.id == BuiltinProc_mem_zero_volatile) &&
			    ce->args.count >= 2) {
				i32 ptr_off = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[0]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr_off), X64Reg_RAX);
				i32 len_off = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[1]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RAX);
				// RDI is nonvolatile in Win64 — save/restore it around REP STOSB.
				x64_emit_push_r(&p->asm_, X64Reg_RDI);
				x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RDI, x64_rbp_mem(ptr_off));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(len_off));
				x64_enc_b(&p->asm_, 0xF3); // REP
				x64_enc_b(&p->asm_, 0xAA); // STOSB
				x64_emit_pop_r(&p->asm_, X64Reg_RDI);
				return x64v_none();
			}
			// raw_data(x): slice/dyn-array/string → .data ptr (offset 0);
			// pointer/multipointer/cstring → the value itself (mirrors LLVM).
			if (be != nullptr && be->kind == Entity_Builtin &&
			    be->Builtin.id == BuiltinProc_raw_data && ce->args.count >= 1) {
				Ast  *arg0 = ce->args[0];
				Type *raw  = x64_typed(arg0->tav.type);
				Type *at   = base_type(raw);
				x64Value av = x64_build_expr(p, arg0);
				bool load_data = at != nullptr &&
				    (at->kind == Type_Slice || at->kind == Type_DynamicArray ||
				     (at->kind == Type_Basic &&
				      (at->Basic.kind == Basic_string || at->Basic.kind == Basic_string16)));
				if (load_data) {
					if (av.kind == x64Value_Mem) {
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, av.mem);
					} else {
						x64_value_to_reg(p, av, X64Reg_RAX);
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 0));
					}
					return x64v_reg(x64_typed(tav.type), X64Reg_RAX);
				}
				return av; // already a raw pointer (^T / [^]T / cstring)
			}
			return x64v_none();
		}

		// Resolve via entity_of_node, not entity_procedure_of (the latter is only set by
		// BuiltinProc_procedure_of; null for normal/pkg-qualified calls).
		Ast    *proc_node  = unparen_expr(ce->proc);
		Entity *callee_ent = entity_of_node(proc_node);

		Type   *callee_type_raw;
		String  callee_sym = {};
		bool    indirect   = false;

		if (callee_ent != nullptr && callee_ent->kind == Entity_Procedure) {
			// Skip LLVM intrinsics — not real linker symbols (e.g. llvm.sin.f64).
			if (callee_ent->Procedure.is_foreign) {
				String fl = callee_ent->Procedure.link_name;
				if (fl.len >= 5 && gb_strncmp((char const *)fl.text, "llvm.", 5) == 0) {
					return x64v_none();
				}
			}
			callee_type_raw = callee_ent->type;
			callee_sym      = x64_get_entity_name(callee_ent);

			// On-demand: a proc nothing depends on (min_dep_count == 0) — a #force_inline
			// runtime helper (bounds_check_error_loc, …) that LLVM inlines and never emits
			// standalone. We don't inline, so queue it; the driver compiles it into its owner
			// module after the parallel pass (exactly one external definition).
			if (!callee_ent->Procedure.is_foreign &&
			    callee_ent->min_dep_count.load(std::memory_order_relaxed) == 0) {
				x64_enqueue_oncall(p, callee_ent);
			}
		} else {
			callee_type_raw = ce->proc->tav.type;
			indirect = true;
		}

		Type *ct = base_type(callee_type_raw);
		if (ct == nullptr || ct->kind != Type_Proc) return x64v_none();

		bool needs_rbp  = x64_returns_by_pointer(callee_type_raw);
		bool needs_ctx  = ct->Proc.calling_convention == ProcCC_Odin;

		int param_count = ct->Proc.params ? (int)ct->Proc.params->Tuple.variables.count : (int)ce->args.count;
		// c_vararg can pass MORE args than params; size generously (params + all args).
		int n_pos_max   = (int)(ce->split_args != nullptr ? ce->split_args->positional.count : ce->args.count);
		int total_slots = (needs_rbp ? 1 : 0) + param_count + n_pos_max + (needs_ctx ? 1 : 0) + 8;
		x64Value *args = gb_alloc_array(temporary_allocator(), x64Value, total_slots);
		int slot = 0;

		// Split returns (mirror LLVM): for N results, results[0..N-2] come back via hidden
		// pointer args (after the params, before context); result[N-1] is the real return
		// (sret slot 0 if by-pointer, else RAX/XMM0). Allocate the landing locals + pointer
		// args up front. Pointers live in memory locals so arg-building (which hammers RAX)
		// can't clobber them.
		Type *last_rt   = x64_last_result_type(ct);
		int   npartial  = x64_num_partial_returns(ct);
		i32  *partial_off     = (npartial > 0) ? gb_alloc_array(temporary_allocator(), i32, npartial) : nullptr;
		x64Value *partial_ptr = (npartial > 0) ? gb_alloc_array(temporary_allocator(), x64Value, npartial) : nullptr;
		for (int i = 0; i < npartial; i++) {
			Type *pty = ct->Proc.results->Tuple.variables[i]->type;
			i64 psz = type_size_of(pty); if (psz <= 0) psz = 1;
			i64 pal = type_align_of(pty); if (pal <= 0) pal = 1;
			partial_off[i]  = x64_alloc_local(p, psz, pal);
			i32 pptr_off    = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(partial_off[i]));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(pptr_off), X64Reg_RAX);
			partial_ptr[i] = x64v_mem(alloc_type_pointer(pty), x64_rbp_mem(pptr_off));
		}

		// Hidden return pointer (sret) for the REAL (last) result, in param slot 0.
		i32 ret_local_off = 0;
		if (needs_rbp) {
			ret_local_off = x64_alloc_local(p, type_size_of(last_rt), type_align_of(last_rt));
			i32 ret_ptr_off = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(ret_local_off));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ret_ptr_off), X64Reg_RAX);
			args[slot++] = x64v_mem(alloc_type_pointer(last_rt), x64_rbp_mem(ret_ptr_off));
		}

		// Explicit args (mirror lb_build_call_expr_internal): build a PARAMETER-INDEXED value
		// array from the checker-resolved split_args (positional, then named, then defaults),
		// then lower each to ABI slots. Handles defaults/named anywhere relative to the
		// variadic, and never indexes raw ce->args out of bounds.
		{
			TypeTuple *params = ct->Proc.params ? &ct->Proc.params->Tuple : nullptr;
			int   n_params    = params ? (int)params->variables.count : 0;
			bool  is_c_vararg = ct->Proc.c_vararg;
			bool  vari_expand = (ce->ellipsis.pos.line != 0); // caller wrote ..slice

			// Odin variadic param `..T`: the caller PACKS the trailing positional args into a
			// []T backing array + {ptr,len} slice (mirrors lb_build_call_expr_internal). `..any`
			// elements are each an `any` {data_ptr, typeid}; any other `..T` stores the T value
			// directly. (c_vararg handled separately below.)
			bool  has_variadic        = false;
			bool  variadic_is_any     = false;
			int   variadic_index      = ct->Proc.variadic ? (int)ct->Proc.variadic_index : -1;
			Type *variadic_slice_type = nullptr;
			Type *variadic_elem_type  = nullptr;
			if (ct->Proc.variadic && !is_c_vararg && params &&
			    variadic_index >= 0 && variadic_index < n_params) {
				Entity *ve = params->variables[variadic_index];
				Type   *st = base_type(ve->type);
				if (st && st->kind == Type_Slice) {
					has_variadic        = true;
					variadic_slice_type = ve->type;
					variadic_elem_type  = st->Slice.elem;
					variadic_is_any     = is_type_any(variadic_elem_type);
				}
			}

			// Checker-resolved arg lists (defaults/named already separated). Fall back
			// to the raw arg list defensively.
			Slice<Ast *> positional = (ce->split_args != nullptr) ? ce->split_args->positional : ce->args;

			// Parameter-indexed values, each stabilised to memory/imm so a later arg
			// evaluation (which reuses RAX) can't clobber it. None = not yet filled.
			x64Value *pvals = gb_alloc_array(temporary_allocator(), x64Value, n_params > 0 ? n_params : 1);
			for (int i = 0; i < n_params; i++) pvals[i] = x64v_none();

			// C varargs are appended as their own ABI slots after the fixed params.
			x64Value *cvar = is_c_vararg ? gb_alloc_array(temporary_allocator(), x64Value, positional.count + 1) : nullptr;
			int cvar_count = 0;

			// Pass 1: positional args (index i == parameter index).
			for (isize i = 0; i < positional.count; i++) {
				Ast *arg = positional[i];
				if (has_variadic && (int)i == variadic_index) {
					// Pack positional[variadic_index..] into a []T slice value.
					int vari_count = (int)positional.count - variadic_index;
					if (vari_count < 0) vari_count = 0;
					if (vari_expand && vari_count >= 1) {
						// caller wrote `..slice` — pass the existing slice directly
						x64Value sv = x64_stabilize_value(p, x64_build_expr(p, positional[variadic_index]));
						if (sv.type == nullptr) sv.type = variadic_slice_type;
						pvals[variadic_index] = sv;
					} else if (vari_count == 0) {
						i32 slice_off = x64_alloc_local(p, 16, 8);
						x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off),     0);
						x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off + 8), 0);
						pvals[variadic_index] = x64v_mem(variadic_slice_type, x64_rbp_mem(slice_off));
					} else {
						// Backing array of `variadic_elem_type[vari_count]`, then a {ptr,len}.
						Type *et  = x64_typed(variadic_elem_type);
						i64   esz = type_size_of(et); if (esz <= 0) esz = 1;
						i64   eal = type_align_of(et); if (eal <= 0) eal = 1;
						i32 arr_off = x64_alloc_local(p, (i64)vari_count * esz, gb_max(eal, (i64)8));
						for (int vi = 0; vi < vari_count; vi++) {
							i32 slot_off = arr_off + vi * (i32)esz;
							if (variadic_is_any) {
								// `any` element = {data_ptr → spilled value, typeid}
								x64Value av = x64_build_expr(p, positional[variadic_index + vi]);
								Type *at = av.type ? av.type : t_any;
								at = x64_typed(at);
								i64 asz = type_size_of(at); if (asz <= 0) asz = 1;
								i64 aal = type_align_of(at); if (aal <= 0) aal = 1;
								i32 val_off = x64_alloc_local(p, asz, aal);
								x64_store_value(p, x64addr(x64_rbp_mem(val_off), at), av);
								x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(val_off));
								x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slot_off), X64Reg_RAX);
								u64 tid = type_hash_canonical_type(at);
								x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)tid);
								x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slot_off + 8), X64Reg_RAX);
							} else {
								// `..T` element = the T value (mirrors LLVM lb_emit_conv to elem)
								x64Value ev = x64_build_expr(p, positional[variadic_index + vi]);
								ev = x64_apply_using_subtype(p, ev, et);
								x64_store_value(p, x64addr(x64_rbp_mem(slot_off), et), ev);
							}
						}
						i32 slice_off = x64_alloc_local(p, 16, 8);
						x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(arr_off));
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off), X64Reg_RAX);
						x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)vari_count);
						x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off + 8), X64Reg_RAX);
						pvals[variadic_index] = x64v_mem(variadic_slice_type, x64_rbp_mem(slice_off));
					}
					break; // positional beyond variadic_index are consumed here
				}
				if (is_c_vararg && ct->Proc.variadic && variadic_index >= 0 && (int)i >= variadic_index) {
					cvar[cvar_count++] = x64_stabilize_value(p, x64_build_expr(p, arg));
					continue;
				}
				if (arg->tav.mode == Addressing_Type) {
					// Poly type param ($T) → Entity_TypeName, no runtime slot. A runtime
					// `typeid` VALUE param instead gets the type's typeid (built below).
					if (i >= n_params || params->variables[i]->kind == Entity_TypeName) continue;
				}
				if (i >= n_params) continue;                    // safety (non-variadic overflow)
				x64Value av = x64_stabilize_value(p, x64_build_expr(p, arg));
				if (av.type == nullptr) av.type = params->variables[i]->type;
				pvals[i] = av;
			}

			// Pass 2: named args → param-indexed slots.
			if (ce->split_args != nullptr) {
				for (isize ni = 0; ni < ce->split_args->named.count; ni++) {
					Ast *fv_node = ce->split_args->named[ni];
					if (fv_node->kind != Ast_FieldValue) continue;
					ast_node(fv, FieldValue, fv_node);
					if (!fv->field || fv->field->kind != Ast_Ident) continue;
					isize pidx = lookup_procedure_parameter(ct, fv->field->Ident.token.string);
					if (pidx < 0 || pidx >= n_params) continue;
					x64Value av = x64_stabilize_value(p, x64_build_expr(p, fv->value));
					if (av.type == nullptr) av.type = params->variables[pidx]->type;
					pvals[pidx] = av;
				}
			}

			// Pass 3: fill defaults for params still empty (mirrors LLVM's final loop
			// over all params at llvm_backend_proc.cpp:5026).
			for (int i = 0; i < n_params; i++) {
				Entity *pe = params->variables[i];
				if (has_variadic && i == variadic_index) {
					if (pvals[i].kind == x64Value_None) { // empty variadic → nil slice
						i32 slice_off = x64_alloc_local(p, 16, 8);
						x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off),     0);
						x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off + 8), 0);
						pvals[i] = x64v_mem(variadic_slice_type, x64_rbp_mem(slice_off));
					}
					continue;
				}
				if (pe->kind != Entity_Variable) continue;
				if (pvals[i].kind != x64Value_None) continue; // already positional/named
				x64Value dv = x64v_none();
				if (pe->Variable.param_value.kind != ParameterValue_Invalid) {
					dv = x64_handle_param_value(p, pe->type, pe->Variable.param_value);
				}
				if (dv.kind == x64Value_None || dv.type == nullptr) {
					dv = x64v_imm(pe->type ? pe->type : t_int, 0);
				}
				if (dv.type == nullptr) dv.type = pe->type;
				pvals[i] = x64_stabilize_value(p, dv);
			}

			// Lower each parameter value to ABI slots in parameter order. Zero-sized
			// params consume no slot; Win64 indirect aggregates are passed by pointer.
			for (int i = 0; i < n_params; i++) {
				Entity *pe = params->variables[i];
				if (pe->kind == Entity_TypeName) continue;              // no runtime slot
				Type *pt2 = pe->type;
				if (pt2 != nullptr && type_size_of(pt2) == 0) continue; // zero-sized
				x64Value v = pvals[i];
				if (v.kind == x64Value_None) v = x64v_imm(pt2 ? pt2 : t_int, 0);
				if (v.type == nullptr) v.type = pt2;
				// `using`-field subtype: a Temp_Allocator passed where Allocator is wanted
				// narrows to the embedded `allocator` field (mirrors lb_emit_conv subtype path).
				v = x64_apply_using_subtype(p, v, pt2);
				if (x64_arg_is_indirect(pt2)) {
					if (v.kind != x64Value_Mem) {
						i64 asz = type_size_of(pt2); if (asz <= 0) asz = 8;
						i64 aal = type_align_of(pt2); if (aal <= 0) aal = 8;
						i32 tmp = x64_alloc_local(p, asz, aal);
						x64_zero_mem(p, x64_rbp_mem(tmp), asz); // covers imm/None defaults
						v = x64v_mem(pt2, x64_rbp_mem(tmp));
					}
					i32 ptr = x64_alloc_local(p, 8, 8);
					x64_emit_lea(&p->asm_, X64Reg_RAX, v.mem);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
					args[slot++] = x64v_mem(alloc_type_pointer(pt2), x64_rbp_mem(ptr));
				} else {
					args[slot++] = v;
				}
			}

			// C varargs follow the fixed parameters as their own slots.
			for (int i = 0; i < cvar_count; i++) args[slot++] = cvar[i];

		}

		// Split-return hidden pointers for results[0..N-2] — after params, before context
		// (mirrors LLVM's arg order: [sret][params][partial-rets][context]).
		for (int i = 0; i < npartial; i++) args[slot++] = partial_ptr[i];

		// Context — keep as memory ref so x64_emit_call loads it fresh (arg-building
		// above clobbers RAX, so we can't snapshot it here).
		if (needs_ctx) {
			args[slot++] = x64_context_ptr_value(p);
		}

		if (!indirect) {
			x64Value rv = x64_emit_call(p, callee_sym, callee_type_raw, args, slot);

			// Capture/assemble the result NOW, before the deferred-procedure block below
			// (which allocates locals and clobbers RAX) can destroy a register result.
			x64Value call_result = x64_reconstruct_call_result(p, ct, npartial, last_rt, needs_rbp, ret_local_off, partial_off, rv);

			// @(deferred_*): register a deferred call at scope exit (mirrors lb_emit_call's
			// deferred handling). Covers sync.guard's deferred_in unlock on return.
			if (callee_ent != nullptr && entity_has_deferred_procedure(callee_ent)) {
				DeferredProcedureKind dk = callee_ent->Procedure.deferred_procedure.kind;
				Entity *dproc = callee_ent->Procedure.deferred_procedure.entity;
				bool is_in_kind = (dk == DeferredProcedure_in || dk == DeferredProcedure_in_by_ptr);
				if (dproc != nullptr && is_in_kind) {
					bool   by_ptr = (dk == DeferredProcedure_in_by_ptr);
					Type  *dct    = base_type(dproc->type);
					bool   dctx   = dct != nullptr && dct->kind == Type_Proc &&
					                dct->Proc.calling_convention == ProcCC_Odin;
					int first_explicit = needs_rbp ? 1 : 0;
					int last_explicit  = slot - (needs_ctx ? 1 : 0);
					int dcount = last_explicit - first_explicit;
					x64Value *dargs = gb_alloc_array(permanent_allocator(), x64Value, dcount + 1);
					int di = 0;
					for (int ai = first_explicit; ai < last_explicit; ai++) {
						x64Value av = args[ai];
						// Spill to a fresh local so the value survives until the
						// deferred call is emitted (at return / scope exit).
						Type *at = av.type ? av.type : t_rawptr;
						i64 asz = type_size_of(at); if (asz <= 0) asz = 8;
						i64 aal = type_align_of(at); if (aal <= 0) aal = 8;
						i32 soff = x64_alloc_local(p, asz, aal);
						x64_store_value(p, x64addr(x64_rbp_mem(soff), at), av);
						x64Value sv = x64v_mem(at, x64_rbp_mem(soff));
						if (by_ptr) {
							i32 poff = x64_alloc_local(p, 8, 8);
							x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(soff));
							x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(poff), X64Reg_RAX);
							sv = x64v_mem(alloc_type_pointer(at), x64_rbp_mem(poff));
						}
						dargs[di++] = sv;
					}
					if (dctx) {
						dargs[di++] = x64_context_ptr_value(p);
					}
					x64Procedure::DeferEntry de = {};
					de.call_proc = dproc;
					de.call_type = dproc->type;
					de.call_args = dargs;
					de.call_argc = di;
					array_add(&p->deferred, de);
				}
			}

			return call_result;
		}

		// Indirect call: load callee into R10, then set up args
		x64Value proc_v = x64_build_expr(p, ce->proc);
		x64_value_to_reg(p, proc_v, X64Reg_R10);

		// Place args (re-use the same logic as direct call, minus CALL sym)
		int reg_count = gb_min(slot, 4);
		for (int i = reg_count - 1; i >= 0; i--) {
			if (x64_arg_is_float(args[i].type)) x64_value_to_xmm(p, args[i], X64_XMM_ARG_REGS[i]);
			else                                 x64_value_to_reg (p, args[i], X64_INT_ARG_REGS[i]);
		}
		for (int i = 4; i < slot; i++) {
			x64_value_to_reg(p, args[i], X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_mem(X64Reg_RSP, 32 + (i-4)*8), X64Reg_RAX);
		}
		x64_emit_call_r(&p->asm_, X64Reg_R10);

		// Single-value register result (multi/sret handled inside reconstruct_call_result).
		x64Value ind_rv = x64v_none();
		if (!needs_rbp && ct->Proc.result_count == 1) {
			Type *ret_t = ct->Proc.results->Tuple.variables[0]->type;
			if (ret_t && type_size_of(ret_t) != 0) {
				ind_rv = x64_is_float(ret_t) ? x64v_xmm(ret_t, X64XmmReg_XMM0)
				                             : x64v_reg(ret_t, X64Reg_RAX);
			}
		}
		return x64_reconstruct_call_result(p, ct, npartial, last_rt, needs_rbp, ret_local_off, partial_off, ind_rv);
	} case_end;

	case_ast_node(te, TernaryIfExpr, expr); {
		x64Value cond_v = x64_build_expr(p, te->cond);
		x64_value_to_reg(p, cond_v, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, x64_op_size_of(te->cond->tav.type), X64Reg_RAX, X64Reg_RAX);
		isize lbl_false = x64_label_alloc(&p->asm_);
		isize lbl_end   = x64_label_alloc(&p->asm_);
		x64_emit_jcc(&p->asm_, X64Cc_E, lbl_false);

		i32 off = x64_alloc_local(p, type_size_of(tav.type), type_align_of(tav.type));
		x64Value tv = x64_build_expr(p, te->x);
		x64_store_value(p, x64addr(x64_rbp_mem(off), tav.type), tv);
		x64_emit_jmp(&p->asm_, lbl_end);
		x64_label_bind(&p->asm_, lbl_false);
		x64Value fv = x64_build_expr(p, te->y);
		x64_store_value(p, x64addr(x64_rbp_mem(off), tav.type), fv);
		x64_label_bind(&p->asm_, lbl_end);
		return x64_load_addr(p, x64addr(x64_rbp_mem(off), tav.type));
	} case_end;

	// ── SliceExpr (arr[lo:hi] / slice[lo:hi]) ────────────────────────────
	case_ast_node(se, SliceExpr, expr); {
		Type *src_raw = x64_typed(se->expr->tav.type);
		Type *src_bt  = base_type(src_raw);
		if (src_bt == nullptr) return x64v_none();

		x64Addr src_addr = x64_build_addr(p, se->expr);

		// Stabilise effective address before any sub-expression clobbers registers
		i32 src_ea_off = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, src_addr.mem);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(src_ea_off), X64Reg_RAX);

		Type *elem_t = t_u8;
		i64   esz    = 1;
		if      (src_bt->kind == Type_Array)         { elem_t = src_bt->Array.elem;         esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_Slice)          { elem_t = src_bt->Slice.elem;          esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_DynamicArray)   { elem_t = src_bt->DynamicArray.elem;   esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_MultiPointer)   { elem_t = src_bt->MultiPointer.elem;   esz = type_size_of(elem_t); }
		if (esz <= 0) esz = 1;

		i32 base_ptr_off = x64_alloc_local(p, 8, 8);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
		if (src_bt->kind != Type_Array) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 0)); // load .data field
		}
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base_ptr_off), X64Reg_RAX);

		// Low index (default 0); high index (default: array count or slice .len)
		i32 lo_off = x64_alloc_local(p, 8, 8);
		if (se->low != nullptr) {
			x64Value lv = x64_build_expr(p, se->low);
			x64_value_to_reg(p, lv, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(lo_off), X64Reg_RAX);
		} else {
			x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(lo_off), 0);
		}

		i32 hi_off = x64_alloc_local(p, 8, 8);
		if (se->high != nullptr) {
			x64Value hv = x64_build_expr(p, se->high);
			x64_value_to_reg(p, hv, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(hi_off), X64Reg_RAX);
		} else if (src_bt->kind == Type_Array) {
			x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)src_bt->Array.count);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(hi_off), X64Reg_RAX);
		} else if (src_bt->kind == Type_MultiPointer) {
			// [^]T has no length field; hi is unused when result is also [^]T
			x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(hi_off), 0);
		} else {
			// Load .len from offset +8 of slice/DA struct
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 8));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(hi_off), X64Reg_RAX);
		}

		i32 slice_off = x64_alloc_local(p, 16, 8);

		// data = base_ptr + lo * esz
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_ptr_off));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(lo_off));
		if (esz > 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX, (i32)esz);
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off), X64Reg_RAX);

		// len = hi - lo
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(hi_off));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(lo_off));
		x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slice_off + 8), X64Reg_RAX);

		return x64v_mem(tav.type ? x64_typed(tav.type) : t_rawptr, x64_rbp_mem(slice_off));
	} case_end;

	// ── CompoundLit ({field: val} or {v0, v1, ...}) ──────────────────────
	case_ast_node(cl, CompoundLit, expr); {
		Type *type = tav.type ? x64_typed(tav.type) : nullptr;
		if (type == nullptr) return x64v_none();
		return x64_build_compound_lit(p, expr, type);
	} case_end;

	// ── TypeAssertion (expr.(T)  /  expr.?) ──────────────────────────────
	case_ast_node(ta, TypeAssertion, expr); {
		bool is_tuple = is_type_tuple(tav.type);

		// `x.?` (Maybe/union unwrap): its type node is a `?` UnaryExpr and type_of_expr(`?`)
		// is null, so derive the asserted type from the result (tav.type = union variant;
		// comma-ok promotes to (value, bool)). Was type_of_expr(`?`)→null→x64v_none.
		bool maybe_unwrap = ta->type != nullptr && ta->type->kind == Ast_UnaryExpr &&
		                    ta->type->UnaryExpr.op.kind == Token_Question;
		Type *dst_type = nullptr;
		if (ta->type != nullptr && !maybe_unwrap) {
			dst_type = type_of_expr(ta->type);
		} else if (is_tuple && tav.type != nullptr && base_type(tav.type)->Tuple.variables.count > 0) {
			dst_type = base_type(tav.type)->Tuple.variables[0]->type;
		} else {
			dst_type = tav.type;
		}
		if (dst_type == nullptr) return x64v_none();

		x64Value src     = x64_build_expr(p, ta->expr);
		Type    *src_raw = x64_typed(ta->expr->tav.type);
		Type    *src_bt  = base_type(src_raw);
		if (src_bt == nullptr) return x64v_none();

		// Ensure src is in memory so we can take its effective address
		if (src.kind != x64Value_Mem) {
			i64 ssz = x64_type_size(src_bt);
			i64 sal = x64_type_align(src_bt);
			if (ssz <= 0) ssz = 8;
			if (sal <= 0) sal = 8;
			i32 soff = x64_alloc_local(p, ssz, sal);
			x64_store_value(p, x64addr(x64_rbp_mem(soff), src_bt), src);
			src = x64v_mem(src_bt, x64_rbp_mem(soff));
		}

		// Stabilise effective address so multi-step loads don't clobber the base
		i32 src_ea_off = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, src.mem);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(src_ea_off), X64Reg_RAX);

		i32 data_ptr_off = x64_alloc_local(p, 8, 8); // will hold ^T result

		if (is_type_any(src_bt)) {
			// any = {data: rawptr @ +0, id: typeid @ +8}
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 0)); // RCX = any.data
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_ptr_off), X64Reg_RCX);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8)); // RCX = any.id
			u64 expected_id = type_hash_canonical_type(dst_type);
			x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)expected_id);
			x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
			x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);
		} else if (is_type_union(src_bt)) {
			if (is_type_union_maybe_pointer(src_bt)) {
				// Single-pointer-variant union: the value IS the pointer at offset 0. Keep
				// data_ptr_off = &union (uniform with the tagged case); ok = (ptr != nil).
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(src_ea_off)); // RCX = &union
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_ptr_off), X64Reg_RCX);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RCX, 0));  // RAX = ptr
				x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
				x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RAX);                             // AL = ok
			} else {
				i64 tag_off_b  = src_bt->Union.variant_block_size;
				i64 tag_sz     = union_tag_size(src_bt);
				X64OpSize tag_opsz = tag_sz <= 1 ? X64OpSize_8 :
				                     tag_sz == 2  ? X64OpSize_16 :
				                     tag_sz == 4  ? X64OpSize_32 : X64OpSize_64;
				i64 expected_tag = union_variant_index_checked(src_bt, dst_type);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(src_ea_off)); // RCX = &union
				X64Mem tag_m = x64_mem(X64Reg_RCX, (i32)tag_off_b);
				if (tag_opsz == X64OpSize_64) x64_emit_mov_rm  (&p->asm_, X64OpSize_64, X64Reg_RAX, tag_m);
				else                          x64_emit_movzx_rm(&p->asm_, tag_opsz,     X64Reg_RAX, tag_m);
				x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)expected_tag);
				x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(data_ptr_off), X64Reg_RCX); // data at offset 0
			}
		} else {
			return x64v_none();
		}

		// Both forms yield the variant VALUE copied out, NOT its address (`s` in `s, ok := x.(T)`
		// is a T by value). data_ptr_off → pointer to the value; AL = ok. Mirrors lb_emit_union_cast:
		// zero-init the result, copy the value only on a match (else return the zero value).
		Type *res_t  = tav.type;
		Type *res_bt = base_type(res_t);
		if (is_tuple) {
			Entity *v0   = res_bt->Tuple.variables[0];
			Entity *v1   = res_bt->Tuple.variables[1];
			i64     dsz  = type_size_of(v0->type); if (dsz < 0)  dsz  = 0;
			i64     okal = type_align_of(v1->type); if (okal <= 0) okal = 1;
			i64     okoff = (dsz + okal - 1) & ~(okal - 1);
			i64     tsz  = type_size_of(res_t); if (tsz <= 0) tsz = okoff + 1;

			i32 res_off = x64_alloc_local(p, tsz, type_align_of(res_t));
			i32 ok_tmp  = x64_alloc_local(p, 1, 1);
			x64_emit_mov_mr(&p->asm_, X64OpSize_8, x64_rbp_mem(ok_tmp), X64Reg_RAX); // save ok (AL)
			x64_zero_mem(p, x64_rbp_mem(res_off), tsz);                              // zero value + ok

			// Copy the value only on a match (else leave the zero value).
			isize skip = x64_label_alloc(&p->asm_);
			x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_RAX, x64_rbp_mem(ok_tmp));
			x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
			x64_emit_jcc(&p->asm_, X64Cc_E, skip);
			if (dsz > 0) {
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(data_ptr_off));
				x64_copy_fixed(p, x64_rbp_mem(res_off), x64_mem(X64Reg_RCX, 0), dsz);
			}
			x64_label_bind(&p->asm_, skip);

			x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_RAX, x64_rbp_mem(ok_tmp));
			x64_emit_mov_mr(&p->asm_, X64OpSize_8, x64_rbp_mem(res_off + (i32)okoff), X64Reg_RAX);
			return x64v_mem(res_t, x64_rbp_mem(res_off));
		} else {
			// Unsafe assertion (Odin panics on mismatch; no panic emitted yet) — copy value.
			i64 dsz = type_size_of(res_t); if (dsz <= 0) dsz = 8;
			i32 res_off = x64_alloc_local(p, dsz, type_align_of(res_t));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(data_ptr_off));
			x64_copy_fixed(p, x64_rbp_mem(res_off), x64_mem(X64Reg_RCX, 0), dsz);
			return x64v_mem(res_t, x64_rbp_mem(res_off));
		}
	} case_end;

	// ── OrReturnExpr ─────────────────────────────────────────────────────────
	case_ast_node(oe, OrReturnExpr, expr); {
		// Build inner expression (call, type assertion, etc.)
		x64Value rv = x64_build_expr(p, oe->expr);

		// Extract rhs (the bool/ok indicator) and lhs (the success value, if any)
		x64Value rhs = rv;
		x64Value lhs = x64v_none();

		Type *inner_type = rv.type;
		if (inner_type && is_type_tuple(inner_type)) {
			TypeTuple *tup = &inner_type->Tuple;
			int n = (int)tup->variables.count;
			if (rv.kind == x64Value_Mem) {
				// Compute byte offset of the last tuple field (the bool/ok)
				i64 rhs_off = 0;
				for (int i = 0; i < n - 1; i++) {
					Entity *e = tup->variables[i];
					i64 al = type_align_of(e->type);
					rhs_off = (rhs_off + al - 1) & ~(al - 1);
					rhs_off += type_size_of(e->type);
				}
				Entity *last_e = tup->variables[n - 1];
				i64 al = type_align_of(last_e->type);
				rhs_off = (rhs_off + al - 1) & ~(al - 1);
				rhs = x64v_mem(last_e->type,
				               x64_mem(rv.mem.base, rv.mem.disp + (i32)rhs_off));
				if (tav.type != nullptr && n >= 2) {
					lhs = x64v_mem(tup->variables[0]->type, rv.mem);
				}
			}
			// else rv.kind == x64Value_None: inner expr didn't expose result buffer;
			// x64_value_to_reg on x64v_none() will XOR-zero RAX → always fail path
		}

		// Load rhs into RAX; spill it so deferred stmts can't clobber it
		Type *rhs_t = rhs.type ? rhs.type : t_bool;
		x64_value_to_reg(p, rhs, X64Reg_RAX);
		X64OpSize rhs_opsz = x64_op_size_of(rhs_t);
		i32 rhs_spill = x64_alloc_local(p, type_size_of(rhs_t), type_align_of(rhs_t));
		x64_emit_mov_mr(&p->asm_, rhs_opsz, x64_rbp_mem(rhs_spill), X64Reg_RAX);

		// Success condition depends on the indicator TYPE (mirrors lb_emit_try_has_value):
		// a BOOLEAN ok-flag succeeds when TRUE; an ERROR / nil-able value when NIL. The old
		// code only did the boolean direction, so `or_return` on an Error proc was INVERTED
		// (continued on error, bailed on success — broke all of math/big).
		x64_emit_test_rr(&p->asm_, rhs_opsz, X64Reg_RAX, X64Reg_RAX);
		isize lbl_continue = x64_label_alloc(&p->asm_);
		if (is_type_boolean(rhs_t)) {
			x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_continue); // bool true → success
		} else {
			x64_emit_jcc(&p->asm_, X64Cc_E, lbl_continue);  // error == nil → success
		}

		// ── Failure path: return from this procedure ──────────────────────────
		x64_run_deferred(p);

		Type *pt = p->type;
		if (pt->Proc.results != nullptr) {
			if (pt->Proc.has_named_results) {
				// Store rhs into the last named result entity's local, then emit
				// named returns which copies locals to the return area.
				TypeTuple *res_tup = &pt->Proc.results->Tuple;
				int nres = (int)res_tup->variables.count;
				if (nres > 0) {
					Entity *last_res = res_tup->variables[nres - 1];
					i32 *loff = x64_var_get(&p->var_offsets, last_res);
					if (loff != nullptr) {
						x64_emit_mov_rm(&p->asm_, rhs_opsz, X64Reg_RAX,
						                x64_rbp_mem(rhs_spill));
						x64_emit_mov_mr(&p->asm_, x64_op_size_of(last_res->type),
						                x64_rbp_mem(*loff), X64Reg_RAX);
					}
				}
				x64_emit_named_returns(p);
			} else if (p->returns_by_pointer) {
				// Write rhs to the last field of the hidden return struct
				TypeTuple *res_tup = &pt->Proc.results->Tuple;
				int nres = (int)res_tup->variables.count;
				if (nres > 0) {
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX,
					                x64_rbp_mem(x64_param_rbp_off(0)));
					i64 last_off = 0;
					for (int i = 0; i < nres - 1; i++) {
						Entity *e = res_tup->variables[i];
						i64 al = type_align_of(e->type);
						last_off = (last_off + al - 1) & ~(al - 1);
						last_off += type_size_of(e->type);
					}
					Entity *last_res = res_tup->variables[nres - 1];
					i64 al = type_align_of(last_res->type);
					last_off = (last_off + al - 1) & ~(al - 1);
					x64_emit_mov_rm(&p->asm_, rhs_opsz, X64Reg_RCX,
					                x64_rbp_mem(rhs_spill));
					x64_emit_mov_mr(&p->asm_, x64_op_size_of(last_res->type),
					                x64_mem(X64Reg_RAX, (i32)last_off), X64Reg_RCX);
				}
			} else {
				// Single register return: load rhs into RAX
				x64_emit_mov_rm(&p->asm_, rhs_opsz, X64Reg_RAX,
				                x64_rbp_mem(rhs_spill));
			}
		}

		x64_proc_emit_epilogue(p);
		x64_emit_ret(&p->asm_);

		// ── Success path ──────────────────────────────────────────────────────
		x64_label_bind(&p->asm_, lbl_continue);

		if (tav.type != nullptr) {
			return lhs;
		}
		return x64v_none();
	} case_end;

	// ── OrElseExpr (x or_else y) ─────────────────────────────────────────────
	case_ast_node(oe, OrElseExpr, expr); {
		Type *rt = tav.type ? x64_typed(tav.type) : nullptr;
		x64Value lhs, rhs;
		x64_try_lhs_rhs(p, oe->x, rt, &lhs, &rhs);
		if (rt == nullptr) rt = lhs.type ? lhs.type : t_int;

		i64 rsz = type_size_of(rt); if (rsz <= 0) rsz = 8;
		i64 ral = type_align_of(rt); if (ral <= 0) ral = 8;
		i32 res_off = x64_alloc_local(p, rsz, ral);

		x64_try_has_value(p, rhs); // RAX = has_value
		x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		isize lbl_else = x64_label_alloc(&p->asm_);
		isize lbl_end  = x64_label_alloc(&p->asm_);
		x64_emit_jcc(&p->asm_, X64Cc_E, lbl_else); // has_value == 0 → else expr

		// success: result = lhs (or nil when the operand had no value part —
		// mirrors lb_emit_or_else's lb_const_nil fallback).
		if (lhs.kind != x64Value_None) {
			x64_store_value(p, x64addr(x64_rbp_mem(res_off), rt), lhs);
		} else {
			x64_zero_mem(p, x64_rbp_mem(res_off), rsz);
		}
		x64_emit_jmp(&p->asm_, lbl_end);

		// failure: result = else expression
		x64_label_bind(&p->asm_, lbl_else);
		x64Value ev = x64_build_expr(p, oe->y);
		x64_store_value(p, x64addr(x64_rbp_mem(res_off), rt), ev);

		x64_label_bind(&p->asm_, lbl_end);
		return x64v_mem(rt, x64_rbp_mem(res_off));
	} case_end;

	// ── OrBranchExpr (x or_break / x or_continue) ────────────────────────────
	case_ast_node(be, OrBranchExpr, expr); {
		bool   is_break = (be->token.kind == Token_or_break);
		String lbl_name = (be->label != nullptr && be->label->kind == Ast_Ident)
		                ? be->label->Ident.token.string : String{};

		// Resolve the target loop/switch's break/continue label + defer floor.
		isize target = -1, floor = 0;
		if (lbl_name.len == 0) {
			for (isize i = p->loops.count - 1; i >= 0; i--) {
				isize t = is_break ? p->loops[i].lbl_break : p->loops[i].lbl_continue;
				if (t >= 0) { target = t; floor = p->loops[i].defer_base; break; }
			}
		} else {
			x64Procedure::LoopInfo *li = x64_find_loop(p, lbl_name);
			if (li != nullptr) {
				target = is_break ? li->lbl_break : li->lbl_continue;
				floor  = li->defer_base;
			}
		}

		Type *rt = tav.type ? x64_typed(tav.type) : nullptr;
		x64Value lhs, rhs;
		x64_try_lhs_rhs(p, be->expr, rt, &lhs, &rhs);

		x64_try_has_value(p, rhs); // RAX = has_value
		x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		isize lbl_then = x64_label_alloc(&p->asm_);
		x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_then); // success → continue with lhs

		// failure: unwind defers down to the loop floor, then break/continue.
		x64_run_deferred_from(p, floor);
		if (target >= 0) x64_emit_jmp(&p->asm_, target);

		x64_label_bind(&p->asm_, lbl_then);
		return (tav.type != nullptr) ? lhs : x64v_none();
	} case_end;

	default:
		if (tav.mode == Addressing_Variable) {
			x64Addr addr = x64_build_addr(p, expr);
			return x64_load_addr(p, addr);
		}
		return x64v_none();
	}
	return x64v_none();
}
