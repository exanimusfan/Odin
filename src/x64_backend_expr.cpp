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
// {ptr, len} for a string / []u8 / []T constant. `elem_size` > 1 means a non-u8 slice (#load
// to []u32 etc.): len is the ELEMENT count = bytes / elem_size (mirrors LLVM's data_len /= sz).
gb_internal x64Value x64_const_string(x64Procedure *p, String sv, i64 elem_size = 1) {
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
	i64 len = (elem_size > 1) ? sv.len / elem_size : sv.len;
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, len);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(str_off + 8), X64Reg_RAX);
	return x64v_mem(t_string, x64_rbp_mem(str_off));
}

// A constant string used where a cstring/cstring16 is expected: a bare NUL-terminated data
// POINTER (8 bytes), NOT a {data,len} header. `wide` UTF-16-encodes the bytes (cstring16, e.g.
// passing "en-US" to a Win32 LPCWSTR param). Mirrors LLVM lb_emit_conv string->cstring and the
// constant_utf16_cstring builtin. Without this a const string arg to a cstring16 param was built
// as a 16-byte string struct → the callee read UTF-8 bytes as UTF-16 (Win32 LocaleNameToLCID → 0).
gb_internal x64Value x64_const_cstring_ptr(x64Procedure *p, String sv, bool wide, Type *result_type) {
	x64Module *m = p->module;
	if (sv.len == 0 && !wide) {
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 0); // empty cstring → null pointer
		return x64v_reg(result_type, X64Reg_RAX);
	}
	String sym;
	if (wide) {
		u16  *buf = gb_alloc_array(temporary_allocator(), u16, sv.len + 2);
		isize n = 0; u8 const *text = sv.text; isize len = sv.len;
		while (len > 0) {
			Rune  r = 0;
			isize w = gb_utf8_decode(text, len, &r);
			text += w; len -= w;
			if ((0 <= r && r < 0xd800) || (0xe000 <= r && r < 0x10000)) {
				buf[n++] = (u16)r;
			} else if (0x10000 <= r && r <= 0x10ffff) {
				Rune rr = r - 0x10000;
				buf[n++] = (u16)(0xd800 + ((rr >> 10) & 0x3ff));
				buf[n++] = (u16)(0xdc00 + (rr & 0x3ff));
			} else {
				buf[n++] = 0xfffd;
			}
		}
		buf[n++] = 0; // NUL terminator
		sym = x64_const_intern_string(m, make_string((u8 const *)buf, n * 2));
	} else {
		sym = x64_const_intern_string(m, sv);
	}
	x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
	return x64v_reg(result_type, X64Reg_RAX);
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
	p->named_seq++; // scope-lived slot: must survive later statements' temp reclamation

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
	p->named_seq++; // scope-lived slot: must survive later statements' temp reclamation
}

// Materialise a runtime.Source_Code_Location {file: string, line, column: i32, procedure: string}
// on the stack and return it as a mem value (mirrors lb_const_source_code_location_const). Used for
// a `loc := #caller_location` default parameter: the value describes the CALL SITE, so `proc_name`
// is the CALLER's proc and `pos` is the call expression's token position.
gb_internal x64Value x64_build_source_code_location(x64Procedure *p, String proc_name, TokenPos pos) {
	String file = get_file_path_string(pos.file_id);
	i32 line = pos.line, column = pos.column;
	switch (build_context.source_code_location_info) {
	case SourceCodeLocationInfo_Normal: break;
	case SourceCodeLocationInfo_Obfuscated:
		file = obfuscate_string(file, "F"); proc_name = obfuscate_string(proc_name, "P");
		line = obfuscate_i32(line); column = obfuscate_i32(column); break;
	case SourceCodeLocationInfo_Filename: file = last_path_element(file); break;
	case SourceCodeLocationInfo_None:
		file = str_lit(""); proc_name = str_lit(""); line = 0; column = 0; break;
	}

	Type *scl = t_source_code_location;
	i64 sz = type_size_of(scl); if (sz <= 0) sz = 8;
	i64 al = type_align_of(scl); if (al <= 0) al = 8;
	i32 off = x64_alloc_local(p, sz, al);
	x64_zero_mem(p, x64_rbp_mem(off), sz);

	x64Value fs = x64_const_string(p, file);     // {data,len} 16B
	x64_copy_fixed(p, x64_rbp_mem(off + (i32)type_offset_of(scl, 0)), fs.mem, 16);
	x64_emit_mov_mi(&p->asm_, X64OpSize_32, x64_rbp_mem(off + (i32)type_offset_of(scl, 1)), line);
	x64_emit_mov_mi(&p->asm_, X64OpSize_32, x64_rbp_mem(off + (i32)type_offset_of(scl, 2)), column);
	x64Value ps = x64_const_string(p, proc_name);
	x64_copy_fixed(p, x64_rbp_mem(off + (i32)type_offset_of(scl, 3)), ps.mem, 16);

	return x64v_mem(scl, x64_rbp_mem(off));
}

// Bounds checking off for this proc? (mirrors lb_bounds_check_disabled; x64Procedure has no per-proc
// state_flags yet, so only the global -no-bounds-check is honoured.) x64_emit_bounds_check itself is
// defined after x64_emit_conv/x64_spill_value (which it uses) and forward-declared in x64_backend.hpp.
gb_internal bool x64_bounds_check_disabled(x64Procedure *p) {
	if (build_context.no_bounds_check) return true;
	return (p->state_flags & StateFlag_no_bounds_check) != 0;
}

// Fold a node's `#no_bounds_check`/`#bounds_check` directive into p->state_flags, INHERITING the
// parent's flags (mirrors lb_build_stmt/lb_build_expr). Returns the previous flags to restore after
// building the node. node->state_flags carries only the directly-attached directive; the inherited
// accumulation happens here at build time.
gb_internal u16 x64_push_state_flags(x64Procedure *p, Ast *node) {
	u16 prev = p->state_flags;
	u16 in = (node != nullptr) ? (u16)node->state_flags : 0;
	if (in != 0) {
		u16 out = prev;
		if (in & StateFlag_bounds_check)         { out |=  StateFlag_bounds_check;    out &= ~StateFlag_no_bounds_check; }
		else if (in & StateFlag_no_bounds_check) { out |=  StateFlag_no_bounds_check; out &= ~StateFlag_bounds_check;    }
		p->state_flags = out;
	}
	return prev;
}

// Byte-swap the low 16 bits of `r` (for f16be <-> LE f16 conversion). r ends holding the swapped u16.
gb_internal void x64_emit_bswap16(x64Procedure *p, X64Reg r) {
	x64_emit_mov_rr(&p->asm_, X64OpSize_32, X64Reg_RCX, r);
	x64_emit_shl_ri(&p->asm_, X64OpSize_32, X64Reg_RCX, 8);   // RCX = bits << 8
	x64_emit_shr_ri(&p->asm_, X64OpSize_32, r, 8);            // r   = bits >> 8
	x64_emit_or_rr(&p->asm_, X64OpSize_32, r, X64Reg_RCX);
	x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RCX, 0xFFFF);
	x64_emit_and_rr(&p->asm_, X64OpSize_32, r, X64Reg_RCX);   // keep low 16
}

gb_internal x64Value x64_build_compound_lit(x64Procedure *p, Ast *lit, Type *type); // defined below

// A compile-time integer constant of a BIG-ENDIAN type (u16be/u32be/u64be) is materialised as a
// host-order immediate; pre-swap to the type's byte width so the little-endian store lays the bytes
// out big-endian (mirrors x64_const_int_bytes' static path). e.g. net's IP6_Address = [8]u16be, whose
// `{0x2620, …}` literals compared unequal to DNS-parsed addresses because the constants weren't swapped.
gb_internal i64 x64_endian_fix_const_int(Type *type, i64 v) {
	Type *bt = type ? base_type(x64_typed(type)) : nullptr;
	if (bt == nullptr || !is_type_integer(bt) || is_type_endian_little(bt)) return v;
	i64 n = type_size_of(bt); if (n <= 1 || n > 8) return v;
	u64 in = (u64)v, out = 0;
	for (i64 i = 0; i < n; i++) { out = (out << 8) | (in & 0xff); in >>= 8; }
	return (i64)out;
}

// Value for an omitted default parameter (mirrors lb_handle_param_value). For
// ParameterValue_Constant, materialise the stored ExactValue directly rather than
// re-evaluating original_ast_expr — that AST node often lacks a usable tav/entity
// in the backend, silently yielding 0 (e.g. `flush := true` arriving as false).
// `call_expr` is the CallExpr Ast (for #caller_location / #caller_expression, which describe
// the call site); may be null for synthetic calls.
gb_internal x64Value x64_handle_param_value(x64Procedure *p, Type *ptype,
                                            ParameterValue const &pv, Ast *call_expr) {
	Type *t  = x64_typed(ptype);
	Type *bt = t ? base_type(t) : nullptr;
	switch (pv.kind) {
	case ParameterValue_Constant: {
		ExactValue ev = pv.value;
		switch (ev.kind) {
		case ExactValue_Bool:
			return x64v_imm(t, ev.value_bool ? 1 : 0);
		case ExactValue_Integer:
			return x64v_imm(t, x64_endian_fix_const_int(ptype, big_int_to_i64(&ev.value_integer)));
		case ExactValue_Float:
			x64_emit_float_const(p, bt, ev.value_float, (f32)ev.value_float);
			return x64v_xmm(t, X64XmmReg_XMM0);
		case ExactValue_String:
			if (is_type_cstring(bt))   return x64_const_cstring_ptr(p, ev.value_string, false, t);
			if (is_type_cstring16(bt)) return x64_const_cstring_ptr(p, ev.value_string, true,  t);
			return x64_const_string(p, ev.value_string);
		case ExactValue_Procedure: {
			Entity *pe = entity_from_expr(ev.value_procedure);
			if (pe != nullptr && pe->kind == Entity_Procedure) {
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(pe));
				return x64v_reg(t, X64Reg_RAX);
			}
			return x64v_imm(t, 0);
		}
		case ExactValue_Compound:
			// Aggregate default value (`vp: [2]Size = {…}`): materialise the stored CompoundLit
			// (mirrors lb_const_value(ExactValue_Compound)). The scalar cases above + this share
			// the build_expr constant path. Was zero-filled by the default below → an omitted
			// aggregate default arrived all-zero (THE blick timeline viewport_size={fill,fill}
			// default → {children,children} → scrollview viewport collapsed to 0).
			if (ev.value_compound != nullptr) return x64_build_compound_lit(p, ev.value_compound, t);
			break; // null compound → zero-fill (handled by caller)
		default:
			return x64v_imm(t, 0);
		}
	}
	case ParameterValue_Value:
		return x64_build_expr(p, pv.ast_value);
	case ParameterValue_Nil:
		break; // zero — handled by caller (imm 0 / zero-fill)
	case ParameterValue_Location: {
		// `loc := #caller_location`: build the CALL SITE's Source_Code_Location (caller proc +
		// call expr position). Was zero-filled before → identical locations → e.g. UI hashes
		// built from `loc` all collided.
		String proc_name = (p->entity != nullptr) ? p->entity->token.string : str_lit("");
		TokenPos pos = {};
		if (call_expr != nullptr) {
			ast_node(ce, CallExpr, call_expr);
			pos = ast_token(ce->proc).pos;
		}
		return x64_build_source_code_location(p, proc_name, pos);
	}
	case ParameterValue_Expression: {
		// `expr := #caller_expression`: the source text of the call (best-effort; the
		// per-target-arg form falls back to the whole call string).
		if (call_expr != nullptr) {
			gbString s = expr_to_string(call_expr, temporary_allocator());
			return x64_const_string(p, make_string_c(s));
		}
		break;
	}
	default:
		if (pv.original_ast_expr != nullptr) {
			x64Value v = x64_build_expr(p, pv.original_ast_expr);
			if (v.kind != x64Value_None) return v;
		}
		break;
	}
	return x64v_none();
}

gb_internal x64Value x64_spill_value(x64Procedure *p, x64Value v, Type *t);

// `#subtype`/`using`-field subtype conversion: narrow a struct value `v` to one of its
// subtype field types when that field type is expected (e.g. `Temp_Allocator{using
// allocator}` passed as `Allocator`, or `^Chan` passed as `^Raw_Chan`). Mirrors
// lb_emit_conv's check_is_assignable_to_using_subtype path (value + pointer sources).
gb_internal x64Value x64_apply_using_subtype(x64Procedure *p, x64Value v, Type *want) {
	if (v.type == nullptr || want == nullptr) return v;
	if (are_types_identical(v.type, want)) return v;
	if (check_is_assignable_to_using_subtype(v.type, want) == 0) return v;
	Selection sel = {};
	sel.index.allocator = temporary_allocator();
	if (!lookup_subtype_polymorphic_selection(want, v.type, &sel)) return v;
	if (sel.entity == nullptr || sel.index.count == 0) return v;

	// `#subtype`/`using`-field narrowing. Two source forms (mirror lb_emit_conv):
	//   value source (Allocator{using ...}): offset into the value's memory;
	//   pointer source (^Chan → ^Raw_Chan via `#subtype impl: ^Raw_Chan`): walk the
	//   field path on the POINTED-TO struct, then LOAD the field (the old code bailed
	//   here, passing &struct unchanged → reading the field as the struct → crash).
	bool src_is_ptr = is_type_pointer(v.type);
	i64 off = 0;
	Type *cur = base_type(src_is_ptr ? type_deref(v.type) : v.type);
	Type *last_field_type = nullptr;
	for_array(i, sel.index) {
		if (cur == nullptr || cur->kind != Type_Struct) return v; // raw_union path: unsupported here
		type_set_offsets(cur);
		i32 idx = sel.index[i];
		if (idx < 0 || idx >= (i32)cur->Struct.fields.count) return v;
		off += cur->Struct.offsets[idx];
		last_field_type = cur->Struct.fields[idx]->type;
		cur = base_type(last_field_type);
	}
	if (src_is_ptr) {
		// Two pointer-source forms (mirror lb_emit_conv):
		//   (a) `using base: Inner` — base is an EMBEDDED VALUE subobject; `^Outer → ^Inner` is the
		//       ADDRESS of that subobject (ptr + off), NO load. (want == ^last_field_type)
		//   (b) `#subtype impl: ^Inner` — the field IS the wanted pointer; LOAD the field value.
		// Was: always (b) → for case (a) it dereferenced the pointer and read the subobject's first
		// bytes as the pointer (blick gfx_init_null_procedures got *gfx not gfx → AV at startup).
		x64_value_to_reg(p, v, X64Reg_RAX);
		bool want_addr_of_field = is_type_pointer(want) && last_field_type != nullptr &&
			are_types_identical(base_type(type_deref(want)), base_type(last_field_type));
		if (want_addr_of_field) {
			if (off != 0) x64_emit_lea(&p->asm_, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)off));
			return x64_spill_value(p, x64v_reg(want, X64Reg_RAX), want);
		}
		// field is at [addr + off]; spill its value to a fresh stack slot (clobber-safe).
		return x64_spill_value(p, x64v_mem(want, x64_mem(X64Reg_RAX, (i32)off)), want);
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
	// Replace a with b when: min→a>b, max→a<b. Load/extend each operand to a full register
	// (sub-8-byte slots hold only `sz` bytes — a 64-bit load/store would read/CLOBBER neighbours;
	// the store especially corrupted the adjacent stack, e.g. the reduce fold's vector base → crash).
	X64Cc cc = is_min ? (sgn ? X64Cc_G : X64Cc_A) : (sgn ? X64Cc_L : X64Cc_B);
	x64_value_to_reg(p, x64v_mem(rt, x64_rbp_mem(a_off)), X64Reg_RAX);
	x64_value_to_reg(p, x64v_mem(rt, x64_rbp_mem(b_off)), X64Reg_RCX);
	x64_emit_cmp_rr(&p->asm_, osz, X64Reg_RAX, X64Reg_RCX);
	x64_emit_cmov_rr(&p->asm_, cc, osz, X64Reg_RAX, X64Reg_RCX);
	x64_emit_mov_mr(&p->asm_, osz, x64_rbp_mem(r_off), X64Reg_RAX);
	return x64v_mem(rt, x64_rbp_mem(r_off));
}

// Two-operand min / max / clamp as values (mirror lb_emit_min / lb_emit_max / lb_emit_clamp). Thin
// over x64_emit_minmax2; clamp(x,lo,hi) == min(max(x,lo),hi).
gb_internal x64Value x64_emit_min(x64Procedure *p, Type *t, x64Value x, x64Value y) {
	return x64_emit_minmax2(p, t, x, y, /*is_min*/true);
}
gb_internal x64Value x64_emit_max(x64Procedure *p, Type *t, x64Value x, x64Value y) {
	return x64_emit_minmax2(p, t, x, y, /*is_min*/false);
}
gb_internal x64Value x64_emit_clamp(x64Procedure *p, Type *t, x64Value x, x64Value lo, x64Value hi) {
	return x64_emit_min(p, t, x64_emit_max(p, t, x, lo), hi);
}

// cond ? x : y as a value (mirrors lb_emit_select). Used only for TRIVIAL ternary operands
// (lb_is_expr_trivial → scalar, side-effect-free), so x and y are both already-built scalars. GPR
// scalars use CMOV (branchless); floats fall back to a small branch (x86 has no XMM cmov). Operands
// must be STABLE (spilled) — the caller spills them first.
gb_internal x64Value x64_emit_select(x64Procedure *p, x64Value cond, x64Value x, x64Value y) {
	Type *t = x.type ? x.type : (y.type ? y.type : t_int);
	if (x64_is_float(t)) {
		i64 sz = type_size_of(t); if (sz <= 0) sz = 8;
		i64 al = type_align_of(t); if (al <= 0) al = 8;
		i32 off = x64_alloc_local(p, sz, al);
		x64_store_value(p, x64addr(x64_rbp_mem(off), t), x);  // default = x
		x64_value_to_reg(p, cond, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		isize keep = x64_label_alloc(&p->asm_);
		x64_emit_jcc(&p->asm_, X64Cc_NE, keep);               // cond true → keep x
		x64_store_value(p, x64addr(x64_rbp_mem(off), t), y);  // else y
		x64_label_bind(&p->asm_, keep);
		return x64v_mem(t, x64_rbp_mem(off));
	}
	x64_value_to_reg(p, y, X64Reg_RCX);                       // RCX = y (false value)
	x64_value_to_reg(p, x, X64Reg_RAX);                       // RAX = x (true value)
	x64_value_to_reg(p, cond, X64Reg_RDX);
	x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RDX, X64Reg_RDX);
	x64_emit_cmov_rr(&p->asm_, X64Cc_E, x64_op_size_of(t), X64Reg_RAX, X64Reg_RCX); // cond==0 → y
	return x64v_reg(t, X64Reg_RAX);
}

// ptr + index*size_of(elem) as a pointer value (mirrors lb_emit_ptr_offset); esz from ptr's element
// type. Caller must pass STABLE (spilled) operands — x64 has no SSA, so a register-held ptr would be
// clobbered while building index. (The IndexExpr path uses the hand-optimised x64_index_elem_addr.)
gb_internal x64Value x64_emit_ptr_offset(x64Procedure *p, x64Value ptr, x64Value index) {
	Type *pt  = ptr.type ? ptr.type : t_rawptr;
	Type *pbt = base_type(pt);
	Type *elem = (pbt && pbt->kind == Type_MultiPointer) ? pbt->MultiPointer.elem :
	             (pbt && pbt->kind == Type_Pointer)       ? pbt->Pointer.elem : nullptr;
	i64 esz = elem ? type_size_of(elem) : 1; if (esz <= 0) esz = 1;
	x64_value_to_reg(p, index, X64Reg_RCX);
	if (esz > 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX, (i32)esz);
	x64_value_to_reg(p, ptr, X64Reg_RAX);
	x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
	return x64v_reg(pt, X64Reg_RAX);
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

// Byte offset of the `len` field in a fixed-capacity dynamic array `[dynamic; N]E`. Layout is
// {data: [N]E @0, len: int} (runtime.Raw_Fixed_Capacity_Dynamic_Array): len sits right AFTER the
// data array, at align_up(N*size_of(E), align_of(int)). NOT type_size-8 — when E is over-aligned the
// struct gets TRAILING padding (size aligned up to align_of(E)), so type_size-8 points into that
// padding, not len. THE blick `media.streams` bug: Media_Stream is 64-byte aligned → 56 bytes of
// padding after len → `append` wrote len at the real field while `len()` read the padding (always 0).
gb_internal i64 x64_fca_len_offset(Type *fca) {
	Type *bt = base_type(fca);
	i64 esz = type_size_of(bt->FixedCapacityDynamicArray.elem);
	i64 cap = bt->FixedCapacityDynamicArray.capacity;
	i64 off = esz * cap;
	i64 ial = type_align_of(t_int); if (ial <= 0) ial = 8;
	return (off + ial - 1) & ~(ial - 1);
}

// Materialise a `CompoundLit` AST into a zeroed stack slot. Shared by the CompoundLit
// expr case and the constant path (an aggregate constant is ExactValue_Compound over the
// same AST) — mirrors lb_const_value / lb_build_addr_compound_lit. Covers every kind:
// struct, [N]T/[E]T/matrix/#simd arrays, slice, bit_set, bit_field, any, map, [dynamic]T,
// and [dynamic;N]T (fixed-capacity).
// Store one struct compound-literal field value `fval` (AST type `fet`) into `dst` (field type
// `ftype`). Mirrors lb_build_struct_compound_lit_field_assignment: a variant boxed into a union
// FIELD needs its discriminant tag written, and an EMPTY-struct variant builds to a typeless None
// (x64_build_expr carries no type for a zero-sized value) — so the variant type must come from the
// AST (`fet`), not fval.type. Was THE blick shutdown hang: `Message_Quit` is `struct{}`, boxed via
// `Message{variant = variant}` its tag stayed 0 → the app pump never matched Message_Quit → the
// quit never propagated → thread.join hung forever.
gb_internal void x64_store_compound_field(x64Procedure *p, X64Mem dst, Type *ftype, x64Value fval, Type *fet) {
	if (fval.type == nullptr) fval.type = fet;
	Type *ftb = (ftype != nullptr) ? base_type(ftype) : nullptr;
	if (ftb != nullptr && ftb->kind == Type_Union && fet != nullptr &&
	    !are_types_identical(fet, ftype) && !is_type_untyped(fet) && union_is_variant_of(ftb, fet)) {
		x64_store_union_variant(p, dst, fval, ftype);
		return;
	}
	x64_store_value(p, x64addr(dst, ftype), fval);
}

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

	if (bt->kind == Type_Map) {
		// map literal: reserve then set each entry (mirrors lb_build_addr_compound_lit Type_Map).
		i32 mp = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, res_m);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(mp), X64Reg_RAX);
		x64Value map_ptr = x64v_mem(alloc_type_pointer(bt), x64_rbp_mem(mp));
		x64_dynamic_map_reserve(p, map_ptr, bt, 2*cl->elems.count);
		for_array(ei, cl->elems) {
			Ast *elem = cl->elems[ei];
			if (elem == nullptr || elem->kind != Ast_FieldValue) continue;
			AstFieldValue *fv = &elem->FieldValue;
			x64Value v = x64_build_expr(p, fv->value);
			v = x64_spill_value(p, v, v.type ? v.type : bt->Map.value);
			x64_internal_dynamic_map_set(p, map_ptr, bt, fv->field, v, elem);
		}
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
				Type *ftype = nullptr;
				for_array(fi, bt->Struct.fields) {
					Entity *fe = bt->Struct.fields[fi];
					if (fe->token.string == fname) {
						foff  = bt->Struct.offsets[fi];
						ftype = fe->type;
						break;
					}
				}
				if (ftype == nullptr) {
						// Field PROMOTED via `using` (e.g. a `using _: struct #raw_union` member like
						// d3d11 view descs' Texture2D/Buffer): the direct-field loop misses it. Resolve via
						// the shared Selection walk. allow_deref=false: a literal can't write THROUGH a
						// `using p: ^T` (the pointer isn't set yet) → such a field is skipped, not clobbered
						// at offset 0. Was: foff stayed 0 → clobbered field 0 → garbage view descs.
						x64Addr dg = x64_emit_deep_field_gep(p, x64_rbp_mem(res_off), bt,
						                                     fv->field->Ident.interned, /*allow_deref*/false);
						if (dg.type != nullptr) { foff = dg.mem.disp - res_off; ftype = dg.type; }
					}
					if (ftype == nullptr) continue; // unresolved promoted field: skip (don't clobber @0)
					x64Value fval = x64_build_expr(p, fv->value);
				x64_store_compound_field(p, x64_mem(X64Reg_RBP, res_off + (i32)foff), ftype, fval, fv->value->tav.type);
			}
		} else {
			isize n = gb_min((isize)cl->elems.count, (isize)bt->Struct.fields.count);
			for (isize ei = 0; ei < n; ei++) {
				if (cl->elems[ei] == nullptr) continue;
				Entity *fe   = bt->Struct.fields[ei];
				i64    foff  = bt->Struct.offsets[ei];
				x64Value fval = x64_build_expr(p, cl->elems[ei]);
				x64_store_compound_field(p, x64_mem(X64Reg_RBP, res_off + (i32)foff), fe->type, fval, cl->elems[ei]->tav.type);
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
		// The slice header points INTO this backing (.data = &arr), so it must outlive the
		// statement — mark scope-lived (named_seq) so the temp reclaimer never reuses it.
		p->named_seq++;
		x64_zero_mem(p, x64_rbp_mem(arr_off), n * esz);
		x64_store_compound_elems(p, cl->elems, bt, et, esz, arr_off, 0);
		if (p->is_startup) {
			// GLOBAL initializer (runs in __entry_point): a stack backing DANGLES once the function
			// returns → the global slice's .data pointed at freed stack → garbage. Copy the built
			// backing into a STATIC global and point .data there (mirrors the &CompoundLit is_startup
			// path). An all-const slice takes the x64_const_value static path; this covers slices with
			// runtime elements, e.g. `g := []f64{ math.inf_f64(-1), … }` (core:math test _sc tables).
			Type  *arrt = alloc_type_array(et, n);
			String sym  = x64_add_global_generated(p->module, arrt);
			x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
			x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(arr_off), n * esz);
			x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX); // .data = &static
		} else {
			x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(arr_off));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX); // .data = &stack
		}
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, n);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off + 8), X64Reg_RAX); // .len
	} else if (bt->kind == Type_FixedCapacityDynamicArray) {
		// `[dynamic;N]T{a, b, …}`: elements live INLINE in the data array (field 0 @0); set len
		// (field 1, at size-8). cap is implicit in the type. Mirrors LLVM lb_build_addr_compound_lit
		// + the FCA index layout (len at type_size-8). Was left zeroed → len 0 → THE blick keyboard
		// bug: `register_action(…, {{{.ctrl_or_cmd}, .O}})` keybind lists came back empty
		// (len(info.keybinds)==0 skipped every shortcut → ctrl+O etc. did nothing).
		Type *et  = bt->FixedCapacityDynamicArray.elem;
		i64   esz = type_size_of(et); if (esz <= 0) esz = 1;
		i64   cap = bt->FixedCapacityDynamicArray.capacity;
		i64   n   = gb_max((i64)cl->elems.count, (i64)cl->max_count);
		if (n > cap) n = cap;
		x64_store_compound_elems(p, cl->elems, bt, et, esz, res_off, 0);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, n);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off + (i32)x64_fca_len_offset(bt)), X64Reg_RAX); // .len
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
	else if (bt->kind == Type_DynamicArray && cl->elems.count > 0) {
		// [dynamic]T{...}: reserve capacity, build a local items array from the literal, append
		// (mirrors lb_build_addr CompoundLit Type_DynamicArray). res is the zeroed Raw_Dynamic_Array.
		Type *et  = bt->DynamicArray.elem;
		i64   esz = type_size_of(et);  if (esz <= 0) esz = 1;
		i64   eal = type_align_of(et); if (eal <= 0) eal = 1;
		i64   n   = gb_max((i64)cl->elems.count, (i64)cl->max_count);

		i32 dptr = x64_alloc_local(p, 8, 8);               // pin &dynarray (calls clobber regs)
		x64_emit_lea(&p->asm_, X64Reg_RAX, res_m);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(dptr), X64Reg_RAX);

		{ // __dynamic_array_reserve(&dyn, esz, eal, n, loc)
			x64Value a[5];
			a[0] = x64v_mem(t_rawptr, x64_rbp_mem(dptr));
			a[1] = x64v_imm(t_int, esz);
			a[2] = x64v_imm(t_int, eal);
			a[3] = x64v_imm(t_int, n);
			a[4] = x64_map_null_loc(p);
			x64_emit_runtime_call(p, str_lit("__dynamic_array_reserve"), a, 5);
		}

		i32 items_off = x64_alloc_local(p, n * esz, eal);  // [n]et populated from the literal
		x64_zero_mem(p, x64_rbp_mem(items_off), n * esz);
		x64_store_compound_elems(p, cl->elems, bt, et, esz, items_off, 0);

		{ // __dynamic_array_append(&dyn, esz, eal, &items, n, loc)
			i32 iptr = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(items_off));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(iptr), X64Reg_RAX);
			x64Value a[6];
			a[0] = x64v_mem(t_rawptr, x64_rbp_mem(dptr));
			a[1] = x64v_imm(t_int, esz);
			a[2] = x64v_imm(t_int, eal);
			a[3] = x64v_mem(t_rawptr, x64_rbp_mem(iptr));
			a[4] = x64v_imm(t_int, n);
			a[5] = x64_map_null_loc(p);
			x64_emit_runtime_call(p, str_lit("__dynamic_array_append"), a, 6);
		}
	}
	return x64v_mem(type, res_m);
}

// ─────────────────────────────────────────────────────────────────────────────
// bit_field member access (read/write). A bit_field member has no addressable byte
// location, so SelectorExpr-read and field-assignment can't go through x64_build_addr;
// they shift/mask the (single-integer ≤8B) backing in-register. Mirrors lb_addr_bit_field
// load (shift+mask+sign-extend) and store (read-modify-write). Array-of-integer backing
// is a rare TODO. Was the rollback_stack allocator bug: its header is a `bit_field u64`,
// and field writes clobbered the whole backing (no shift/mask) → map grow corrupted →
// `odin test` (tracking allocator uses a map) crashes.
gb_internal bool x64_bit_field_member_info(Ast *expr, Type **field_type, Type **backing_type,
                                           i64 *bit_offset, i64 *bit_size, i64 *byte_offset) {
	Ast *e = unparen_expr(expr);
	if (e == nullptr || e->kind != Ast_SelectorExpr) return false;
	AstSelectorExpr *se = &e->SelectorExpr;
	if (se->selector == nullptr || se->selector->kind != Ast_Ident) return false;
	String fname = se->selector->Ident.token.string;
	Type *outer = se->expr->tav.type;
	Type *obt = base_type(is_type_pointer(outer) ? type_deref(outer) : outer);
	if (obt == nullptr) return false;

	// The bit_field is either `outer` itself, OR a `using`-promoted member of a struct/raw_union
	// (e.g. `struct #raw_union { using _: bit_field u32 {…}, value: u32 }`, possibly through distinct
	// types). Find the bit_field type + its byte offset within `outer`. Was: only the direct case was
	// handled → a promoted member fell back to a full-width field access aliasing the backing (no
	// shift/mask) → garbage (THE blick slot_map handle.index write returned 0).
	Type *bt = nullptr;
	i64 bf_byte_off = 0;
	if (obt->kind == Type_BitField) {
		bt = obt;
	} else if (obt->kind == Type_Struct) {
		type_set_offsets(obt);
		for_array(i, obt->Struct.fields) {
			Entity *f = obt->Struct.fields[i];
			if ((f->flags & EntityFlag_Using) == 0) continue; // only `using`-promoted fields
			Type *fbt = base_type(f->type);
			if (fbt == nullptr || fbt->kind != Type_BitField) continue;
			for_array(fi, fbt->BitField.fields) {
				if (fbt->BitField.fields[fi]->token.string == fname) { bt = fbt; bf_byte_off = obt->Struct.offsets[i]; break; }
			}
			if (bt != nullptr) break;
		}
	}
	if (bt == nullptr) return false;

	Type *bk = core_type(bt->BitField.backing_type);
	if (!is_type_integer(bk) || type_size_of(bt) > 8) return false; // single-int backing only
	for_array(fi, bt->BitField.fields) {
		if (bt->BitField.fields[fi]->token.string == fname) {
			*field_type   = bt->BitField.fields[fi]->type;
			*backing_type = bk;
			*bit_offset   = bt->BitField.bit_offsets[fi];
			*bit_size     = (i64)bt->BitField.bit_sizes[fi];
			*byte_offset  = bf_byte_off;
			return true;
		}
	}
	return false;
}

// Load &backing into `dst`. `base.field`: lea of base's addr; `ptr.field` (ptr→bit_field): the ptr value.
// `byte_offset` is the bit_field's offset within `base` (non-zero for a `using`-promoted member).
gb_internal void x64_bit_field_backing_to_reg(x64Procedure *p, Ast *expr, X64Reg dst, i64 byte_offset) {
	AstSelectorExpr *se = &unparen_expr(expr)->SelectorExpr;
	if (is_type_pointer(se->expr->tav.type)) {
		x64_value_to_reg(p, x64_build_expr(p, se->expr), dst);
	} else {
		x64_emit_lea(&p->asm_, dst, x64_build_addr(p, se->expr).mem);
	}
	if (byte_offset != 0) x64_emit_add_ri(&p->asm_, X64OpSize_64, dst, (i32)byte_offset);
}

gb_internal x64Value x64_bit_field_load(x64Procedure *p, Ast *expr, Type *field_type,
                                        Type *backing_type, i64 bit_offset, i64 bit_size, i64 byte_offset) {
	x64_bit_field_backing_to_reg(p, expr, X64Reg_RCX, byte_offset);                      // RCX = &backing
	x64_value_to_reg(p, x64v_mem(backing_type, x64_mem(X64Reg_RCX, 0)), X64Reg_RAX);     // RAX = backing
	if (bit_offset != 0) x64_emit_shr_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (u8)bit_offset);
	bool is_signed = is_type_integer(field_type) && !is_type_unsigned(field_type);
	if (bit_size < 64) {
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)((1ull << (u64)bit_size) - 1));
		x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);                 // RAX &= mask
		if (is_signed) {
			// sign-extend the bit_size-bit value: (x XOR m) - m, m = 1<<(bit_size-1)
			x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)(1ull << (u64)(bit_size - 1)));
			x64_emit_xor_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
			x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		}
	}
	return x64v_reg(field_type, X64Reg_RAX);
}

gb_internal void x64_bit_field_store(x64Procedure *p, Ast *expr, Ast *rhs_expr, Type *field_type,
                                     Type *backing_type, i64 bit_offset, i64 bit_size, i64 byte_offset) {
	X64OpSize bsz = x64_op_size_of(backing_type);
	// Build RHS + spill (clobber-safe: backing-addr building below uses regs).
	x64Value rvm = x64_spill_value(p, x64_build_expr(p, rhs_expr), field_type);
	x64_bit_field_backing_to_reg(p, expr, X64Reg_RAX, byte_offset);                      // RAX = &backing
	i32 addr_off = x64_alloc_local(p, 8, 8);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(addr_off), X64Reg_RAX);
	x64_value_to_reg(p, rvm, X64Reg_RDX);                                                // RDX = value
	u64 mask = (bit_size < 64) ? ((1ull << (u64)bit_size) - 1) : ~0ull;
	if (bit_size < 64) {
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, (i64)mask);
		x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RCX);                 // RDX = v & mask
	}
	if (bit_offset != 0) x64_emit_shl_ri(&p->asm_, X64OpSize_64, X64Reg_RDX, (u8)bit_offset); // << off
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(addr_off));          // RCX = &backing
	x64_value_to_reg(p, x64v_mem(backing_type, x64_mem(X64Reg_RCX, 0)), X64Reg_RAX);     // RAX = backing
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_R8, (i64)(~(mask << (u64)bit_offset)));
	x64_emit_and_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_R8);                      // clear field bits
	x64_emit_or_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);                      // OR in new bits
	x64_emit_mov_mr(&p->asm_, bsz, x64_mem(X64Reg_RCX, 0), X64Reg_RAX);                  // store backing
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

// Address of a `using`-promoted field (mirrors lb_emit_deep_field_gep): walk the lookup_field
// Selection from `st` (base at `base_mem`), summing each level's struct offset. An intermediate
// `using p: ^T` is dereferenced when `allow_deref` (an lvalue READ); for literal CONSTRUCTION it must
// NOT be (the pointer field isn't initialized yet) → bail. Returns the field addr (type set) on
// success; {type==nullptr} when the name isn't a promoted field or a needed deref was disallowed —
// the caller then falls back / skips. Shared by x64_build_addr (SelectorExpr) and
// x64_build_compound_lit (promoted member of a `using _: struct` field).
gb_internal x64Addr x64_emit_deep_field_gep(x64Procedure *p, X64Mem base_mem, Type *st,
                                            InternedString interned, bool allow_deref) {
	Selection s = lookup_field(st, interned, false);
	if (s.entity == nullptr || s.index.count == 0) return x64addr(base_mem, nullptr);
	X64Mem m   = base_mem;
	Type  *cur = st;
	for_array(i, s.index) {
		Type *cb = base_type(cur);
		if (cb == nullptr || cb->kind != Type_Struct) return x64addr(base_mem, nullptr);
		type_set_offsets(cb);
		i32 fi = s.index[i];
		if (fi < 0 || fi >= (i32)cb->Struct.fields.count) return x64addr(base_mem, nullptr);
		Entity *fe = cb->Struct.fields[fi];
		m.disp += (i32)cb->Struct.offsets[fi];
		cur = fe->type;
		if (i + 1 < s.index.count && is_type_pointer(base_type(cur))) {
			if (!allow_deref) return x64addr(base_mem, nullptr); // can't set through a ptr in a literal
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, m);
			m   = x64_mem(X64Reg_RAX, 0);
			cur = type_deref(cur);
		}
	}
	return x64addr(m, x64_typed(s.entity->type));
}

// Address of field `sel->selector` within aggregate `base` (mirrors lb_emit_struct_ep + the
// pseudo-field resolution): struct direct/`using`-promoted field, slice/string/dynamic-array
// (data/len/cap/allocator), map (data/len/allocator), fixed-cap (data/len), `any` (data/id),
// quaternion (.x/.y/.z/.w) and single-component array swizzle (.x → &v[i]). `result_type` is the
// selector's resolved type (the field's declared type wins when found). Returns the field addr.
gb_internal x64Addr x64_emit_struct_ep(x64Procedure *p, x64Addr base, AstSelectorExpr *sel, Type *result_type) {
		Entity *field = entity_of_node(sel->selector);

		Type *st       = base_type(base.type);
		i64   field_off = 0;
		Type *field_type = result_type;

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
			// field falls through with field_off=0 and aliases offset 0. Resolve via the Selection
			// walk (x64_emit_deep_field_gep). Was the thread bug: Thread.win32_thread_id (promoted,
			// off 8) resolved to 0 and overwrote the handle.
			if (!found && sel->selector->kind == Ast_Ident) {
				x64Addr dg = x64_emit_deep_field_gep(p, base.mem, st, sel->selector->Ident.interned, /*allow_deref*/true);
				if (dg.type != nullptr) return dg;
			}
		} else if (st->kind == Type_Slice || st->kind == Type_DynamicArray || is_type_string(st)) {
			// Built-in pseudo-field on slice / string / dynamic array (data@0, len@8, cap@16,
			// allocator@24). The checker may resolve the selector to the underlying Raw_* field
			// entity (field != nullptr), but `st` is the aggregate type, not its Raw_ struct, so
			// the struct-offset path can't run — apply the offset BY NAME regardless of `field`.
			// (Gating on field==nullptr read .allocator at 0, the data ptr: _make stored it via
			// the ^Raw_Dynamic_Array struct field @24, every read here saw @0 → make/delete crash.)
			String fname = sel->selector->Ident.token.string;
			if      (fname == "data")      { field_off =  0; }
			else if (fname == "len")       { field_off =  8; }
			else if (fname == "cap")       { field_off = 16; }
			else if (fname == "allocator") { field_off = 24; }
			if (field != nullptr) field_type = field->type;
		} else if (st->kind == Type_Map) {
			// Raw_Map: data@0, len@8, allocator@16 (no cap field) — allocator sits earlier than in
			// a dynamic array. Same field!=nullptr gap as the aggregate bases above.
			String fname = sel->selector->Ident.token.string;
			if      (fname == "data")      { field_off =  0; }
			else if (fname == "len")       { field_off =  8; }
			else if (fname == "allocator") { field_off = 16; }
			if (field != nullptr) field_type = field->type;
		} else if (st->kind == Type_FixedCapacityDynamicArray) {
			// Inline layout {data: [N]E @0, len: int after the data array}; cap is the compile-time N.
			String fname = sel->selector->Ident.token.string;
			if      (fname == "data") { field_off = 0; }
			else if (fname == "len")  { field_off = x64_fca_len_offset(st); }
			if (field != nullptr) field_type = field->type;
		} else if (is_type_quaternion(st)) {
			// quaternion components via .x/.y/.z/.w selectors (real/imag/jmag/kmag are builtins, not
			// fields). Layout {imag@0, jmag, kmag, real} (esz=size/4) → x/y/z/w == imag/jmag/kmag/real.
			// Without this .y/.z/.w resolved to offset 0 (same non-struct field gap as above).
			i64 esz = type_size_of(st) / 4;
			String fname = sel->selector->Ident.token.string;
			if      (fname == "y") field_off = 1 * esz;
			else if (fname == "z") field_off = 2 * esz;
			else if (fname == "w") field_off = 3 * esz;
		} else if (field != nullptr) {
			field_type = field->type;
		}

		X64Mem m = base.mem;
		m.disp += (i32)field_off;
		return x64addr(m, field_type ? field_type : t_int);
}

// Address of an already-resolved entity reference (mirrors lb_build_addr_from_entity): local → its
// slot; file-scope global variable → LEA [RIP+sym]; non-addressable kinds (import/library/type) and a
// variable that is neither local nor file-scope (a struct field reached via SelectorExpr) → a dummy
// local; anything else with no storage symbol → materialise into a local. `e` may be nullptr.
// Address of a `using`-promoted field referenced as a BARE identifier (mirrors lb_get_using_variable):
// e.g. `using params` where params: ^Filter_Params, then a bare `channels`/`dest` means params^.channels.
// The field entity has no storage of its own — compute it every time from the using_parent's address +
// the field offset. Handles a pointer parent (`using p: ^T`) by dereferencing. Returns {type==nullptr}
// when it can't (unknown parent) so the caller falls through. Was core:image/png emitting bare `width`/
// `channels`/`dest`/`depth` as unresolved external symbols (fell to the global LEA [RIP+name] path).
gb_internal x64Addr x64_get_using_variable(x64Procedure *p, Entity *e) {
	Entity *parent = e->using_parent;
	if (parent == nullptr) return x64addr(X64Mem{}, nullptr);
	X64Mem base = {};
	if (x64_entity_is_local(p, parent)) {
		base = x64_entity_addr(p, parent).mem;
	} else if (e->using_expr != nullptr) {
		base = x64_build_addr(p, e->using_expr).mem; // address of the using_expr lvalue
	} else {
		return x64addr(X64Mem{}, nullptr);
	}
	Type *pt = parent->type;
	if (is_type_pointer(pt)) { // `using p: ^T` → load the pointer, index off the pointee
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, base);
		base = x64_mem(X64Reg_RAX, 0);
		pt = type_deref(pt);
	}
	return x64_emit_deep_field_gep(p, base, base_type(pt), entity_interned_name(e), /*allow_deref*/true);
}

gb_internal x64Addr x64_build_addr_from_entity(x64Procedure *p, Entity *e, Ast *expr) {
	Type *t = expr->tav.type;
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
	// `using`-promoted field referenced by bare name → compute params^.field (mirrors LLVM's
	// lb_build_addr_from_entity EntityFlag_Using branch). Must precede the dummy/global fallbacks.
	if (e->kind == Entity_Variable && (e->flags & EntityFlag_Using) && !x64_entity_is_local(p, e)) {
		x64Addr ua = x64_get_using_variable(p, e);
		if (ua.type != nullptr) return ua;
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
	Type *ct = t ? x64_typed(t) : t_int;
	i64 sz = x64_type_size(ct); if (sz <= 0) sz = 8;
	i64 al = x64_type_align(ct); if (al <= 0) al = 8;
	i32 off = x64_alloc_local(p, sz, al);
	x64_zero_mem(p, x64_rbp_mem(off), sz);
	x64Value v = x64_build_expr(p, expr);
	if (v.kind != x64Value_None) x64_store_value(p, x64addr(x64_rbp_mem(off), ct), v);
	return x64addr(x64_rbp_mem(off), ct);
}

// `a[i]` lvalue address (mirrors lb_build_addr_index_expr): array/enum-array/FCA index off the inline
// data; slice/dynarray/multipointer off .data; string off the byte data. Auto-derefs an indexed pointer
// (p[i], p: ^[N]T/^[]T). The index is spilled BEFORE building the base — the base load reuses RAX/RCX
// and would clobber a register-held index.
gb_internal x64Addr x64_build_addr_index_expr(x64Procedure *p, Ast *expr) {
	ast_node(idx, IndexExpr, expr);
	Type *t     = expr->tav.type;
	Type *arr_t = base_type(idx->expr->tav.type);
	bool via_ptr = is_type_pointer(arr_t);
	if (via_ptr) arr_t = base_type(type_deref(idx->expr->tav.type));
	x64Value index_v = x64_build_expr(p, idx->index);
	Type *it = x64_typed(index_v.type ? index_v.type : t_int);
	x64Value idx_mem = x64_spill_value(p, index_v, it);
	i64 esz = type_size_of(t);

	if (arr_t->kind == Type_Array) {
		// Fixed array: const-len bounds check (skip constant indices — checked at compile time),
		// emitted BEFORE building the base into RAX (the runtime call would clobber it).
		if (idx->index->tav.mode != Addressing_Constant) {
			x64_emit_bounds_check(p, ast_token(idx->index).pos, idx_mem, x64v_imm(t_int, arr_t->Array.count));
		}
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
	// Runtime-length kinds (slice / dynarray / string / fixed-capacity): pin the aggregate's address,
	// read its length, bounds-check, then index. Length is at offset 8 (slice/dynarray/string) or
	// size-8 (fixed-capacity). FCA's data is the INLINE array @0; the others index off the .data ptr.
	if (arr_t->kind == Type_Slice || arr_t->kind == Type_DynamicArray ||
	    arr_t->kind == Type_FixedCapacityDynamicArray ||
	    (arr_t->kind == Type_Basic && (arr_t->Basic.flags & BasicFlag_String) && !is_type_cstring(arr_t))) {
		bool fca = arr_t->kind == Type_FixedCapacityDynamicArray;
		bool str = arr_t->kind == Type_Basic;
		i64  len_off = fca ? x64_fca_len_offset(arr_t) : 8;
		i64  use_esz = str ? 1 : esz; // strings index by byte (preserve prior behaviour)

		i32 base_slot = x64_alloc_local(p, 8, 8);
		if (via_ptr) x64_value_to_reg(p, x64_build_expr(p, idx->expr), X64Reg_RAX);      // RAX = &aggregate
		else         x64_emit_lea(&p->asm_, X64Reg_RAX, x64_build_addr(p, idx->expr).mem);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base_slot), X64Reg_RAX);
		i32 len_slot = x64_alloc_local(p, 8, 8);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)len_off)); // len
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_slot), X64Reg_RCX);
		x64_emit_bounds_check(p, ast_token(idx->index).pos, idx_mem, x64v_mem(t_int, x64_rbp_mem(len_slot)));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_slot));      // RAX = &aggregate
		if (!fca) x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 0)); // RAX = .data (FCA: inline @0)
		return x64_index_elem_addr(p, idx_mem, use_esz, /*sub*/0, t);
	}
	if (arr_t->kind == Type_MultiPointer ||
	    (arr_t->kind == Type_Basic && (arr_t->Basic.flags & BasicFlag_String))) {
		// MultiPointer: the pointer IS .data. cstring: a bare pointer (no length) → no bounds check.
		x64_index_base_to_rax(p, idx->expr, via_ptr, /*data_ptr*/true);
		i64 use_esz = (arr_t->kind == Type_Basic) ? 1 : esz;
		return x64_index_elem_addr(p, idx_mem, use_esz, /*sub*/0, t);
	}
	if (arr_t->kind == Type_Map) {
		// A map index used as an lvalue BASE — the inner `m[k1]` of a chained `m[k1][k2]` read, or
		// `m[k1].field`. Materialize the map VALUE (zero/nil map if the key is absent, which
		// __dynamic_map_get handles safely) into a temp and return its address; the outer op then
		// operates on that addressable copy. Mirrors LLVM's lb_addr_load(lbAddr_Map) for the inner
		// index. (Using &m[k1]'s element pointer would be nil for an absent key → deref crash.) Was:
		// this fell to the garbage-temp fallback below → `m["LOG"]["level"]` read a bogus inner map →
		// segfault (core:encoding/ini parse_ini). Chained map STORE still prefers the &m[k] path.
		x64Value v = x64_build_map_index_load(p, idx->expr, idx->index, t);
		if (v.kind != x64Value_Mem) v = x64_spill_value(p, v, t);
		return x64addr(v.mem, t);
	}
	// Fallback: materialise to a temp
	i64 fsz = t ? x64_type_size(t) : 8;
	i64 fal = t ? x64_type_align(t) : 8;
	i32 off = x64_alloc_local(p, fsz, fal);
	return x64addr(x64_rbp_mem(off), t ? t : t_u8);
}

// Address of `soa[idx].field` — a SelectorExpr whose base is an IndexExpr on a #soa value. x64 has no
// Addressing_SoaVariable machinery (lb_addr_soa_variable); compute the element address directly. The #soa
// struct is { f0:[^]F0, f1:[^]F1, …, __$len:int } for slice/dynamic (each field a pointer), or
// { f0:[N]F0, … } inline for fixed. `soa[idx].fk` = (slice/dyn: load the [^]Fk at the field's offset;
// fixed: soa_base + field_offset) + idx*size(Fk). Was UNIMPLEMENTED → read as an AoS slice → garbage/AV
// (reflect.struct_fields_zipped returns #soa[]Struct_Field → core:encoding/json crash → config_load fail).
gb_internal x64Addr x64_build_soa_index_field(x64Procedure *p, Ast *ie_expr, Ast *sel_ast, Type *field_type) {
	ast_node(ie, IndexExpr, ie_expr);
	Type *soa_ptr_t = ie->expr->tav.type;
	bool  deref     = is_type_pointer(soa_ptr_t);
	Type *soa       = base_type(type_deref(soa_ptr_t));
	GB_ASSERT(soa != nullptr && soa->kind == Type_Struct && soa->Struct.soa_kind != StructSoa_None);

	String fname = unparen_expr(sel_ast->SelectorExpr.selector)->Ident.token.string;
	isize k = -1;
	for_array(i, soa->Struct.fields) {
		if (soa->Struct.fields[i]->token.string == fname) { k = i; break; }
	}
	GB_ASSERT_MSG(k >= 0, "soa field not found: %.*s", LIT(fname));
	i64 foff = soa->Struct.offsets[k];
	i64 esz  = type_size_of(field_type); if (esz <= 0) esz = 1;

	// soa base address → spill
	if (deref) { x64_value_to_reg(p, x64_build_expr(p, ie->expr), X64Reg_RAX); }
	else       { x64_emit_lea(&p->asm_, X64Reg_RAX, x64_build_addr(p, ie->expr).mem); }
	i32 sb = x64_alloc_local(p, 8, 8);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sb), X64Reg_RAX);

	// idx → spill
	x64_value_to_reg(p, x64_build_expr(p, ie->index), X64Reg_RCX);
	i32 ib = x64_alloc_local(p, 8, 8);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ib), X64Reg_RCX);

	// field-array base pointer
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(sb));
	if (soa->Struct.soa_kind == StructSoa_Fixed) {
		if (foff != 0) x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)foff); // inline array
	} else {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)foff)); // load [^] ptr
	}
	// addr = base + idx*esz
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(ib));
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RDX, esz);
	x64_emit_imul_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
	x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
	return x64addr(x64_mem(X64Reg_RAX, 0), field_type);
}

// Copy `csz` bytes between the pointer held in stack slot `ptr_slot` (points at the component's [idx]
// element) and the RBP-relative `elem_comp`. Unrolled 8/4/2/1-byte chunks (SOA components are small),
// clobber-safe: the component address stays in RAX, values pass through RDX. `store`: elem→[ptr].
gb_internal void x64_soa_copy_component(x64Procedure *p, i32 ptr_slot, X64Mem elem_comp, i64 csz, bool store) {
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ptr_slot)); // RAX = &component[idx]
	for (i64 o = 0; o < csz; ) {
		i64 chunk = (csz - o >= 8) ? 8 : (csz - o >= 4) ? 4 : (csz - o >= 2) ? 2 : 1;
		X64OpSize os = (chunk == 8) ? X64OpSize_64 : (chunk == 4) ? X64OpSize_32 : (chunk == 2) ? X64OpSize_16 : X64OpSize_8;
		X64Mem ec = elem_comp; ec.disp += (i32)o;
		X64Mem cm = x64_mem(X64Reg_RAX, (i32)o);
		if (store) { x64_emit_mov_rm(&p->asm_, os, X64Reg_RDX, ec); x64_emit_mov_mr(&p->asm_, os, cm, X64Reg_RDX); }
		else       { x64_emit_mov_rm(&p->asm_, os, X64Reg_RDX, cm); x64_emit_mov_mr(&p->asm_, os, ec, X64Reg_RDX); }
		o += chunk;
	}
}

// Whole-element access to a #soa array element (`soa[idx]` gather / `soa[idx] = v` scatter). x64 has no
// Addressing_SoaVariable, so the AoS element is gathered into / scattered from a stack buffer. Component
// k of the element lives in the k-th SOA field: a [^]Ck (slice/dynamic — load the ptr @field-offset) or
// an inline [N]Ck (fixed); its [idx] address = that base + idx*size(Ck). In the AoS element, component k
// is at sub-offset `coff` (array elem: k*size(elem); struct: field k's offset). Mirrors lb_addr_soa_variable.
gb_internal x64Value x64_soa_index_element(x64Procedure *p, Ast *ie_expr, Type *elem_type, bool store, x64Value src) {
	ast_node(ie, IndexExpr, ie_expr);
	Type *soa_ptr_t = ie->expr->tav.type;
	bool  deref     = is_type_pointer(soa_ptr_t);
	Type *soa       = base_type(type_deref(soa_ptr_t));
	Type *ebt       = base_type(elem_type);
	GB_ASSERT(soa != nullptr && soa->kind == Type_Struct && soa->Struct.soa_kind != StructSoa_None);
	type_set_offsets(soa);

	int ncomp = 0;
	if      (ebt->kind == Type_Struct) { ncomp = (int)ebt->Struct.fields.count; type_set_offsets(ebt); }
	else if (ebt->kind == Type_Array)  ncomp = (int)ebt->Array.count;

	// soa base ptr + idx → slots (calls below clobber regs)
	if (deref) x64_value_to_reg(p, x64_build_expr(p, ie->expr), X64Reg_RAX);
	else       x64_emit_lea(&p->asm_, X64Reg_RAX, x64_build_addr(p, ie->expr).mem);
	i32 sb = x64_alloc_local(p, 8, 8);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sb), X64Reg_RAX);
	x64_value_to_reg(p, x64_build_expr(p, ie->index), X64Reg_RCX);
	i32 ib = x64_alloc_local(p, 8, 8);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ib), X64Reg_RCX);

	// element buffer (rbp-relative): the source for scatter, or the gather destination. Gather zeroes
	// first so a padded element (e.g. struct{x:int, y:u8}) has zero padding bytes, matching a normally
	// constructed value — else a memory-wise struct compare sees garbage in the padding.
	X64Mem ebuf;
	if (store) { x64Value sv = x64_spill_value(p, src, elem_type); ebuf = sv.mem; }
	else       { i64 esz = gb_max(type_size_of(elem_type), (i64)1); i32 e_off = x64_alloc_local(p, esz, gb_max(type_align_of(elem_type), (i64)1)); ebuf = x64_rbp_mem(e_off); x64_zero_mem(p, ebuf, esz); }

	for (int k = 0; k < ncomp; k++) {
		Type *comp_t; i64 coff;
		if (ebt->kind == Type_Struct) { comp_t = ebt->Struct.fields[k]->type; coff = ebt->Struct.offsets[k]; }
		else                          { comp_t = ebt->Array.elem;             coff = (i64)k * type_size_of(ebt->Array.elem); }
		i64 csz  = type_size_of(comp_t); if (csz <= 0) csz = 1;
		i64 foff = soa->Struct.offsets[k];

		// &component[idx] → RAX, pin to a slot
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(sb));
		if (soa->Struct.soa_kind == StructSoa_Fixed) { if (foff != 0) x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)foff); }
		else                                          x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)foff)); // load [^] ptr
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(ib));
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RDX, csz);
		x64_emit_imul_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		i32 caddr = x64_alloc_local(p, 8, 8);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(caddr), X64Reg_RAX);

		X64Mem ecomp = ebuf; ecomp.disp += (i32)coff;
		x64_soa_copy_component(p, caddr, ecomp, csz, store);
	}
	return store ? x64v_none() : x64v_mem(elem_type, ebuf);
}

gb_internal x64Addr x64_build_addr(x64Procedure *p, Ast *expr) {
	expr = unparen_expr(expr);
	Type *t = expr->tav.type;

	// A constant has no storage or linker symbol. AGGREGATE constants → emitted once into
	// .rdata and referenced by address (mirrors lb_add_global_generated_from_procedure for
	// &CONST). SCALAR constants → a stack local. Without this, &named/compound constant fell
	// through to the Ident global path → LEA [RIP+name] → unresolved.
	//
	// EXCEPT a CompoundLit operand: `&T{…}` is a pointer to a FRESH MUTABLE temporary (Odin
	// semantics), even when the literal is a compile-time constant. Routing it to the read-only
	// .rdata constant made writes through the pointer fault — `copy_n`'s `&Limited_Reader{}`
	// then `l.r = r` wrote to .rdata → AV (THE blick cbor.decode / project-open crash). Let it
	// fall to the default case below, which spills a WRITABLE stack copy.
	if (expr->tav.mode == Addressing_Constant && expr->kind != Ast_CompoundLit) {
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
		return x64_build_addr_from_entity(p, e, expr);
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
		// `soa[idx].field`: base is an IndexExpr on a #soa value — no contiguous element address, so
		// compute the field-array element address directly (mirrors lb's Addressing_SoaVariable path).
		{
			Ast *be = unparen_expr(sel->expr);
			if (be->kind == Ast_IndexExpr) {
				Type *bt = base_type(type_deref(be->IndexExpr.expr->tav.type));
				if (bt != nullptr && bt->kind == Type_Struct && bt->Struct.soa_kind != StructSoa_None) {
					return x64_build_soa_index_field(p, be, expr, x64_typed(t));
				}
			}
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

		return x64_emit_struct_ep(p, base, sel, x64_typed(t));
	} case_end;

	case_ast_node(idx, IndexExpr, expr); {
		return x64_build_addr_index_expr(p, expr);
	} case_end;

	// Matrix element `m[row, col]` (runtime indices). Column-major + stride layout; element byte offset
	// = (row + stride*col)*esz (row-major: (col + stride*row)*esz). Mirrors lb's matrix index addr.
	case_ast_node(mie, MatrixIndexExpr, expr); {
		bool  via_ptr = is_type_pointer(mie->expr->tav.type);
		Type *mt   = base_type(via_ptr ? type_deref(mie->expr->tav.type) : mie->expr->tav.type);
		Type *elem = base_type(mt->Matrix.elem);
		i64   esz    = type_size_of(elem); if (esz <= 0) esz = 1;
		i64   stride = matrix_type_stride_in_elems(mt);
		bool  row_major = mt->Matrix.is_row_major;

		// Pin the matrix base address, then the two indices (each build clobbers RAX).
		x64Value bv = via_ptr ? x64_build_expr(p, mie->expr)
		                      : (x64_emit_lea(&p->asm_, X64Reg_RAX, x64_build_addr(p, mie->expr).mem), x64v_reg(t_rawptr, X64Reg_RAX));
		i32 base_off = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, bv, X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base_off), X64Reg_RAX);
		i32 row_off = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, x64_build_expr(p, mie->row_index), X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(row_off), X64Reg_RAX);
		i32 col_off = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, x64_build_expr(p, mie->column_index), X64Reg_RAX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(col_off), X64Reg_RAX);

		// offset_elems = inner + stride*outer  (col-major: inner=row, outer=col; row-major swapped)
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(row_major ? col_off : row_off)); // inner
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(row_major ? row_off : col_off)); // outer
		if (stride > 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX, (i32)stride);
		x64_emit_add_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);                                  // offset_elems
		if (esz > 1) x64_emit_imul_rri(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX, (i32)esz);         // offset_bytes
		x64_emit_add_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_off));                       // + base
		return x64addr(x64_mem(X64Reg_RAX, 0), elem);
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

	case_ast_node(ta, TypeAssertion, expr); {
		// `&x.(T)`: address of the variant DATA inside the union/any — NOT a copy. Falling through to
		// default spilled a COPY (x64_build_type_assertion copies the value out), so writes through the
		// pointer never reached the real aggregate (e.g. a worker doing `v := &stream.variant.(Video);
		// v.ready = true` set the flag on a throwaway temp → the reader's union field stayed false).
		// The variant value lives at offset 0 of a tagged/maybe-pointer union, so &union is the data
		// ptr; for `any` the data ptr is the `.data` field. Mirrors LLVM lb_build_addr TypeAssertion.
		// (Tag/id mismatch panic is not emitted, matching the existing value path.)
		bool maybe_unwrap = ta->type != nullptr && ta->type->kind == Ast_UnaryExpr &&
		                    ta->type->UnaryExpr.op.kind == Token_Question;
		Type *dst_type = (ta->type != nullptr && !maybe_unwrap) ? type_of_expr(ta->type) : t;
		if (dst_type == nullptr) dst_type = t ? t : t_int;
		Type *dt = x64_typed(dst_type);
		Type *src_bt = base_type(x64_typed(ta->expr->tav.type));
		if (src_bt != nullptr && is_type_any(src_bt)) {
			x64Addr any_addr = x64_build_addr(p, ta->expr);           // &any
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, any_addr.mem); // RAX = any.data ptr
			return x64addr(x64_mem(X64Reg_RAX, 0), dt);
		}
		x64Addr u = x64_build_addr(p, ta->expr);                     // &union; variant data @ offset 0
		return x64addr(u.mem, dt);
	} case_end;

	default: {
		// Evaluate, spill to a stack local, return its address. Covers BasicLit, CompoundLit, etc.
		// The returned address may escape the statement (e.g. `p := &Foo{}`), so the slot must
		// be scope-lived — mark it (named_seq) so the temp reclaimer never reuses it.
		Type *t = expr->tav.type ? x64_typed(expr->tav.type) : t_int;
		i64 sz = x64_type_size(t);
		i64 al = x64_type_align(t);
		if (sz <= 0) sz = 8;
		if (al <= 0) al = 8;
		i32 off = x64_alloc_local(p, sz, al);
		p->named_seq++;
		if (p->local_size > p->escape_floor) p->escape_floor = p->local_size; // slot outlives its block (LLVM hoists)
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

// Scalar compiler intrinsics (intrinsics.odin) lowered inline, mirroring lb_build_builtin_proc.
// These are NOT runtime calls in LLVM either (LLVM lowers them to llvm.* intrinsics) — without
// them they silently no-op and return garbage (e.g. overflow_add → 0 broke the temp arena).
gb_internal bool x64_is_simple_intrinsic(i32 id) {
	switch (id) {
	case BuiltinProc_count_ones: case BuiltinProc_count_zeros:
	case BuiltinProc_count_trailing_zeros: case BuiltinProc_count_leading_zeros:
	case BuiltinProc_count_trailing_ones:  case BuiltinProc_count_leading_ones:
	case BuiltinProc_reverse_bits: case BuiltinProc_byte_swap:
	case BuiltinProc_overflow_add: case BuiltinProc_overflow_sub: case BuiltinProc_overflow_mul:
	case BuiltinProc_saturating_add: case BuiltinProc_saturating_sub:
	case BuiltinProc_sqrt: case BuiltinProc_fused_mul_add:
	case BuiltinProc_cpu_relax: case BuiltinProc_trap: case BuiltinProc_debug_trap:
	case BuiltinProc_read_cycle_counter: case BuiltinProc_read_cycle_counter_frequency:
	case BuiltinProc_expect: case BuiltinProc_likely: case BuiltinProc_unlikely:
	case BuiltinProc_volatile_load:     case BuiltinProc_volatile_store:
	case BuiltinProc_non_temporal_load: case BuiltinProc_non_temporal_store:
	case BuiltinProc_unaligned_load:    case BuiltinProc_unaligned_store:
	case BuiltinProc_prefetch_read_instruction:  case BuiltinProc_prefetch_read_data:
	case BuiltinProc_prefetch_write_instruction: case BuiltinProc_prefetch_write_data:
	case BuiltinProc_x86_cpuid: case BuiltinProc_x86_xgetbv:
	case BuiltinProc_alloca:
		return true;
	}
	return false;
}

// Zero-extend RAX from `bytes` to a full 64-bit register so bit-count ops see clean upper bits.
gb_internal void x64_zext_rax(x64Procedure *p, i64 bytes) {
	X64Assembler *a = &p->asm_;
	if      (bytes == 1) x64_emit_movzx_rr(a, X64OpSize_8,  X64Reg_RAX, X64Reg_RAX);
	else if (bytes == 2) x64_emit_movzx_rr(a, X64OpSize_16, X64Reg_RAX, X64Reg_RAX);
	else if (bytes == 4) x64_emit_mov_rr  (a, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
}

gb_internal x64Value x64_build_intrinsic(x64Procedure *p, Ast *expr, i32 id) {
	ast_node(ce, CallExpr, expr);
	X64Assembler *a = &p->asm_;

	// alloca(size, align): grow the stack by `size` (16-aligned) and return a pointer ABOVE the
	// fixed call area (shadow 32 + outgoing 64 = 96), which is rsp-relative and follows rsp down — so
	// later calls write their shadow/args below the alloca region without clobbering it. rsp is
	// restored to rbp in the epilogue, freeing it. Was UNIMPLEMENTED → returned 0 (nil) → callers
	// crashed (core:encoding/json unmarshal_object's `field_used := alloca(...)` → mem_zero(nil) → AV →
	// config_load failed → wrong window size). Result is 16-aligned (covers align ≤ 16; json uses 1).
	if (id == BuiltinProc_alloca) {
		x64Value sz = x64_build_expr(p, ce->args[0]);
		x64_value_to_reg(p, sz, X64Reg_RAX);
		x64_emit_add_ri(a, X64OpSize_64, X64Reg_RAX, 15);
		x64_emit_and_ri(a, X64OpSize_64, X64Reg_RAX, -16);          // round size up to 16
		x64_emit_sub_rr(a, X64OpSize_64, X64Reg_RSP, X64Reg_RAX);   // grow the stack
		x64_emit_lea(a, X64Reg_RAX, x64_mem(X64Reg_RSP, 96));       // skip the 96-byte call area
		return x64v_reg(t_rawptr, X64Reg_RAX);
	}

	// --- control / misc (no integer operand) ---
	switch (id) {
	case BuiltinProc_cpu_relax:  x64_emit_pause(a); return x64v_none();
	case BuiltinProc_debug_trap: x64_emit_int3(a);  return x64v_none();
	case BuiltinProc_trap:       x64_emit_ud2(a);   return x64v_none();
	case BuiltinProc_read_cycle_counter: {
		x64_emit_rdtsc(a);                                          // EDX:EAX
		x64_emit_mov_rr(a, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);   // zero-extend low 32
		x64_emit_shl_ri(a, X64OpSize_64, X64Reg_RDX, 32);
		x64_emit_or_rr (a, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
		return x64v_reg(x64_typed(expr->tav.type), X64Reg_RAX);
	}
	case BuiltinProc_read_cycle_counter_frequency:
		return x64v_imm(x64_typed(expr->tav.type), 0);             // unknown on x86 (mirrors LLVM)
	case BuiltinProc_expect: case BuiltinProc_likely: case BuiltinProc_unlikely:
		return x64_build_expr(p, ce->args[0]);                     // hint only — pass the value through
	case BuiltinProc_prefetch_read_instruction:  case BuiltinProc_prefetch_read_data:
	case BuiltinProc_prefetch_write_instruction: case BuiltinProc_prefetch_write_data:
		x64_build_expr(p, ce->args[0]);                            // evaluate ptr; prefetch is a hint → no-op
		return x64v_none();
	}

	// --- memory load/store intrinsics (volatile/non_temporal/unaligned ≈ plain load/store on x86) ---
	if (id == BuiltinProc_volatile_load || id == BuiltinProc_non_temporal_load || id == BuiltinProc_unaligned_load) {
		Type *T = x64_typed(type_deref(x64_typed(ce->args[0]->tav.type)));
		x64Value dv = x64_build_expr(p, ce->args[0]);
		x64_value_to_reg(p, dv, X64Reg_RAX);
		return x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), T));
	}
	if (id == BuiltinProc_volatile_store || id == BuiltinProc_non_temporal_store || id == BuiltinProc_unaligned_store) {
		Type *T = x64_typed(type_deref(x64_typed(ce->args[0]->tav.type)));
		x64Value dv = x64_build_expr(p, ce->args[0]);
		i32 poff = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, dv, X64Reg_RAX);
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(poff), X64Reg_RAX);
		x64Value vv = x64_build_expr(p, ce->args[1]);
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8, x64_rbp_mem(poff)); // R8 = ptr (copy clobbers RSI/RDI/RCX, not R8)
		x64_store_value(p, x64addr(x64_mem(X64Reg_R8, 0), T), vv);
		return x64v_none();
	}

	// --- x86 CPUID / XGETBV (return a small result struct) ---
	if (id == BuiltinProc_x86_cpuid) {
		Type *st = x64_typed(expr->tav.type);
		x64Value a0 = x64_build_expr(p, ce->args[0]); i32 o0 = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, a0, X64Reg_RAX); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(o0), X64Reg_RAX);
		x64Value a1 = x64_build_expr(p, ce->args[1]); i32 o1 = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, a1, X64Reg_RAX); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(o1), X64Reg_RAX);
		i32 t = x64_alloc_local(p, type_size_of(st), type_align_of(st));
		x64_emit_mov_rm(a, X64OpSize_32, X64Reg_RAX, x64_rbp_mem(o0)); // leaf
		x64_emit_mov_rm(a, X64OpSize_32, X64Reg_RCX, x64_rbp_mem(o1)); // subleaf
		x64_emit_push_r(a, X64Reg_RBX);                                // RBX is callee-saved; CPUID writes it
		x64_emit_cpuid(a);
		x64_emit_mov_mr(a, X64OpSize_32, x64_rbp_mem(t + (i32)type_offset_of(st, 0)), X64Reg_RAX);
		x64_emit_mov_mr(a, X64OpSize_32, x64_rbp_mem(t + (i32)type_offset_of(st, 1)), X64Reg_RBX);
		x64_emit_mov_mr(a, X64OpSize_32, x64_rbp_mem(t + (i32)type_offset_of(st, 2)), X64Reg_RCX);
		x64_emit_mov_mr(a, X64OpSize_32, x64_rbp_mem(t + (i32)type_offset_of(st, 3)), X64Reg_RDX);
		x64_emit_pop_r(a, X64Reg_RBX);
		return x64v_mem(st, x64_rbp_mem(t));
	}
	if (id == BuiltinProc_x86_xgetbv) {
		Type *st = x64_typed(expr->tav.type);
		x64Value a0 = x64_build_expr(p, ce->args[0]);
		x64_value_to_reg(p, a0, X64Reg_RCX);
		i32 t = x64_alloc_local(p, type_size_of(st), type_align_of(st));
		x64_emit_xgetbv(a);
		x64_emit_mov_mr(a, X64OpSize_32, x64_rbp_mem(t + (i32)type_offset_of(st, 0)), X64Reg_RAX);
		x64_emit_mov_mr(a, X64OpSize_32, x64_rbp_mem(t + (i32)type_offset_of(st, 1)), X64Reg_RDX);
		return x64v_mem(st, x64_rbp_mem(t));
	}

	// --- floating-point ---
	if (id == BuiltinProc_sqrt) {
		Type *t = x64_typed(expr->tav.type);
		x64Value xv = x64_build_expr(p, ce->args[0]);
		x64_value_to_xmm(p, xv, X64XmmReg_XMM0);
		if (x64_is_double(t)) x64_emit_sqrtsd(a, X64XmmReg_XMM0, X64XmmReg_XMM0);
		else                  x64_emit_sqrtss(a, X64XmmReg_XMM0, X64XmmReg_XMM0);
		return x64v_xmm(t, X64XmmReg_XMM0);
	}
	if (id == BuiltinProc_fused_mul_add) {
		Type *t = x64_typed(expr->tav.type);
		bool dbl = x64_is_double(t);
		i32 s0 = x64_alloc_local(p, 8, 8), s1 = x64_alloc_local(p, 8, 8), s2 = x64_alloc_local(p, 8, 8);
		x64_value_to_xmm(p, x64_build_expr(p, ce->args[0]), X64XmmReg_XMM0);
		if (dbl) x64_emit_movsd_mr(a, x64_rbp_mem(s0), X64XmmReg_XMM0); else x64_emit_movss_mr(a, x64_rbp_mem(s0), X64XmmReg_XMM0);
		x64_value_to_xmm(p, x64_build_expr(p, ce->args[1]), X64XmmReg_XMM0);
		if (dbl) x64_emit_movsd_mr(a, x64_rbp_mem(s1), X64XmmReg_XMM0); else x64_emit_movss_mr(a, x64_rbp_mem(s1), X64XmmReg_XMM0);
		x64_value_to_xmm(p, x64_build_expr(p, ce->args[2]), X64XmmReg_XMM0);
		if (dbl) x64_emit_movsd_mr(a, x64_rbp_mem(s2), X64XmmReg_XMM0); else x64_emit_movss_mr(a, x64_rbp_mem(s2), X64XmmReg_XMM0);
		if (dbl) {
			x64_emit_movsd_rm(a, X64XmmReg_XMM0, x64_rbp_mem(s0)); x64_emit_movsd_rm(a, X64XmmReg_XMM1, x64_rbp_mem(s1));
			x64_emit_mulsd(a, X64XmmReg_XMM0, X64XmmReg_XMM1);     x64_emit_movsd_rm(a, X64XmmReg_XMM1, x64_rbp_mem(s2));
			x64_emit_addsd(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
		} else {
			x64_emit_movss_rm(a, X64XmmReg_XMM0, x64_rbp_mem(s0)); x64_emit_movss_rm(a, X64XmmReg_XMM1, x64_rbp_mem(s1));
			x64_emit_mulss(a, X64XmmReg_XMM0, X64XmmReg_XMM1);     x64_emit_movss_rm(a, X64XmmReg_XMM1, x64_rbp_mem(s2));
			x64_emit_addss(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
		}
		return x64v_xmm(t, X64XmmReg_XMM0);
	}

	// --- integer: 2-operand (overflow_* / saturating_*) ---
	if (id == BuiltinProc_overflow_add || id == BuiltinProc_overflow_sub || id == BuiltinProc_overflow_mul ||
	    id == BuiltinProc_saturating_add || id == BuiltinProc_saturating_sub) {
		Type *main = x64_typed(expr->tav.type);
		bool tup = main != nullptr && main->kind == Type_Tuple;
		Type *ot = tup ? x64_typed(main->Tuple.variables[0]->type) : main;
		X64OpSize sz = x64_op_size_of(ot);
		i64 width = 8 * type_size_of(ot); if (width <= 0) width = 64;
		bool uns = !x64_is_signed_integer(ot);

		x64Value xv = x64_build_expr(p, ce->args[0]);
		i32 xoff = x64_alloc_local(p, 8, 8);
		x64_value_to_reg(p, xv, X64Reg_RAX);
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(xoff), X64Reg_RAX);
		x64Value yv = x64_build_expr(p, ce->args[1]);
		x64_value_to_reg(p, yv, X64Reg_RCX);
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(xoff)); // RAX=x, RCX=y

		bool is_add = (id == BuiltinProc_overflow_add || id == BuiltinProc_saturating_add);
		bool is_sub = (id == BuiltinProc_overflow_sub || id == BuiltinProc_saturating_sub);

		if (id == BuiltinProc_overflow_add || id == BuiltinProc_overflow_sub || id == BuiltinProc_overflow_mul) {
			if      (is_add) x64_emit_add_rr(a, sz, X64Reg_RAX, X64Reg_RCX);
			else if (is_sub) x64_emit_sub_rr(a, sz, X64Reg_RAX, X64Reg_RCX);
			else if (uns)    x64_emit_mul_r (a, sz, X64Reg_RCX);  // RDX:RAX = RAX*RCX
			else             x64_emit_imul_r(a, sz, X64Reg_RCX);
			X64Cc cc = uns ? X64Cc_C : X64Cc_O; // unsigned wrap = carry, signed = overflow
			x64_emit_setcc_r(a, cc, X64Reg_RDX);
			if (!tup) return x64v_reg(ot, X64Reg_RAX);
			i32 t = x64_alloc_local(p, type_size_of(main), type_align_of(main));
			x64_emit_mov_mr(a, sz, x64_rbp_mem(t), X64Reg_RAX);  // field 0 = result
			i64 vsz  = type_size_of(ot);     if (vsz  <= 0) vsz  = 1;
			i64 b_al = type_align_of(t_bool); if (b_al <= 0) b_al = 1;
			i64 off1 = (vsz + (b_al - 1)) & ~(b_al - 1);
			x64_emit_mov_mr(a, X64OpSize_8, x64_rbp_mem(t + (i32)off1), X64Reg_RDX); // field 1 = did_overflow
			return x64v_mem(main, x64_rbp_mem(t));
		}

		// saturating
		if (uns) {
			if (is_add) {
				x64_emit_add_rr(a, sz, X64Reg_RAX, X64Reg_RCX);
				x64_emit_setcc_r(a, X64Cc_C, X64Reg_RDX);
				x64_emit_movzx_rr(a, X64OpSize_8, X64Reg_RDX, X64Reg_RDX);
				x64_emit_neg_r(a, X64OpSize_64, X64Reg_RDX);            // -1 on carry, else 0
				x64_emit_or_rr (a, sz, X64Reg_RAX, X64Reg_RDX);        // carry → all-ones (MAX)
			} else {
				x64_emit_sub_rr(a, sz, X64Reg_RAX, X64Reg_RCX);
				x64_emit_setcc_r(a, X64Cc_C, X64Reg_RDX);
				x64_emit_movzx_rr(a, X64OpSize_8, X64Reg_RDX, X64Reg_RDX);
				x64_emit_neg_r(a, X64OpSize_64, X64Reg_RDX);
				x64_emit_not_r(a, X64OpSize_64, X64Reg_RDX);            // 0 on borrow, else all-ones
				x64_emit_and_rr(a, sz, X64Reg_RAX, X64Reg_RDX);        // borrow → 0 (MIN)
			}
			return x64v_reg(ot, X64Reg_RAX);
		}
		// signed saturating: on overflow clamp to INT_MAX/INT_MIN by sign of the (wrapped) sum
		if (is_add) x64_emit_add_rr(a, sz, X64Reg_RAX, X64Reg_RCX);
		else        x64_emit_sub_rr(a, sz, X64Reg_RAX, X64Reg_RCX);
		x64_emit_setcc_r(a, X64Cc_O, X64Reg_RDX);                      // DL = overflow (before flags clobbered)
		x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX);
		x64_emit_sar_ri(a, sz, X64Reg_R8, (u8)(width - 1));            // R8 = 0 / -1 by sum sign
		i64 imin = (i64)((u64)1 << (u64)(width - 1));
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, imin);
		x64_emit_xor_rr(a, sz, X64Reg_R8, X64Reg_RCX);                 // R8 = INT_MAX (pos) / INT_MIN (neg)
		x64_emit_test_rr(a, X64OpSize_8, X64Reg_RDX, X64Reg_RDX);
		x64_emit_cmov_rr(a, X64Cc_NZ, X64OpSize_64, X64Reg_RAX, X64Reg_R8);
		return x64v_reg(ot, X64Reg_RAX);
	}

	// --- integer: 1-operand bit ops ---
	Type *t = x64_typed(expr->tav.type);
	i64 bytes = type_size_of(t); if (bytes <= 0) bytes = 8;
	i64 width = 8 * bytes;
	X64OpSize wsz = (bytes >= 8) ? X64OpSize_64 : X64OpSize_32;
	x64Value xv = x64_build_expr(p, ce->args[0]);

	// 128-bit bit-count intrinsics: no 128b GPR, so split into hi/lo 64-bit halves and combine.
	// Result type is the 16-byte input type → return a Mem (count in low 8B, high 8B zeroed).
	// Returning x64v_none() here (the old bail) fed a bogus operand into callers like floattidf's
	// `N - count_leading_zeros(a)` → garbage/crash (reflect.as_f64(i128) segfault).
	if (bytes == 16 && (id == BuiltinProc_count_ones || id == BuiltinProc_count_zeros ||
	    id == BuiltinProc_count_leading_zeros  || id == BuiltinProc_count_trailing_zeros ||
	    id == BuiltinProc_count_leading_ones   || id == BuiltinProc_count_trailing_ones)) {
		x64Value sp = x64_spill_value(p, xv, t);
		GB_ASSERT(sp.kind == x64Value_Mem);
		X64Mem lo_m = sp.mem, hi_m = sp.mem; hi_m.disp += 8;
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8, lo_m); // lo
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R9, hi_m); // hi
		if (id == BuiltinProc_count_leading_ones || id == BuiltinProc_count_trailing_ones) {
			x64_emit_not_r(a, X64OpSize_64, X64Reg_R8);     // count_*_ones(x) = count_*_zeros(~x)
			x64_emit_not_r(a, X64OpSize_64, X64Reg_R9);
		}
		switch (id) {
		case BuiltinProc_count_ones:
		case BuiltinProc_count_zeros:
			x64_emit_popcnt_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_R8);
			x64_emit_popcnt_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_R9);
			x64_emit_add_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
			if (id == BuiltinProc_count_zeros) {
				x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, 128);
				x64_emit_sub_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
				x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
			}
			break;
		case BuiltinProc_count_trailing_zeros:
		case BuiltinProc_count_trailing_ones:
			x64_emit_tzcnt_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_R8);    // tzcnt(lo)
			x64_emit_tzcnt_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_R9);
			x64_emit_add_ri(a, X64OpSize_64, X64Reg_RCX, 64);            // 64 + tzcnt(hi)
			x64_emit_test_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_R8);
			x64_emit_cmov_rr(a, X64Cc_Z, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); // lo==0 → use hi
			break;
		default: // count_leading_zeros / count_leading_ones
			x64_emit_lzcnt_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_R9);    // lzcnt(hi)
			x64_emit_lzcnt_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_R8);
			x64_emit_add_ri(a, X64OpSize_64, X64Reg_RAX, 64);            // 64 + lzcnt(lo)
			x64_emit_test_rr(a, X64OpSize_64, X64Reg_R9, X64Reg_R9);
			x64_emit_cmov_rr(a, X64Cc_NZ, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); // hi!=0 → use hi
			break;
		}
		i32 ro = x64_alloc_local(p, 16, 16);
		X64Mem res_lo = x64_rbp_mem(ro), res_hi = x64_rbp_mem(ro); res_hi.disp += 8;
		x64_emit_mov_mr(a, X64OpSize_64, res_lo, X64Reg_RAX);
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, 0);
		x64_emit_mov_mr(a, X64OpSize_64, res_hi, X64Reg_RCX); // zero high 8B
		return x64v_mem(t, x64_rbp_mem(ro));
	}

	x64_value_to_reg(p, xv, X64Reg_RAX);
	if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) return x64v_none(); // byte_swap/reverse i128: not lowered

	switch (id) {
	case BuiltinProc_count_ones:
		x64_zext_rax(p, bytes);
		x64_emit_popcnt_rr(a, wsz, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_count_zeros:
		x64_zext_rax(p, bytes);
		x64_emit_popcnt_rr(a, wsz, X64Reg_RAX, X64Reg_RAX);
		x64_emit_mov_ri(a, X64OpSize_32, X64Reg_RCX, width);
		x64_emit_sub_rr(a, X64OpSize_32, X64Reg_RCX, X64Reg_RAX);
		x64_emit_mov_rr(a, X64OpSize_32, X64Reg_RAX, X64Reg_RCX);
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_count_trailing_zeros:
		x64_zext_rax(p, bytes);
		if      (bytes == 1) x64_emit_or_ri(a, X64OpSize_32, X64Reg_RAX, 0x100);    // x==0 → tz=8=width
		else if (bytes == 2) x64_emit_or_ri(a, X64OpSize_32, X64Reg_RAX, 0x10000);  // x==0 → tz=16
		x64_emit_tzcnt_rr(a, wsz, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_count_leading_zeros:
		x64_zext_rax(p, bytes);
		x64_emit_lzcnt_rr(a, wsz, X64Reg_RAX, X64Reg_RAX);
		if (bytes < 8 && (32 - width) != 0) x64_emit_sub_ri(a, X64OpSize_32, X64Reg_RAX, (i32)(32 - width));
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_count_trailing_ones:
		x64_zext_rax(p, bytes);                                    // upper bits 0 → become 1 after NOT → caps tz at width
		x64_emit_not_r(a, X64OpSize_64, X64Reg_RAX);
		x64_emit_tzcnt_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_count_leading_ones:
		x64_zext_rax(p, bytes);
		if      (bytes == 8) x64_emit_not_r(a, X64OpSize_64, X64Reg_RAX);
		else                 x64_emit_xor_ri(a, X64OpSize_32, X64Reg_RAX, (i32)((u32)((u64)1 << width) - 1)); // flip low `width` bits
		x64_emit_lzcnt_rr(a, wsz, X64Reg_RAX, X64Reg_RAX);
		if (bytes < 8 && (32 - width) != 0) x64_emit_sub_ri(a, X64OpSize_32, X64Reg_RAX, (i32)(32 - width));
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_byte_swap:
		if      (bytes >= 4) x64_emit_bswap_r(a, (X64OpSize)bytes, X64Reg_RAX);
		else if (bytes == 2) x64_emit_rol_ri(a, X64OpSize_16, X64Reg_RAX, 8);
		return x64v_reg(t, X64Reg_RAX);
	case BuiltinProc_reverse_bits: {
		// full 64-bit bit-reverse (swap 1s/2s/4s + bswap), then shift the result down to `width`
		struct { i64 m; u8 s; } steps[3] = {
			{ (i64)0x5555555555555555ull, 1 },
			{ (i64)0x3333333333333333ull, 2 },
			{ (i64)0x0F0F0F0F0F0F0F0Full, 4 },
		};
		for (isize i = 0; i < 3; i++) {
			x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RDX, steps[i].m);
			x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
			x64_emit_shr_ri(a, X64OpSize_64, X64Reg_RAX, steps[i].s);
			x64_emit_and_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
			x64_emit_and_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
			x64_emit_shl_ri(a, X64OpSize_64, X64Reg_RCX, steps[i].s);
			x64_emit_or_rr (a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		}
		x64_emit_bswap_r(a, X64OpSize_64, X64Reg_RAX);
		if (width < 64) x64_emit_shr_ri(a, X64OpSize_64, X64Reg_RAX, (u8)(64 - width));
		return x64v_reg(t, X64Reg_RAX);
	}
	}
	return x64v_none();
}

// Value-producing compiler builtins (complex/quaternion math, swizzle, unreachable) — mirror
// lb_build_builtin_proc. These build aggregates/fields directly (no runtime call).
gb_internal x64Value x64_emit_arith_matrix(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type, bool component_wise);
gb_internal x64Value x64_emit_matrix_transpose(x64Procedure *p, x64Value m, Type *type);

gb_internal bool x64_is_value_builtin(i32 id) {
	switch (id) {
	case BuiltinProc_unreachable:
	case BuiltinProc_complex: case BuiltinProc_quaternion:
	case BuiltinProc_real: case BuiltinProc_imag: case BuiltinProc_jmag: case BuiltinProc_kmag:
	case BuiltinProc_conj: case BuiltinProc_swizzle: case BuiltinProc_expand_values:
	case BuiltinProc_transpose: case BuiltinProc_hadamard_product:
		return true;
	}
	return false;
}

// Flip the IEEE sign bit of the float at [base+foff] (esz = 4 → f32, 8 → f64) via an integer XOR
// of the high dword — used to negate a complex/quaternion imaginary component (conj).
gb_internal void x64_negate_float_mem(x64Procedure *p, i32 base_off, i64 foff, i64 esz) {
	i32 dword = (i32)(base_off + foff + (esz == 8 ? 4 : 0));
	x64_emit_xor_mi(&p->asm_, X64OpSize_32, x64_rbp_mem(dword), (i32)0x80000000);
}

gb_internal x64Value x64_build_value_builtin(x64Procedure *p, Ast *expr, i32 id) {
	ast_node(ce, CallExpr, expr);

	if (id == BuiltinProc_unreachable) { x64_emit_ud2(&p->asm_); return x64v_none(); }

	// Matrix builtins. transpose (matrix form); hadamard_product = component-wise matrix `*`.
	// (matrix_flatten / outer_product still GB_PANIC below — findable gaps.)
	if (id == BuiltinProc_transpose) {
		x64Value m = x64_build_expr(p, ce->args[0]);
		if (m.type != nullptr && is_type_matrix(m.type)) {
			return x64_emit_matrix_transpose(p, m, x64_typed(expr->tav.type));
		}
		GB_PANIC("x64 transpose of non-matrix (rank-2 array) unimplemented");
	}
	if (id == BuiltinProc_hadamard_product) {
		x64Value a = x64_build_expr(p, ce->args[0]);
		x64Value b = x64_build_expr(p, ce->args[1]);
		return x64_emit_arith_matrix(p, Token_Mul, a, b, x64_typed(expr->tav.type), /*component_wise*/true);
	}

	if (id == BuiltinProc_complex) {
		Type *ct = x64_typed(expr->tav.type);
		Type *ft = x64_typed(base_complex_elem_type(ct));
		i64 esz = type_size_of(ft);
		i32 off = x64_alloc_local(p, type_size_of(ct), type_align_of(ct));
		x64_store_value(p, x64addr(x64_rbp_mem(off),            ft), x64_build_expr(p, ce->args[0])); // real @0
		x64_store_value(p, x64addr(x64_rbp_mem(off + (i32)esz), ft), x64_build_expr(p, ce->args[1])); // imag @esz
		return x64v_mem(ct, x64_rbp_mem(off));
	}

	if (id == BuiltinProc_quaternion) {
		Type *qt = x64_typed(expr->tav.type);
		Type *ft = x64_typed(base_complex_elem_type(qt));
		i64 esz = type_size_of(ft);
		i32 off = x64_alloc_local(p, type_size_of(qt), type_align_of(qt));
		for (isize i = 0; i < ce->args.count; i++) {
			ast_node(fv, FieldValue, ce->args[i]);
			String nm = fv->field->Ident.token.string;   // @QuaternionLayout: x/imag=0 y/jmag=1 z/kmag=2 w/real=3
			i32 idx = (nm == "x" || nm == "imag") ? 0 : (nm == "y" || nm == "jmag") ? 1 :
			          (nm == "z" || nm == "kmag") ? 2 : 3;
			x64_store_value(p, x64addr(x64_rbp_mem(off + idx*(i32)esz), ft), x64_build_expr(p, fv->value));
		}
		return x64v_mem(qt, x64_rbp_mem(off));
	}

	if (id == BuiltinProc_real || id == BuiltinProc_imag || id == BuiltinProc_jmag || id == BuiltinProc_kmag) {
		Type *vt = base_type(x64_typed(ce->args[0]->tav.type));
		Type *ft = x64_typed(expr->tav.type);
		i64 esz = type_size_of(base_complex_elem_type(vt));
		bool cplx = is_type_complex(vt);
		i32 idx = (id == BuiltinProc_real) ? (cplx ? 0 : 3) :  // complex: {real@0, imag@1}; quaternion @QuaternionLayout {x/imag@0,y/jmag@1,z/kmag@2,w/real@3}
		          (id == BuiltinProc_imag) ? (cplx ? 1 : 0) :
		          (id == BuiltinProc_jmag) ? 1 : 2;
		i32 voff = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
		x64_store_value(p, x64addr(x64_rbp_mem(voff), vt), x64_build_expr(p, ce->args[0]));
		return x64v_mem(ft, x64_rbp_mem(voff + idx*(i32)esz));
	}

	if (id == BuiltinProc_conj) {
		Type *vt = x64_typed(expr->tav.type);
		Type *bt = base_type(vt);
		i64 esz = type_size_of(base_complex_elem_type(bt));
		i32 off = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
		x64_store_value(p, x64addr(x64_rbp_mem(off), vt), x64_build_expr(p, ce->args[0]));
		if (is_type_complex(bt)) {
			x64_negate_float_mem(p, off, esz, esz);                       // imag @esz
		} else { // quaternion: negate x,y,z (imag parts), keep w (real)
			x64_negate_float_mem(p, off, 0,       esz);
			x64_negate_float_mem(p, off, esz,     esz);
			x64_negate_float_mem(p, off, 2*esz,   esz);
		}
		return x64v_mem(vt, x64_rbp_mem(off));
	}

	if (id == BuiltinProc_expand_values) { // struct/array → tuple of its fields (mirrors lb_expand_values)
		Type *rt = x64_typed(expr->tav.type);
		Type *st = base_type(x64_typed(ce->args[0]->tav.type));
		i32 soff = x64_alloc_local(p, type_size_of(st), type_align_of(st));
		x64_store_value(p, x64addr(x64_rbp_mem(soff), st), x64_build_expr(p, ce->args[0]));
		bool is_struct = st->kind == Type_Struct;
		i64 esz = is_struct ? 0 : type_size_of(base_array_type(st));
		if (rt == nullptr || rt->kind != Type_Tuple) { // single value → field 0 (offset 0)
			Type *ft = is_struct ? x64_typed(st->Struct.fields[0]->type) : x64_typed(base_array_type(st));
			return x64v_mem(ft, x64_rbp_mem(soff));
		}
		i32 toff = x64_alloc_local(p, type_size_of(rt), type_align_of(rt));
		x64Value dst_tuple = x64v_mem(rt, x64_rbp_mem(toff));
		for (isize i = 0; i < rt->Tuple.variables.count; i++) {
			Type *ft = rt->Tuple.variables[i]->type;
			i64 src_off = is_struct ? type_offset_of(st, st->Struct.fields[i]->Variable.field_index) : (i64)i*esz;
			x64Addr dst = x64_emit_tuple_ep(p, dst_tuple, (i32)i); // dst field i @ canonical type_offset_of
			x64_store_value(p, dst, x64v_mem(ft, x64_rbp_mem(soff + (i32)src_off)));
		}
		return x64v_mem(rt, x64_rbp_mem(toff));
	}

	if (id == BuiltinProc_swizzle) { // array form: swizzle(arr, i0, i1, ...) → [arr[i0], arr[i1], ...]
		Type *rt = x64_typed(expr->tav.type);
		Type *et = x64_typed(base_array_type(rt));
		i64 esz = type_size_of(et);
		Type *st = base_type(x64_typed(ce->args[0]->tav.type));
		i32 soff = x64_alloc_local(p, type_size_of(st), type_align_of(st));
		x64_store_value(p, x64addr(x64_rbp_mem(soff), st), x64_build_expr(p, ce->args[0]));
		i32 roff = x64_alloc_local(p, type_size_of(rt), type_align_of(rt));
		for (isize i = 1; i < ce->args.count; i++) {
			i64 idx = exact_value_to_i64(ce->args[i]->tav.value);
			x64_store_value(p, x64addr(x64_rbp_mem(roff + (i32)((i-1)*esz)), et),
			                x64v_mem(et, x64_rbp_mem(soff + (i32)(idx*esz))));
		}
		return x64v_mem(rt, x64_rbp_mem(roff));
	}

	return x64v_none();
}

// cstring → string (mirrors runtime.cstring_to_string): nil → "" ({0,0}), else
// {data=ptr, len=strlen(ptr)}, inlined as a NUL byte-scan (SIMD-widen later). Without
// this, `string(cstr)` kept only the ptr and left len=0 (os.args[i] came out empty).
// cstring → string / cstring16 → string16: scan for the NUL terminator to compute the length,
// then build a {data, len} value. `elem_size` is the code-unit size (1 = byte/UTF-8, 2 = u16/
// UTF-16) and `len` is in ELEMENTS. Mirrors the runtime cstring_to_string / cstring16_to_string16
// calls LLVM emits (lb_emit_conv); inlined here to avoid a runtime-proc dependency.
gb_internal x64Value x64_cstring_to_string(x64Procedure *p, x64Value src, i64 elem_size, Type *result_type) {
	x64_value_to_reg(p, src, X64Reg_RAX); // RAX = cstring data ptr
	i32 off = x64_alloc_local(p, 16, 8);  // string result: data@0, len@8
	X64OpSize cmp_sz = (elem_size == 2) ? X64OpSize_16 : X64OpSize_8;

	isize l_nil  = x64_label_alloc(&p->asm_);
	isize l_loop = x64_label_alloc(&p->asm_);
	isize l_done = x64_label_alloc(&p->asm_);
	isize l_end  = x64_label_alloc(&p->asm_);

	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
	x64_emit_jcc(&p->asm_, X64Cc_E, l_nil); // nil cstring → ""

	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX); // data = ptr
	x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);       // cursor = ptr
	x64_label_bind(&p->asm_, l_loop);
	x64_emit_cmp_mi(&p->asm_, cmp_sz, x64_mem(X64Reg_RCX, 0), 0);          // code unit == 0?
	x64_emit_jcc(&p->asm_, X64Cc_E, l_done);
	if (elem_size == 2) x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, 2);
	else                x64_emit_inc_r(&p->asm_, X64OpSize_64, X64Reg_RCX);
	x64_emit_jmp(&p->asm_, l_loop);
	x64_label_bind(&p->asm_, l_done);
	x64_emit_sub_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);       // byte span = cursor - ptr
	if (elem_size == 2) x64_emit_shr_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, 1); // → element count
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_RCX);
	x64_emit_jmp(&p->asm_, l_end);

	x64_label_bind(&p->asm_, l_nil);
	x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(off),     0); // data = 0
	x64_emit_mov_mi(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), 0); // len  = 0

	x64_label_bind(&p->asm_, l_end);
	return x64v_mem(result_type, x64_rbp_mem(off));
}

// A type used as a value → its typeid (mirrors lb_typeid): the canonical type hash, matching
// typeid_of and the emitted Type_Info.id.
gb_internal x64Value x64_typeid(Type *type) {
	return x64v_imm(t_typeid, (i64)type_hash_canonical_type(default_type(type)));
}

// Fits in a GPR: moved/extended rather than memcpy'd.
gb_internal bool x64_is_reg_scalar(Type *bt) {
	if (bt == nullptr) return false;
	if (type_size_of(bt) > 8) return false;
	if (x64_is_float(bt))     return false;
	return x64_is_integer(bt) || is_type_boolean(bt) || is_type_rune(bt) || bt->kind == Type_Enum ||
	       is_type_pointer(bt) || is_type_multi_pointer(bt) || is_type_proc(bt) ||
	       is_type_bit_set(bt) || is_type_typeid(bt);
}

// Mirrors lb_emit_conv. Unhandled pairs fall through to a reinterpret (x64 is incomplete vs lb's PANIC).
gb_internal x64Value x64_emit_conv(x64Procedure *p, x64Value src, Type *from, Type *to) {
	if (to == nullptr) return src;
	if (from != nullptr && are_types_identical(from, to)) { src.type = x64_typed(to); return src; }

	Type *dst_base = base_type(to);
	Type *src_base = (from != nullptr) ? base_type(from) : nullptr;

	// Same representation, distinct named/typed wrappers → reinterpret.
	if (src_base != nullptr && are_types_identical(src_base, dst_base)) {
		src.type = x64_typed(to);
		return src;
	}

	if (src_base != nullptr && is_type_cstring(src_base) && is_type_string(dst_base)) {
		return x64_cstring_to_string(p, src, 1, x64_typed(to));
	}
	if (src_base != nullptr && is_type_cstring16(src_base) && is_type_string16(dst_base)) {
		return x64_cstring_to_string(p, src, 2, x64_typed(to));
	}

	// f16 conversions via the runtime libgcc-style helpers (x64 has no native f16; F16C not assumed).
	// extendhfsf2(f16)->f32, truncsfhf2(f32)->f16, truncdfhf2(f64)->f16 (Odin entity names; link __…).
	// f16be is byte-swapped to/from LE around the helper. (Helper bodies now compile — see the float
	// compound-assign fix; gnu_h2f_ieee's `v.f *= magic.f` previously gave 0.)
	{
		bool dst_f16 = x64_is_f16(dst_base);
		bool src_f16 = (src_base != nullptr) && x64_is_f16(src_base);
		// SCALAR f16 conversions only. A `f16 -> [N]f16` / `f16 -> #simd[N]f16` is a scalar→aggregate
		// BROADCAST (e.g. `vec_f16 * f16_scalar`) and must fall through to the array/#simd broadcast cases
		// below — NOT be treated as an f16→scalar conv (which returned a 4-byte f32 typed as the array →
		// movss-spilled → low half landed in lane 0 = 0, high half in lane 1). Was the blick rounded-corner
		// bug: corner_roundness ([4]f16) * f16(dpi) → garbage lanes.
		// Only a genuine f16 *numeric* conversion (both sides float/int). Boxing an f16 INTO a union
		// variant / any / struct — or broadcasting to [N]f16 / #simd — is NOT an f16 scalar conv and
		// must fall through to the union/aggregate construction below. The old guard only excluded
		// arrays/#simd, so `f16 -> union` relabeled an f32 as the union type and skipped
		// x64_store_union_variant → variant value + tag never written → cbor's half-float decode of
		// 0.5 boxed into cbor.Value read back as 0 (blick's saved layout fractions collapsed to 0).
		bool dst_num = is_type_float(dst_base) || is_type_integer(dst_base);
		bool src_num = src_base != nullptr && (is_type_float(src_base) || is_type_integer(src_base));
		if ((src_f16 && dst_num) || (dst_f16 && src_num)) {
			bool dst_be = dst_f16 && dst_base->kind == Type_Basic && (dst_base->Basic.flags & BasicFlag_EndianBig) != 0;
			bool src_be = src_f16 && src_base->kind == Type_Basic && (src_base->Basic.flags & BasicFlag_EndianBig) != 0;
			if (src_f16 && dst_f16) { // f16 -> f16 (endianness change / alias)
				x64_value_to_reg(p, src, X64Reg_RAX);
				if (src_be != dst_be) x64_emit_bswap16(p, X64Reg_RAX);
				return x64v_reg(x64_typed(to), X64Reg_RAX);
			}
			if (src_f16) { // f16 -> f32/f64/int: extendhfsf2 then f32->dst
				x64_value_to_reg(p, src, X64Reg_RAX);
				if (src_be) x64_emit_bswap16(p, X64Reg_RAX);
				x64Value harg = x64_spill_value(p, x64v_reg(t_f16, X64Reg_RAX), t_f16);
				x64Value f32v = x64_emit_runtime_call(p, str_lit("extendhfsf2"), &harg, 1); // f32
				if (x64_is_double(dst_base)) {
					x64_value_to_xmm(p, f32v, X64XmmReg_XMM0);
					x64_emit_cvtss2sd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM0);
					return x64v_xmm(x64_typed(to), X64XmmReg_XMM0);
				}
				if (x64_is_integer(dst_base)) {
					x64_value_to_xmm(p, f32v, X64XmmReg_XMM0);
					x64_emit_cvttss2si(&p->asm_, X64Reg_RAX, X64OpSize_64, X64XmmReg_XMM0);
					return x64v_reg(x64_typed(to), X64Reg_RAX);
				}
				f32v.type = x64_typed(to); // dst f32
				return f32v;
			}
			// dst_f16: f32/f64/int -> f16
			x64Value f16v;
			if (x64_is_double(src_base)) {
				x64Value harg = x64_spill_value(p, src, t_f64);
				f16v = x64_emit_runtime_call(p, str_lit("truncdfhf2"), &harg, 1); // f16
			} else {
				x64Value f32src = src;
				if (x64_is_integer(src_base)) {
					x64_value_to_reg(p, src, X64Reg_RAX);
					x64_emit_cvtsi2ss(&p->asm_, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64);
					f32src = x64v_xmm(t_f32, X64XmmReg_XMM0);
				}
				x64Value harg = x64_spill_value(p, f32src, t_f32);
				f16v = x64_emit_runtime_call(p, str_lit("truncsfhf2"), &harg, 1); // f16
			}
			x64_value_to_reg(p, f16v, X64Reg_RAX);
			if (dst_be) x64_emit_bswap16(p, X64Reg_RAX);
			return x64v_reg(x64_typed(to), X64Reg_RAX);
		}
	}

	// Complex/quaternion conversions (component-wise float convert; mirrors lb_emit_conv). x64 has no
	// native complex/quaternion — they're aggregates of 2/4 floats — so a widening/narrowing conv
	// (e.g. `complex128(complex64)` in fmt's fmt_value) must convert EACH float component, not
	// reinterpret bits. Without this it fell to the reinterpret fallback → garbage (fmt printed
	// complex64 as 32768.008+0i; quaternion widening then segfaulted). complex layout {real@0, imag@1};
	// quaternion @QuaternionLayout {x/imag@0, y/jmag@1, z/kmag@2, w/real@3}.
	if (src_base != nullptr && (is_type_complex(dst_base) || is_type_quaternion(dst_base))) {
		Type *dft  = x64_typed(base_complex_elem_type(dst_base));
		i64   desz = type_size_of(dft);
		i32   doff = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		if (is_type_complex(src_base) || is_type_quaternion(src_base)) {
			Type *sft  = x64_typed(base_complex_elem_type(src_base));
			i64   sesz = type_size_of(sft);
			i32   soff = x64_alloc_local(p, type_size_of(from), type_align_of(from));
			x64_store_value(p, x64addr(x64_rbp_mem(soff), x64_typed(from)), src);
			bool scplx = is_type_complex(src_base), dcplx = is_type_complex(dst_base);
			if (scplx == dcplx) {
				// complex→complex (2 comps) or quaternion→quaternion (4 comps): same component order.
				int n = dcplx ? 2 : 4;
				for (int i = 0; i < n; i++) {
					x64Value ev = x64v_mem(sft, x64_rbp_mem(soff + i*(i32)sesz));
					x64_store_value(p, x64addr(x64_rbp_mem(doff + i*(i32)desz), dft), x64_emit_conv(p, ev, sft, dft));
				}
			} else {
				// complex→quaternion: {imag→x@0, real→w@3}, others 0 (mirrors LLVM {imag,0,0,real}).
				x64_zero_mem(p, x64_rbp_mem(doff), type_size_of(to));
				x64Value rv = x64v_mem(sft, x64_rbp_mem(soff));               // real @0
				x64Value iv = x64v_mem(sft, x64_rbp_mem(soff + (i32)sesz));   // imag @sesz
				x64_store_value(p, x64addr(x64_rbp_mem(doff),                 dft), x64_emit_conv(p, iv, sft, dft)); // x/imag @0
				x64_store_value(p, x64addr(x64_rbp_mem(doff + 3*(i32)desz),   dft), x64_emit_conv(p, rv, sft, dft)); // w/real @3
			}
		} else {
			// scalar (int/float) → complex/quaternion: real component only, rest zero.
			x64_zero_mem(p, x64_rbp_mem(doff), type_size_of(to));
			x64Value rv = x64_emit_conv(p, src, from, dft);
			i32 real_idx = is_type_complex(dst_base) ? 0 : 3; // complex real @0; quaternion w/real @3
			x64_store_value(p, x64addr(x64_rbp_mem(doff + real_idx*(i32)desz), dft), rv);
		}
		return x64v_mem(x64_typed(to), x64_rbp_mem(doff));
	}

	if (x64_is_float(dst_base) && src_base != nullptr && x64_is_float(src_base)) {
		x64_value_to_xmm(p, src, X64XmmReg_XMM0);
		if (x64_is_double(dst_base) && !x64_is_double(src_base))      x64_emit_cvtss2sd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM0);
		else if (!x64_is_double(dst_base) && x64_is_double(src_base)) x64_emit_cvtsd2ss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM0);
		return x64v_xmm(x64_typed(to), X64XmmReg_XMM0);
	}
	if (x64_is_integer(dst_base) && src_base != nullptr && x64_is_float(src_base)) {
		x64_value_to_xmm(p, src, X64XmmReg_XMM0);
		if (x64_is_double(src_base)) x64_emit_cvttsd2si(&p->asm_, X64Reg_RAX, X64OpSize_64, X64XmmReg_XMM0);
		else                         x64_emit_cvttss2si(&p->asm_, X64Reg_RAX, X64OpSize_64, X64XmmReg_XMM0);
		return x64v_reg(x64_typed(to), X64Reg_RAX);
	}
	// 128-bit integer → float: no SSE cvtsi2s* for 128-bit operands. Call the runtime compiler-rt
	// helpers (__floattidf / __floattidf_unsigned) for f64, then narrow for f32/f16. Was reflect's
	// as_f64(i128) → garbage (the cvtsi2sd path below saw only the low 64 bits). Must precede it.
	if (x64_is_float(dst_base) && src_base != nullptr && is_type_integer(src_base) && type_size_of(src_base) == 16) {
		bool uns = !x64_is_signed_integer(src_base);
		// floattidf/_unsigned take the i128/u128 by value, but Win64 passes a 16-byte scalar param
		// INDIRECTLY (a pointer to a caller copy). x64_emit_call has no per-arg indirect lowering, so
		// pass the address ourselves as a pointer-typed arg (mirrors the normal call path's indirect
		// slot). Passing the raw 16B value loaded only its low 8B as the "pointer" → floattidf deref'd
		// a small int as an address → segfault (reflect.as_f64(i128) crash).
		x64Value slot = x64_spill_value(p, src, x64_typed(from));
		GB_ASSERT(slot.kind == x64Value_Mem);
		i32 ptr_off = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, slot.mem);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr_off), X64Reg_RAX);
		x64Value arg = x64v_mem(alloc_type_pointer(x64_typed(from)), x64_rbp_mem(ptr_off));
		x64Value f64v = x64_emit_runtime_call(p, uns ? str_lit("floattidf_unsigned") : str_lit("floattidf"), &arg, 1);
		if (x64_is_double(dst_base)) { f64v.type = x64_typed(to); return f64v; }
		return x64_emit_conv(p, f64v, t_f64, to); // f64 → f32/f16
	}
	if (x64_is_float(dst_base) && src_base != nullptr && x64_is_integer(src_base)) {
		x64_value_to_reg(p, src, X64Reg_RAX);
		// Re-extend RAX to a full 64 bits from the SOURCE integer width before cvtsi2s*, which reads
		// the whole 64-bit reg. An upstream sub-64-bit op can leave stale high bits — e.g. `i8 & 15`
		// compiles to `and al, 15`, masking AL but leaving the sign-extended upper bytes → for i8 -128
		// RAX = 0xFF..FF00 and cvtsi2sd read -256 not 0 (math/fixed.to_f64). value_to_reg can't fix a
		// register source (no-op into the same reg), so extend explicitly by type here.
		i64 isz = type_size_of(src_base); if (isz <= 0) isz = 8;
		bool sgn = x64_is_signed_integer(src_base);
		if (isz == 1 || isz == 2) {
			X64OpSize osz = (isz == 1) ? X64OpSize_8 : X64OpSize_16;
			if (sgn) x64_emit_movsx_rr(&p->asm_, osz, X64Reg_RAX, X64Reg_RAX);
			else     x64_emit_movzx_rr(&p->asm_, osz, X64Reg_RAX, X64Reg_RAX);
		} else if (isz == 4) {
			if (sgn) x64_emit_movsx_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX); // MOVSXD
			else     x64_emit_mov_rr  (&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX); // 32-bit mov zero-extends
		}
		if (x64_is_double(dst_base)) x64_emit_cvtsi2sd(&p->asm_, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64);
		else                         x64_emit_cvtsi2ss(&p->asm_, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64);
		return x64v_xmm(x64_typed(to), X64XmmReg_XMM0);
	}

	// Array element-wise conversion ([N]From → [N]To with differing element types): convert each
	// element (mirrors lb_emit_conv's array path). Without this, `cast([2]u32)(some [2]f32)` fell to
	// the reinterpret fallback below and BIT-CAST the floats instead of truncating them — e.g.
	// `cast([2]u32)(window.size)` produced f32 bit patterns (1608.0 → 0x44C90000 = 1153957888) →
	// ResizeBuffers E_INVALIDARG. Must precede the >8B same-size reinterpret (arrays ≥16B hit it too).
	if (dst_base->kind == Type_Array && src_base != nullptr && src_base->kind == Type_Array &&
	    dst_base->Array.count == src_base->Array.count &&
	    !are_types_identical(base_type(src_base->Array.elem), base_type(dst_base->Array.elem))) {
		Type *se = src_base->Array.elem;
		Type *de = dst_base->Array.elem;
		i64 n   = dst_base->Array.count;
		i64 ssz = type_size_of(se); if (ssz <= 0) ssz = 1;
		i64 dsz = type_size_of(de); if (dsz <= 0) dsz = 1;
		// Pin the source to a stable rbp slot — the per-element convert clobbers RAX/XMM0 and
		// src.mem may be register-relative.
		i32 src_off = x64_alloc_local(p, type_size_of(from), type_align_of(from));
		x64_store_value(p, x64addr(x64_rbp_mem(src_off), x64_typed(from)), src);
		i32 dst_off = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		for (i64 i = 0; i < n; i++) {
			x64Value ev = x64v_mem(se, x64_rbp_mem(src_off + (i32)(i * ssz)));
			x64Value cv = x64_emit_conv(p, ev, se, de);
			x64_store_value(p, x64addr(x64_rbp_mem(dst_off + (i32)(i * dsz)), de), cv);
		}
		return x64v_mem(x64_typed(to), x64_rbp_mem(dst_off));
	}

	// #simd[N]From → #simd[N]To lane-wise conversion (differing element types): lanes are contiguous in
	// memory like an array, so convert each lane scalar-wise (mirrors the array→array case + LLVM's simd
	// conv). Without this it fell to the reinterpret fallback — e.g. xxhash's `u64xW(u32xW(v))` (narrow
	// each u64 lane to u32 then widen back) kept the full 64-bit lanes instead of zeroing the high 32.
	if (dst_base->kind == Type_SimdVector && src_base != nullptr && src_base->kind == Type_SimdVector &&
	    dst_base->SimdVector.count == src_base->SimdVector.count &&
	    !are_types_identical(base_type(src_base->SimdVector.elem), base_type(dst_base->SimdVector.elem))) {
		Type *se = src_base->SimdVector.elem;
		Type *de = dst_base->SimdVector.elem;
		i64 n   = dst_base->SimdVector.count;
		i64 ssz = type_size_of(se); if (ssz <= 0) ssz = 1;
		i64 dsz = type_size_of(de); if (dsz <= 0) dsz = 1;
		i32 src_off = x64_alloc_local(p, type_size_of(from), type_align_of(from));
		x64_store_value(p, x64addr(x64_rbp_mem(src_off), x64_typed(from)), src);
		i32 dst_off = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		for (i64 i = 0; i < n; i++) {
			x64Value ev = x64v_mem(se, x64_rbp_mem(src_off + (i32)(i * ssz)));
			x64Value cv = x64_emit_conv(p, ev, se, de);
			x64_store_value(p, x64addr(x64_rbp_mem(dst_off + (i32)(i * dsz)), de), cv);
		}
		return x64v_mem(x64_typed(to), x64_rbp_mem(dst_off));
	}

	// Scalar → array broadcast (`vec * scalar` reaches x64_emit_arith_array, which convs the scalar to
	// the array type): convert the scalar to the element type and replicate into every element. Mirrors
	// lb_emit_conv's is_type_array_like(dst) path. Must follow the array→array case above.
	if (dst_base->kind == Type_Array && (src_base == nullptr || src_base->kind != Type_Array)) {
		Type *elem = base_array_type(to);
		i64 n   = get_array_type_count(to);
		i64 dsz = type_size_of(elem); if (dsz <= 0) dsz = 1;
		x64Value ev = x64_spill_value(p, x64_emit_conv(p, src, from, elem), elem);
		i32 dst_off = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		for (i64 i = 0; i < n; i++) {
			x64_store_value(p, x64addr(x64_rbp_mem(dst_off + (i32)(i * dsz)), elem), ev);
		}
		return x64v_mem(x64_typed(to), x64_rbp_mem(dst_off));
	}

	// Scalar → #simd[N]T broadcast (`v: #simd[N]T = scalar` / runtime memory_compare's sentinel splat):
	// replicate the scalar into every lane. Without this the conv fell through to a scalar store that
	// only wrote lane 0 — e.g. `#simd[8]u8 = u8(0xFF)` became [0xFF,0,0,…]. Mirrors the array case.
	if (dst_base->kind == Type_SimdVector && (src_base == nullptr || src_base->kind != Type_SimdVector)) {
		Type *elem = base_type(dst_base->SimdVector.elem);
		i64 n   = dst_base->SimdVector.count;
		i64 dsz = type_size_of(elem); if (dsz <= 0) dsz = 1;
		x64Value ev = x64_spill_value(p, x64_emit_conv(p, src, from, elem), elem);
		i32 dst_off = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		for (i64 i = 0; i < n; i++) {
			x64_store_value(p, x64addr(x64_rbp_mem(dst_off + (i32)(i * dsz)), elem), ev);
		}
		return x64v_mem(x64_typed(to), x64_rbp_mem(dst_off));
	}

	if (dst_base->kind == Type_Union && from != nullptr && union_is_variant_of(dst_base, from)) {
		i32 off = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		x64_store_union_variant(p, x64_rbp_mem(off), src, x64_typed(to));
		return x64v_mem(x64_typed(to), x64_rbp_mem(off));
	}

	// Must precede the pointer/scalar paths below.
	if (from != nullptr && check_is_assignable_to_using_subtype(from, to)) {
		return x64_apply_using_subtype(p, src, to);
	}

	if (is_type_any(dst_base)) {
		i32 off = x64_alloc_local(p, type_size_of(to), type_align_of(to));
		x64_box_any(p, x64_rbp_mem(off), src, from);
		return x64v_mem(x64_typed(to), x64_rbp_mem(off));
	}

	// >8B same-size reinterpret ([]u8↔string, transmute…): retype in place; the GPR path truncates.
	// EXCEPT a 16-byte int with a mismatched endianness (u128be↔u128) — that needs a real byte-swap
	// (handled just below), not a bit-preserving retype, else `u128(u128be)` in fmt kept the raw bytes.
	i64 cast_dsz = type_size_of(x64_typed(to));
	i64 cast_ssz = (from != nullptr) ? type_size_of(x64_typed(from)) : cast_dsz;
	bool endian_i128 = src_base != nullptr && is_type_integer(src_base) && is_type_integer(dst_base) &&
	                   (is_type_different_to_arch_endianness(src_base) || is_type_different_to_arch_endianness(dst_base));
	if (cast_dsz > 8 && cast_dsz == cast_ssz && !endian_i128) {
		src.type = x64_typed(to);
		return src;
	}

	// Widen a ≤8-byte integer to a 128-bit integer (i128/u128): the reg-scalar path can't represent
	// 16 bytes, so without this it fell to the reinterpret fallback → high 64 bits left GARBAGE (THE
	// udivmod128 `u128(n[low]/d[low])` bug → bad i128 div/mod + strconv u128 format). Build a 16-byte
	// temp: low = the (extended) source, high = sign-extension (signed) or 0 (unsigned).
	if (is_type_integer(dst_base) && type_size_of(dst_base) == 16 &&
	    from != nullptr && is_type_integer(src_base) && type_size_of(src_base) <= 8) {
		bool src_signed = x64_is_signed_integer(src_base);
		x64_value_to_reg(p, src, X64Reg_RAX); // loads + extends to 64-bit per the source type
		i32 off = x64_alloc_local(p, 16, 16);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
		if (src_signed) { x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RAX); x64_emit_sar_ri(&p->asm_, X64OpSize_64, X64Reg_RDX, 63); }
		else            { x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); }
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_RDX);
		return x64v_mem(x64_typed(to), x64_rbp_mem(off));
	}

	// 128-bit endian integer conversion (u128be↔u128, net's IP6 transmute→fmt): reverse all 16 bytes —
	// bswap each 8-byte half and swap the halves. Handled here since the reg path can't hold 16 bytes.
	if (src_base != nullptr && is_type_integer(src_base) && is_type_integer(dst_base) &&
	    type_size_of(src_base) == 16 && type_size_of(dst_base) == 16 &&
	    (is_type_different_to_arch_endianness(src_base) || is_type_different_to_arch_endianness(dst_base))) {
		i32 soff = x64_alloc_local(p, 16, 16);
		x64_store_value(p, x64addr(x64_rbp_mem(soff), x64_typed(from)), src);
		i32 doff = x64_alloc_local(p, 16, 16);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(soff));     // src low 8
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(soff + 8)); // src high 8
		x64_emit_bswap_r(&p->asm_, X64OpSize_64, X64Reg_RAX);
		x64_emit_bswap_r(&p->asm_, X64OpSize_64, X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(doff),     X64Reg_RCX); // dst low  = swap(src high)
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(doff + 8), X64Reg_RAX); // dst high = swap(src low)
		return x64v_mem(x64_typed(to), x64_rbp_mem(doff));
	}

	// Endian integer conversions: u16be/u32be/u64be (and *le on a BE arch) store bytes in the non-native
	// order, so converting to/from one must byte-swap (mirrors lb_emit_conv int↔int endian handling).
	// Without this `u64(u32be)` read the bytes natively → net's parse_ip4_address formatted 0100007f
	// instead of 7f000001. 128-bit endian ints (IP6 u128be) are left to the fallthrough (TODO).
	if (src_base != nullptr && is_type_integer(src_base) && is_type_integer(dst_base) &&
	    type_size_of(src_base) <= 8 && type_size_of(dst_base) <= 8 &&
	    (is_type_different_to_arch_endianness(src_base) || is_type_different_to_arch_endianness(dst_base))) {
		i64 ssz = type_size_of(src_base), dsz2 = type_size_of(dst_base);
		x64_value_to_reg(p, src, X64Reg_RAX); // loads + zero/sign-extends the stored bytes
		if (ssz > 1 && is_type_different_to_arch_endianness(src_base)) {           // stored → native
			if (ssz == 2) x64_emit_bswap16(p, X64Reg_RAX); else x64_emit_bswap_r(&p->asm_, x64_op_size_of(src_base), X64Reg_RAX);
		}
		if (dsz2 > 1 && is_type_different_to_arch_endianness(dst_base)) {          // native → dst endian
			if (dsz2 == 2) x64_emit_bswap16(p, X64Reg_RAX); else x64_emit_bswap_r(&p->asm_, x64_op_size_of(dst_base), X64Reg_RAX);
		}
		return x64v_reg(x64_typed(to), X64Reg_RAX);
	}

	if (x64_is_reg_scalar(src_base) && x64_is_reg_scalar(dst_base)) {
		X64OpSize src_sz = x64_op_size_of(src_base);
		X64OpSize dst_sz = x64_op_size_of(dst_base);
		x64_value_to_reg(p, src, X64Reg_RAX);
		if ((int)dst_sz > (int)src_sz && x64_is_signed_integer(src_base) && src_sz < X64OpSize_64) {
			x64_emit_movsx_rr(&p->asm_, src_sz, X64Reg_RAX, X64Reg_RAX);
		} else if ((int)dst_sz > (int)src_sz) {
			// Unsigned widen = zero-extend. MOVZX encodes 8/16→N; there is no MOVZX r64,r/m32, so a
			// 32→64 widen uses a 32-bit MOV (auto-zeroes the upper 32). Required because a prior
			// truncating cast (u64→u32) leaves the upper bits dirty — `u64(u32(x))` must clear them.
			if (src_sz <= X64OpSize_16) x64_emit_movzx_rr(&p->asm_, src_sz, X64Reg_RAX, X64Reg_RAX);
			else                        x64_emit_mov_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
		}
		return x64v_reg(x64_typed(to), X64Reg_RAX);
	}

	// Fallback: reinterpret (retype, preserve kind/bits). x64 is incomplete vs lb_emit_conv's
	// PANIC; retyping matches the prior store behavior for unhandled pairs (no regression).
	src.type = x64_typed(to);
	return src;
}

// Call a runtime bounds/slice error proc `name` with [file, line, col, <extra ints>] (mirrors
// lb_set_file_line_col + lb_emit_runtime_call). The error proc does the actual range test + panic. The
// `file: string` (16B aggregate) is passed BY POINTER (Odin ABI), like string_eq. `extra` must already
// be STABLE (spilled) int values — building the file string clobbers RAX.
gb_internal void x64_emit_bounds_runtime_call(x64Procedure *p, TokenPos pos, String name, x64Value *extra, int n_extra) {
	AstPackage *rt = p->module->gen->info->runtime_package;
	Entity *e = (rt != nullptr) ? scope_lookup_current(rt->scope, string_interner_insert(name)) : nullptr;
	if (e == nullptr || e->kind != Entity_Procedure) return;

	String file = get_file_path_string(pos.file_id);
	i32 line = pos.line, column = pos.column;
	switch (build_context.source_code_location_info) {
	case SourceCodeLocationInfo_Normal: break;
	case SourceCodeLocationInfo_Obfuscated: file = obfuscate_string(file, "F"); line = obfuscate_i32(line); column = obfuscate_i32(column); break;
	case SourceCodeLocationInfo_Filename: file = last_path_element(file); break;
	case SourceCodeLocationInfo_None: file = str_lit(""); line = 0; column = 0; break;
	}
	i32 file_slot = x64_alloc_local(p, 16, 8);
	x64Value fs = x64_const_string(p, file);
	x64_copy_fixed(p, x64_rbp_mem(file_slot), fs.mem, 16);
	i32 fileptr_slot = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(file_slot));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(fileptr_slot), X64Reg_RAX);

	if (!e->Procedure.is_foreign && e->min_dep_count.load(std::memory_order_relaxed) == 0) {
		x64_enqueue_oncall(p, e);
	}
	x64Value cargs[8];
	cargs[0] = x64v_mem(t_rawptr, x64_rbp_mem(fileptr_slot)); // &string (indirect param)
	cargs[1] = x64v_imm(t_i32, line);
	cargs[2] = x64v_imm(t_i32, column);
	for (int i = 0; i < n_extra; i++) cargs[3 + i] = extra[i];
	x64_emit_call(p, x64_get_entity_name(e), e->type, cargs, 3 + n_extra);
}

// `index` in [0, len) or panic (mirrors lb_emit_bounds_check → runtime.bounds_check_error, which does
// the `uint(index) < uint(count)` test itself).
gb_internal void x64_emit_bounds_check(x64Procedure *p, TokenPos pos, x64Value index, x64Value len) {
	if (x64_bounds_check_disabled(p)) return;
	x64Value extra[2];
	extra[0] = x64_spill_value(p, x64_emit_conv(p, index, index.type, t_int), t_int);
	extra[1] = x64_spill_value(p, x64_emit_conv(p, len,   len.type,   t_int), t_int);
	x64_emit_bounds_runtime_call(p, pos, str_lit("bounds_check_error"), extra, 2);
}

// `s[low:high]` bounds: 0 ≤ high ≤ len (no low) or 0 ≤ low ≤ high ≤ len (mirrors lb_emit_slice_bounds_check
// → runtime.slice_expr_error_hi / slice_expr_error_lo_hi).
gb_internal void x64_emit_slice_bounds_check(x64Procedure *p, TokenPos pos, x64Value low, x64Value high, x64Value len, bool lower_value_used) {
	if (x64_bounds_check_disabled(p)) return;
	if (!lower_value_used) {
		x64Value extra[2];
		extra[0] = x64_spill_value(p, x64_emit_conv(p, high, high.type, t_int), t_int);
		extra[1] = x64_spill_value(p, x64_emit_conv(p, len,  len.type,  t_int), t_int);
		x64_emit_bounds_runtime_call(p, pos, str_lit("slice_expr_error_hi"), extra, 2);
	} else {
		x64Value extra[3];
		extra[0] = x64_spill_value(p, x64_emit_conv(p, low,  low.type,  t_int), t_int);
		extra[1] = x64_spill_value(p, x64_emit_conv(p, high, high.type, t_int), t_int);
		extra[2] = x64_spill_value(p, x64_emit_conv(p, len,  len.type,  t_int), t_int);
		x64_emit_bounds_runtime_call(p, pos, str_lit("slice_expr_error_lo_hi"), extra, 3);
	}
}

// `p[low:high]` on a [^]T (no length): just low ≤ high (mirrors lb_emit_multi_pointer_slice_bounds_check
// → runtime.multi_pointer_slice_expr_error).
gb_internal void x64_emit_multi_pointer_slice_bounds_check(x64Procedure *p, TokenPos pos, x64Value low, x64Value high) {
	if (x64_bounds_check_disabled(p)) return;
	x64Value extra[2];
	extra[0] = x64_spill_value(p, x64_emit_conv(p, low,  low.type,  t_int), t_int);
	extra[1] = x64_spill_value(p, x64_emit_conv(p, high, high.type, t_int), t_int);
	x64_emit_bounds_runtime_call(p, pos, str_lit("multi_pointer_slice_expr_error"), extra, 2);
}

// Bit reinterpret of `value` as type `t` (mirrors lb_emit_transmute). Same byte size; NO numeric
// conversion (unlike x64_emit_conv): spill the bits and reload as `t`, so int↔float go through
// memory rather than a value-changing cvt/move.
gb_internal x64Value x64_emit_transmute(x64Procedure *p, x64Value value, Type *t) {
	Type *tt = x64_typed(t);
	if (value.type != nullptr && are_types_identical(value.type, tt)) return value;
	Type *st = (value.type != nullptr) ? value.type : tt;
	i64 sz = type_size_of(tt); if (sz <= 0) sz = 8;
	i32 off = x64_alloc_local(p, sz, type_align_of(tt));
	x64_store_value(p, x64addr(x64_rbp_mem(off), st), value); // store source bits (own representation)
	return x64v_mem(tt, x64_rbp_mem(off));                    // reload as `t` on use
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
		// A large indirect-ABI aggregate is passed BY POINTER (immutable param), so only its
		// ADDRESS needs to survive — copying the whole value would put it on the stack (a huge
		// by-value arg like `slot_map^` overflowed the stack). Spill the address; mark by_ref.
		if (x64_arg_is_indirect(v.type)) {
			i32 ptr_off = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, v.mem);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr_off), X64Reg_RAX);
			x64Value r = x64v_mem(v.type, x64_rbp_mem(ptr_off));
			r.by_ref = true;
			return r;
		}
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

// Manually inline a #force_inline DIRECT call. Mirrors what LLVM's always-inliner does (it
// runs even at -O0): emit the callee body into the caller, with params bound to caller-frame
// locals and `return` redirected to a join label + result slots (see x64_build_return_stmt).
// Conservative: only the common, safe shape is handled — everything else returns false and the
// normal call path runs. Inlined instructions are attributed to the call-site line for clean
// step-over debugging (Tier-1; S_INLINESITE inline-frames are a planned Tier-2 addition).
#define X64_MAX_INLINE_DEPTH 8
gb_internal bool x64_try_inline_call(x64Procedure *p, AstCallExpr *ce, Entity *callee, Type *ct, x64Value *out) {
	// Mirror LLVM EXACTLY: at -O0 + -debug the module pass manager runs NOTHING — including the
	// always-inliner (llvm_backend_opt.cpp:189 returns before LLVMAddAlwaysInlinerPass) — so LLVM
	// leaves #force_inline procs as REAL CALLS in debug builds. A real call has its own frame and is
	// trivially steppable in every debugger; a manual inline produces S_INLINESITE inline frames that
	// VS/RemedyBG/RAD don't surface (verified: the records are byte-correct but no debugger steps in).
	// So DON'T inline in debug builds — keep the real call. Inline only where LLVM would: opt>0, or
	// -O0 without -debug (then the always-inliner runs). Callees stay reachable via the normal call
	// path's on-demand min_dep==0 compilation.
	if (build_context.optimization_level <= 0 && build_context.ODIN_DEBUG) return false;

	if (callee == nullptr || callee->kind != Entity_Procedure || callee->Procedure.is_foreign) return false;

	DeclInfo *d = decl_info_of_entity(callee);
	if (d == nullptr || d->proc_lit == nullptr || d->proc_lit->kind != Ast_ProcLit) return false;
	Ast *pl   = d->proc_lit;
	Ast *body = pl->ProcLit.body;
	if (body == nullptr || body->kind != Ast_BlockStmt) return false;

	// #force_inline on the proc OR the call site (mirrors LLVM's alwaysinline on either).
	if (pl->ProcLit.inlining != ProcInlining_inline && ce->inlining != ProcInlining_inline) return false;

	if (ct->Proc.variadic || ct->Proc.c_vararg || ct->Proc.is_polymorphic) return false;
	ProcCallingConvention cc = ct->Proc.calling_convention;
	if (!is_calling_convention_odin(cc) && !is_calling_convention_none(cc)) return false;
	// An Odin-cc body uses `context`; inlining it into a caller that has none would change
	// which context it sees. Only inline when the caller can supply one.
	if (is_calling_convention_odin(cc) && !p->has_context) return false;

	// Recursion / depth guard: never inline ourselves or an entity already on the stack.
	if (p->inline_frames.count >= X64_MAX_INLINE_DEPTH || callee == p->entity) return false;
	for_array(fi, p->inline_frames) if (p->inline_frames[fi].entity == callee) return false;

	// Only all-positional calls with exactly one arg per param (no defaults / named / spread).
	int param_count = (ct->Proc.params != nullptr) ? (int)ct->Proc.params->Tuple.variables.count : 0;
	if ((int)ce->args.count != param_count) return false;
	for_array(ai, ce->args) if (ce->args[ai] == nullptr || ce->args[ai]->kind == Ast_FieldValue) return false;
	for (int i = 0; i < param_count; i++) {
		Entity *pv = ct->Proc.params->Tuple.variables[i];
		if (pv == nullptr || pv->kind != Entity_Variable) return false;
	}

	// Register this inline site for S_INLINESITE (debug builds only). NOT activated yet
	// (cur_inline_site stays the enclosing scope) — arg expressions below are CALLER code and must
	// keep the caller's lines. site_idx == -1 when not emitting debug info.
	i32 saved_site = p->cur_inline_site;
	i32 site_idx   = -1;
	if (p->module->debug_s != nullptr) {
		site_idx = (i32)p->inline_sites.count;
		x64Procedure::InlineSiteRec sr = {};
		sr.callee = callee;
		sr.parent = saved_site;
		TokenPos dp = ast_token(pl).pos; // callee declaration (base for line deltas)
		sr.decl_file_id = dp.file_id;
		sr.decl_line    = dp.line;
		array_init(&sr.lines,  p->alloc, 0, 8);
		array_init(&sr.locals, p->alloc, 0, 4);
		array_add(&p->inline_sites, sr);
	}

	// Bind each arg to a caller-frame local. A LARGE param is bound BY POINTER (alias the arg's
	// storage — Odin params are immutable, so no copy is needed), mirroring the call path's
	// by-pointer passing; copying it would drop the whole aggregate into the caller's frame PER
	// inline site (a Slot_Map-sized value × N sites = MBs each → frame blow-up). Small params copy.
	for (int i = 0; i < param_count; i++) {
		Entity *pv = ct->Proc.params->Tuple.variables[i];
		Type *pt = pv->type;
		i64 sz = type_size_of(pt); if (sz <= 0) sz = 1;
		i64 al = type_align_of(pt); if (al <= 0) al = 1;
		x64Value av = x64_build_expr(p, ce->args[i]);
		i32 loc_off; bool loc_ptr;
		if (sz > X64_INDIRECT_PARAM_COPY_MAX && av.kind == x64Value_Mem) {
			i32 ptr_off = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, av.mem);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr_off), X64Reg_RAX);
			x64_var_set(&p->var_offsets, pv, ptr_off);
			array_add(&p->indirect_params, pv); // x64_entity_addr derefs through the pointer
			loc_off = ptr_off; loc_ptr = true;
		} else {
			i32 off = x64_alloc_local(p, sz, al);
			x64_store_value(p, x64addr(x64_rbp_mem(off), pt), av);
			x64_var_set(&p->var_offsets, pv, off);
			loc_off = off; loc_ptr = false;
		}
		// Record as a scoped local of THIS site (per-site offset → correct value even when the same
		// callee is inlined at several sites, unlike the flat var_offsets dump).
		if (site_idx >= 0 && pv->token.string.len > 0) {
			x64Procedure::InlineSiteLocal L; L.entity = pv; L.name = pv->token.string; L.offset = loc_off;
			L.type = pt; L.is_ptr = loc_ptr;
			array_add(&p->inline_sites[site_idx].locals, L);
		}
		p->named_seq++; // scope-lived binding — must survive temp reclamation
	}

	// Result slots, zero-inited; named results bound so the body can read/assign them.
	Type *results = ct->Proc.results;
	int   nres    = (results != nullptr) ? (int)results->Tuple.variables.count : 0;
	i32   *roffs  = (nres > 0) ? gb_alloc_array(temporary_allocator(), i32,    nres) : nullptr;
	Type **rtypes = (nres > 0) ? gb_alloc_array(temporary_allocator(), Type *, nres) : nullptr;
	for (int i = 0; i < nres; i++) {
		Entity *re = results->Tuple.variables[i];
		Type *rt = re->type;
		i64 sz = type_size_of(rt); if (sz <= 0) sz = 1;
		i64 al = type_align_of(rt); if (al <= 0) al = 1;
		i32 off = x64_alloc_local(p, sz, al);
		x64_zero_mem(p, x64_rbp_mem(off), type_size_of(rt));
		roffs[i] = off; rtypes[i] = rt;
		if (re->token.string.len > 0) {
			x64_var_set(&p->var_offsets, re, off); p->named_seq++;
			if (site_idx >= 0) {
				x64Procedure::InlineSiteLocal L; L.entity = re; L.name = re->token.string; L.offset = off;
				L.type = rt; L.is_ptr = false;
				array_add(&p->inline_sites[site_idx].locals, L);
			}
		}
	}

	// Push the frame, emit the body (returns redirect to `join`), bind `join`, pop.
	isize join = x64_label_alloc(&p->asm_);
	x64Procedure::InlineFrame fr = {};
	fr.join_label   = join;
	fr.result_offs  = roffs;
	fr.result_types = rtypes;
	fr.nres         = nres;
	fr.defer_base   = p->deferred.count;
	fr.entity       = callee;
	array_add(&p->inline_frames, fr);

	i32 saved_line = p->inline_call_line;
	if (p->inline_frames.count == 1) p->inline_call_line = (i32)ast_token(ce->proc).pos.line;

	// Activate the site: body statements now record their (callee) lines into it. Done AFTER arg
	// building so caller-side arg code keeps the caller's lines.
	if (site_idx >= 0) p->inline_sites[site_idx].code_start = (u32)p->asm_.code.count;
	p->cur_inline_site = site_idx;
	x64_build_stmt(p, body);
	if (site_idx >= 0) p->inline_sites[site_idx].code_end = (u32)p->asm_.code.count;
	p->cur_inline_site = saved_site;

	x64_label_bind(&p->asm_, join);
	p->inline_frames.count -= 1;
	p->inline_call_line = saved_line;

	if (nres == 0) { *out = x64v_none(); return true; }
	if (nres == 1) { *out = x64v_mem(rtypes[0], x64_rbp_mem(roffs[0])); return true; }
	// Multi-result: reassemble the tuple so callers (multi-assign / unpack) read it uniformly.
	i32 tup = x64_alloc_local(p, type_size_of(results), type_align_of(results));
	for (int i = 0; i < nres; i++) {
		i64 fsz = type_size_of(rtypes[i]); if (fsz <= 0) continue;
		x64_copy_fixed(p, x64_rbp_mem(tup + (i32)type_offset_of(results, i)), x64_rbp_mem(roffs[i]), fsz);
	}
	*out = x64v_mem(results, x64_rbp_mem(tup));
	return true;
}

// Arithmetic binary op on pre-built (spilled) operands — mirrors lb_emit_arith. lhs/rhs are stable
// (loaded from memory, no build between them). Handles float (add/sub/mul/quo) and integer/bit_set
// arith (bit_set: + → union/OR, - → difference/AND-NOT). `type` is the result type.
gb_internal x64Value x64_emit_arith_array(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type);
gb_internal x64Value x64_emit_arith_matrix(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type, bool component_wise);
enum X64SimdBin { X64SimdBin_Add, X64SimdBin_Sub, X64SimdBin_Mul, X64SimdBin_Div,
                  X64SimdBin_And, X64SimdBin_Or,  X64SimdBin_Xor, X64SimdBin_AndN,
                  X64SimdBin_Min, X64SimdBin_Max };
gb_internal bool     x64_simd_packed_ok(int op, Type *elem);
gb_internal x64Value x64_simd_binop_vec(x64Procedure *p, int op, Type *vt, Type *rt, Type *elem, X64Mem amem, X64Mem bmem, bool swap);

// Materialise a value as a stable 16-byte stack temp (lo@+0, hi@+8). A real 128-bit Mem is copied; a
// narrower scalar is loaded and sign/zero-extended into the high half (so `u128_var + 1` etc. don't read
// garbage high bits). Returns the RBP offset.
gb_internal i32 x64_i128_operand(x64Procedure *p, x64Value v, Type *wide_t, bool signed_) {
	X64Assembler *a = &p->asm_;
	i32 off = x64_alloc_local(p, 16, 16);
	bool wide = v.kind == x64Value_Mem && v.type != nullptr &&
	            type_size_of(x64_typed(base_type(v.type))) >= 16;
	if (wide) {
		x64_store_value(p, x64addr(x64_rbp_mem(off), wide_t), v); // 16-byte copy
	} else {
		x64_value_to_reg(p, v, X64Reg_RAX); // extends to 64-bit per its own type
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
		if (signed_) { x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_RAX); x64_emit_sar_ri(a, X64OpSize_64, X64Reg_RDX, 63); }
		else         { x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); }
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_RDX);
	}
	return off;
}

// 128-bit integer (i128/u128, 16-byte bit_set) arithmetic via two-register (lo:hi) lowering — x64 has no
// 128-bit GP register, so x64_emit_arith's single-register path truncated+crashed. add/sub use ADC/SBB;
// bitwise is per-half; shifts use SHLD/SHRD with a count≥64 branch (Odin: count≥width → 0); mul is the
// truncated 3-product; quo/mod call the runtime __udivti3/__divti3/__umodti3/__modti3 helpers (sret ABI).
gb_internal x64Value x64_emit_arith_i128(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *result_type, bool signed_) {
	X64Assembler *a = &p->asm_;
	Type *lbt = base_type(result_type);
	if (lbt != nullptr && lbt->kind == Type_BitSet) {
		if      (op == Token_Add) op = Token_Or;
		else if (op == Token_Sub) op = Token_AndNot;
	}

	i32 la = x64_i128_operand(p, lhs, result_type, signed_);
	i32 res = x64_alloc_local(p, 16, 16);
	X64Mem rlo = x64_rbp_mem(res), rhi = x64_rbp_mem(res + 8);

	switch (op) {
	case Token_Add: case Token_Sub:
	case Token_And: case Token_Or: case Token_Xor: case Token_AndNot: {
		i32 rb = x64_i128_operand(p, rhs, result_type, signed_);
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(la + 8));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(rb));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8,  x64_rbp_mem(rb + 8));
		switch (op) {
		case Token_Add: x64_emit_add_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); x64_emit_adc_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_R8); break;
		case Token_Sub: x64_emit_sub_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); x64_emit_sbb_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_R8); break;
		case Token_And: x64_emit_and_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); x64_emit_and_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_R8); break;
		case Token_Or:  x64_emit_or_rr (a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); x64_emit_or_rr (a, X64OpSize_64, X64Reg_RDX, X64Reg_R8); break;
		case Token_Xor: x64_emit_xor_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); x64_emit_xor_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_R8); break;
		case Token_AndNot:
			x64_emit_not_r(a, X64OpSize_64, X64Reg_RCX); x64_emit_not_r(a, X64OpSize_64, X64Reg_R8);
			x64_emit_and_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); x64_emit_and_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_R8); break;
		}
		x64_emit_mov_mr(a, X64OpSize_64, rlo, X64Reg_RAX); x64_emit_mov_mr(a, X64OpSize_64, rhi, X64Reg_RDX);
		break;
	}
	case Token_Shl: case Token_Shr: {
		bool is_shl = op == Token_Shl;
		bool arith  = !is_shl && signed_; // SAR vs SHR
		x64_value_to_reg(p, rhs, X64Reg_RCX); // count
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la));     // lo
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(la + 8)); // hi
		isize l_big = x64_label_alloc(a), l_fill = x64_label_alloc(a), l_done = x64_label_alloc(a);
		x64_emit_cmp_ri(a, X64OpSize_64, X64Reg_RCX, 128);
		x64_emit_jcc(a, X64Cc_AE, l_fill);
		x64_emit_cmp_ri(a, X64OpSize_64, X64Reg_RCX, 64);
		x64_emit_jcc(a, X64Cc_AE, l_big);
		// 0 <= count < 64
		if (is_shl) { x64_emit_shld_rcl(a, X64OpSize_64, X64Reg_RDX, X64Reg_RAX); x64_emit_shl_rcl(a, X64OpSize_64, X64Reg_RAX); }
		else        { x64_emit_shrd_rcl(a, X64OpSize_64, X64Reg_RAX, X64Reg_RDX); if (arith) x64_emit_sar_rcl(a, X64OpSize_64, X64Reg_RDX); else x64_emit_shr_rcl(a, X64OpSize_64, X64Reg_RDX); }
		x64_emit_jmp(a, l_done);
		// 64 <= count < 128 (CL & 63 == count-64)
		x64_label_bind(a, l_big);
		if (is_shl) {
			x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RDX, X64Reg_RAX); x64_emit_shl_rcl(a, X64OpSize_64, X64Reg_RDX); x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
		} else {
			x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
			if (arith) { x64_emit_sar_rcl(a, X64OpSize_64, X64Reg_RAX); x64_emit_sar_ri(a, X64OpSize_64, X64Reg_RDX, 63); }
			else       { x64_emit_shr_rcl(a, X64OpSize_64, X64Reg_RAX); x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); }
		}
		x64_emit_jmp(a, l_done);
		// count >= 128 : 0 (logical) or all-sign (arithmetic)
		x64_label_bind(a, l_fill);
		if (arith) { x64_emit_sar_ri(a, X64OpSize_64, X64Reg_RDX, 63); x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RDX); }
		else       { x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RAX, X64Reg_RAX); x64_emit_xor_rr(a, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); }
		x64_label_bind(a, l_done);
		x64_emit_mov_mr(a, X64OpSize_64, rlo, X64Reg_RAX); x64_emit_mov_mr(a, X64OpSize_64, rhi, X64Reg_RDX);
		break;
	}
	case Token_Mul: {
		i32 rb = x64_i128_operand(p, rhs, result_type, signed_);
		// res.lo = a.lo*b.lo ; res.hi = high64(a.lo*b.lo) + a.lo*b.hi + a.hi*b.lo  (truncated 128-bit)
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(rb));
		x64_emit_mul_r(a, X64OpSize_64, X64Reg_RCX);            // RDX:RAX = a.lo*b.lo
		x64_emit_mov_mr(a, X64OpSize_64, rlo, X64Reg_RAX);
		x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RDX); // R8 = high(a.lo*b.lo)
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(rb + 8));
		x64_emit_imul_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); // a.lo*b.hi (low 64)
		x64_emit_add_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX);
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la + 8));
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(rb));
		x64_emit_imul_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); // a.hi*b.lo (low 64)
		x64_emit_add_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX);
		x64_emit_mov_mr(a, X64OpSize_64, rhi, X64Reg_R8);
		break;
	}
	case Token_Quo: case Token_Mod: case Token_ModMod: {
		// runtime call: proc "c" (a, b: u128) -> u128  (Win64 sret: RCX=&res, RDX=&a, R8=&b)
		i32 rb = x64_i128_operand(p, rhs, result_type, signed_);
		String name = (op == Token_Quo) ? (signed_ ? str_lit("divti3") : str_lit("udivti3"))
		                                 : (signed_ ? str_lit("modti3") : str_lit("umodti3"));
		AstPackage *rt = p->module->gen->info->runtime_package;
		Entity *e = (rt != nullptr) ? scope_lookup_current(rt->scope, string_interner_insert(name)) : nullptr;
		if (e != nullptr && e->kind == Entity_Procedure) {
			if (!e->Procedure.is_foreign && e->min_dep_count.load(std::memory_order_relaxed) == 0) x64_enqueue_oncall(p, e);
			i32 pr = x64_alloc_local(p, 8, 8), pa = x64_alloc_local(p, 8, 8), pb = x64_alloc_local(p, 8, 8);
			x64_emit_lea(a, X64Reg_RAX, rlo);              x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(pr), X64Reg_RAX);
			x64_emit_lea(a, X64Reg_RAX, x64_rbp_mem(la));  x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(pa), X64Reg_RAX);
			x64_emit_lea(a, X64Reg_RAX, x64_rbp_mem(rb));  x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(pb), X64Reg_RAX);
			x64Value cargs[3];
			cargs[0] = x64v_mem(t_rawptr, x64_rbp_mem(pr));
			cargs[1] = x64v_mem(t_rawptr, x64_rbp_mem(pa));
			cargs[2] = x64v_mem(t_rawptr, x64_rbp_mem(pb));
			x64_emit_call(p, x64_get_entity_name(e), e->type, cargs, 3);
		} else {
			GB_PANIC("x64 i128 quo/mod: runtime helper '%.*s' not found", LIT(name));
		}
		break;
	}
	default:
		GB_PANIC("x64 i128 arith op %d not supported", op);
	}
	return x64v_mem(result_type, x64_rbp_mem(res));
}

gb_internal x64Value x64_emit_arith(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type) {
	Type *lt          = x64_typed(lhs.type ? lhs.type : type);
	Type *result_type = x64_typed(type ? type : lt);

	// Matrix arithmetic (component-wise +/-/scalar-mul, matrix*matrix, matrix*vector) — checked BEFORE
	// the array path because matrix*vector produces an ARRAY result yet must go the matrix route.
	if ((lhs.type != nullptr && is_type_matrix(lhs.type)) || (rhs.type != nullptr && is_type_matrix(rhs.type))) {
		return x64_emit_arith_matrix(p, op, lhs, rhs, result_type, false);
	}
	// Element-wise array arithmetic (vec/array `+`/`-`/`*`/`/`); mirrors lb_emit_arith → lb_emit_arith_array.
	Type *rbt = base_type(result_type);
	if (rbt != nullptr && rbt->kind == Type_Array) {
		return x64_emit_arith_array(p, op, lhs, rhs, result_type);
	}
	// #simd operator arithmetic (`v + w`, `v * w`, `v ~ w`, …): packed SSE/AVX per x64_simd_packed_ok,
	// else scalar per-lane. Without this a #simd fell to the scalar/i128 path below — e.g. #simd[2]u64
	// `+`/`*` did a 16-byte i128 op that carried/multiplied ACROSS lanes (xxhash hashLong accumulate).
	if (rbt != nullptr && rbt->kind == Type_SimdVector) {
		Type *elem = base_type(rbt->SimdVector.elem);
		int bop = -1;
		switch (op) {
		case Token_Add: bop = X64SimdBin_Add;  break;
		case Token_Sub: bop = X64SimdBin_Sub;  break;
		case Token_Mul: bop = X64SimdBin_Mul;  break;
		case Token_Quo: bop = X64SimdBin_Div;  break;
		case Token_And: bop = X64SimdBin_And;  break;
		case Token_Or:  bop = X64SimdBin_Or;   break;
		case Token_Xor: bop = X64SimdBin_Xor;  break;
		case Token_AndNot: bop = X64SimdBin_AndN; break;
		}
		if (bop >= 0 && x64_simd_packed_ok(bop, elem)) {
			x64Value la = x64_spill_value(p, x64_emit_conv(p, lhs, lhs.type, result_type), result_type);
			x64Value ra = x64_spill_value(p, x64_emit_conv(p, rhs, rhs.type ? rhs.type : result_type, result_type), result_type);
			bool swap = (bop == X64SimdBin_AndN); // pandn = (~dst)&src → dst=b, src=a for a&~b
			return x64_simd_binop_vec(p, bop, result_type, result_type, elem, la.mem, ra.mem, swap);
		}
		return x64_emit_arith_array(p, op, lhs, rhs, result_type); // no packed insn (u64 mul, rem, …) → scalar lanes
	}

	// f16 arithmetic via f32 promotion (x64 has no native f16 arith): a op b = f16(f32(a) op f32(b)).
	// Was: f16 fell to the integer path → int ops on the raw bits. (pow2_f16's denormal `*`/`/`.)
	if (x64_is_f16(lt)) {
		x64Value la = x64_spill_value(p, x64_emit_conv(p, lhs, lt, t_f32), t_f32);
		x64Value ra = x64_spill_value(p, x64_emit_conv(p, rhs, rhs.type ? rhs.type : lt, t_f32), t_f32);
		x64Value fr = x64_emit_arith(p, op, la, ra, t_f32); // f32 arith (recurses into the float branch)
		return x64_emit_conv(p, fr, t_f32, result_type);    // f32 → f16/f16le/f16be
	}

	if (x64_is_float(lt)) {
		x64_value_to_xmm(p, lhs, X64XmmReg_XMM0);
		x64_value_to_xmm(p, rhs, X64XmmReg_XMM1);
		bool is_d = x64_is_double(lt);
		switch (op) {
		case Token_Add: if (is_d) x64_emit_addsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); else x64_emit_addss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); return x64v_xmm(result_type, X64XmmReg_XMM0);
		case Token_Sub: if (is_d) x64_emit_subsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); else x64_emit_subss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); return x64v_xmm(result_type, X64XmmReg_XMM0);
		case Token_Mul: if (is_d) x64_emit_mulsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); else x64_emit_mulss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); return x64v_xmm(result_type, X64XmmReg_XMM0);
		case Token_Quo: if (is_d) x64_emit_divsd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); else x64_emit_divss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1); return x64v_xmm(result_type, X64XmmReg_XMM0);
		default: GB_PANIC("x64 float arith op %d not supported", op);
		}
	}

	// 128-bit integer / 16-byte bit_set: no single GP register can hold it — use two-register lowering.
	{
		Type *lbt0 = base_type(lt);
		if (lbt0 != nullptr && type_size_of(lt) == 16 &&
		    (is_type_integer(lt) || lbt0->kind == Type_BitSet)) {
			return x64_emit_arith_i128(p, op, lhs, rhs, result_type, x64_is_signed_integer(lt));
		}
	}

	x64_value_to_reg(p, lhs, X64Reg_RAX);
	x64_value_to_reg(p, rhs, X64Reg_RCX);
	X64OpSize sz = x64_op_size_of(lt);
	bool signed_ = x64_is_signed_integer(lt);
	Type *lbt = base_type(lt);
	if (lbt != nullptr && lbt->kind == Type_BitSet) {
		if      (op == Token_Add) op = Token_Or;     // union
		else if (op == Token_Sub) op = Token_AndNot; // difference
	}
	switch (op) {
	case Token_Add:  x64_emit_add_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_Sub:  x64_emit_sub_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_Mul:
		if (signed_) x64_emit_imul_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
		else         x64_emit_mul_r  (&p->asm_, sz, X64Reg_RCX);
		return x64v_reg(result_type, X64Reg_RAX);
	case Token_Quo:
	case Token_Mod: case Token_ModMod: {
		// 8-bit DIV/IDIV are special: dividend is AX (not DX:AX), quotient→AL, remainder→AH — NOT DX.
		// The RDX-based sequence below then read RDX(=0) for the remainder → `u8 % 10` returned 0 (the
		// month-as-00 bug). Promote 8-bit to 32-bit (operands are already extended in RAX/RCX by
		// value_to_reg) so quotient lands in EAX and remainder in EDX like the wider sizes.
		X64OpSize dsz = sz;
		if (sz == X64OpSize_8) {
			if (signed_) { x64_emit_movsx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX); x64_emit_movsx_rr(&p->asm_, X64OpSize_8, X64Reg_RCX, X64Reg_RCX); }
			else         { x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX); x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RCX, X64Reg_RCX); }
			dsz = X64OpSize_32;
		}
		if (signed_) { if (dsz==X64OpSize_64) x64_emit_cqo(&p->asm_); else x64_emit_cdq(&p->asm_); x64_emit_idiv_r(&p->asm_, dsz, X64Reg_RCX); }
		else         { x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX); x64_emit_div_r(&p->asm_, dsz, X64Reg_RCX); }
		if (op == Token_Mod || op == Token_ModMod) x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
		return x64v_reg(result_type, X64Reg_RAX);
	}
	case Token_And:    x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_Or:     x64_emit_or_rr (&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_Xor:    x64_emit_xor_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_AndNot: x64_emit_not_r(&p->asm_, sz, X64Reg_RCX); x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_Shl:  x64_emit_shl_rcl(&p->asm_, sz, X64Reg_RAX); return x64v_reg(result_type, X64Reg_RAX);
	case Token_Shr:  if (signed_) x64_emit_sar_rcl(&p->asm_, sz, X64Reg_RAX); else x64_emit_shr_rcl(&p->asm_, sz, X64Reg_RAX); return x64v_reg(result_type, X64Reg_RAX);
	default: GB_PANIC("x64 integer arith op %d not supported", op);
	}
	return x64v_none();
}

// Element-wise arithmetic on arrays (vec + vec, etc.); mirrors lb_emit_arith_array's inline path: for
// each element compute lhs[i] op rhs[i] into a fresh result local. Nested arrays (e.g. matrix rows)
// recurse naturally through x64_emit_arith → x64_emit_arith_array. Small arrays unroll; lb uses a
// runtime loop for very large arrays (revisit if huge-array arith ever bloats .text).
gb_internal x64Value x64_emit_arith_array(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type) {
	Type *elem = base_array_type(type);
	i64   n    = get_array_type_count(type);
	i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;

	// Convert both operands to the result array type, then stabilise to memory.
	x64Value lv = x64_emit_conv(p, lhs, lhs.type, type);
	x64Value rv = x64_emit_conv(p, rhs, rhs.type, type);
	x64Value la = (lv.kind == x64Value_Mem) ? lv : x64_spill_value(p, lv, type);
	x64Value ra = (rv.kind == x64Value_Mem) ? rv : x64_spill_value(p, rv, type);

	// Pin both source base addresses in slots (the per-element op clobbers RAX).
	i32 la_off = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, la.mem);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(la_off), X64Reg_RAX);
	i32 ra_off = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, ra.mem);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ra_off), X64Reg_RAX);

	i32 res_off = x64_alloc_local(p, type_size_of(type), type_align_of(type));
	for (i64 i = 0; i < n; i++) {
		i32 eoff = cast(i32)(i*esz);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la_off));
		x64Value a = x64_spill_value(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, eoff), elem)), elem);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ra_off));
		x64Value b = x64_spill_value(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, eoff), elem)), elem);
		x64Value c = x64_emit_arith(p, op, a, b, elem);
		x64_store_value(p, x64addr(x64_rbp_mem(res_off + eoff), elem), c);
	}
	return x64v_mem(x64_typed(type), x64_rbp_mem(res_off));
}

// Matrix element load at [row, col] from an RBP-relative matrix `mat` (column-major + stride; mirrors
// lb_emit_matrix_ev). `mt` is the matrix type; the slot must be rbp-based and stable across the loop.
gb_internal x64Value x64_matrix_ev(x64Procedure *p, X64Mem mat, Type *mt, i64 row, i64 col) {
	Type *bt   = base_type(mt);
	Type *elem = base_type(bt->Matrix.elem);
	i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;
	X64Mem m = mat; m.disp += (i32)(matrix_indices_to_offset(bt, row, col) * esz);
	return x64_load_addr(p, x64addr(m, elem));
}

// C = A * B (matrix×matrix): C[i,j] = Σ_k A[i,k]·B[k,j]. Mirrors lb_emit_matrix_mul.
gb_internal x64Value x64_emit_matrix_mul(x64Procedure *p, x64Value lhs, x64Value rhs, Type *type) {
	Type *ct = base_type(type), *lt = base_type(lhs.type), *rt = base_type(rhs.type);
	Type *elem = base_type(ct->Matrix.elem);
	i64 Lr = lt->Matrix.row_count, Lc = lt->Matrix.column_count, Rc = rt->Matrix.column_count;
	i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
	x64Value lm = x64_spill_value(p, lhs, lhs.type);
	x64Value rm = x64_spill_value(p, rhs, rhs.type);
	i32 res = x64_alloc_local(p, type_size_of(type), type_align_of(type));
	x64_zero_mem(p, x64_rbp_mem(res), type_size_of(type));
	for (i64 i = 0; i < Lr; i++) for (i64 j = 0; j < Rc; j++) {
		x64Value acc = {};
		for (i64 k = 0; k < Lc; k++) {
			x64Value a = x64_spill_value(p, x64_matrix_ev(p, lm.mem, lt, i, k), elem);
			x64Value b = x64_spill_value(p, x64_matrix_ev(p, rm.mem, rt, k, j), elem);
			x64Value prod = x64_spill_value(p, x64_emit_arith(p, Token_Mul, a, b, elem), elem);
			acc = (k == 0) ? prod : x64_spill_value(p, x64_emit_arith(p, Token_Add, acc, prod, elem), elem);
		}
		x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(matrix_indices_to_offset(ct, i, j) * esz)), elem), acc);
	}
	return x64v_mem(x64_typed(type), x64_rbp_mem(res));
}

// C = A * v (matrix×vector → array): C[i] = Σ_j A[i,j]·v[j]. Mirrors lb_emit_matrix_mul_vector.
gb_internal x64Value x64_emit_matrix_mul_vector(x64Procedure *p, x64Value lhs, x64Value rhs, Type *type) {
	Type *lt = base_type(lhs.type);
	Type *elem = base_type(base_array_type(type));
	i64 Lr = lt->Matrix.row_count, Lc = lt->Matrix.column_count;
	i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
	x64Value lm = x64_spill_value(p, lhs, lhs.type);
	x64Value rv = x64_spill_value(p, rhs, rhs.type); // array [Lc]T
	i32 res = x64_alloc_local(p, type_size_of(type), type_align_of(type));
	for (i64 i = 0; i < Lr; i++) {
		x64Value acc = {};
		for (i64 j = 0; j < Lc; j++) {
			x64Value a = x64_spill_value(p, x64_matrix_ev(p, lm.mem, lt, i, j), elem);
			X64Mem vm = rv.mem; vm.disp += (i32)(j * esz);
			x64Value b = x64_spill_value(p, x64_load_addr(p, x64addr(vm, elem)), elem);
			x64Value prod = x64_spill_value(p, x64_emit_arith(p, Token_Mul, a, b, elem), elem);
			acc = (j == 0) ? prod : x64_spill_value(p, x64_emit_arith(p, Token_Add, acc, prod, elem), elem);
		}
		x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(i * esz)), elem), acc);
	}
	return x64v_mem(x64_typed(type), x64_rbp_mem(res));
}

// Matrix arithmetic (mirrors lb_emit_arith_matrix): `*` → matrix*matrix / matrix*vector; component-wise
// for +/-/scalar-`*` (a scalar operand broadcasts to every element). vector*matrix not yet implemented.
gb_internal x64Value x64_emit_arith_matrix(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type, bool component_wise) {
	bool lm = lhs.type != nullptr && is_type_matrix(lhs.type);
	bool rm = rhs.type != nullptr && is_type_matrix(rhs.type);
	if (op == Token_Mul && !component_wise) {
		if (lm && rm) return x64_emit_matrix_mul(p, lhs, rhs, type);
		if (lm && rhs.type != nullptr && is_type_array(rhs.type)) return x64_emit_matrix_mul_vector(p, lhs, rhs, type);
		if (rm && lhs.type != nullptr && is_type_array(lhs.type)) {
			GB_PANIC("x64 vector*matrix unimplemented — mirror lb_emit_matrix_mul (vector path)");
		}
		// scalar*matrix / matrix*scalar fall through to the component-wise broadcast below.
	}

	Type *mt = base_type(type);
	GB_ASSERT(mt != nullptr && mt->kind == Type_Matrix);
	Type *elem = base_type(mt->Matrix.elem);
	i64 rows = mt->Matrix.row_count, cols = mt->Matrix.column_count;
	i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;

	x64Value la = lm ? x64_spill_value(p, lhs, lhs.type) : x64v_none();
	x64Value ra = rm ? x64_spill_value(p, rhs, rhs.type) : x64v_none();
	x64Value ls = lm ? x64v_none() : x64_spill_value(p, x64_emit_conv(p, lhs, lhs.type, elem), elem);
	x64Value rs = rm ? x64v_none() : x64_spill_value(p, x64_emit_conv(p, rhs, rhs.type, elem), elem);

	i32 res = x64_alloc_local(p, type_size_of(type), type_align_of(type));
	x64_zero_mem(p, x64_rbp_mem(res), type_size_of(type));
	for (i64 i = 0; i < rows; i++) for (i64 j = 0; j < cols; j++) {
		x64Value a = lm ? x64_spill_value(p, x64_matrix_ev(p, la.mem, base_type(lhs.type), i, j), elem) : ls;
		x64Value b = rm ? x64_spill_value(p, x64_matrix_ev(p, ra.mem, base_type(rhs.type), i, j), elem) : rs;
		x64Value c = x64_emit_arith(p, op, a, b, elem);
		x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(matrix_indices_to_offset(mt, i, j) * esz)), elem), c);
	}
	return x64v_mem(x64_typed(type), x64_rbp_mem(res));
}

// transpose(m): result[j,i] = m[i,j] (scalar form of lb_emit_matrix_transpose; input Rr×Rc → Rc×Rr).
gb_internal x64Value x64_emit_matrix_transpose(x64Procedure *p, x64Value m, Type *type) {
	Type *mt = base_type(m.type), *ct = base_type(type);
	GB_ASSERT(mt->kind == Type_Matrix && ct->kind == Type_Matrix);
	Type *elem = base_type(ct->Matrix.elem);
	i64 Rr = mt->Matrix.row_count, Rc = mt->Matrix.column_count;
	i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
	x64Value sm = x64_spill_value(p, m, m.type);
	i32 res = x64_alloc_local(p, type_size_of(type), type_align_of(type));
	x64_zero_mem(p, x64_rbp_mem(res), type_size_of(type));
	for (i64 i = 0; i < Rr; i++) for (i64 j = 0; j < Rc; j++) {
		x64Value v = x64_spill_value(p, x64_matrix_ev(p, sm.mem, mt, i, j), elem);
		x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(matrix_indices_to_offset(ct, j, i) * esz)), elem), v);
	}
	return x64v_mem(x64_typed(type), x64_rbp_mem(res));
}

// Comparison binary op on pre-built (spilled) operands — mirrors lb_emit_comp. Handles strings
// (runtime string_*), aggregates (array/struct/union memory compare), and float/int/bit_set
// scalar comparisons. Nil comparison is handled by the caller (AST-based, before this). Returns bool.
gb_internal x64Value x64_emit_comp(x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs) {
	Type *lt  = x64_typed(lhs.type ? lhs.type : t_int);
	Type *lbt = base_type(lt);

	// f16 comparison: f16 is moved as a raw 2-byte integer (x64_is_float excludes size-2), so it would
	// fall to the INTEGER compare path below and compare raw BITS — wrong for floats: neg-zero(0x8000)
	// != 0, NaN==NaN reads true, and </>/<=/>= treat the sign bit as an unsigned magnitude. Convert both
	// operands to f32 and compare as floats (NaN-correct). Was math.classify_f16 returning Inf for
	// Neg_Zero / NaN / Neg_Inf (x==0, x*0.25==x, x<0 all mis-evaluated).
	if (x64_is_f16(lbt)) {
		x64Value lf = x64_emit_conv(p, lhs, lt, t_f32);
		lf = x64_spill_value(p, lf, t_f32); // stabilize: the rhs conversion below reuses XMM0
		x64Value rf = x64_emit_conv(p, rhs, rhs.type ? rhs.type : lt, t_f32);
		rf = x64_spill_value(p, rf, t_f32); // both must be in memory: the float compare loads lhs into XMM0 first, which would clobber rf if it were still a live register
		return x64_emit_comp(p, op, lf, rf);
	}

	// String comparison → runtime helpers string_eq/ne/lt/gt/le/ge (compare content, not the data
	// ptr). A `string`/`string16` arg is 16B → indirect, so pass the ADDRESS of each spilled operand.
	// EXCLUDE cstring/cstring16: those are 8B POINTERS (BasicFlag_String is set on them too), NOT
	// 16B {data,len} — comparing them as strings transmutes the pointer to a Raw_String and reads
	// garbage for .len. They fall through to the scalar pointer compare below (correct for `== nil`
	// and pointer identity). Was the `wkey == nil` crash (wkey: cstring16) → string_eq →
	// memory_equal(garbage_len) → SIMD overrun.
	if (is_type_string(lbt) && !is_type_cstring(lbt) && !is_type_cstring16(lbt)) {
		bool is16 = is_type_string16(lbt);
		String name = {};
		switch (op) {
		case Token_CmpEq: name = is16 ? str_lit("string16_eq") : str_lit("string_eq"); break;
		case Token_NotEq: name = is16 ? str_lit("string16_ne") : str_lit("string_ne"); break;
		case Token_Lt:    name = is16 ? str_lit("string16_lt") : str_lit("string_lt"); break;
		case Token_Gt:    name = is16 ? str_lit("string16_gt") : str_lit("string_gt"); break;
		case Token_LtEq:  name = is16 ? str_lit("string16_le") : str_lit("string_le"); break;
		case Token_GtEq:  name = is16 ? str_lit("string16_ge") : str_lit("string_ge"); break;
		default: break;
		}
		if (name.len != 0) {
			AstPackage *rt = p->module->gen->info->runtime_package;
			Entity *e = (rt != nullptr) ? scope_lookup_current(rt->scope, string_interner_insert(name)) : nullptr;
			if (e != nullptr && e->kind == Entity_Procedure) {
				if (!e->Procedure.is_foreign && e->min_dep_count.load(std::memory_order_relaxed) == 0) {
					x64_enqueue_oncall(p, e); // called by name; on-demand path won't see it
				}
				x64Value lm = x64_spill_value(p, lhs, t_string);
				x64Value rm = x64_spill_value(p, rhs, t_string);
				i32 lp = x64_alloc_local(p, 8, 8);
				x64_emit_lea(&p->asm_, X64Reg_RAX, lm.mem);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(lp), X64Reg_RAX);
				i32 rp = x64_alloc_local(p, 8, 8);
				x64_emit_lea(&p->asm_, X64Reg_RAX, rm.mem);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(rp), X64Reg_RAX);
				x64Value cargs[2];
				cargs[0] = x64v_mem(t_rawptr, x64_rbp_mem(lp));
				cargs[1] = x64v_mem(t_rawptr, x64_rbp_mem(rp));
				return x64_emit_call(p, x64_get_entity_name(e), e->type, cargs, 2);
			}
		}
	}

	// cstring comparison: CONTENT compare via runtime cstring_eq/ne/lt/le/gt/ge (cstrings are 8-byte
	// pointers passed DIRECTLY, unlike 16-byte strings). These helpers are nil-safe (check the
	// pointers first), so `cstring == nil` works too — distinct-pointer same-content cstrings now
	// compare EQUAL (was a pointer compare → unequal; the flags test_all_strings failure).
	if ((is_type_cstring(lbt) || is_type_cstring16(lbt)) &&
	    (op == Token_CmpEq || op == Token_NotEq || op == Token_Lt || op == Token_LtEq || op == Token_Gt || op == Token_GtEq)) {
		bool is16 = is_type_cstring16(lbt);
		String name = {};
		switch (op) {
		case Token_CmpEq: name = is16 ? str_lit("cstring16_eq") : str_lit("cstring_eq"); break;
		case Token_NotEq: name = is16 ? str_lit("cstring16_ne") : str_lit("cstring_ne"); break;
		case Token_Lt:    name = is16 ? str_lit("cstring16_lt") : str_lit("cstring_lt"); break;
		case Token_LtEq:  name = is16 ? str_lit("cstring16_le") : str_lit("cstring_le"); break;
		case Token_Gt:    name = is16 ? str_lit("cstring16_gt") : str_lit("cstring_gt"); break;
		case Token_GtEq:  name = is16 ? str_lit("cstring16_ge") : str_lit("cstring_ge"); break;
		default: break;
		}
		AstPackage *rt = p->module->gen->info->runtime_package;
		Entity *e = (rt != nullptr) ? scope_lookup_current(rt->scope, string_interner_insert(name)) : nullptr;
		if (e != nullptr && e->kind == Entity_Procedure) {
			if (!e->Procedure.is_foreign && e->min_dep_count.load(std::memory_order_relaxed) == 0) {
				x64_enqueue_oncall(p, e);
			}
			x64Value cargs[2] = { lhs, rhs }; // cstring values (8-byte pointers, direct)
			return x64_emit_call(p, x64_get_entity_name(e), e->type, cargs, 2);
		}
	}

	// Aggregate equality (array/struct/union) via flat byte compare. Gated on is_type_simple_compare.
	if (op == Token_CmpEq || op == Token_NotEq) {
		bool is_agg = lbt != nullptr &&
		    (lbt->kind == Type_Array || lbt->kind == Type_EnumeratedArray ||
		     lbt->kind == Type_Struct || lbt->kind == Type_Union);
		if (is_agg && is_type_simple_compare(lt)) {
			if (lhs.kind != x64Value_Mem) lhs = x64_spill_value(p, lhs, lt);
			if (rhs.kind != x64Value_Mem) rhs = x64_spill_value(p, rhs, lt);
			x64_emit_lea(&p->asm_, X64Reg_RAX, lhs.mem); // RAX = &lhs (stays through the loop)
			x64_emit_lea(&p->asm_, X64Reg_RCX, rhs.mem); // RCX = &rhs
			isize lbl_ne   = x64_label_alloc(&p->asm_);
			isize lbl_done = x64_label_alloc(&p->asm_);
			i64 total = type_size_of(lt);
			i64 off = 0;
			X64OpSize chunks[4] = { X64OpSize_64, X64OpSize_32, X64OpSize_16, X64OpSize_8 };
			i64 widths[4]       = { 8, 4, 2, 1 };
			for (int ci = 0; ci < 4; ci++) {
				while (total - off >= widths[ci]) {
					x64_emit_mov_rm(&p->asm_, chunks[ci], X64Reg_R8, x64_mem(X64Reg_RAX, (i32)off));
					x64_emit_mov_rm(&p->asm_, chunks[ci], X64Reg_R9, x64_mem(X64Reg_RCX, (i32)off));
					x64_emit_cmp_rr(&p->asm_, chunks[ci], X64Reg_R8, X64Reg_R9);
					x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_ne);
					off += widths[ci];
				}
			}
			x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RAX, (op == Token_CmpEq) ? 1 : 0);
			x64_emit_jmp(&p->asm_, lbl_done);
			x64_label_bind(&p->asm_, lbl_ne);
			x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RAX, (op == Token_CmpEq) ? 0 : 1);
			x64_label_bind(&p->asm_, lbl_done);
			return x64v_reg(t_bool, X64Reg_RAX);
		}
		// Non-simple comparable struct (string / other deep-compare fields): the bitwise path above
		// can't be used — it would compare string HEADERS (ptr,len) not content. Call the per-type
		// synth equal proc `__$equal$<hash>(&lhs, &rhs) -> bool` (field-wise; strings by content,
		// mirrors lb's generated record-equality proc). Was json unmarshal_json's `product == original`
		// comparing only the first field's bytes (a==b false, a==d true). Arrays/unions with
		// deep-compare elems still fall through (rare; would need element-loop in x64_emit_equal_body).
		if (lbt != nullptr && lbt->kind == Type_Struct && is_type_comparable(lt)) {
			if (lhs.kind != x64Value_Mem) lhs = x64_spill_value(p, lhs, lt);
			if (rhs.kind != x64Value_Mem) rhs = x64_spill_value(p, rhs, lt);
			i32 la = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, lhs.mem);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(la), X64Reg_RAX);
			i32 ra = x64_alloc_local(p, 8, 8);
			x64_emit_lea(&p->asm_, X64Reg_RAX, rhs.mem);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ra), X64Reg_RAX);
			String en = x64_synth_proc_name(p->module, X64Synth_Equal, lt);
			x64_enqueue_synth(p->module, X64Synth_Equal, lt);
			x64Value ea[2] = { x64v_mem(t_rawptr, x64_rbp_mem(la)), x64v_mem(t_rawptr, x64_rbp_mem(ra)) };
			x64Value eq = x64_emit_call(p, en, t_equal_proc, ea, 2);
			x64_value_to_reg(p, eq, X64Reg_RAX);
			if (op == Token_NotEq) x64_emit_xor_ri(&p->asm_, X64OpSize_32, X64Reg_RAX, 1); // invert bool
			return x64v_reg(t_bool, X64Reg_RAX);
		}
	}

	// Float comparison. UCOMISD/SS sets ZF/CF/PF; PF=1 marks UNORDERED (a NaN operand). A lone setcc
	// mishandles NaN: ==(E) and <=(BE)/<(B) read true when unordered (e.g. NaN==NaN), and !=(NE) reads
	// false. Combine with the parity flag so ordered ops are false for NaN and != is true (mirrors
	// LLVM's ordered f-compares). >(A)/>=(AE) are already NaN-safe (they need CF=0, but NaN sets CF=1).
	if (x64_is_float(lbt)) {
		x64_value_to_xmm(p, lhs, X64XmmReg_XMM0);
		x64_value_to_xmm(p, rhs, X64XmmReg_XMM1);
		if (x64_is_double(lbt)) x64_emit_ucomisd(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
		else                    x64_emit_ucomiss(&p->asm_, X64XmmReg_XMM0, X64XmmReg_XMM1);
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
		// ==,<,<= : AND with NP (false when unordered). != : OR with P (true when unordered).
		if (op == Token_CmpEq || op == Token_Lt || op == Token_LtEq) {
			x64_emit_setcc_r(&p->asm_, X64Cc_NP, X64Reg_RCX);
			x64_emit_and_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RCX);
		} else if (op == Token_NotEq) {
			x64_emit_setcc_r(&p->asm_, X64Cc_P, X64Reg_RCX);
			x64_emit_or_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RCX);
		}
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t_bool, X64Reg_RAX);
	}

	// 128-bit integer comparison (two-register); also 16-byte bit_set == / !=. The scalar path below
	// only compares the low 64 bits (op_size clamps to 64) → wrong when the high half differs.
	if (type_size_of(lt) == 16 &&
	    (is_type_integer(lt) ||
	     (lbt != nullptr && lbt->kind == Type_BitSet && (op == Token_CmpEq || op == Token_NotEq)))) {
		bool i128_signed = x64_is_signed_integer(lt);
		i32 la = x64_i128_operand(p, lhs, lt, i128_signed);
		i32 rb = x64_i128_operand(p, rhs, lt, i128_signed);
		if (op == Token_CmpEq || op == Token_NotEq) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la));
			x64_emit_xor_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(rb));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(la + 8));
			x64_emit_xor_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(rb + 8));
			x64_emit_or_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
			x64_emit_setcc_r(&p->asm_, (op == Token_CmpEq) ? X64Cc_E : X64Cc_NE, X64Reg_RAX);
			x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
			return x64v_reg(t_bool, X64Reg_RAX);
		}
		// Ordering: full 128-bit subtract (cmp lo + sbb hi) leaves flags reflecting x-y; setcc picks the
		// condition. a<b≡sub(a,b)L/B; a>b≡sub(b,a)L/B; a>=b≡sub(a,b)GE/AE; a<=b≡sub(b,a)GE/AE.
		i32 x = la, y = rb; X64Cc cc;
		bool ge = (op == Token_GtEq || op == Token_LtEq);
		if (op == Token_Gt || op == Token_LtEq) { x = rb; y = la; }
		if (i128_signed) cc = ge ? X64Cc_GE : X64Cc_L;
		else             cc = ge ? X64Cc_AE : X64Cc_B;
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x));
		x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(y));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x + 8));
		x64_emit_sbb_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(y + 8));
		x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX);
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t_bool, X64Reg_RAX);
	}

	// Integer / bit_set comparison.
	x64_value_to_reg(p, lhs, X64Reg_RAX);
	x64_value_to_reg(p, rhs, X64Reg_RCX);
	X64OpSize sz = x64_op_size_of(lt);
	bool signed_ = x64_is_signed_integer(lt);
	if (lbt != nullptr && lbt->kind == Type_BitSet &&
	    (op == Token_Lt || op == Token_LtEq || op == Token_Gt || op == Token_GtEq)) {
		bool subset = (op == Token_Lt || op == Token_LtEq); // ⊆ vs ⊇
		bool strict = (op == Token_Lt || op == Token_Gt);   // proper subset/superset
		x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RAX);
		x64_emit_and_rr(&p->asm_, sz, X64Reg_RDX, X64Reg_RCX);            // RDX = a & b
		x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RDX, subset ? X64Reg_RAX : X64Reg_RCX);
		x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_R8);                  // (a&b)==a → ⊆ (or ==b → ⊇)
		if (strict) {                                                    // && a != b
			x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
			x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_R9);
			x64_emit_and_rr(&p->asm_, X64OpSize_8, X64Reg_R8, X64Reg_R9);
		}
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_R8);
		return x64v_reg(t_bool, X64Reg_RAX);
	}
	X64Cc cc = X64Cc_E;
	switch (op) {
	case Token_CmpEq: cc = X64Cc_E;  break;
	case Token_NotEq: cc = X64Cc_NE; break;
	case Token_Lt:    cc = signed_ ? X64Cc_L  : X64Cc_B;  break;
	case Token_LtEq:  cc = signed_ ? X64Cc_LE : X64Cc_BE; break;
	case Token_Gt:    cc = signed_ ? X64Cc_G  : X64Cc_A;  break;
	case Token_GtEq:  cc = signed_ ? X64Cc_GE : X64Cc_AE; break;
	default: GB_PANIC("x64 comparison op %d not supported", op);
	}
	x64_emit_cmp_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
	x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX);
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	return x64v_reg(t_bool, X64Reg_RAX);
}

// Unary arithmetic/logical op on an already-built value (mirrors lb_emit_unary_arith): `+` (no-op),
// `-` (negate; float = XOR sign bit), `~` (bitwise not), `!` (logical not → bool). Address-of (`&`) is
// NOT here — it needs the lvalue and is handled in the UnaryExpr case via x64_build_addr.
gb_internal x64Value x64_emit_unary_arith(x64Procedure *p, TokenKind op, x64Value val, Type *t) {
	// Aggregate unary op (array / #simd): apply the scalar op element-wise. Without this, `-v` on a
	// [3]f32 fell to the scalar path below → value_to_reg loaded only the low 8 of 12 bytes into RAX
	// and integer-negated float bits → garbage (THE core:math/noise 3D bug: `a0 := f_sign * -ri` with
	// ri:[3]f32; 2D/[2] and 4D/[4] "worked" only because power-of-2 sizes hid it less). Binary arith
	// already has x64_emit_arith_array; this is its unary counterpart.
	Type *ubt = (t != nullptr) ? base_type(t) : nullptr;
	if (ubt != nullptr && (ubt->kind == Type_Array || ubt->kind == Type_SimdVector)) {
		Type *et = (ubt->kind == Type_Array) ? ubt->Array.elem : ubt->SimdVector.elem;
		i64   n  = (ubt->kind == Type_Array) ? ubt->Array.count : ubt->SimdVector.count;
		i64  esz = type_size_of(et); if (esz <= 0) esz = 1;
		x64Value src = x64_spill_value(p, val, t);       // stable rbp base for the elements
		i32 ro = x64_alloc_local(p, type_size_of(t), type_align_of(t));
		for (i64 i = 0; i < n; i++) {
			X64Mem em = src.mem; em.disp += (i32)(i * esz);
			x64Value nv = x64_emit_unary_arith(p, op, x64v_mem(et, em), et); // scalar elem (no re-entry)
			X64Mem rm = x64_rbp_mem(ro); rm.disp += (i32)(i * esz);
			x64_store_value(p, x64addr(rm, et), nv);
		}
		return x64v_mem(t, x64_rbp_mem(ro));
	}
	// 128-bit integer unary negate / bitwise-not (two-register; the single-register path below only
	// touches the low 64 bits). Needed e.g. for strconv's negative-i128 formatting.
	if (t != nullptr && type_size_of(t) == 16 && is_type_integer(t) && (op == Token_Sub || op == Token_Xor)) {
		i32 vo  = x64_i128_operand(p, val, t, x64_is_signed_integer(t));
		i32 off = x64_alloc_local(p, 16, 16);
		if (op == Token_Xor) { // bitwise NOT each half
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));     x64_emit_not_r(&p->asm_, X64OpSize_64, X64Reg_RAX); x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo + 8)); x64_emit_not_r(&p->asm_, X64OpSize_64, X64Reg_RAX); x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_RAX);
		} else { // negate: 0 - val
			x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
			x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX);
			x64_emit_sub_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));
			x64_emit_sbb_rm(&p->asm_, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(vo + 8));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_RDX);
		}
		return x64v_mem(t, x64_rbp_mem(off));
	}
	switch (op) {
	case Token_Add:
		return val; // unary + is a no-op
	case Token_Sub: {
		if (x64_is_f16(base_type(t))) {
			// f16 is moved as a raw 2-byte integer, so x64_is_float is false — but negate must flip the
			// SIGN BIT (XOR 0x8000), not two's-complement the bits. Was math.trunc_f16 wrong for every
			// negative input (trunc uses -trunc_internal(-f)): NEG turned e.g. -8.07 into +Inf-ish garbage.
			x64_value_to_reg(p, val, X64Reg_RAX);
			x64_emit_xor_ri(&p->asm_, X64OpSize_16, X64Reg_RAX, 0x8000);
			return x64v_reg(t, X64Reg_RAX);
		}
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
		GB_PANIC("x64_emit_unary_arith: unhandled op %d", op);
	}
	return x64v_none();
}

// Load a tagged union's variant index (mirrors lb_emit_union_tag_value): the tag lives at
// `variant_block_size` with width `union_tag_size`, zero-extended into `dst`. `union_mem` is the
// union's base address.
gb_internal void x64_emit_union_tag_value(x64Procedure *p, X64Mem union_mem, Type *ubt, X64Reg dst) {
	X64UnionTag tag = x64_emit_union_tag_ptr(ubt, union_mem);
	if (tag.opsz == X64OpSize_64) x64_emit_mov_rm  (&p->asm_, X64OpSize_64, dst, tag.mem);
	else                          x64_emit_movzx_rm(&p->asm_, tag.opsz,     dst, tag.mem);
}

// `x.(T)` / `s, ok := x.(T)` / `x.?` on a union or `any` (mirrors lb_emit_union_cast + lb_emit_any_cast):
// compute a pointer to the variant data + an `ok` flag (tag/id compare), then zero-init the result and
// copy the value out only on a match (comma-ok form keeps `ok`; single form copies unconditionally).
gb_internal x64Value x64_build_type_assertion(x64Procedure *p, Ast *expr) {
	ast_node(ta, TypeAssertion, expr);
	TypeAndValue tav = expr->tav;
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
				i64 expected_tag = union_variant_index_checked(src_bt, dst_type);
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(src_ea_off)); // RCX = &union
				x64_emit_union_tag_value(p, x64_mem(X64Reg_RCX, 0), src_bt, X64Reg_RAX);
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
}

// `a[lo:hi]` / `a[:]` (mirrors lb_build_slice_expr): auto-deref a pointer source (slice THROUGH it),
// then compute data = base + lo*esz and len = hi - lo over array / slice / dynamic-array /
// multi-pointer / fixed-cap sources; returns the {data,len} result.
gb_internal x64Value x64_build_slice_expr(x64Procedure *p, Ast *expr) {
	ast_node(se, SliceExpr, expr);
	TypeAndValue tav = expr->tav;
		Type *src_raw = x64_typed(se->expr->tav.type);
		Type *src_bt  = base_type(src_raw);
		if (src_bt == nullptr) return x64v_none();

		// Auto-deref a POINTER source: `p[lo:hi]` (p: ^[]T / ^[N]T / ^string) slices THROUGH the
		// pointer (like IndexExpr; mirrors lb_build_addr). The effective aggregate address is the
		// pointer VALUE, not &p. Without this `.data`/`.len` were read from &p / &p+8 (garbage len)
		// — was os.read_directory→…→_split_iterator's `s[:]` (s: ^string) returning a huge len → hang.
		bool via_ptr = is_type_pointer(src_bt);
		i32 src_ea_off = x64_alloc_local(p, 8, 8);
		if (via_ptr) {
			x64Value pv = x64_build_expr(p, se->expr);          // the pointer = aggregate address
			x64_value_to_reg(p, pv, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(src_ea_off), X64Reg_RAX);
			src_raw = type_deref(src_raw);
			src_bt  = base_type(src_raw);
			if (src_bt == nullptr) return x64v_none();
		} else {
			x64Addr src_addr = x64_build_addr(p, se->expr);
			x64_emit_lea(&p->asm_, X64Reg_RAX, src_addr.mem);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(src_ea_off), X64Reg_RAX);
		}

		Type *elem_t = t_u8;
		i64   esz    = 1;
		if      (src_bt->kind == Type_Array)         { elem_t = src_bt->Array.elem;         esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_Slice)          { elem_t = src_bt->Slice.elem;          esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_DynamicArray)   { elem_t = src_bt->DynamicArray.elem;   esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_MultiPointer)   { elem_t = src_bt->MultiPointer.elem;   esz = type_size_of(elem_t); }
		else if (src_bt->kind == Type_FixedCapacityDynamicArray) { elem_t = src_bt->FixedCapacityDynamicArray.elem; esz = type_size_of(elem_t); }
		if (esz <= 0) esz = 1;

		i32 base_ptr_off = x64_alloc_local(p, 8, 8);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
		if (src_bt->kind != Type_Array && src_bt->kind != Type_FixedCapacityDynamicArray) {
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
		} else if (src_bt->kind == Type_FixedCapacityDynamicArray) {
			// len is the int field after the data array (data is inline @0).
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)x64_fca_len_offset(src_bt)));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(hi_off), X64Reg_RAX);
		} else {
			// Load .len from offset +8 of slice/DA struct
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 8));
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(hi_off), X64Reg_RAX);
		}

		// Bounds check (mirrors lb_emit_slice_bounds_check / multi_pointer_slice_expr_error): verify
		// 0 ≤ lo ≤ hi ≤ len against the SOURCE length before forming the result slice.
		if (src_bt->kind == Type_MultiPointer) {
			if (se->high != nullptr) {
				x64_emit_multi_pointer_slice_bounds_check(p, ast_token(expr).pos,
					x64v_mem(t_int, x64_rbp_mem(lo_off)), x64v_mem(t_int, x64_rbp_mem(hi_off)));
			}
		} else {
			i32 len_off = x64_alloc_local(p, 8, 8);
			if (src_bt->kind == Type_Array) {
				x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)src_bt->Array.count);
			} else if (src_bt->kind == Type_FixedCapacityDynamicArray) {
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)x64_fca_len_offset(src_bt)));
			} else { // slice / dynarray / string: .len @8
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(src_ea_off));
				x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 8));
			}
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(len_off), X64Reg_RAX);
			x64_emit_slice_bounds_check(p, ast_token(expr).pos,
				x64v_mem(t_int, x64_rbp_mem(lo_off)), x64v_mem(t_int, x64_rbp_mem(hi_off)),
				x64v_mem(t_int, x64_rbp_mem(len_off)), se->low != nullptr);
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
}

// Short-circuit `a && b` / `a || b` in a VALUE context (result materialised as a bool 0/1); mirrors
// lb_emit_logical_binary_expr. Condition contexts use x64_build_cond for direct branching instead.
gb_internal x64Value x64_emit_logical_binary_expr(x64Procedure *p, TokenKind op, Ast *left, Ast *right, Type *final_type) {
	x64Value lv = x64_build_expr(p, left);
	x64_value_to_reg(p, lv, X64Reg_RAX);
	x64_emit_test_rr(&p->asm_, x64_op_size_of(left->tav.type), X64Reg_RAX, X64Reg_RAX);

	isize lbl_end    = x64_label_alloc(&p->asm_);
	isize lbl_result = x64_label_alloc(&p->asm_);

	if (op == Token_CmpAnd) x64_emit_jcc(&p->asm_, X64Cc_E,  lbl_result); // false → skip right
	else                    x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_result); // true  → skip right

	x64Value rv = x64_build_expr(p, right);
	x64_value_to_reg(p, rv, X64Reg_RAX);
	x64_emit_test_rr(&p->asm_, x64_op_size_of(right->tav.type), X64Reg_RAX, X64Reg_RAX);
	x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RAX);
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	x64_emit_jmp(&p->asm_, lbl_end);

	x64_label_bind(&p->asm_, lbl_result);
	if (op == Token_CmpAnd) x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
	else                    x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
	x64_label_bind(&p->asm_, lbl_end);
	return x64v_reg(final_type ? final_type : t_bool, X64Reg_RAX);
}

// `elem in set` / `elem not_in set` for a bit_set (mirrors lb_build_binary_in). Takes the Ast operands
// (not pre-built values like lb): x64 must build+spill the SET after the elem — keeping the set live in
// a register made value_to_reg a no-op, degenerating the test to `mask & mask` (every elem read "in").
gb_internal x64Value x64_build_binary_in(x64Procedure *p, Ast *left, Ast *right, TokenKind op) {
	// type_deref: `k in m` also accepts a POINTER to the map/bit_set (auto-deref), e.g. `k in section`
	// where `section := &m[k1]` is a ^map. base_type alone leaves rt as Type_Pointer → the Type_Map check
	// below misses → the "unknown type" fallback returned false → duplicate keys never detected (was
	// core:text/i18n's Duplicate_Key never firing). x64_map_addr_of handles the pointer operand itself.
	Type *rt = base_type(x64_typed(type_deref(right->tav.type)));
	if (rt != nullptr && rt->kind == Type_BitSet) {
		Type     *it = bit_set_to_int(rt);
		X64OpSize sz = x64_op_size_of(it);
		i64 lower = rt->BitSet.lower;

		x64Value lv = x64_build_expr(p, left);
		Type *et   = lv.type ? lv.type : t_int;
		x64Value elem_m = x64_spill_value(p, lv, et);
		x64Value set_m  = x64_spill_value(p, x64_build_expr(p, right), rt);

		x64_value_to_reg(p, elem_m, X64Reg_RCX);
		x64_emit_bit_set_mask(p, sz, lower);          // RAX = 1 << (elem - lower)

		x64_value_to_reg(p, set_m, X64Reg_RCX);       // RCX = set value (mask stays in RAX)
		x64_emit_and_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RCX);
		x64_emit_test_rr(&p->asm_, sz, X64Reg_RAX, X64Reg_RAX);
		X64Cc cc = (op == Token_in) ? X64Cc_NE : X64Cc_E;
		x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX);
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t_bool, X64Reg_RAX);
	}
	// `key in map` / `key not_in map`: get_ptr (^V, nil if absent), then test against nil
	// (mirrors lb_build_binary_in Type_Map). `right` is the map, `left` the key.
	if (rt != nullptr && rt->kind == Type_Map) {
		x64Value map_ptr = x64_map_addr_of(p, right, rt);
		x64Value ptr = x64_internal_dynamic_map_get_ptr(p, map_ptr, rt, left); // ^V, spilled; nil if absent
		x64_value_to_reg(p, ptr, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
		X64Cc cc = (op == Token_in) ? X64Cc_NE : X64Cc_E;
		x64_emit_setcc_r(&p->asm_, cc, X64Reg_RAX);
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t_bool, X64Reg_RAX);
	}

	// Unknown type — not yet implemented; return false
	(void)x64_build_expr(p, left);
	x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
	return x64v_reg(t_bool, X64Reg_RAX);
}

// `x == nil` / `x != nil` (mirrors lb_emit_comp_against_nil) for the cases the generic scalar compare
// gets wrong: a TAGGED union (nil ⇔ tag==0, tag lives AFTER the variant block, so comparing the first
// word — the variant DATA — reads stale bytes as non-nil) and #shared_nil (nil ⇔ the variant data
// block is all-zero; the tag is unreliable). Returns {none} when `nn` is not such a union, so the
// caller falls through to the scalar path (pointer/slice/map/maybe-ptr keep their discriminant @0).
gb_internal x64Value x64_emit_comp_against_nil(x64Procedure *p, TokenKind op, Ast *nn) {
	Type *nbt = base_type(nn->tav.type);
	if (nbt == nullptr || nbt->kind != Type_Union ||
	    is_type_union_maybe_pointer(nbt) || type_size_of(nbt) <= 0) {
		return x64v_none();
	}
	// Build the VALUE (handles call results / non-lvalues — can't take &of an rvalue); a tagged union
	// is an aggregate so this is a Mem, but spill defensively.
	x64Value uv = x64_build_expr(p, nn);
	if (uv.kind != x64Value_Mem) {
		i32 off = x64_alloc_local(p, type_size_of(nbt), type_align_of(nbt));
		x64_store_value(p, x64addr(x64_rbp_mem(off), nbt), uv);
		uv = x64v_mem(nbt, x64_rbp_mem(off));
	}
	if (nbt->Union.kind == UnionType_shared_nil) {
		i64 vbs = nbt->Union.variant_block_size; if (vbs <= 0) vbs = type_size_of(nbt);
		x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RCX, X64Reg_RCX);
		i64 b = 0;
		while (vbs - b >= 8) { x64_emit_or_chunk(p, uv, b, X64OpSize_64, false); b += 8; }
		if (vbs - b >= 4)    { x64_emit_or_chunk(p, uv, b, X64OpSize_32, false); b += 4; }
		if (vbs - b >= 2)    { x64_emit_or_chunk(p, uv, b, X64OpSize_16, true);  b += 2; }
		if (vbs - b >= 1)    { x64_emit_or_chunk(p, uv, b, X64OpSize_8,  true); }
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX);
		x64_emit_setcc_r(&p->asm_, (op == Token_CmpEq) ? X64Cc_E : X64Cc_NE, X64Reg_RAX);
		x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t_bool, X64Reg_RAX);
	}
	x64_emit_union_tag_value(p, uv.mem, nbt, X64Reg_RCX);
	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX);
	x64_emit_setcc_r(&p->asm_, (op == Token_CmpEq) ? X64Cc_E : X64Cc_NE, X64Reg_RAX);
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	return x64v_reg(t_bool, X64Reg_RAX);
}

// `&x.(T)` in COMMA-OK context → (^T, bool): the IN-PLACE variant address (nil on a tag/id mismatch)
// plus ok. Mirrors lb_build_unary_and's Ast_TypeAssertion tuple branch. NOT a copy — the generic
// build_unary_and returned a single ^T with no `ok`, so `if v, ok := &u.(T); ok {…}` saw ok=0 and the
// body never ran (blick: `if v,ok := &stream.variant.(Video_Stream); ok { slot_map.init(&v.decode_map) }`
// never initialised the decode_map → allocate_decoder bailed on nil .items → no frames → black monitor).
gb_internal x64Value x64_build_addr_type_assertion_tuple(x64Procedure *p, Ast *ta_expr, Type *tuple_type) {
	ast_node(ta, TypeAssertion, ta_expr);
	Type *dst_type = type_of_expr(ta->type);
	Type *src_bt   = base_type(x64_typed(ta->expr->tav.type));
	i64   okoff    = type_offset_of(tuple_type, 1);

	i32 res_off  = x64_alloc_local(p, type_size_of(tuple_type), type_align_of(tuple_type));
	x64_zero_mem(p, x64_rbp_mem(res_off), type_size_of(tuple_type));
	i32 dptr_off = x64_alloc_local(p, 8, 8);
	i32 ok_off   = x64_alloc_local(p, 1, 1);

	// Effective address of the union/any (in-place, NOT a copy).
	x64Addr src_addr = x64_build_addr(p, ta->expr);
	x64_emit_lea(&p->asm_, X64Reg_RAX, src_addr.mem);
	i32 ea_off = x64_alloc_local(p, 8, 8);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ea_off), X64Reg_RAX);

	if (src_bt != nullptr && is_type_any(src_bt)) {
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ea_off));
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 0)); // any.data → dptr
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(dptr_off), X64Reg_RCX);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, 8)); // any.id
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)type_hash_canonical_type(dst_type));
		x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
		x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);
	} else {
		// union: data_ptr is &union (variant data @ offset 0), uniform for tagged & maybe-pointer.
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(ea_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(dptr_off), X64Reg_RCX);
		if (is_type_union_maybe_pointer(src_bt)) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RCX, 0)); // ptr value
			x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
			x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RAX);
		} else {
			i64 expected = union_variant_index_checked(src_bt, dst_type);
			x64_emit_union_tag_value(p, x64_mem(X64Reg_RCX, 0), src_bt, X64Reg_RAX); // RAX = tag
			x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i32)expected);
			x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);
		}
	}
	x64_emit_mov_mr(&p->asm_, X64OpSize_8, x64_rbp_mem(ok_off), X64Reg_RAX); // ok (AL)

	// field0 = ok ? data_ptr : nil ; field1 = ok
	isize skip = x64_label_alloc(&p->asm_);
	x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_RAX, x64_rbp_mem(ok_off));
	x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	x64_emit_jcc(&p->asm_, X64Cc_E, skip);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(dptr_off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RCX);
	x64_label_bind(&p->asm_, skip);
	x64_emit_mov_rm(&p->asm_, X64OpSize_8, X64Reg_RAX, x64_rbp_mem(ok_off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_8, x64_rbp_mem(res_off + (i32)okoff), X64Reg_RAX);
	return x64v_mem(tuple_type, x64_rbp_mem(res_off));
}

// `&x` address-of as a VALUE (mirrors lb_build_unary_and): the address of the operand's lvalue.
// Special-cases &CompoundLit in the startup runtime — the address escapes (e.g. `INT_ZERO := &Int{}`),
// so it's backed by a STATIC global rather than a stack local (mirrors lb_build_expr's is_startup path).
gb_internal x64Value x64_build_unary_and(x64Procedure *p, Ast *expr) {
	ast_node(ue, UnaryExpr, expr);
	Ast *sub = unparen_expr(ue->expr);
	// `&x.(T)` comma-ok → (^T, bool) tuple with the in-place pointer + tag-match ok.
	if (sub != nullptr && sub->kind == Ast_TypeAssertion && is_type_tuple(expr->tav.type)) {
		return x64_build_addr_type_assertion_tuple(p, sub, expr->tav.type);
	}
	// `&m[k]` — map element pointer (^V, nil if absent; comma-ok → (^V, bool)). The generic build_addr
	// path has no Type_Map case → returned a garbage temp. Mirrors lb_build_addr(lbAddr_Map) + get_ptr.
	if (sub != nullptr && sub->kind == Ast_IndexExpr) {
		Ast *ix = sub;
		Type *ibt = ix->IndexExpr.expr->tav.type ? base_type(type_deref(ix->IndexExpr.expr->tav.type)) : nullptr;
		if (ibt != nullptr && ibt->kind == Type_Map) {
			return x64_build_map_index_ptr(p, ix->IndexExpr.expr, ix->IndexExpr.index, expr->tav.type);
		}
	}
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

// Anonymous proc literal used as a value (mirrors lb_generate_anonymous_proc_lit): generate the proc
// on-demand (enqueue into the module proc-queue) and return its address. Without this a `proc(){…}`
// value was none → a NULL fn pointer → DEP exec at 0 when called.
// Materialize an anonymous `proc(){…}` literal as a real proc entity (deterministic name, enqueued
// into the module's proc-queue for compilation) and return it. Shared by the expr path (value use)
// and the const path (a proc-lit as a global-variable initializer). Idempotent via pl->decl->entity.
gb_internal Entity *x64_anon_proc_entity(x64Module *m, Ast *expr) {
	ast_node(pl, ProcLit, expr);
	Entity *e = (pl->decl != nullptr) ? pl->decl->entity.load() : nullptr;
	if (e == nullptr && pl->decl != nullptr) {
		static std::atomic<i32> x64_anon_proc_id;
		char buf[24];
		gb_snprintf(buf, gb_size_of(buf), "anon_proc$%d", 1 + (int)x64_anon_proc_id.fetch_add(1));
		Token token = {};
		token.pos    = ast_token(expr).pos;
		token.kind   = Token_Ident;
		token.string = copy_string(permanent_allocator(), make_string((u8 const *)buf, gb_strlen(buf)));
		Entity *ne = alloc_entity_procedure(nullptr, token, type_of_expr(expr), pl->tags);
		ne->file             = expr->file();
		ne->scope            = ne->file ? ne->file->scope : nullptr;
		ne->decl_info        = pl->decl;
		ne->parent_proc_decl = pl->decl->parent;
		ne->Procedure.is_anonymous = true;
		ne->flags |= EntityFlag_ProcBodyChecked;
		Entity *expected = nullptr;
		if (pl->decl->entity.compare_exchange_strong(expected, ne)) {
			e = ne;
			mpsc_enqueue(&m->proc_queue, e); // compiled by the module's proc-queue drain
		} else {
			e = expected; // another thread generated it first
		}
	}
	return e;
}

gb_internal x64Value x64_generate_anonymous_proc_lit(x64Procedure *p, Ast *expr) {
	TypeAndValue tav = expr->tav;
	Entity *e = x64_anon_proc_entity(p->module, expr);
	if (e == nullptr) return x64v_none();
	x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(e));
	return x64v_reg(x64_typed(tav.type ? tav.type : e->type), X64Reg_RAX);
}

// `x or_else y` (mirrors lb_emit_or_else): if x has a value, result = x's value part; else result = y.
gb_internal x64Value x64_emit_or_else(x64Procedure *p, Ast *expr) {
	ast_node(oe, OrElseExpr, expr);
	TypeAndValue tav = expr->tav;
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
}

// `x or_return` (mirrors lb_emit_or_return): if x's indicator signals failure, return it from the
// current procedure (running defers, writing the indicator into the last result via the named /
// by-pointer / single-register path); otherwise continue with x's value part.
gb_internal x64Value x64_emit_or_return(x64Procedure *p, Ast *expr) {
	ast_node(oe, OrReturnExpr, expr);
	TypeAndValue tav = expr->tav;
	// Build inner expression (call, type assertion, etc.)
	x64Value rv = x64_build_expr(p, oe->expr);

	// Extract rhs (the bool/ok indicator) and lhs (the success value, if any)
	x64Value rhs = rv;
	x64Value lhs = x64v_none();

	Type *inner_type = rv.type ? base_type(rv.type) : nullptr;
	if (inner_type && is_type_tuple(inner_type)) {
		int count = (int)inner_type->Tuple.variables.count;
		if (rv.kind == x64Value_Mem) {
			// rhs = the last tuple field (the bool/ok or error indicator).
			x64Addr rhs_a = x64_emit_tuple_ep(p, rv, count - 1);
			rhs = x64v_mem(rhs_a.type, rhs_a.mem);
			if (tav.type != nullptr && count >= 2) {
				if (count == 2) {
					// single success value = field 0.
					x64Addr lhs_a = x64_emit_tuple_ep(p, rv, 0);
					lhs = x64v_mem(lhs_a.type, lhs_a.mem);
				} else {
					// count > 2: assemble the (count-1)-value success TUPLE into a local of tv.type
					// (mirrors lb_emit_try_lhs_rhs). Was only capturing field 0 → 3+-value `or_return`
					// (e.g. strconv `parse_components(str) or_return`) lost every value → parse failed.
					i32 sub = x64_alloc_local(p, type_size_of(tav.type), gb_max(type_align_of(tav.type), (i64)1));
					for (int i = 0; i < count - 1; i++) {
						x64Addr fa = x64_emit_tuple_ep(p, rv, i);
						x64_store_value(p, x64addr(x64_rbp_mem(sub + (i32)type_offset_of(tav.type, i)), fa.type),
						                x64v_mem(fa.type, fa.mem));
					}
					lhs = x64v_mem(tav.type, x64_rbp_mem(sub));
				}
			}
		}
		// else rv.kind == x64Value_None: inner expr didn't expose result buffer;
		// x64_value_to_reg on x64v_none() will XOR-zero RAX → always fail path
	}

	// Spill rhs to a stable stack slot (FULL size) so the failure path can return it and deferred stmts
	// can't clobber it. The old code spilled only op_size (≤8) bytes via value_to_reg → a union/aggregate
	// error was truncated AND value_to_reg loaded the wrong bytes.
	Type *rhs_t  = rhs.type ? rhs.type : t_bool;
	Type *rhs_bt = base_type(rhs_t);
	X64OpSize rhs_opsz = x64_op_size_of(rhs_t);
	i32 rhs_spill = x64_alloc_local(p, gb_max(type_size_of(rhs_t), (i64)1), gb_max(type_align_of(rhs_t), (i64)1));
	x64_store_value(p, x64addr(x64_rbp_mem(rhs_spill), rhs_t), rhs);
	x64Value rhs_mem = x64v_mem(rhs_t, x64_rbp_mem(rhs_spill));

	// Success condition depends on the indicator TYPE (mirrors lb_emit_try_has_value): a BOOLEAN ok-flag
	// succeeds when TRUE; an ERROR / nil-able value when NIL. A tagged UNION error is nil ⟺ tag==0 — the
	// tag lives AFTER the variant block, so testing the first word (the variant DATA) is wrong. Was THE
	// core:flags `set_option() or_return` bug: a non-nil Parse_Error union read as nil → never propagated.
	// #shared_nil: nil ⟺ variant block all-zero. Enum/pointer/maybe-pointer errors keep a discriminant @0.
	isize lbl_continue = x64_label_alloc(&p->asm_);
	if (is_type_boolean(rhs_t)) {
		x64_value_to_reg(p, rhs_mem, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, rhs_opsz, X64Reg_RAX, X64Reg_RAX);
		x64_emit_jcc(&p->asm_, X64Cc_NE, lbl_continue); // bool true → success
	} else if (rhs_bt != nullptr && rhs_bt->kind == Type_Union && !is_type_union_maybe_pointer(rhs_bt)) {
		if (rhs_bt->Union.kind == UnionType_shared_nil) {
			i64 vbs = rhs_bt->Union.variant_block_size; if (vbs <= 0) vbs = type_size_of(rhs_bt);
			x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RCX, X64Reg_RCX);
			i64 b = 0;
			while (vbs - b >= 8) { x64_emit_or_chunk(p, rhs_mem, b, X64OpSize_64, false); b += 8; }
			if (vbs - b >= 4)    { x64_emit_or_chunk(p, rhs_mem, b, X64OpSize_32, false); b += 4; }
			if (vbs - b >= 2)    { x64_emit_or_chunk(p, rhs_mem, b, X64OpSize_16, true);  b += 2; }
			if (vbs - b >= 1)    { x64_emit_or_chunk(p, rhs_mem, b, X64OpSize_8,  true); }
			x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX);
		} else {
			x64_emit_union_tag_value(p, rhs_mem.mem, rhs_bt, X64Reg_RCX);
			x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RCX);
		}
		x64_emit_jcc(&p->asm_, X64Cc_E, lbl_continue);  // tag/block == 0 → nil → success
	} else {
		x64_value_to_reg(p, rhs_mem, X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, rhs_opsz, X64Reg_RAX, X64Reg_RAX);
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
					if (type_size_of(rhs_t) > 8) { // aggregate (union) error: copy the FULL value
						x64_store_value(p, x64addr(x64_rbp_mem(*loff), last_res->type), rhs_mem);
					} else {
						x64_emit_mov_rm(&p->asm_, rhs_opsz, X64Reg_RAX,
						                x64_rbp_mem(rhs_spill));
						x64_emit_mov_mr(&p->asm_, x64_op_size_of(last_res->type),
						                x64_rbp_mem(*loff), X64Reg_RAX);
					}
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
				if (type_size_of(rhs_t) > 8) { // aggregate (union) error: copy the FULL value
					i32 dptr = x64_alloc_local(p, 8, 8);
					x64_emit_lea(&p->asm_, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)last_off));
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(dptr), X64Reg_RAX);
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(dptr));
					x64_store_value(p, x64addr(x64_mem(X64Reg_RAX, 0), last_res->type), rhs_mem);
				} else {
					x64_emit_mov_rm(&p->asm_, rhs_opsz, X64Reg_RCX,
					                x64_rbp_mem(rhs_spill));
					x64_emit_mov_mr(&p->asm_, x64_op_size_of(last_res->type),
					                x64_mem(X64Reg_RAX, (i32)last_off), X64Reg_RCX);
				}
			}
		} else {
			// Single register return: load rhs into RAX (extended to the full register).
			x64_value_to_reg(p, x64v_mem(rhs_t, x64_rbp_mem(rhs_spill)), X64Reg_RAX);
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
}

// Packed binary-op kinds for #simd lowering.

// Does SSE have a SINGLE packed instruction for (op, elem)? (SSE lacks packed integer division and
// 8/64-bit integer multiply; integer min/max only ≤32-bit.) Callers fall back to a scalar lane loop.
gb_internal bool x64_simd_packed_ok(int op, Type *elem) {
	bool flt = is_type_float(elem);
	i64  esz = type_size_of(elem);
	switch (op) {
	case X64SimdBin_Add: case X64SimdBin_Sub: return flt || esz == 1 || esz == 2 || esz == 4 || esz == 8;
	case X64SimdBin_Mul: return flt || esz == 2 || esz == 4;
	case X64SimdBin_Div: return flt;
	case X64SimdBin_And: case X64SimdBin_Or: case X64SimdBin_Xor: case X64SimdBin_AndN: return true;
	case X64SimdBin_Min: case X64SimdBin_Max: return flt || esz == 1 || esz == 2 || esz == 4;
	}
	return false;
}

// Map an `llvm.<op>.f<width>` math intrinsic (used by core:math's foreign block — sin/cos/pow/exp/
// fmuladd, etc.) to the equivalent C-runtime (ucrt) symbol. LLVM lowers these intrinsics to libm
// calls / hardware; the x64 backend has no LLVM, so it emits a real call to the CRT (already linked).
// Returns {} for anything unmapped (e.g. f16 variants, bit-manipulation intrinsics) → caller no-ops.
gb_internal String x64_map_llvm_math_intrinsic(String ln) {
	static struct { char const *llvm; char const *libm; } const table[] = {
		{"llvm.sin.f64","sin"},   {"llvm.sin.f32","sinf"},
		{"llvm.cos.f64","cos"},   {"llvm.cos.f32","cosf"},
		{"llvm.tan.f64","tan"},   {"llvm.tan.f32","tanf"},
		{"llvm.pow.f64","pow"},   {"llvm.pow.f32","powf"},
		{"llvm.exp.f64","exp"},   {"llvm.exp.f32","expf"},
		{"llvm.exp2.f64","exp2"}, {"llvm.exp2.f32","exp2f"},
		{"llvm.log.f64","log"},   {"llvm.log.f32","logf"},
		{"llvm.log2.f64","log2"}, {"llvm.log2.f32","log2f"},
		{"llvm.log10.f64","log10"},{"llvm.log10.f32","log10f"},
		{"llvm.fma.f64","fma"},   {"llvm.fma.f32","fmaf"},
		{"llvm.fmuladd.f64","fma"},{"llvm.fmuladd.f32","fmaf"},
		{"llvm.floor.f64","floor"},{"llvm.floor.f32","floorf"},
		{"llvm.ceil.f64","ceil"}, {"llvm.ceil.f32","ceilf"},
		{"llvm.trunc.f64","trunc"},{"llvm.trunc.f32","truncf"},
		{"llvm.round.f64","round"},{"llvm.round.f32","roundf"},
		{"llvm.rint.f64","rint"}, {"llvm.rint.f32","rintf"},
		{"llvm.nearbyint.f64","nearbyint"},{"llvm.nearbyint.f32","nearbyintf"},
		{"llvm.fabs.f64","fabs"}, {"llvm.fabs.f32","fabsf"},
		{"llvm.sqrt.f64","sqrt"}, {"llvm.sqrt.f32","sqrtf"},
		{"llvm.copysign.f64","copysign"},{"llvm.copysign.f32","copysignf"},
		{"llvm.minnum.f64","fmin"},{"llvm.minnum.f32","fminf"},
		{"llvm.maxnum.f64","fmax"},{"llvm.maxnum.f32","fmaxf"},
	};
	for (isize i = 0; i < gb_count_of(table); i++) {
		String l = make_string_c(table[i].llvm);
		if (ln.len == l.len && gb_strncmp((char const *)ln.text, table[i].llvm, ln.len) == 0) {
			return make_string_c(table[i].libm);
		}
	}
	return {};
}

// Compute an `llvm.<op>.f16` math intrinsic (no ucrt symbol for f16) by promoting to f32: convert
// each f16 arg → f32, call the .f32 ucrt fn, convert the f32 result → f16. Was: unmapped llvm.*.f16
// no-op'd → 0, so math.pow(2, f16(n)) returned 0. Returns none if not an f16 intrinsic / .f32 unmapped.
gb_internal x64Value x64_try_llvm_f16_intrinsic(x64Procedure *p, AstCallExpr *ce, String fl, Type *result_type) {
	if (fl.len < 5 || gb_strncmp((char const *)fl.text + fl.len - 4, ".f16", 4) != 0) return x64v_none();
	int argc = (int)ce->args.count;
	if (argc < 1 || argc > 3) return x64v_none();
	char buf[64];
	if (fl.len >= (isize)gb_size_of(buf)) return x64v_none();
	gb_memmove(buf, fl.text, fl.len);
	buf[fl.len - 2] = '3'; buf[fl.len - 1] = '2'; // ".f16" → ".f32"
	String f32sym = x64_map_llvm_math_intrinsic(make_string((u8 const *)buf, fl.len));
	if (f32sym.len == 0) return x64v_none();
	x64Value fargs[3];
	for (int i = 0; i < argc; i++) {
		x64Value a = x64_emit_conv(p, x64_build_expr(p, ce->args[i]), x64_typed(ce->args[i]->tav.type), t_f32);
		fargs[i] = x64_spill_value(p, a, t_f32);
	}
	Type *ptypes[3] = { t_f32, t_f32, t_f32 };
	Type *ftype = alloc_type_proc_from_types(ptypes, (unsigned)argc, t_f32, false, ProcCC_CDecl);
	x64Value r = x64_emit_call(p, f32sym, ftype, fargs, argc); // f32 in XMM0
	return x64_emit_conv(p, r, t_f32, result_type);            // f32 → f16
}

// AVX2 (256-bit YMM) SIMD codegen is gated on the SAME target-feature query the LLVM backend uses
// (-target-features:avx2 or a -microarch implying it), so a given source behaves identically on both
// backends (LLVM↔x64 parity). When false, the legacy 128-bit SSE path is emitted.
gb_internal bool x64_use_avx2(void) {
	return check_target_feature_is_enabled(str_lit("avx2"), nullptr);
}

// AVX VEX-encoded packed binop, width-parameterized: XMM0/YMM0 = XMM0/YMM0 <op> XMM1/YMM1. Same
// register model as the destructive SSE x64_emit_simd_binop, so all caller operand-ordering carries
// over unchanged; only the width (L) and encoding (VEX, non-destructive 3-operand) differ. opcodes
// mirror the SSE emitter table (pp: 0=none/ps 1=66; mm: 1=0F 2=0F38; W=0/WIG).
gb_internal void x64_emit_vsimd_binop(X64Assembler *a, bool ymm, int op, Type *elem) {
	bool flt = is_type_float(elem);
	bool uns = is_type_unsigned(elem);
	i64  esz = type_size_of(elem);
	bool d64 = esz == 8;
	X64XmmReg d = X64XmmReg_XMM0, v = X64XmmReg_XMM0, s = X64XmmReg_XMM1; // d = v <op> s
	u8 pp = 1, mm = 1, opc = 0; // default: 66 0F (packed integer)
	switch (op) {
	case X64SimdBin_Add: if (flt) { pp=d64?1u:0u; opc=0x58u; } else { opc = esz==1?0xFCu:esz==2?0xFDu:esz==4?0xFEu:0xD4u; } break;
	case X64SimdBin_Sub: if (flt) { pp=d64?1u:0u; opc=0x5Cu; } else { opc = esz==1?0xF8u:esz==2?0xF9u:esz==4?0xFAu:0xFBu; } break;
	case X64SimdBin_Mul: if (flt) { pp=d64?1u:0u; opc=0x59u; } else if (esz==2) { opc=0xD5u; } else { mm=2; opc=0x40u; } break;
	case X64SimdBin_Div: pp=d64?1u:0u; opc=0x5Eu; break;
	case X64SimdBin_And: opc=0xDBu; break;
	case X64SimdBin_Or:  opc=0xEBu; break;
	case X64SimdBin_Xor: opc=0xEFu; break;
	case X64SimdBin_AndN:opc=0xDFu; break; // vpandn d,v,s = (~v)&s; caller sets v=b,s=a → a&~b
	case X64SimdBin_Min:
		if (flt) { pp=d64?1u:0u; opc=0x5Du; }
		else if (esz==1) { if (uns) opc=0xDAu; else { mm=2; opc=0x38u; } }
		else if (esz==2) { if (uns) { mm=2; opc=0x3Au; } else opc=0xEAu; }
		else             { mm=2; opc = uns?0x3Bu:0x39u; }
		break;
	case X64SimdBin_Max:
		if (flt) { pp=d64?1u:0u; opc=0x5Fu; }
		else if (esz==1) { if (uns) opc=0xDEu; else { mm=2; opc=0x3Cu; } }
		else if (esz==2) { if (uns) { mm=2; opc=0x3Eu; } else opc=0xEEu; }
		else             { mm=2; opc = uns?0x3Fu:0x3Du; }
		break;
	}
	x64_emit_vex_rr(a, pp, mm, opc, false, ymm, d, v, s);
}

// VEX vmovups load (XMM0/YMM0/1 ← mem) and store (mem ← XMM0/YMM0). pure 256/128-bit unaligned move.
gb_internal void x64_emit_vmovups_rm(X64Assembler *a, bool ymm, X64XmmReg dst, X64Mem src) {
	x64_emit_vex_rm(a, 0u, 1u, 0x10u, false, ymm, dst, X64XmmReg_XMM0, src);
}
gb_internal void x64_emit_vmovups_mr(X64Assembler *a, bool ymm, X64Mem dst, X64XmmReg src) {
	x64_emit_vex_mr(a, 0u, 1u, 0x11u, false, ymm, dst, X64XmmReg_XMM0, src);
}

// Emit XMM0 = XMM0 <op> XMM1 (packed) for the given lane type. Assumes x64_simd_packed_ok.
gb_internal void x64_emit_simd_binop(X64Assembler *a, int op, Type *elem) {
	bool flt = is_type_float(elem);
	bool d64 = type_size_of(elem) == 8;
	bool uns = is_type_unsigned(elem);
	i64  esz = type_size_of(elem);
	X64XmmReg d = X64XmmReg_XMM0, s = X64XmmReg_XMM1;
	switch (op) {
	case X64SimdBin_Add:
		if (flt) { d64 ? x64_emit_addpd(a, d, s) : x64_emit_addps(a, d, s); return; }
		if (esz == 1) x64_emit_paddb(a, d, s); else if (esz == 2) x64_emit_paddw(a, d, s); else if (esz == 4) x64_emit_paddd(a, d, s); else x64_emit_paddq(a, d, s);
		return;
	case X64SimdBin_Sub:
		if (flt) { d64 ? x64_emit_subpd(a, d, s) : x64_emit_subps(a, d, s); return; }
		if (esz == 1) x64_emit_psubb(a, d, s); else if (esz == 2) x64_emit_psubw(a, d, s); else if (esz == 4) x64_emit_psubd(a, d, s); else x64_emit_psubq(a, d, s);
		return;
	case X64SimdBin_Mul:
		if (flt) { d64 ? x64_emit_mulpd(a, d, s) : x64_emit_mulps(a, d, s); return; }
		if (esz == 2) x64_emit_pmullw(a, d, s); else x64_emit_pmulld(a, d, s);
		return;
	case X64SimdBin_Div: d64 ? x64_emit_divpd(a, d, s) : x64_emit_divps(a, d, s); return;
	case X64SimdBin_And:  x64_emit_pand(a, d, s);  return;
	case X64SimdBin_Or:   x64_emit_por (a, d, s);  return;
	case X64SimdBin_Xor:  x64_emit_pxor(a, d, s);  return;
	case X64SimdBin_AndN: x64_emit_pandn(a, d, s); return; // (~d)&s; caller loads d=b, s=a → a&~b
	case X64SimdBin_Min:
		if (flt) { d64 ? x64_emit_minpd(a, d, s) : x64_emit_minps(a, d, s); return; }
		if (esz == 1) { uns ? x64_emit_pminub(a, d, s) : x64_emit_pminsb(a, d, s); }
		else if (esz == 2) { uns ? x64_emit_pminuw(a, d, s) : x64_emit_pminsw(a, d, s); }
		else { uns ? x64_emit_pminud(a, d, s) : x64_emit_pminsd(a, d, s); }
		return;
	case X64SimdBin_Max:
		if (flt) { d64 ? x64_emit_maxpd(a, d, s) : x64_emit_maxps(a, d, s); return; }
		if (esz == 1) { uns ? x64_emit_pmaxub(a, d, s) : x64_emit_pmaxsb(a, d, s); }
		else if (esz == 2) { uns ? x64_emit_pmaxuw(a, d, s) : x64_emit_pmaxsw(a, d, s); }
		else { uns ? x64_emit_pmaxud(a, d, s) : x64_emit_pmaxsd(a, d, s); }
		return;
	}
}

// res[chunk] = a[chunk] OP b[chunk], packed. AVX2 → 256-bit YMM chunks + a 128-bit tail; else 128-bit
// SSE. `swap` loads b into the dst reg and a into the src reg (for AndN: vpandn/pandn = (~dst)&src, so
// dst=b,src=a yields a&~b). Caller must ensure x64_simd_packed_ok(op, elem); total is a multiple of 16.
gb_internal void x64_simd_copy_lane(x64Procedure *p, X64Mem dst, X64Mem src, i64 esz); // defined below

gb_internal x64Value x64_simd_binop_vec(x64Procedure *p, int op, Type *vt, Type *rt, Type *elem, X64Mem amem, X64Mem bmem, bool swap) {
	X64Assembler *a = &p->asm_;
	bool avx = x64_use_avx2();
	i64 total = type_size_of(vt);
	i32 res = x64_alloc_local(p, gb_max(total, (i64)16), gb_max(type_align_of(vt), (i64)16));
	// Sub-128-bit vector: copy operands into zero-padded 16-byte temps and do ONE 128-bit packed op (the
	// extra padding lanes are computed but ignored). Still a real SIMD instruction, not scalar.
	if (total < 16) {
		i32 ta = x64_alloc_local(p, 16, 16); x64_zero_mem(p, x64_rbp_mem(ta), 16); x64_simd_copy_lane(p, x64_rbp_mem(ta), swap ? bmem : amem, total);
		i32 tb = x64_alloc_local(p, 16, 16); x64_zero_mem(p, x64_rbp_mem(tb), 16); x64_simd_copy_lane(p, x64_rbp_mem(tb), swap ? amem : bmem, total);
		if (avx) {
			x64_emit_vmovups_rm(a, false, X64XmmReg_XMM0, x64_rbp_mem(ta));
			x64_emit_vmovups_rm(a, false, X64XmmReg_XMM1, x64_rbp_mem(tb));
			x64_emit_vsimd_binop(a, false, op, elem);
			x64_emit_vmovups_mr(a, false, x64_rbp_mem(res), X64XmmReg_XMM0);
		} else {
			x64_emit_movups_rm(a, X64XmmReg_XMM0, x64_rbp_mem(ta));
			x64_emit_movups_rm(a, X64XmmReg_XMM1, x64_rbp_mem(tb));
			x64_emit_simd_binop(a, op, elem);
			x64_emit_movups_mr(a, x64_rbp_mem(res), X64XmmReg_XMM0);
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}
	i64 off = 0;
	if (avx) {
		for (; off + 32 <= total; off += 32) {
			X64Mem ac = amem; ac.disp += (i32)off;
			X64Mem bc = bmem; bc.disp += (i32)off;
			x64_emit_vmovups_rm(a, true, X64XmmReg_XMM0, swap ? bc : ac);
			x64_emit_vmovups_rm(a, true, X64XmmReg_XMM1, swap ? ac : bc);
			x64_emit_vsimd_binop(a, true, op, elem);
			x64_emit_vmovups_mr(a, true, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
		}
	}
	for (; off + 16 <= total; off += 16) {
		X64Mem ac = amem; ac.disp += (i32)off;
		X64Mem bc = bmem; bc.disp += (i32)off;
		if (avx) {
			x64_emit_vmovups_rm(a, false, X64XmmReg_XMM0, swap ? bc : ac);
			x64_emit_vmovups_rm(a, false, X64XmmReg_XMM1, swap ? ac : bc);
			x64_emit_vsimd_binop(a, false, op, elem);
			x64_emit_vmovups_mr(a, false, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
		} else {
			x64_emit_movups_rm(a, X64XmmReg_XMM0, swap ? bc : ac);
			x64_emit_movups_rm(a, X64XmmReg_XMM1, swap ? ac : bc);
			x64_emit_simd_binop(a, op, elem);
			x64_emit_movups_mr(a, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
		}
	}
	return x64v_mem(rt, x64_rbp_mem(res));
}

// Copy one SIMD lane (esz ∈ {1,2,4,8} bytes) src→dst as a raw bit move through RAX.
gb_internal void x64_simd_copy_lane(x64Procedure *p, X64Mem dst, X64Mem src, i64 esz) {
	X64OpSize os = esz <= 1 ? X64OpSize_8 : esz == 2 ? X64OpSize_16 : esz <= 4 ? X64OpSize_32 : X64OpSize_64;
	x64_emit_mov_rm(&p->asm_, os, X64Reg_RAX, src);
	x64_emit_mov_mr(&p->asm_, os, dst, X64Reg_RAX);
}

// Gather lanes by CONSTANT indices into concat(a,b): res[i] = (idx[i] < ca) ? a[idx[i]] : b[idx[i]-ca].
// Single-operand shuffles (reverse/rotate/lanes_*) pass b_mem == a_mem with all idx < ca. Mirrors the
// LLVM ShuffleVector lowering (lb uses LLVMBuildShuffleVector; SSE has no general shuffle so we gather).
gb_internal x64Value x64_simd_shuffle_const(x64Procedure *p, X64Mem a_mem, X64Mem b_mem, i64 ca,
                                            i64 esz, i64 const *idx, i64 n, Type *res_type) {
	i32 res = x64_alloc_local(p, gb_max(type_size_of(res_type), (i64)1), gb_max(type_align_of(res_type), (i64)1));
	for (i64 i = 0; i < n; i++) {
		i64 k = idx[i];
		X64Mem s = (k < ca) ? a_mem : b_mem;
		s.disp += (i32)((k < ca ? k : k - ca) * esz);
		x64_simd_copy_lane(p, x64_rbp_mem(res + (i32)(i*esz)), s, esz);
	}
	return x64v_mem(res_type, x64_rbp_mem(res));
}

// Scalar per-lane min/max/clamp (kind 0/1/2) for the cases with NO packed instruction even on AVX2:
// 8-byte integer min/max (VPMINSQ is AVX-512) and sub-128-bit vectors. x64_emit_min/max/clamp handle
// float and signed/unsigned int, so this is correct for every element type.
gb_internal x64Value x64_simd_scalar_minmax(x64Procedure *p, int kind, Type *vt, Type *rt, Type *elem,
                                            X64Mem m0, X64Mem m1, X64Mem m2) {
	i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
	i64 n = base_type(vt)->SimdVector.count;
	i32 res = x64_alloc_local(p, type_size_of(vt), gb_max(type_align_of(vt), (i64)16));
	for (i64 i = 0; i < n; i++) {
		X64Mem a0 = m0; a0.disp += (i32)(i*esz);
		x64Value av = x64_spill_value(p, x64_load_addr(p, x64addr(a0, elem)), elem);
		X64Mem b0 = m1; b0.disp += (i32)(i*esz);
		x64Value bv = x64_spill_value(p, x64_load_addr(p, x64addr(b0, elem)), elem);
		x64Value r;
		if (kind == 2) {
			X64Mem c0 = m2; c0.disp += (i32)(i*esz);
			x64Value cv = x64_spill_value(p, x64_load_addr(p, x64addr(c0, elem)), elem);
			r = x64_emit_clamp(p, elem, av, bv, cv);
		} else {
			r = (kind == 0) ? x64_emit_min(p, elem, av, bv) : x64_emit_max(p, elem, av, bv);
		}
		x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(i*esz)), elem), r);
	}
	return x64v_mem(rt, x64_rbp_mem(res));
}

// AVX2 64-bit integer min/max: VPCMPGTQ + VPBLENDVB (VPMINSQ/VPMAXSQ are AVX-512). Unsigned flips the
// sign bit before the signed compare. Packed over 256/128-bit chunks; total must be a multiple of 16.
gb_internal x64Value x64_simd_minmax64_avx2(x64Procedure *p, bool is_min, Type *vt, Type *rt, Type *elem, X64Mem amem, X64Mem bmem) {
	X64Assembler *a = &p->asm_;
	bool uns = is_type_unsigned(elem);
	i64 total = type_size_of(vt);
	i32 sb = -1;
	if (uns) {
		sb = x64_alloc_local(p, 8, 8);
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, (i64)0x8000000000000000ull);
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(sb), X64Reg_RAX);
	}
	i32 res = x64_alloc_local(p, total, gb_max(type_align_of(vt), (i64)16));
	for (i64 off = 0; off < total; ) {
		bool ymm = (total - off) >= 32;
		X64Mem ac = amem; ac.disp += (i32)off;
		X64Mem bc = bmem; bc.disp += (i32)off;
		x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, ac); // a
		x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, bc); // b
		X64XmmReg ca = X64XmmReg_XMM0, cb = X64XmmReg_XMM1;
		if (uns) { // a^signbit, b^signbit → signed-compare gives the unsigned ordering
			x64_emit_vex_rm(a, 1u, 2u, 0x59u, false, ymm, X64XmmReg_XMM3, X64XmmReg_XMM0, x64_rbp_mem(sb)); // vpbroadcastq signbit
			x64_emit_vex_rr(a, 1u, 1u, 0xEFu, false, ymm, X64XmmReg_XMM2, X64XmmReg_XMM0, X64XmmReg_XMM3);  // a^sb
			x64_emit_vex_rr(a, 1u, 1u, 0xEFu, false, ymm, X64XmmReg_XMM3, X64XmmReg_XMM1, X64XmmReg_XMM3);  // b^sb
			ca = X64XmmReg_XMM2; cb = X64XmmReg_XMM3;
		}
		x64_emit_vex_rr(a, 1u, 2u, 0x37u, false, ymm, X64XmmReg_XMM4, ca, cb); // YMM4 = (a > b) signed, all-ones per lane
		// VPBLENDVB dst, src1, src2, mask(imm[7:4]): result = mask ? src2 : src1 (per byte; lane mask is uniform).
		if (is_min) x64_emit_vex_rr_i(a, 1u, 3u, 0x4Cu, false, ymm, X64XmmReg_XMM5, X64XmmReg_XMM0, X64XmmReg_XMM1, (u8)(4u<<4)); // a>b ? b : a
		else        x64_emit_vex_rr_i(a, 1u, 3u, 0x4Cu, false, ymm, X64XmmReg_XMM5, X64XmmReg_XMM1, X64XmmReg_XMM0, (u8)(4u<<4)); // a>b ? a : b
		x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM5);
		off += ymm ? 32 : 16;
	}
	return x64v_mem(rt, x64_rbp_mem(res));
}

// Scalar per-lane neg/abs fallback (8-byte int — no AVX2 64-bit min/max; sub-128-bit vectors). Floats
// are done by bit manipulation (neg = XOR sign bit, abs = AND ~sign bit) so they're exact; signed ints
// use 0-x and max(x,-x). Unsigned abs (identity) is handled by the caller before this.
gb_internal x64Value x64_simd_scalar_neg_abs(x64Procedure *p, bool is_abs, Type *vt, Type *rt, Type *elem, X64Mem m0) {
	X64Assembler *a = &p->asm_;
	bool flt = is_type_float(elem);
	i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
	i64 n = base_type(vt)->SimdVector.count;
	X64OpSize os = x64_op_size_of(elem);
	i32 res = x64_alloc_local(p, type_size_of(vt), gb_max(type_align_of(vt), (i64)16));
	for (i64 i = 0; i < n; i++) {
		X64Mem c = m0; c.disp += (i32)(i*esz);
		if (flt) {
			i64 signbit = (esz == 8) ? (i64)0x8000000000000000ull : (i64)0x80000000;
			x64_emit_mov_rm(a, os, X64Reg_RAX, c);
			x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, is_abs ? ~signbit : signbit);
			if (is_abs) x64_emit_and_rr(a, os, X64Reg_RAX, X64Reg_RCX);
			else        x64_emit_xor_rr(a, os, X64Reg_RAX, X64Reg_RCX);
			x64_emit_mov_mr(a, os, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RAX);
		} else {
			x64Value xv  = x64_spill_value(p, x64_load_addr(p, x64addr(c, elem)), elem);
			x64Value neg = x64_spill_value(p, x64_emit_arith(p, Token_Sub, x64v_imm(elem, 0), xv, elem), elem);
			x64Value r   = is_abs ? x64_emit_max(p, elem, xv, neg) : neg;
			x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(i*esz)), elem), r);
		}
	}
	return x64v_mem(rt, x64_rbp_mem(res));
}

// SIMD vector builtins (#simd[N]T) — mirrors lb_build_builtin_simd_proc. A #simd value is N*esz
// contiguous bytes in memory (built by the compound-lit path; simd.to_array is a transmute). Ops are
// emitted as REAL packed SSE/SSE4.1 instructions over 128-bit chunks (developers building -x64-backend
// have these). A scalar lane loop is used ONLY where SSE has no instruction at all (integer div/rem,
// 8/64-bit integer multiply, unsigned ordered compares), for sub-128-bit vectors, or for the
// shuffle/gather family (no general SSE shuffle). gather/scatter/masked_*/interleave/deinterleave hit
// the GB_PANIC — a findable gap; fill in from lb case-for-case.
gb_internal x64Value x64_build_builtin_simd_proc_inner(x64Procedure *p, Ast *expr, i32 id) {
	ast_node(ce, CallExpr, expr);
	X64Assembler *a = &p->asm_;
	Type *rt = x64_typed(expr->tav.type);
	Type *vt = base_type(rt);

	// Packed rounding (ROUNDPS/ROUNDPD, per 128-bit chunk).
	switch (id) {
	case BuiltinProc_simd_floor:
	case BuiltinProc_simd_ceil:
	case BuiltinProc_simd_trunc:
	case BuiltinProc_simd_nearest: {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		bool d64 = type_size_of(elem) == 8;
		i64  total = type_size_of(vt);
		u8 mode = 0;
		switch (id) {
		case BuiltinProc_simd_floor:   mode = 0x09; break; // toward -inf
		case BuiltinProc_simd_ceil:    mode = 0x0A; break; // toward +inf
		case BuiltinProc_simd_trunc:   mode = 0x0B; break; // toward zero
		case BuiltinProc_simd_nearest: mode = 0x0C; break; // current MXCSR (nearbyint)
		}
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		bool avx = x64_use_avx2();
		i32 res = x64_alloc_local(p, gb_max(total, (i64)16), gb_max(type_align_of(vt), (i64)16));
		for (i64 off = 0; off < total; ) {
			bool ymm = avx && (total - off) >= 32;
			X64Mem c = vm.mem; c.disp += (i32)off;
			if (avx) { // vroundps/pd (66 0F3A 08/09): 2-operand+imm, vvvv unused (XMM0 → 1111)
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, c);
				x64_emit_vex_rr_i(a, 1u, 3u, d64?0x09u:0x08u, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM0, mode);
				x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
			} else {
				x64_emit_movups_rm(a, X64XmmReg_XMM0, c);
				if (d64) x64_emit_roundpd(a, X64XmmReg_XMM0, X64XmmReg_XMM0, mode);
				else     x64_emit_roundps(a, X64XmmReg_XMM0, X64XmmReg_XMM0, mode);
				x64_emit_movups_mr(a, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
			}
			off += ymm ? 32 : 16;
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}
	}

	// Element-wise binary arithmetic / bitwise → packed SSE over 128-bit chunks; scalar fallback only
	// where SSE has no packed instruction (x64_simd_packed_ok) or for sub-128-bit vectors.
	int bop = -1; TokenKind aop = Token_Invalid;
	switch (id) {
	case BuiltinProc_simd_add:         bop = X64SimdBin_Add;  aop = Token_Add;    break;
	case BuiltinProc_simd_sub:         bop = X64SimdBin_Sub;  aop = Token_Sub;    break;
	case BuiltinProc_simd_mul:         bop = X64SimdBin_Mul;  aop = Token_Mul;    break;
	case BuiltinProc_simd_div:         bop = X64SimdBin_Div;  aop = Token_Quo;    break;
	case BuiltinProc_simd_rem:         bop = -1;              aop = Token_Mod;    break; // no packed int rem
	case BuiltinProc_simd_bit_and:     bop = X64SimdBin_And;  aop = Token_And;    break;
	case BuiltinProc_simd_bit_or:      bop = X64SimdBin_Or;   aop = Token_Or;     break;
	case BuiltinProc_simd_bit_xor:     bop = X64SimdBin_Xor;  aop = Token_Xor;    break;
	case BuiltinProc_simd_bit_and_not: bop = X64SimdBin_AndN; aop = Token_AndNot; break;
	}
	if (aop != Token_Invalid) {
		Type *elem = base_type(vt->SimdVector.elem);
		if (bop >= 0 && x64_simd_packed_ok(bop, elem)) { // binop_vec pads sub-128 to a 128-bit op
			// Build+spill each operand in sequence: building rhs may reuse lhs's temp slot (esp. for
			// small sub-128 vectors), so lhs MUST be spilled before rhs is built. (Was the sub-128 bug.)
			x64Value la = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
			x64Value ra = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);
			bool swap = (bop == X64SimdBin_AndN); // (v)pandn = (~v)&s → v=b, s=a to get a&~b
			return x64_simd_binop_vec(p, bop, vt, rt, elem, la.mem, ra.mem, swap);
		}
		x64Value lhs = x64_build_expr(p, ce->args[0]);
		x64Value rhs = x64_build_expr(p, ce->args[1]);
		return x64_emit_arith_array(p, aop, lhs, rhs, rt); // no packed insn → scalar lanes
	}

	// Per-lane comparison → all-ones / all-zeros mask. Float: CMPPS/CMPPD imm8. Signed int: PCMPEQ /
	// PCMPGT (with operand swap + invert for ne/lt/le/ge). Unsigned ordered or sub-128-bit → scalar.
	{
		bool is_cmp = true, base_eq = false, swap = false, invert = false; u8 fimm = 0;
		switch (id) {
		case BuiltinProc_simd_lanes_eq: base_eq = true;  fimm = 0; break;
		case BuiltinProc_simd_lanes_ne: base_eq = true;  invert = true; fimm = 4; break;
		case BuiltinProc_simd_lanes_lt: swap = true;     fimm = 1; break; // a<b = b>a
		case BuiltinProc_simd_lanes_le: invert = true;   fimm = 2; break; // a<=b = !(a>b)
		case BuiltinProc_simd_lanes_gt:                  fimm = 6; break; // a>b
		case BuiltinProc_simd_lanes_ge: swap = true; invert = true; fimm = 5; break; // a>=b = !(b>a)
		default: is_cmp = false; break;
		}
		if (is_cmp) {
			// The OPERANDS are #simd[N]T (T may be float); the RESULT rt is the integer mask #simd[N]V.
			// Derive the lane element from the ARGUMENT, not rt — else float compares pick PCMP* not CMPPS.
			Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
			GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
			Type *elem = base_type(avt->SimdVector.elem);
			bool flt = is_type_float(elem), d64 = type_size_of(elem) == 8, uns = is_type_unsigned(elem);
			i64  esz = type_size_of(elem); if (esz <= 0) esz = 1;
			i64  total = type_size_of(avt);
			// Build+spill each operand before building the next (rhs build may reuse lhs's small temp slot).
			x64Value la = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
			x64Value ra = x64_spill_value(p, x64_build_expr(p, ce->args[1]), avt);
			// Unsigned ordered compare (PCMPGT is signed) → scalar lane loop; packable handles sub-128.
			bool packable = (flt || base_eq || !uns);
			if (packable) {
				bool avx = x64_use_avx2();
				i32 res = x64_alloc_local(p, gb_max(total, (i64)16), gb_max(type_align_of(vt), (i64)16));
				for (i64 off = 0; off < total; ) {
					bool ymm = avx && (total - off) >= 32;
					X64Mem ac = la.mem; ac.disp += (i32)off;
					X64Mem bc = ra.mem; bc.disp += (i32)off;
					if (avx) {
						x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, swap ? bc : ac);
						x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, swap ? ac : bc);
						if (flt) { // vcmpps/pd imm: XMM0 = cmp(XMM0, XMM1)
							x64_emit_vex_rr_i(a, d64?1u:0u, 1u, 0xC2u, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1, fimm);
						} else {
							u8 mm_i = 1, opc = 0;
							if (base_eq) { if (esz==1){mm_i=1;opc=0x74;} else if(esz==2){mm_i=1;opc=0x75;} else if(esz==4){mm_i=1;opc=0x76;} else {mm_i=2;opc=0x29;} }
							else         { if (esz==1){mm_i=1;opc=0x64;} else if(esz==2){mm_i=1;opc=0x65;} else if(esz==4){mm_i=1;opc=0x66;} else {mm_i=2;opc=0x37;} }
							x64_emit_vex_rr(a, 1u, mm_i, opc, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1);
							if (invert) {
								x64_emit_vex_rr(a, 1u, 1u, 0x76u, false, ymm, X64XmmReg_XMM2, X64XmmReg_XMM2, X64XmmReg_XMM2); // all-ones
								x64_emit_vex_rr(a, 1u, 1u, 0xEFu, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM2); // ^= → NOT
							}
						}
						x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
					} else {
						x64_emit_movups_rm(a, X64XmmReg_XMM0, swap ? bc : ac);
						x64_emit_movups_rm(a, X64XmmReg_XMM1, swap ? ac : bc);
						if (flt) {
							if (d64) x64_emit_cmppd(a, X64XmmReg_XMM0, X64XmmReg_XMM1, fimm);
							else     x64_emit_cmpps(a, X64XmmReg_XMM0, X64XmmReg_XMM1, fimm);
						} else {
							if (base_eq) {
								if      (esz == 1) x64_emit_pcmpeqb(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
								else if (esz == 2) x64_emit_pcmpeqw(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
								else if (esz == 4) x64_emit_pcmpeqd(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
								else               x64_emit_pcmpeqq(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
							} else {
								if      (esz == 1) x64_emit_pcmpgtb(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
								else if (esz == 2) x64_emit_pcmpgtw(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
								else if (esz == 4) x64_emit_pcmpgtd(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
								else               x64_emit_pcmpgtq(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
							}
							if (invert) { // NOT via XOR with all-ones (pcmpeqd reg,reg)
								x64_emit_pcmpeqd(a, X64XmmReg_XMM2, X64XmmReg_XMM2);
								x64_emit_pxor(a, X64XmmReg_XMM0, X64XmmReg_XMM2);
							}
						}
						x64_emit_movups_mr(a, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
					}
					off += ymm ? 32 : 16;
				}
				return x64v_mem(rt, x64_rbp_mem(res));
			}
			// Scalar fallback: per-lane x64_emit_comp (0/1) → neg → mask.
			TokenKind cop = Token_CmpEq;
			switch (id) {
			case BuiltinProc_simd_lanes_eq: cop = Token_CmpEq; break;
			case BuiltinProc_simd_lanes_ne: cop = Token_NotEq; break;
			case BuiltinProc_simd_lanes_lt: cop = Token_Lt;    break;
			case BuiltinProc_simd_lanes_le: cop = Token_LtEq;  break;
			case BuiltinProc_simd_lanes_gt: cop = Token_Gt;    break;
			case BuiltinProc_simd_lanes_ge: cop = Token_GtEq;  break;
			}
			i64       n   = vt->SimdVector.count;
			Type     *me  = base_array_type(rt);
			i64       msz = type_size_of(me); if (msz <= 0) msz = esz;
			X64OpSize mos = x64_op_size_of(me);
			i32 la_off = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, la.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(la_off), X64Reg_RAX);
			i32 ra_off = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, ra.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(ra_off), X64Reg_RAX);
			i32 res_off = x64_alloc_local(p, type_size_of(rt), type_align_of(rt));
			for (i64 i = 0; i < n; i++) {
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(la_off));
				x64Value ae = x64_spill_value(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, (i32)(i*esz)), elem)), elem);
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ra_off));
				x64Value be = x64_spill_value(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, (i32)(i*esz)), elem)), elem);
				x64Value cmp = x64_emit_comp(p, cop, ae, be);
				x64_value_to_reg(p, cmp, X64Reg_RAX);
				x64_emit_neg_r(a, X64OpSize_64, X64Reg_RAX);
				x64_emit_mov_mr(a, mos, x64_rbp_mem(res_off + (i32)(i*msz)), X64Reg_RAX);
			}
			return x64v_mem(rt, x64_rbp_mem(res_off));
		}
	}

	// Horizontal reductions → scalar T. Packed: fold all 128-bit chunks into one accumulator, then
	// fold lanes within it via PSRLDQ (whole-register byte shift) + the packed op; read lane 0. Scalar
	// fallback where SSE lacks the packed op. The ARG is the #simd; RESULT (rt) is the scalar element.
	{
		int rb = -1; TokenKind rop = Token_Invalid;
		switch (id) {
		case BuiltinProc_simd_reduce_add_bisect:
		case BuiltinProc_simd_reduce_add_ordered:
		case BuiltinProc_simd_reduce_add_pairs: rb = X64SimdBin_Add; rop = Token_Add; break;
		case BuiltinProc_simd_reduce_mul_bisect:
		case BuiltinProc_simd_reduce_mul_ordered:
		case BuiltinProc_simd_reduce_mul_pairs: rb = X64SimdBin_Mul; rop = Token_Mul; break;
		case BuiltinProc_simd_reduce_and:       rb = X64SimdBin_And; rop = Token_And; break;
		case BuiltinProc_simd_reduce_or:        rb = X64SimdBin_Or;  rop = Token_Or;  break;
		case BuiltinProc_simd_reduce_xor:       rb = X64SimdBin_Xor; rop = Token_Xor; break;
		case BuiltinProc_simd_reduce_min:       rb = X64SimdBin_Min; break;
		case BuiltinProc_simd_reduce_max:       rb = X64SimdBin_Max; break;
		}
		if (rb >= 0) {
			Type *avt  = base_type(x64_typed(ce->args[0]->tav.type));
			GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
			Type *elem = base_type(avt->SimdVector.elem);
			i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;
			i64   total = type_size_of(avt);
			x64Value v = x64_build_expr(p, ce->args[0]);
			if (total % 16 == 0 && x64_simd_packed_ok(rb, elem)) {
				bool avx = x64_use_avx2();
				x64Value vm = x64_spill_value(p, v, avt);
				i32 acc = x64_alloc_local(p, 16, 16);
				if (avx) {
					if (total >= 32) {
						x64_emit_vmovups_rm(a, true, X64XmmReg_XMM0, vm.mem);   // YMM0 = chunk 0 (32B)
						for (i64 off = 32; off + 32 <= total; off += 32) {      // fold 32B chunks
							X64Mem c = vm.mem; c.disp += (i32)off;
							x64_emit_vmovups_rm(a, true, X64XmmReg_XMM1, c);
							x64_emit_vsimd_binop(a, true, rb, elem);
						}
						// vextracti128 XMM1, YMM0, 1 (reg=src YMM0, rm=dst XMM1) → fold hi128 into lo128.
						x64_emit_vex_rr_i(a, 1u, 3u, 0x39u, false, true, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1, 1u);
						x64_emit_vsimd_binop(a, false, rb, elem);              // XMM0 = lo op hi
					} else {
						x64_emit_vmovups_rm(a, false, X64XmmReg_XMM0, vm.mem);  // single 128-bit chunk
					}
					x64_emit_vmovups_mr(a, false, x64_rbp_mem(acc), X64XmmReg_XMM0);
					for (i64 sh = 8; sh >= esz; sh /= 2) {                      // fold lanes within 128 bits
						x64_emit_vmovups_rm(a, false, X64XmmReg_XMM0, x64_rbp_mem(acc));
						x64_emit_vmovups_rm(a, false, X64XmmReg_XMM1, x64_rbp_mem(acc));
						// vpsrldq XMM1, XMM1, sh (NDD: /3 in reg, dst in vvvv, src in rm)
						x64_emit_vex_rr_i(a, 1u, 1u, 0x73u, false, false, X64XmmReg_XMM3, X64XmmReg_XMM1, X64XmmReg_XMM1, (u8)sh);
						x64_emit_vsimd_binop(a, false, rb, elem);
						x64_emit_vmovups_mr(a, false, x64_rbp_mem(acc), X64XmmReg_XMM0);
					}
					return x64v_mem(elem, x64_rbp_mem(acc));                    // lane 0
				}
				x64_emit_movups_rm(a, X64XmmReg_XMM0, vm.mem);              // chunk 0
				for (i64 off = 16; off < total; off += 16) {                // fold remaining chunks
					X64Mem c = vm.mem; c.disp += (i32)off;
					x64_emit_movups_rm(a, X64XmmReg_XMM1, c);
					x64_emit_simd_binop(a, rb, elem);
				}
				x64_emit_movups_mr(a, x64_rbp_mem(acc), X64XmmReg_XMM0);
				for (i64 sh = 8; sh >= esz; sh /= 2) {                      // fold lanes within 128 bits
					x64_emit_movups_rm(a, X64XmmReg_XMM0, x64_rbp_mem(acc));
					x64_emit_movups_rm(a, X64XmmReg_XMM1, x64_rbp_mem(acc));
					x64_emit_psrldq(a, X64XmmReg_XMM1, (u8)sh);
					x64_emit_simd_binop(a, rb, elem);
					x64_emit_movups_mr(a, x64_rbp_mem(acc), X64XmmReg_XMM0);
				}
				return x64v_mem(elem, x64_rbp_mem(acc));                    // lane 0
			}
			// Scalar fold.
			i64 n = avt->SimdVector.count;
			x64Value vm = x64_spill_value(p, v, avt);
			i32 base_off = x64_alloc_local(p, 8, 8);
			x64_emit_lea(a, X64Reg_RAX, vm.mem);
			x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(base_off), X64Reg_RAX);
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_off));
			x64Value accv = x64_spill_value(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, 0), elem)), elem);
			for (i64 i = 1; i < n; i++) {
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_off));
				x64Value lane = x64_spill_value(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, (i32)(i*esz)), elem)), elem);
				x64Value r;
				if      (rb == X64SimdBin_Min) r = x64_emit_min(p, elem, accv, lane);
				else if (rb == X64SimdBin_Max) r = x64_emit_max(p, elem, accv, lane);
				else                           r = x64_emit_arith(p, rop, accv, lane, elem);
				accv = x64_spill_value(p, r, elem);
			}
			return accv;
		}
	}

	// simd_select(cond, x, y): per-lane cond!=0 ? x : y. A mask blend (cond is all-ones/all-zeros per
	// lane): result = (x & cond) | (~cond & y). Bitwise → element-type-agnostic; packed over chunks.
	if (id == BuiltinProc_simd_select) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		i64 total = type_size_of(vt);
		// Build+spill each operand in sequence: building a later operand can clobber an
		// earlier one's register (cond was read as zero → select always returned y).
		x64Value cm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), base_type(x64_typed(ce->args[0]->tav.type)));
		x64Value xm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);
		x64Value ym = x64_spill_value(p, x64_build_expr(p, ce->args[2]), vt);
		bool avx = x64_use_avx2();
		i32 res = x64_alloc_local(p, gb_max(total, (i64)16), gb_max(type_align_of(vt), (i64)16));
		for (i64 off = 0; off < total; ) {
			bool ymm = avx && (total - off) >= 32;
			X64Mem cc = cm.mem; cc.disp += (i32)off;
			X64Mem xc = xm.mem; xc.disp += (i32)off;
			X64Mem yc = ym.mem; yc.disp += (i32)off;
			if (avx) { // (x & cond) | (~cond & y), VEX 3-operand
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, cc);  // cond
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, xc);  // x
				x64_emit_vex_rr(a, 1u, 1u, 0xDBu, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1);  // x & cond
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM3, yc);  // y
				x64_emit_vex_rr(a, 1u, 1u, 0xDFu, false, ymm, X64XmmReg_XMM2, X64XmmReg_XMM1, X64XmmReg_XMM3);  // (~cond) & y
				x64_emit_vex_rr(a, 1u, 1u, 0xEBu, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM2);  // |
				x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
			} else {
				x64_emit_movups_rm(a, X64XmmReg_XMM0, xc);
				x64_emit_movups_rm(a, X64XmmReg_XMM1, cc);
				x64_emit_pand(a, X64XmmReg_XMM0, X64XmmReg_XMM1);  // x & cond
				x64_emit_movups_rm(a, X64XmmReg_XMM2, cc);
				x64_emit_movups_rm(a, X64XmmReg_XMM3, yc);
				x64_emit_pandn(a, X64XmmReg_XMM2, X64XmmReg_XMM3); // (~cond) & y
				x64_emit_por(a, X64XmmReg_XMM0, X64XmmReg_XMM2);
				x64_emit_movups_mr(a, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
			}
			off += ymm ? 32 : 16;
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// simd_indices(T) → the constant lane-index vector {0, 1, …, N-1} (T's element may be int or float).
	if (id == BuiltinProc_simd_indices) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64       esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64       n   = vt->SimdVector.count;
		bool      flt = is_type_float(elem);
		X64OpSize os  = x64_op_size_of(elem);
		i32 res = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
		for (i64 i = 0; i < n; i++) {
			if (flt) {
				x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, i);
				if (esz == 8) { x64_emit_cvtsi2sd(a, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64); x64_emit_movsd_mr(a, x64_rbp_mem(res + (i32)(i*esz)), X64XmmReg_XMM0); }
				else          { x64_emit_cvtsi2ss(a, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64); x64_emit_movss_mr(a, x64_rbp_mem(res + (i32)(i*esz)), X64XmmReg_XMM0); }
			} else {
				x64_emit_mov_ri(a, os, X64Reg_RAX, i);
				x64_emit_mov_mr(a, os, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RAX);
			}
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// to_bits / to_bits_signed: reinterpret #simd[N]float → #simd[N]int (identical bits) → a transmute.
	if (id == BuiltinProc_simd_to_bits || id == BuiltinProc_simd_to_bits_signed) {
		return x64_emit_transmute(p, x64_build_expr(p, ce->args[0]), rt);
	}

	// simd_extract(v, idx) → scalar lane v[idx] (idx may be a runtime value).
	if (id == BuiltinProc_simd_extract) {
		Type *avt  = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		Ast *idx = ce->args[1];
		if (idx->tav.mode == Addressing_Constant) {
			X64Mem m = vm.mem; m.disp += (i32)(exact_value_to_i64(idx->tav.value) * esz);
			return x64_load_addr(p, x64addr(m, elem));
		}
		i32 base_off = x64_alloc_local(p, 8, 8);
		x64_emit_lea(a, X64Reg_RAX, vm.mem);
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(base_off), X64Reg_RAX);
		x64_value_to_reg(p, x64_build_expr(p, idx), X64Reg_RCX);          // idx
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(base_off)); // &v (reload; build clobbered)
		return x64_load_addr(p, x64addr(x64_mem_idx(X64Reg_RAX, X64Reg_RCX, (u8)esz, 0), elem));
	}

	// simd_replace(v, idx, val) → copy of v with lane idx set to val.
	if (id == BuiltinProc_simd_replace) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;
		i64   total = type_size_of(vt);
		x64Value vv  = x64_build_expr(p, ce->args[0]);
		i32 res = x64_alloc_local(p, total, type_align_of(vt));
		x64_store_value(p, x64addr(x64_rbp_mem(res), vt), vv); // copy the whole vector
		x64Value val = x64_build_expr(p, ce->args[2]);
		Ast *idx = ce->args[1];
		if (idx->tav.mode == Addressing_Constant) {
			i64 k = exact_value_to_i64(idx->tav.value);
			x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(k*esz)), elem), val);
		} else {
			x64Value vs = x64_spill_value(p, val, elem);
			x64_value_to_reg(p, x64_build_expr(p, idx), X64Reg_RCX);
			x64_emit_lea(a, X64Reg_RAX, x64_rbp_mem(res));
			x64_emit_lea(a, X64Reg_RAX, x64_mem_idx(X64Reg_RAX, X64Reg_RCX, (u8)esz, 0)); // &res[idx]
			x64_emit_mov_rm(a, x64_op_size_of(elem), X64Reg_RDX, vs.mem);
			x64_emit_mov_mr(a, x64_op_size_of(elem), x64_mem(X64Reg_RAX, 0), X64Reg_RDX);
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// Per-lane shifts. SSE has no packed VARIABLE per-lane shift (that's AVX2 VPSLLVD), so this is a
	// scalar lane loop. Odin logic (shl/shr): count ≥ bitwidth → 0. C/masked logic: count &= width-1.
	{
		bool is_shl = false, masked = false, is_shift = true;
		switch (id) {
		case BuiltinProc_simd_shl:        is_shl = true;  break;
		case BuiltinProc_simd_shr:        is_shl = false; break;
		case BuiltinProc_simd_shl_masked: is_shl = true;  masked = true; break;
		case BuiltinProc_simd_shr_masked: is_shl = false; masked = true; break;
		default: is_shift = false; break;
		}
		if (is_shift) {
			GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
			Type *elem = base_type(vt->SimdVector.elem);
			i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;
			i64   n    = vt->SimdVector.count;
			i64   bits = esz * 8;
			bool  sgn  = !is_type_unsigned(elem);
			X64OpSize os = x64_op_size_of(elem);
			x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
			x64Value bm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);

			// AVX2 native variable per-lane shift (VPSLLVD/Q, VPSRLVD/Q, VPSRAVD). Native semantics for
			// the logical ops (count≥bits → 0) match Odin's non-masked rule; VPSRAVD's count≥bits gives
			// sign-fill ≠ Odin's 0, so arithmetic shr only takes this path when masked (count pre-AND'd
			// to <bits) and esz==4. 8/16-bit lanes and arithmetic 64-bit shr (no VPSRAVQ in AVX2) stay
			// scalar below. The Win64 count vector is broadcast-masked for the `_masked` variants.
			bool avx = x64_use_avx2();
			bool can_avx = avx && (esz == 4 || esz == 8) && (type_size_of(vt) % 16 == 0); // 16-byte op needs ≥128-bit
			if (can_avx && !is_shl && sgn) can_avx = (esz == 4 && masked);
			if (can_avx) {
				u8 sop = is_shl ? 0x47u : (sgn ? 0x46u : 0x45u);
				bool W = (esz == 8);
				i64 total = type_size_of(vt);
				i32 mslot = -1;
				if (masked) { // broadcast (bits-1) so count &= bits-1 per lane
					mslot = x64_alloc_local(p, 8, 8);
					x64_emit_mov_mi(a, W ? X64OpSize_64 : X64OpSize_32, x64_rbp_mem(mslot), (i32)(bits - 1));
				}
				i32 res = x64_alloc_local(p, total, gb_max(type_align_of(vt), (i64)16));
				for (i64 off = 0; off < total; ) {
					bool ymm = (total - off) >= 32;
					X64Mem ac = am.mem; ac.disp += (i32)off;
					X64Mem bc = bm.mem; bc.disp += (i32)off;
					x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, ac); // values
					x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, bc); // counts
					if (masked) {
						x64_emit_vex_rm(a, 1u, 2u, W?0x59u:0x58u, false, ymm, X64XmmReg_XMM2, X64XmmReg_XMM0, x64_rbp_mem(mslot)); // vpbroadcastd/q
						x64_emit_vex_rr(a, 1u, 1u, 0xDBu, false, ymm, X64XmmReg_XMM1, X64XmmReg_XMM1, X64XmmReg_XMM2);             // count &= bits-1
					}
					x64_emit_vex_rr(a, 1u, 2u, sop, W, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1); // VPSxLV: YMM0 = values shift counts
					x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
					off += ymm ? 32 : 16;
				}
				return x64v_mem(rt, x64_rbp_mem(res));
			}

			i32 ao = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, am.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(ao), X64Reg_RAX);
			i32 bo = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, bm.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(bo), X64Reg_RAX);
			i32 res = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
			for (i64 i = 0; i < n; i++) {
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ao));
				x64_value_to_reg(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, (i32)(i*esz)), elem)), X64Reg_RAX);
				x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX); // a-lane
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(bo));
				x64_value_to_reg(p, x64_load_addr(p, x64addr(x64_mem(X64Reg_RAX, (i32)(i*esz)), elem)), X64Reg_RCX); // count
				x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_R8);
				if (masked) {
					x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, bits - 1);
					x64_emit_and_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_R9);
				}
				if (is_shl)   x64_emit_shl_rcl(a, os, X64Reg_RAX);
				else if (sgn) x64_emit_sar_rcl(a, os, X64Reg_RAX);
				else          x64_emit_shr_rcl(a, os, X64Reg_RAX);
				if (!masked) { // Odin: count >= bits → 0
					x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, bits);
					x64_emit_cmp_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_R9);
					x64_emit_setcc_r(a, X64Cc_B, X64Reg_RDX);            // 1 if count < bits
					x64_emit_movzx_rr(a, X64OpSize_8, X64Reg_RDX, X64Reg_RDX);
					x64_emit_neg_r(a, X64OpSize_64, X64Reg_RDX);          // -1 if count<bits else 0
					x64_emit_and_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_RDX);
				}
				x64_emit_mov_mr(a, os, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RAX);
			}
			return x64v_mem(rt, x64_rbp_mem(res));
		}
	}

	// min / max (element-wise, packed). 8-byte integer min/max has no SSE op → GB_PANIC (narrow gap).
	if (id == BuiltinProc_simd_min || id == BuiltinProc_simd_max) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		int   op   = (id == BuiltinProc_simd_min) ? X64SimdBin_Min : X64SimdBin_Max;
		Type *elem = base_type(vt->SimdVector.elem);
		x64Value la = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		x64Value ra = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);
		if (x64_simd_packed_ok(op, elem)) { // binop_vec pads sub-128
			return x64_simd_binop_vec(p, op, vt, rt, elem, la.mem, ra.mem, /*swap*/false);
		}
		if (x64_use_avx2() && type_size_of(elem) == 8 && type_size_of(vt) % 16 == 0) { // 64-bit: VPCMPGTQ+VPBLENDVB
			return x64_simd_minmax64_avx2(p, id == BuiltinProc_simd_min, vt, rt, elem, la.mem, ra.mem);
		}
		return x64_simd_scalar_minmax(p, (id == BuiltinProc_simd_min) ? 0 : 1, vt, rt, elem, la.mem, ra.mem, la.mem); // sub-128
	}

	// clamp(x, lo, hi) = min(max(x, lo), hi), packed.
	if (id == BuiltinProc_simd_clamp) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		x64Value xv  = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		x64Value lov = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);
		x64Value hiv = x64_spill_value(p, x64_build_expr(p, ce->args[2]), vt);
		if (x64_simd_packed_ok(X64SimdBin_Min, elem) && x64_simd_packed_ok(X64SimdBin_Max, elem)) { // binop_vec pads sub-128
			x64Value t  = x64_simd_binop_vec(p, X64SimdBin_Max, vt, vt, elem, xv.mem, lov.mem, false); // max(x,lo)
			x64Value tt = x64_spill_value(p, t, vt);
			return x64_simd_binop_vec(p, X64SimdBin_Min, vt, rt, elem, tt.mem, hiv.mem, false);        // min(.,hi)
		}
		if (x64_use_avx2() && type_size_of(elem) == 8 && type_size_of(vt) % 16 == 0) { // 64-bit clamp
			x64Value t  = x64_simd_minmax64_avx2(p, /*is_min*/false, vt, vt, elem, xv.mem, lov.mem); // max(x,lo)
			x64Value tt = x64_spill_value(p, t, vt);
			return x64_simd_minmax64_avx2(p, /*is_min*/true, vt, rt, elem, tt.mem, hiv.mem);          // min(.,hi)
		}
		return x64_simd_scalar_minmax(p, 2, vt, rt, elem, xv.mem, lov.mem, hiv.mem); // sub-128
	}

	// neg = 0 - x (packed). abs = max(x, -x) (float + signed int 1/2/4); |unsigned| = identity.
	if (id == BuiltinProc_simd_neg || id == BuiltinProc_simd_abs) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		bool  flt  = is_type_float(elem);
		bool  uns  = is_type_unsigned(elem);
		i64   total = type_size_of(vt);
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		if (id == BuiltinProc_simd_abs && !flt && uns) return x64v_mem(rt, vm.mem);
		bool can = x64_simd_packed_ok(X64SimdBin_Sub, elem) &&
		           (id == BuiltinProc_simd_neg || x64_simd_packed_ok(X64SimdBin_Max, elem)); // pads sub-128
		if (can) {
			bool avx = x64_use_avx2();
			i32 res = x64_alloc_local(p, gb_max(total, (i64)16), gb_max(type_align_of(vt), (i64)16));
			for (i64 off = 0; off < total; ) {
				bool ymm = avx && (total - off) >= 32;
				X64Mem c = vm.mem; c.disp += (i32)off;
				if (avx) {
					x64_emit_vex_rr(a, 1u, 1u, 0xEFu, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM0); // vpxor → 0
					x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, c);                 // x
					x64_emit_vsimd_binop(a, ymm, X64SimdBin_Sub, elem);            // YMM0 = -x
					if (id == BuiltinProc_simd_abs) {
						x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, c);
						x64_emit_vsimd_binop(a, ymm, X64SimdBin_Max, elem);        // YMM0 = max(-x, x) = |x|
					}
					x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
				} else {
					x64_emit_pxor(a, X64XmmReg_XMM0, X64XmmReg_XMM0);  // 0
					x64_emit_movups_rm(a, X64XmmReg_XMM1, c);          // x
					x64_emit_simd_binop(a, X64SimdBin_Sub, elem);      // XMM0 = -x
					if (id == BuiltinProc_simd_abs) {
						x64_emit_movups_rm(a, X64XmmReg_XMM1, c);      // x
						x64_emit_simd_binop(a, X64SimdBin_Max, elem);  // XMM0 = max(-x, x) = |x|
					}
					x64_emit_movups_mr(a, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
				}
				off += ymm ? 32 : 16;
			}
			return x64v_mem(rt, x64_rbp_mem(res));
		}
		if (x64_use_avx2() && type_size_of(elem) == 8 && total % 16 == 0) {
			// neg(64) is the packed path above (VPSUBQ); only abs reaches here. abs = max(x, 0-x).
			i32 zo = x64_alloc_local(p, total, gb_max(type_align_of(vt), (i64)16));
			x64_zero_mem(p, x64_rbp_mem(zo), total);
			x64Value negx = x64_spill_value(p, x64_simd_binop_vec(p, X64SimdBin_Sub, vt, vt, elem, x64_rbp_mem(zo), vm.mem, false), vt);
			return x64_simd_minmax64_avx2(p, /*is_min*/false, vt, rt, elem, vm.mem, negx.mem);
		}
		return x64_simd_scalar_neg_abs(p, id == BuiltinProc_simd_abs, vt, rt, elem, vm.mem); // sub-128
	}

	// reduce_any / reduce_all → bool. any = OR-reduce, all = AND-reduce (a false lane = 0 zeros the
	// AND); the folded lane 0 is then tested != 0. Mirrors lb's reduce.or/and → bool.
	if (id == BuiltinProc_simd_reduce_any || id == BuiltinProc_simd_reduce_all) {
		Type *avt  = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64   esz  = type_size_of(elem); if (esz <= 0) esz = 1;
		i64   total = type_size_of(avt);
		int   rb   = (id == BuiltinProc_simd_reduce_any) ? X64SimdBin_Or : X64SimdBin_And;
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		GB_ASSERT_MSG(total % 16 == 0, "x64 simd_reduce_any/all: sub-128-bit unimplemented");
		bool avx = x64_use_avx2();
		i32 acc = x64_alloc_local(p, 16, 16);
		if (avx) {
			if (total >= 32) {
				x64_emit_vmovups_rm(a, true, X64XmmReg_XMM0, vm.mem);
				for (i64 off = 32; off + 32 <= total; off += 32) {
					X64Mem c = vm.mem; c.disp += (i32)off;
					x64_emit_vmovups_rm(a, true, X64XmmReg_XMM1, c);
					x64_emit_vsimd_binop(a, true, rb, elem);
				}
				x64_emit_vex_rr_i(a, 1u, 3u, 0x39u, false, true, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1, 1u); // vextracti128
				x64_emit_vsimd_binop(a, false, rb, elem);
			} else {
				x64_emit_vmovups_rm(a, false, X64XmmReg_XMM0, vm.mem);
			}
			x64_emit_vmovups_mr(a, false, x64_rbp_mem(acc), X64XmmReg_XMM0);
			for (i64 sh = 8; sh >= esz; sh /= 2) {
				x64_emit_vmovups_rm(a, false, X64XmmReg_XMM0, x64_rbp_mem(acc));
				x64_emit_vmovups_rm(a, false, X64XmmReg_XMM1, x64_rbp_mem(acc));
				x64_emit_vex_rr_i(a, 1u, 1u, 0x73u, false, false, X64XmmReg_XMM3, X64XmmReg_XMM1, X64XmmReg_XMM1, (u8)sh); // vpsrldq
				x64_emit_vsimd_binop(a, false, rb, elem);
				x64_emit_vmovups_mr(a, false, x64_rbp_mem(acc), X64XmmReg_XMM0);
			}
		} else {
			x64_emit_movups_rm(a, X64XmmReg_XMM0, vm.mem);
			for (i64 off = 16; off < total; off += 16) {
				X64Mem c = vm.mem; c.disp += (i32)off;
				x64_emit_movups_rm(a, X64XmmReg_XMM1, c);
				x64_emit_simd_binop(a, rb, elem);
			}
			x64_emit_movups_mr(a, x64_rbp_mem(acc), X64XmmReg_XMM0);
			for (i64 sh = 8; sh >= esz; sh /= 2) {
				x64_emit_movups_rm(a, X64XmmReg_XMM0, x64_rbp_mem(acc));
				x64_emit_movups_rm(a, X64XmmReg_XMM1, x64_rbp_mem(acc));
				x64_emit_psrldq(a, X64XmmReg_XMM1, (u8)sh);
				x64_emit_simd_binop(a, rb, elem);
				x64_emit_movups_mr(a, x64_rbp_mem(acc), X64XmmReg_XMM0);
			}
		}
		x64_value_to_reg(p, x64_load_addr(p, x64addr(x64_rbp_mem(acc), elem)), X64Reg_RAX);
		x64_emit_test_rr(a, x64_op_size_of(elem), X64Reg_RAX, X64Reg_RAX);
		x64_emit_setcc_r(a, X64Cc_NE, X64Reg_RAX);
		x64_emit_movzx_rr(a, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
		return x64v_reg(t_bool, X64Reg_RAX);
	}

	// extract_lsbs / extract_msbs → pack one bit per lane into an integer bitmask (lane i → bit i).
	// lsbs = bit 0 of each lane; msbs = the sign bit (bit esz*8-1). Mirrors lb's trunc-to-i1 + bitcast.
	if (id == BuiltinProc_simd_extract_lsbs || id == BuiltinProc_simd_extract_msbs) {
		Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 n = avt->SimdVector.count;
		i64 bit = (id == BuiltinProc_simd_extract_msbs) ? (esz*8 - 1) : 0;
		X64OpSize os = x64_op_size_of(elem);

		// AVX2: extract each lane's sign bit directly — VPMOVMSKB (8-bit) / VMOVMSKPS (32) / VMOVMSKPD
		// (64). lsbs shifts bit0 → sign bit first (VPSLLD/Q). Result masked to n bits (sub-128 padding /
		// wider result types). 16-bit msbs and 8/16-bit lsbs have no movmsk → scalar loop below.
		bool msbs = (id == BuiltinProc_simd_extract_msbs);
		if (x64_use_avx2() && type_size_of(avt) <= 32 &&
		    ((msbs && (esz == 1 || esz == 4 || esz == 8)) || (!msbs && (esz == 4 || esz == 8)))) {
			i64 total = type_size_of(avt);
			bool ymm = (total == 32);
			x64Value vm2 = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
			x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, vm2.mem);
			if (!msbs) { // VPSLLD/Q (NDD: /6 in reg, dst in vvvv, src in rm): bit0 → sign bit
				x64_emit_vex_rr_i(a, 1u, 1u, esz==8?0x73u:0x72u, false, ymm, X64XmmReg_XMM6, X64XmmReg_XMM0, X64XmmReg_XMM0, (u8)(esz*8 - 1));
			}
			if      (esz == 1) x64_emit_vex_rr(a, 1u, 1u, 0xD7u, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM0); // VPMOVMSKB → EAX
			else if (esz == 4) x64_emit_vex_rr(a, 0u, 1u, 0x50u, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM0); // VMOVMSKPS
			else               x64_emit_vex_rr(a, 1u, 1u, 0x50u, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM0); // VMOVMSKPD
			if (n < 32) x64_emit_and_ri(a, X64OpSize_64, X64Reg_RAX, (i32)(((i64)1 << n) - 1));
			return x64v_reg(rt, X64Reg_RAX);
		}

		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		i32 bo = x64_alloc_local(p, 8, 8);
		x64_emit_lea(a, X64Reg_RAX, vm.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(bo), X64Reg_RAX);
		x64_emit_xor_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_R8); // bitmask accumulator
		for (i64 i = 0; i < n; i++) {
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(bo));
			X64Mem lane = x64_mem(X64Reg_RAX, (i32)(i*esz));
			if (os == X64OpSize_64 || os == X64OpSize_32) x64_emit_mov_rm(a, os, X64Reg_RAX, lane); // 32 auto-zext
			else                                          x64_emit_movzx_rm(a, os, X64Reg_RAX, lane);
			if (bit != 0) { x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, bit); x64_emit_shr_rcl(a, X64OpSize_64, X64Reg_RAX); }
			x64_emit_and_ri(a, X64OpSize_64, X64Reg_RAX, 1);
			if (i != 0)   { x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, i);   x64_emit_shl_rcl(a, X64OpSize_64, X64Reg_RAX); }
			x64_emit_or_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX);
		}
		x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RAX, X64Reg_R8);
		return x64v_reg(rt, X64Reg_RAX);
	}

	// shuffle(a, b, idx...) → gather constant lanes from concat(a,b). Result lane count = #index args.
	if (id == BuiltinProc_simd_shuffle) {
		Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 ca = avt->SimdVector.count;
		i64 n = ce->args.count - 2;
		GB_ASSERT(n >= 0 && n <= 256);
		i64 idx[256];
		for (i64 i = 0; i < n; i++) idx[i] = exact_value_to_i64(ce->args[i+2]->tav.value);
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		x64Value bm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), avt);
		return x64_simd_shuffle_const(p, am.mem, bm.mem, ca, esz, idx, n, rt);
	}

	// odd_even(a, b): result[0..N/2) = odd lanes of a (concat idx 2i+1); [N/2..N) = even lanes of b (2i+N).
	if (id == BuiltinProc_simd_odd_even) {
		Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = avt->SimdVector.count; GB_ASSERT(N <= 256);
		i64 idx[256];
		for (i64 i = 0; i < N/2; i++) idx[i]       = 2*i + 1;
		for (i64 i = 0; i < N/2; i++) idx[i + N/2] = 2*i + N;
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		x64Value bm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), avt);
		return x64_simd_shuffle_const(p, am.mem, bm.mem, N, esz, idx, N, rt);
	}

	// lanes_reverse(a): res[i] = a[N-1-i].
	if (id == BuiltinProc_simd_lanes_reverse) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vt->SimdVector.count; GB_ASSERT(N <= 256);
		i64 idx[256];
		for (i64 i = 0; i < N; i++) idx[i] = N-1-i;
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		return x64_simd_shuffle_const(p, am.mem, am.mem, N, esz, idx, N, rt);
	}

	// lanes_rotate_left/right(a, n): res[i] = a[(i+left) mod N], left = ±n mod N (N power of two).
	if (id == BuiltinProc_simd_lanes_rotate_left || id == BuiltinProc_simd_lanes_rotate_right) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vt->SimdVector.count; GB_ASSERT(is_power_of_two(N) && N <= 256);
		i64 r = exact_value_to_i64(ce->args[1]->tav.value);
		if (id == BuiltinProc_simd_lanes_rotate_right) r = -r;
		r = ((r % N) + N) % N;
		i64 idx[256];
		for (i64 i = 0; i < N; i++) idx[i] = (i + r) & (N - 1);
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		return x64_simd_shuffle_const(p, am.mem, am.mem, N, esz, idx, N, rt);
	}

	// runtime_swizzle(a, indices): res[i] = a[indices[i]] (runtime indices, masked to N-1 for pow2 N).
	if (id == BuiltinProc_simd_runtime_swizzle) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vt->SimdVector.count;
		Type *ivt = base_type(x64_typed(ce->args[1]->tav.type));
		GB_ASSERT(ivt != nullptr && ivt->kind == Type_SimdVector);
		Type *ielem = base_type(ivt->SimdVector.elem);
		i64 isz = type_size_of(ielem); if (isz <= 0) isz = 1;
		X64OpSize ios = x64_op_size_of(ielem);
		bool pow2 = is_power_of_two(N);
		X64OpSize os = esz<=1?X64OpSize_8 : esz==2?X64OpSize_16 : esz<=4?X64OpSize_32 : X64OpSize_64;
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		x64Value im = x64_spill_value(p, x64_build_expr(p, ce->args[1]), ivt);

		// AVX2: single-instruction cross-lane permute for 32-bit lanes with 32-bit indices. 256-bit (8
		// lanes) → VPERMD (indices=vvvv, data=rm; selects idx&7). 128-bit (4 lanes) → VPERMILPS variable
		// (data=vvvv, indices=rm; selects idx&3). Both mask the index to the lane count, matching Odin's
		// idx&(N-1). Other lane sizes (8/16/64-bit — no vector-indexed AVX2 permute) fall to the scalar loop.
		if (x64_use_avx2() && esz == 4 && isz == 4 && (type_size_of(vt) == 16 || type_size_of(vt) == 32)) {
			i32 res = x64_alloc_local(p, type_size_of(vt), gb_max(type_align_of(vt), (i64)16));
			if (type_size_of(vt) == 32) {
				x64_emit_vmovups_rm(a, true,  X64XmmReg_XMM1, im.mem); // indices
				x64_emit_vmovups_rm(a, true,  X64XmmReg_XMM2, am.mem); // data
				x64_emit_vex_rr(a, 1u, 2u, 0x36u, false, true, X64XmmReg_XMM0, X64XmmReg_XMM1, X64XmmReg_XMM2); // vpermd
				x64_emit_vmovups_mr(a, true,  x64_rbp_mem(res), X64XmmReg_XMM0);
			} else {
				x64_emit_vmovups_rm(a, false, X64XmmReg_XMM1, am.mem); // data
				x64_emit_vmovups_rm(a, false, X64XmmReg_XMM2, im.mem); // indices
				x64_emit_vex_rr(a, 1u, 2u, 0x0Cu, false, false, X64XmmReg_XMM0, X64XmmReg_XMM1, X64XmmReg_XMM2); // vpermilps
				x64_emit_vmovups_mr(a, false, x64_rbp_mem(res), X64XmmReg_XMM0);
			}
			return x64v_mem(rt, x64_rbp_mem(res));
		}

		i32 ao = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, am.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(ao), X64Reg_RAX);
		i32 io = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, im.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(io), X64Reg_RAX);
		i32 res = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
		for (i64 i = 0; i < N; i++) {
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(io));
			x64_emit_movzx_rm(a, ios, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*isz)));
			if (pow2) x64_emit_and_ri(a, X64OpSize_64, X64Reg_RCX, (i32)(N-1));
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ao));
			x64_emit_mov_rm(a, os, X64Reg_RDX, x64_mem_idx(X64Reg_RAX, X64Reg_RCX, (u8)esz, 0));
			x64_emit_mov_mr(a, os, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RDX);
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// pairwise_add/sub(a, b): x=concat(a,b) even lanes, y=odd lanes, res = x op y (per lane).
	if (id == BuiltinProc_simd_pairwise_add || id == BuiltinProc_simd_pairwise_sub) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vt->SimdVector.count; GB_ASSERT(N <= 256);
		TokenKind op = (id == BuiltinProc_simd_pairwise_add) ? Token_Add : Token_Sub;
		i64 ev[256], od[256];
		for (i64 i = 0; i < N; i++) { ev[i] = 2*i; od[i] = 2*i + 1; }
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		x64Value bm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);
		x64Value xs = x64_spill_value(p, x64_simd_shuffle_const(p, am.mem, bm.mem, N, esz, ev, N, vt), vt);
		x64Value ys = x64_spill_value(p, x64_simd_shuffle_const(p, am.mem, bm.mem, N, esz, od, N, vt), vt);
		i32 res = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
		for (i64 i = 0; i < N; i++) {
			X64Mem xm = xs.mem; xm.disp += (i32)(i*esz);
			X64Mem ym = ys.mem; ym.disp += (i32)(i*esz);
			x64Value la = x64_spill_value(p, x64_load_addr(p, x64addr(xm, elem)), elem);
			x64Value lb = x64_spill_value(p, x64_load_addr(p, x64addr(ym, elem)), elem);
			x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(i*esz)), elem), x64_emit_arith(p, op, la, lb, elem));
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// sums_of_n(a, n): res[g] = sum of lanes a[g*n .. g*n+n-1]; result has N/n lanes.
	if (id == BuiltinProc_simd_sums_of_n) {
		Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = avt->SimdVector.count;
		i64 nn = exact_value_to_i64(ce->args[1]->tav.value);
		GB_ASSERT(nn > 0 && N % nn == 0);
		i64 outn = N / nn;
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		i32 res = x64_alloc_local(p, type_size_of(vt), gb_max(type_align_of(vt), (i64)1));
		x64_zero_mem(p, x64_rbp_mem(res), type_size_of(vt));
		for (i64 g = 0; g < outn; g++) {
			X64Mem m0 = am.mem; m0.disp += (i32)((g*nn)*esz);
			x64Value acc = x64_spill_value(p, x64_load_addr(p, x64addr(m0, elem)), elem);
			for (i64 j = 1; j < nn; j++) {
				X64Mem m = am.mem; m.disp += (i32)((g*nn + j)*esz);
				x64Value lane = x64_spill_value(p, x64_load_addr(p, x64addr(m, elem)), elem);
				acc = x64_spill_value(p, x64_emit_arith(p, Token_Add, acc, lane, elem), elem);
			}
			x64_store_value(p, x64addr(x64_rbp_mem(res + (i32)(g*esz)), elem), acc);
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// saturating_add/sub. Native packed VPADDS/VPSUBS (signed) / VPADDUS/VPSUBUS (unsigned) for 8/16-bit
	// lanes under AVX2 — the only widths with a saturating instruction. 32-bit → extend-to-64 + clamp.
	// 64-bit → scalar overflow detection (no saturating insn at this width on any x86).
	if (id == BuiltinProc_simd_saturating_add || id == BuiltinProc_simd_saturating_sub) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 n = vt->SimdVector.count;
		bool sub = (id == BuiltinProc_simd_saturating_sub);
		bool sgn = !is_type_unsigned(elem);
		X64OpSize os = x64_op_size_of(elem);
		x64Value am = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		x64Value bm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vt);
		bool avx = x64_use_avx2();

		if (avx && (esz == 1 || esz == 2) && type_size_of(vt) % 16 == 0) { // native packed saturating (≥128-bit)
			u8 op = sub ? (sgn ? (esz==1?0xE8u:0xE9u) : (esz==1?0xD8u:0xD9u))
			            : (sgn ? (esz==1?0xECu:0xEDu) : (esz==1?0xDCu:0xDDu));
			i64 total = type_size_of(vt);
			i32 res = x64_alloc_local(p, total, gb_max(type_align_of(vt), (i64)16));
			for (i64 off = 0; off < total; ) {
				bool ymm = (total - off) >= 32;
				X64Mem ac = am.mem; ac.disp += (i32)off;
				X64Mem bc = bm.mem; bc.disp += (i32)off;
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, ac);
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, bc);
				x64_emit_vex_rr(a, 1u, 1u, op, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM0, X64XmmReg_XMM1);
				x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
				off += ymm ? 32 : 16;
			}
			return x64v_mem(rt, x64_rbp_mem(res));
		}

		i32 ao = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, am.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(ao), X64Reg_RAX);
		i32 bo = x64_alloc_local(p, 8, 8); x64_emit_lea(a, X64Reg_RAX, bm.mem); x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(bo), X64Reg_RAX);
		i32 res = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));

		if (esz <= 4) { // extend to 64-bit (can't overflow) → clamp → truncate
			i64 bits = esz*8;
			i64 tmax = sgn ? (((i64)1 << (bits-1)) - 1) : (((i64)1 << bits) - 1);
			i64 tmin = sgn ? -((i64)1 << (bits-1)) : 0;
			for (i64 i = 0; i < n; i++) {
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ao));
				if (sgn) x64_emit_movsx_rm(a, os, X64Reg_R8, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				else     x64_emit_movzx_rm(a, os, X64Reg_R8, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(bo));
				if (sgn) x64_emit_movsx_rm(a, os, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				else     x64_emit_movzx_rm(a, os, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				if (sub) x64_emit_sub_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RCX);
				else     x64_emit_add_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RCX);
				x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, tmax);
				x64_emit_cmp_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_R9);
				x64_emit_cmov_rr(a, X64Cc_G, X64OpSize_64, X64Reg_R8, X64Reg_R9); // > max → max
				x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, tmin);
				x64_emit_cmp_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_R9);
				x64_emit_cmov_rr(a, X64Cc_L, X64OpSize_64, X64Reg_R8, X64Reg_R9); // < min → min
				x64_emit_mov_mr(a, os, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_R8);
			}
			return x64v_mem(rt, x64_rbp_mem(res));
		}

		// 64-bit: r = a±b, then detect overflow and saturate. signed add ovf = (a^r)&(b^r)<0; signed sub
		// ovf = (a^b)&(a^r)<0; unsigned add ovf = carry (r<a); unsigned sub ovf = borrow (a<b → 0).
		i64 smax = (i64)0x7FFFFFFFFFFFFFFFll, smin = (i64)0x8000000000000000ull;
		for (i64 i = 0; i < n; i++) {
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(ao)); x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R8,  x64_mem(X64Reg_RAX, (i32)(i*8))); // a
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(bo)); x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*8))); // b
			x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R10, X64Reg_R8); // save a
			if (!sub) {
				x64_emit_add_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RCX); // r=a+b
				if (sgn) {
					x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_R10); x64_emit_xor_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_R8); // a^r
					x64_emit_xor_rr(a, X64OpSize_64, X64Reg_RCX, X64Reg_R8);                                                          // b^r
					x64_emit_and_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_RCX);                                                         // ovf flag
					x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, smax); x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RDX, smin);
					x64_emit_test_rr(a, X64OpSize_64, X64Reg_R10, X64Reg_R10); x64_emit_cmov_rr(a, X64Cc_S, X64OpSize_64, X64Reg_R9, X64Reg_RDX); // a<0?min:max
					x64_emit_test_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_R11); x64_emit_cmov_rr(a, X64Cc_S, X64OpSize_64, X64Reg_R8, X64Reg_R9);
				} else {
					x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, -1);
					x64_emit_cmp_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_R10); x64_emit_cmov_rr(a, X64Cc_B, X64OpSize_64, X64Reg_R8, X64Reg_R9); // r<a → UINT64_MAX
				}
			} else {
				x64_emit_sub_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RCX); // r=a-b
				if (sgn) {
					x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_R10); x64_emit_xor_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_RCX); // a^b
					x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R9,  X64Reg_R10); x64_emit_xor_rr(a, X64OpSize_64, X64Reg_R9,  X64Reg_R8);  // a^r
					x64_emit_and_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_R9);                                                           // ovf flag
					x64_emit_mov_ri(a, X64OpSize_64, X64Reg_R9, smax); x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RDX, smin);
					x64_emit_test_rr(a, X64OpSize_64, X64Reg_R10, X64Reg_R10); x64_emit_cmov_rr(a, X64Cc_S, X64OpSize_64, X64Reg_R9, X64Reg_RDX);
					x64_emit_test_rr(a, X64OpSize_64, X64Reg_R11, X64Reg_R11); x64_emit_cmov_rr(a, X64Cc_S, X64OpSize_64, X64Reg_R8, X64Reg_R9);
				} else {
					x64_emit_xor_rr(a, X64OpSize_64, X64Reg_R9, X64Reg_R9);
					x64_emit_cmp_rr(a, X64OpSize_64, X64Reg_R10, X64Reg_RCX); x64_emit_cmov_rr(a, X64Cc_B, X64OpSize_64, X64Reg_R8, X64Reg_R9); // a<b → 0
				}
			}
			x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(res + (i32)(i*8)), X64Reg_R8);
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// approx_recip(x) = 1/x ; approx_recip_sqrt(x) = 1/sqrt(x). Exact per-lane (acceptable for an
	// "approximate" builtin); lb falls back to the same 1.0/x when no SSE intrinsic is selected.
	if (id == BuiltinProc_simd_approx_recip || id == BuiltinProc_simd_approx_recip_sqrt) {
		GB_ASSERT(vt != nullptr && vt->kind == Type_SimdVector);
		Type *elem = base_type(vt->SimdVector.elem);
		bool d64 = type_size_of(elem) == 8;
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vt->SimdVector.count;
		bool rsqrt = (id == BuiltinProc_simd_approx_recip_sqrt);
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), vt);
		i32 one = x64_alloc_local(p, 8, 8); // stash 1.0 (div clobbers XMM0)
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, 1);
		if (d64) { x64_emit_cvtsi2sd(a, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64); x64_emit_movsd_mr(a, x64_rbp_mem(one), X64XmmReg_XMM0); }
		else     { x64_emit_cvtsi2ss(a, X64XmmReg_XMM0, X64Reg_RAX, X64OpSize_64); x64_emit_movss_mr(a, x64_rbp_mem(one), X64XmmReg_XMM0); }
		i32 res = x64_alloc_local(p, type_size_of(vt), type_align_of(vt));
		for (i64 i = 0; i < N; i++) {
			X64Mem m = vm.mem; m.disp += (i32)(i*esz);
			if (d64) {
				x64_emit_movsd_rm(a, X64XmmReg_XMM1, m);
				if (rsqrt) x64_emit_sqrtsd(a, X64XmmReg_XMM1, X64XmmReg_XMM1);
				x64_emit_movsd_rm(a, X64XmmReg_XMM0, x64_rbp_mem(one));
				x64_emit_divsd(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
				x64_emit_movsd_mr(a, x64_rbp_mem(res + (i32)(i*esz)), X64XmmReg_XMM0);
			} else {
				x64_emit_movss_rm(a, X64XmmReg_XMM1, m);
				if (rsqrt) x64_emit_sqrtss(a, X64XmmReg_XMM1, X64XmmReg_XMM1);
				x64_emit_movss_rm(a, X64XmmReg_XMM0, x64_rbp_mem(one));
				x64_emit_divss(a, X64XmmReg_XMM0, X64XmmReg_XMM1);
				x64_emit_movss_mr(a, x64_rbp_mem(res + (i32)(i*esz)), X64XmmReg_XMM0);
			}
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// interleave(a, b, …) → result[g*n + s] = arg[s][g] (n = arg count, M lanes each → n*M lanes).
	if (id == BuiltinProc_simd_interleave) {
		i64 n = ce->args.count;
		Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 M = avt->SimdVector.count;
		GB_ASSERT(n <= 64);
		X64Mem mems[64];
		for (i64 s = 0; s < n; s++) mems[s] = x64_spill_value(p, x64_build_expr(p, ce->args[s]), avt).mem;
		i32 res = x64_alloc_local(p, type_size_of(vt), gb_max(type_align_of(vt), (i64)1));
		for (i64 g = 0; g < M; g++) for (i64 s = 0; s < n; s++) {
			X64Mem src = mems[s]; src.disp += (i32)(g*esz);
			x64_simd_copy_lane(p, x64_rbp_mem(res + (i32)((g*n + s)*esz)), src, esz);
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	// gather / scatter (ptr is a #simd[N]rawptr). mask lane true ⟺ bit 0 set (LLVM trunc-to-i1 parity).
	// gather: res[i] = mask[i] ? *ptr[i] : values[i].  scatter: if mask[i] *ptr[i] = values[i] (→ none).
	if (id == BuiltinProc_simd_gather || id == BuiltinProc_simd_scatter) {
		bool is_gather = (id == BuiltinProc_simd_gather);
		Type *pvt = base_type(x64_typed(ce->args[0]->tav.type));
		Type *vvt = base_type(x64_typed(ce->args[1]->tav.type));
		Type *mvt = base_type(x64_typed(ce->args[2]->tav.type));
		Type *elem = base_type(vvt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vvt->SimdVector.count;
		i64 msz = type_size_of(base_type(mvt->SimdVector.elem)); if (msz <= 0) msz = 1;
		X64OpSize eos = esz<=1?X64OpSize_8:esz==2?X64OpSize_16:esz<=4?X64OpSize_32:X64OpSize_64;
		x64Value pm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), pvt);
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vvt);
		x64Value mm = x64_spill_value(p, x64_build_expr(p, ce->args[2]), mvt);
		i32 po=x64_alloc_local(p,8,8); x64_emit_lea(a,X64Reg_RAX,pm.mem); x64_emit_mov_mr(a,X64OpSize_64,x64_rbp_mem(po),X64Reg_RAX);
		i32 vo=x64_alloc_local(p,8,8); x64_emit_lea(a,X64Reg_RAX,vm.mem); x64_emit_mov_mr(a,X64OpSize_64,x64_rbp_mem(vo),X64Reg_RAX);
		i32 mo=x64_alloc_local(p,8,8); x64_emit_lea(a,X64Reg_RAX,mm.mem); x64_emit_mov_mr(a,X64OpSize_64,x64_rbp_mem(mo),X64Reg_RAX);
		i32 res = is_gather ? x64_alloc_local(p, type_size_of(vvt), type_align_of(vvt)) : 0;
		for (i64 i = 0; i < N; i++) {
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(mo));
			if (msz >= 4) x64_emit_mov_rm(a, msz==8?X64OpSize_64:X64OpSize_32, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*msz)));
			else          x64_emit_movzx_rm(a, msz==2?X64OpSize_16:X64OpSize_8, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*msz)));
			x64_emit_and_ri(a, X64OpSize_64, X64Reg_RCX, 1);
			isize els = x64_label_alloc(&p->asm_), end = x64_label_alloc(&p->asm_);
			x64_emit_jcc(&p->asm_, X64Cc_E, els); // bit0==0 → masked off
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(po));
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, (i32)(i*8))); // ptr[i]
			if (is_gather) {
				x64_emit_mov_rm(a, eos, X64Reg_RDX, x64_mem(X64Reg_RAX, 0));
				x64_emit_mov_mr(a, eos, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RDX);
			} else {
				x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX);
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));
				x64_emit_mov_rm(a, eos, X64Reg_RDX, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				x64_emit_mov_mr(a, eos, x64_mem(X64Reg_R8, 0), X64Reg_RDX);
			}
			x64_emit_jmp(&p->asm_, end);
			x64_label_bind(&p->asm_, els);
			if (is_gather) { // res[i] = values[i]
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));
				x64_emit_mov_rm(a, eos, X64Reg_RDX, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				x64_emit_mov_mr(a, eos, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RDX);
			}
			x64_label_bind(&p->asm_, end);
		}
		return is_gather ? x64v_mem(rt, x64_rbp_mem(res)) : x64v_none();
	}

	// masked_load/store/expand_load/compress_store (ptr is a single ^T/rawptr). mask true ⟺ bit 0.
	// load:  res[i] = mask[i] ? ptr[i]        : values[i]      store:  if mask[i] ptr[i] = values[i]
	// expand:res[i] = mask[i] ? ptr[cursor++] : values[i]      compress: if mask[i] ptr[cursor++] = values[i]
	if (id == BuiltinProc_simd_masked_load || id == BuiltinProc_simd_masked_store ||
	    id == BuiltinProc_simd_masked_expand_load || id == BuiltinProc_simd_masked_compress_store) {
		bool is_load    = (id == BuiltinProc_simd_masked_load || id == BuiltinProc_simd_masked_expand_load);
		bool is_packed  = (id == BuiltinProc_simd_masked_expand_load || id == BuiltinProc_simd_masked_compress_store);
		Type *vvt = base_type(x64_typed(ce->args[1]->tav.type));
		Type *mvt = base_type(x64_typed(ce->args[2]->tav.type));
		Type *elem = base_type(vvt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 N = vvt->SimdVector.count;
		i64 msz = type_size_of(base_type(mvt->SimdVector.elem)); if (msz <= 0) msz = 1;
		X64OpSize eos = esz<=1?X64OpSize_8:esz==2?X64OpSize_16:esz<=4?X64OpSize_32:X64OpSize_64;
		x64Value pbv = x64_spill_value(p, x64_build_expr(p, ce->args[0]), t_rawptr);
		x64Value vm  = x64_spill_value(p, x64_build_expr(p, ce->args[1]), vvt);
		x64Value mm  = x64_spill_value(p, x64_build_expr(p, ce->args[2]), mvt);

		// AVX2 native masked_load/store via VPMASKMOVD/Q (32/64-bit; mask lane size must match the data
		// lane size). The mask is converted bit0→all-ones (so its high bit matches the VPMASKMOV/VPBLENDVB
		// convention, matching LLVM's trunc-to-i1). expand_load/compress_store are AVX-512 → scalar below.
		if (x64_use_avx2() && !is_packed && (esz == 4 || esz == 8) && msz == esz) {
			bool W = (esz == 8);
			i64 total = type_size_of(vvt);
			i32 one = x64_alloc_local(p, 8, 8); x64_emit_mov_mi(a, W?X64OpSize_64:X64OpSize_32, x64_rbp_mem(one), 1);
			i32 res = is_load ? x64_alloc_local(p, total, gb_max(type_align_of(vvt), (i64)16)) : 0;
			for (i64 off = 0; off < total; ) {
				bool ymm = (total - off) >= 32;
				x64_emit_vex_rm(a, 1u, 2u, W?0x59u:0x58u, false, ymm, X64XmmReg_XMM3, X64XmmReg_XMM0, x64_rbp_mem(one)); // YMM3 = 1
				X64Mem mc = mm.mem; mc.disp += (i32)off;
				x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM1, mc);
				x64_emit_vex_rr(a, 1u, 1u, 0xDBu, false, ymm, X64XmmReg_XMM1, X64XmmReg_XMM1, X64XmmReg_XMM3);            // mask & 1
				x64_emit_vex_rr(a, 1u, W?2u:1u, W?0x29u:0x76u, false, ymm, X64XmmReg_XMM1, X64XmmReg_XMM1, X64XmmReg_XMM3); // ==1 → all-ones
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, pbv.mem);
				X64Mem pm = x64_mem(X64Reg_RAX, (i32)off);
				if (is_load) {
					x64_emit_vex_rm(a, 1u, 2u, 0x8Cu, W, ymm, X64XmmReg_XMM0, X64XmmReg_XMM1, pm); // load (0 where masked off)
					X64Mem vc = vm.mem; vc.disp += (i32)off;
					x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM2, vc);
					x64_emit_vex_rr_i(a, 1u, 3u, 0x4Cu, false, ymm, X64XmmReg_XMM0, X64XmmReg_XMM2, X64XmmReg_XMM0, (u8)(1u<<4)); // mask?loaded:values
					x64_emit_vmovups_mr(a, ymm, x64_rbp_mem(res + (i32)off), X64XmmReg_XMM0);
				} else {
					X64Mem vc = vm.mem; vc.disp += (i32)off;
					x64_emit_vmovups_rm(a, ymm, X64XmmReg_XMM0, vc);
					x64_emit_vex_mr(a, 1u, 2u, 0x8Eu, W, ymm, pm, X64XmmReg_XMM1, X64XmmReg_XMM0); // masked store
				}
				off += ymm ? 32 : 16;
			}
			return is_load ? x64v_mem(rt, x64_rbp_mem(res)) : x64v_none();
		}

		i32 vo=x64_alloc_local(p,8,8); x64_emit_lea(a,X64Reg_RAX,vm.mem); x64_emit_mov_mr(a,X64OpSize_64,x64_rbp_mem(vo),X64Reg_RAX);
		i32 mo=x64_alloc_local(p,8,8); x64_emit_lea(a,X64Reg_RAX,mm.mem); x64_emit_mov_mr(a,X64OpSize_64,x64_rbp_mem(mo),X64Reg_RAX);
		i32 cur = -1;
		if (is_packed) { cur = x64_alloc_local(p,8,8); x64_emit_mov_mi(a, X64OpSize_64, x64_rbp_mem(cur), 0); }
		i32 res = is_load ? x64_alloc_local(p, type_size_of(vvt), type_align_of(vvt)) : 0;
		for (i64 i = 0; i < N; i++) {
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(mo));
			if (msz >= 4) x64_emit_mov_rm(a, msz==8?X64OpSize_64:X64OpSize_32, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*msz)));
			else          x64_emit_movzx_rm(a, msz==2?X64OpSize_16:X64OpSize_8, X64Reg_RCX, x64_mem(X64Reg_RAX, (i32)(i*msz)));
			x64_emit_and_ri(a, X64OpSize_64, X64Reg_RCX, 1);
			isize els = x64_label_alloc(&p->asm_), end = x64_label_alloc(&p->asm_);
			x64_emit_jcc(&p->asm_, X64Cc_E, els);
			// compute the element address into RAX: base + (packed ? cursor : i)*esz
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, pbv.mem);
			if (is_packed) {
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(cur));
				x64_emit_lea(a, X64Reg_RAX, x64_mem_idx(X64Reg_RAX, X64Reg_RCX, (u8)esz, 0));
			} else {
				X64Mem em = x64_mem(X64Reg_RAX, (i32)(i*esz)); x64_emit_lea(a, X64Reg_RAX, em);
			}
			if (is_load) {
				x64_emit_mov_rm(a, eos, X64Reg_RDX, x64_mem(X64Reg_RAX, 0));
				x64_emit_mov_mr(a, eos, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RDX);
			} else {
				x64_emit_mov_rr(a, X64OpSize_64, X64Reg_R8, X64Reg_RAX);
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));
				x64_emit_mov_rm(a, eos, X64Reg_RDX, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				x64_emit_mov_mr(a, eos, x64_mem(X64Reg_R8, 0), X64Reg_RDX);
			}
			if (is_packed) { // cursor++
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(cur));
				x64_emit_add_ri(a, X64OpSize_64, X64Reg_RCX, 1);
				x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(cur), X64Reg_RCX);
			}
			x64_emit_jmp(&p->asm_, end);
			x64_label_bind(&p->asm_, els);
			if (is_load) { // res[i] = values[i]
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));
				x64_emit_mov_rm(a, eos, X64Reg_RDX, x64_mem(X64Reg_RAX, (i32)(i*esz)));
				x64_emit_mov_mr(a, eos, x64_rbp_mem(res + (i32)(i*esz)), X64Reg_RDX);
			}
			x64_label_bind(&p->asm_, end);
		}
		return is_load ? x64v_mem(rt, x64_rbp_mem(res)) : x64v_none();
	}

	// deinterleave(v, n) → tuple of n vectors (each M/n lanes); stream s = [v[s], v[s+n], …]. Result is
	// a tuple materialized in stack memory with field s at type_offset_of(rt, s) (the consumer — a
	// multi-assign — reads each field by that offset, same convention as a multi-return call result).
	if (id == BuiltinProc_simd_deinterleave) {
		Type *avt = base_type(x64_typed(ce->args[0]->tav.type));
		GB_ASSERT(avt != nullptr && avt->kind == Type_SimdVector);
		Type *elem = base_type(avt->SimdVector.elem);
		i64 esz = type_size_of(elem); if (esz <= 0) esz = 1;
		i64 M = avt->SimdVector.count;
		i64 n = exact_value_to_i64(ce->args[1]->tav.value);
		GB_ASSERT(n >= 1 && M % n == 0);
		i64 per = M / n;
		x64Value vm = x64_spill_value(p, x64_build_expr(p, ce->args[0]), avt);
		i32 res = x64_alloc_local(p, type_size_of(rt), gb_max(type_align_of(rt), (i64)1));
		bool is_tuple = (base_type(rt) != nullptr && base_type(rt)->kind == Type_Tuple);
		for (i64 s = 0; s < n; s++) {
			i32 foff = is_tuple ? (i32)type_offset_of(rt, s) : 0; // n==1 collapses to the vector type
			for (i64 g = 0; g < per; g++) {
				X64Mem src = vm.mem; src.disp += (i32)((g*n + s)*esz);
				x64_simd_copy_lane(p, x64_rbp_mem(res + foff + (i32)(g*esz)), src, esz);
			}
		}
		return x64v_mem(rt, x64_rbp_mem(res));
	}

	GB_PANIC("x64 SIMD builtin '%.*s' (id=%d) unimplemented — mirror lb_build_builtin_simd_proc",
	         LIT(builtin_procs[id].name), id);
	return x64v_none();
}

// All SIMD-builtin results materialize to stack memory or a GP register (never a live YMM reg), so a
// VZEROUPPER here is safe and clears the dirty upper-YMM state before control returns to the backend's
// surrounding legacy-SSE code (avoids the SSE↔AVX transition stall). Only emitted when AVX2 is active.
gb_internal x64Value x64_build_builtin_simd_proc(x64Procedure *p, Ast *expr, i32 id) {
	bool avx = x64_use_avx2();
	x64Value r = x64_build_builtin_simd_proc_inner(p, expr, id);
	if (avx) x64_emit_vzeroupper(&p->asm_);
	return r;
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
		return x64_typeid(tav.type);
	}
	// A typeid CONSTANT whose mode is NOT Addressing_Type — e.g. a POINTER type `^T` used directly as a
	// value (`x.id == ^os.File`): the checker folds it to an ExactValue_Typeid with mode Value/Constant,
	// so the check above misses it and it fell through to a bogus AST path (UnaryExpr `^`) → typeid 0.
	// (A plain named/struct type IS Addressing_Type and is handled above; only the folded forms land here.)
	// Was core:flags `type_info.id == ^os.File` (and datetime/net) always false → Unsupported_Type.
	if (tav.value.kind == ExactValue_Typeid) {
		Type *tt = tav.value.value_typeid ? tav.value.value_typeid : tav.type;
		if (tt != nullptr) return x64_typeid(tt);
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
			// SCALAR >8-byte int (i128/u128) or wide bit_set only. An ARRAY/#simd/matrix-typed integer
			// constant (e.g. `[2]int` is 16B) must NOT be written as one wide scalar — that yields
			// {value, 0, …} instead of a broadcast. Let it fall to the array-broadcast handler below.
			// Was THE blick no-text bug: `base_slot * BASE_SLOT_SIZE` ([2]int * 16) → {x, 0} → every glyph
			// in atlas row 0 → wrong/garbled glyph UVs.
			if (isz > 8 && ct->kind != Type_Array && ct->kind != Type_SimdVector && ct->kind != Type_Matrix) {
				u8 stackbuf[32] = {};
				u8 *buf = stackbuf;
				if (isz > (i64)gb_size_of(stackbuf)) {
					buf = cast(u8 *)gb_alloc(temporary_allocator(), isz);
					gb_zero_size(buf, isz);
				}
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

		// Complex/quaternion constants: emit once into .rdata via the shared x64_const_value (the
		// SINGLE owner of complex/quaternion layout — also used by module globals and by &CONST in
		// x64_build_addr) and load it. Keyed on TYPE and placed BEFORE the float branch, because
		// x64_is_float() is TRUE for complex (BasicFlag_Complex) and would otherwise grab it and
		// emit a single scalar 0.0 word (→ real=0, imag=stale).
		if (is_type_complex(ct) || is_type_quaternion(ct)) {
			x64Module *m = p->module;
			i64 sz = x64_type_size(x64_typed(tav.type)); if (sz <= 0) sz = 8;
			i64 al = x64_type_align(x64_typed(tav.type)); if (al <= 0) al = 8;
			u32 goff = x64_const_reserve(m->rdata, al, sz);
			x64_const_value(m, m->rdata, goff, x64_typed(tav.type), ev, tav.type);
			String sym = x64_const_new_name(m, "__$ccg$");
			x64_const_define_symbol(m, sym, 2 /*.rdata*/, goff);
			x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
			return x64v_mem(x64_typed(tav.type), x64_mem(X64Reg_RAX, 0));
		}

		// Array/#simd-typed constant carrying a SCALAR exact value (e.g. `vec * 0.5`: the checker types
		// the untyped `0.5` operand as the array type [2]f32, not f32). Build the scalar at the element
		// type and broadcast to every lane via x64_emit_conv (mirrors lb_const_value's array path). Was:
		// fell through to the integer fallback → exact_value_to_i64(0.5)=0 → all lanes 0. THE blick
		// all-text-at-origin bug: glyph center_position = (placement_p0+placement_p1)*0.5 → {0,0}.
		if ((ct->kind == Type_Array || ct->kind == Type_SimdVector) &&
		    (ev.kind == ExactValue_Float || ev.kind == ExactValue_Integer)) {
			Type *elem = (ct->kind == Type_Array) ? base_array_type(tav.type) : ct->SimdVector.elem;
			Type *ebt  = base_type(elem);
			f64   fv   = (ev.kind == ExactValue_Float) ? ev.value_float : (f64)big_int_to_i64(&ev.value_integer);
			x64Value sc;
			if (x64_is_f16(ebt)) {
				u16 h = f32_to_f16((f32)fv);
				if (ebt->kind == Type_Basic && (ebt->Basic.flags & BasicFlag_EndianBig) != 0) h = (u16)((h >> 8) | (h << 8));
				sc = x64v_imm(elem, (i64)h);
			} else if (x64_is_float(ebt)) {
				x64_emit_float_const(p, ebt, fv, (f32)fv);
				sc = x64v_xmm(elem, X64XmmReg_XMM0);
			} else {
				sc = x64v_imm(elem, (ev.kind == ExactValue_Integer) ? big_int_to_i64(&ev.value_integer) : (i64)fv);
			}
			return x64_emit_conv(p, sc, elem, x64_typed(tav.type)); // broadcast scalar → every lane
		}

		if (x64_is_float(ct) && !is_type_complex(ct) && !is_type_quaternion(ct)) {
			f64 fv = (ev.kind == ExactValue_Float)   ? ev.value_float :
			         (ev.kind == ExactValue_Integer)  ? (f64)big_int_to_i64(&ev.value_integer) : 0.0;
			x64_emit_float_const(p, ct, fv, (f32)fv);
			return x64v_xmm(x64_typed(tav.type), X64XmmReg_XMM0);
		}

		// f16 constant: x64_is_float excludes size-2, so it fell to the integer fallback below →
		// `(i64)1.0 == 1` (bits 0x0001) instead of the f16 encoding 0x3C00. Convert to f16 bits
		// (host f32_to_f16, shared with LLVM) as a 2-byte immediate; byte-swap for f16be.
		if (x64_is_f16(ct)) {
			f64 fv = (ev.kind == ExactValue_Float)   ? ev.value_float :
			         (ev.kind == ExactValue_Integer)  ? (f64)big_int_to_i64(&ev.value_integer) : 0.0;
			u16 h = f32_to_f16((f32)fv);
			Type *ctb = base_type(ct);
			if (ctb != nullptr && ctb->kind == Type_Basic && (ctb->Basic.flags & BasicFlag_EndianBig) != 0) {
				h = (u16)((h >> 8) | (h << 8));
			}
			return x64v_imm(x64_typed(tav.type), (i64)h);
		}

		if (ev.kind == ExactValue_String) {
			if (is_type_cstring(ct))   return x64_const_cstring_ptr(p, ev.value_string, false, x64_typed(tav.type));
			if (is_type_cstring16(ct)) return x64_const_cstring_ptr(p, ev.value_string, true,  x64_typed(tav.type));
			i64 esz = is_type_slice(ct) ? type_size_of(ct->Slice.elem) : 1;
			return x64_const_string(p, ev.value_string, esz);
		}

		// Constant typeid (type in a typeid context, e.g. poly param to struct_fields_zipped(T)).
		// Must be the canonical hash — same as typeid_of / Type_Info.id, or type_info_of(it)
		// misses the table. Without this it fell through to 0 → nil.
		if (ev.kind == ExactValue_Typeid) {
			Type *tt = ev.value_typeid ? ev.value_typeid : tav.type;
			return x64_typeid(tt);
		}

		// Proc value used as data (e.g. stored into a struct field). Top-level procs are
		// Addressing_Constant w/ ExactValue_Procedure, so they land here, not the Ident
		// Entity_Procedure case — emit lea_sym.
		if (ev.kind == ExactValue_Procedure) {
			// An ANONYMOUS proc literal folded to a constant — e.g. a struct-field value in a
			// compound literal (`{procedure = proc(){…}}`), which the named-proc walk below can't
			// resolve (proc_e is a Constant, not an Entity_Procedure) → it returned 0. Generate the
			// anon proc on-demand + lea, like x64_generate_anonymous_proc_lit. Was crypto.random_generator's
			// `procedure` field → nil → uuid generate_v7 CSPRNG assert (then the runner hung).
			Ast *plast = ev.value_procedure ? unparen_expr(ev.value_procedure) : unparen_expr(expr);
			if (plast != nullptr && plast->kind == Ast_ProcLit) {
				Entity *ae = x64_anon_proc_entity(p->module, plast);
				if (ae != nullptr) {
					x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(ae));
					return x64v_reg(x64_typed(tav.type), X64Reg_RAX);
				}
			}
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
		case ExactValue_Integer: iv = x64_endian_fix_const_int(tav.type, big_int_to_i64(&ev.value_integer)); break;
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
			// bare `using`-promoted field as an rvalue (`channels * width`) → params^.field, not a global.
			if (!x64_entity_is_local(p, e) && (e->flags & EntityFlag_Using)) {
				x64Addr ua = x64_get_using_variable(p, e);
				if (ua.type != nullptr) return x64_load_addr(p, ua);
			}
			x64Addr addr = x64_entity_is_local(p, e)
			             ? x64_entity_addr(p, e)
			             : x64addr(x64_global_mem(p, e), e->type);
			return x64_load_addr(p, addr);
		}
		case Entity_TypeName:
			// A type name used as a VALUE is its typeid (e.g. poly param `T` to a `typeid`
			// arg). e->type is the bound type in an instantiation. Canonical hash.
			return x64_typeid(e->type);
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
			return x64_build_unary_and(p, expr);
		}

		x64Value val = x64_build_expr(p, ue->expr);
		Type    *t   = val.type ? val.type : tav.type;
		return x64_emit_unary_arith(p, op, val, t);
	} case_end;

	// ── Binary expression ────────────────────────────────────────────────
	case_ast_node(be, BinaryExpr, expr); {
		TokenKind op = be->op.kind;
		Type *ltype       = x64_typed(be->left->tav.type);
		Type *result_type = x64_typed(tav.type ? tav.type : ltype);
		if (ltype == nullptr || result_type == nullptr) return x64v_none();

		// Short-circuit logical operators (value context)
		if (op == Token_CmpAnd || op == Token_CmpOr) {
			return x64_emit_logical_binary_expr(p, op, be->left, be->right, result_type);
		}

		// bit_set membership: elem in set / elem not_in set
		if (op == Token_in || op == Token_not_in) {
			return x64_build_binary_in(p, be->left, be->right, op);
		}

		// `x == nil` / `x != nil` where the generic scalar compare gets the union cases wrong.
		if (op == Token_CmpEq || op == Token_NotEq) {
			Ast *nn = nullptr;
			if      (is_type_untyped_nil(be->left->tav.type))  nn = be->right;
			else if (is_type_untyped_nil(be->right->tav.type)) nn = be->left;
			if (nn != nullptr) {
				x64Value r = x64_emit_comp_against_nil(p, op, nn);
				if (r.kind != x64Value_None) return r;
			}
		}

		// Generic binary op: build BOTH operands (spilled, so they're stable), then dispatch to
		// x64_emit_comp (string/aggregate/float/int comparisons) or x64_emit_arith. The special
		// cases above (&&/||, in/not_in, x==nil) returned already. Mirrors lb_build_expr's
		// BinaryExpr -> lb_emit_arith / lb_emit_comp.
		x64Value lhs = x64_spill_value(p, x64_build_expr(p, be->left), ltype);
		x64Value rhs = x64_spill_value(p, x64_build_expr(p, be->right), x64_typed(be->right->tav.type));
		bool is_cmp = op == Token_CmpEq || op == Token_NotEq || op == Token_Lt ||
		              op == Token_Gt    || op == Token_LtEq  || op == Token_GtEq;
		if (is_cmp) return x64_emit_comp(p, op, lhs, rhs);
		return x64_emit_arith(p, op, lhs, rhs, result_type);
	} case_end;

	// ── Type cast ─────────────────────────────────────────────────────────
	case_ast_node(tc, TypeCast, expr); {
		// cast(T)x converts (x64_emit_conv); transmute(T)x reinterprets bits (x64_emit_transmute).
		// Mirrors lb_build_expr's TypeCast switch on tc->token.kind.
		x64Value src = x64_build_expr(p, tc->expr);
		switch (tc->token.kind) {
		case Token_cast:      return x64_emit_conv(p, src, tc->expr->tav.type, tav.type);
		case Token_transmute: return x64_emit_transmute(p, src, tav.type);
		}
		GB_PANIC("Invalid AST TypeCast");
	} case_end;

	case_ast_node(ac, AutoCast, expr); {
		// auto_cast x — convert the operand to the context-inferred type (was a no-op before).
		x64Value src = x64_build_expr(p, ac->expr);
		return x64_emit_conv(p, src, ac->expr->tav.type, tav.type);
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
		// Multi-component swizzle (v.xyz / v.xy / v.wzyx): swizzle_count>0 means a NEW vector,
		// gathered from the selected element indices (swizzle_indices packs 2 bits each — mirrors
		// lb_build_addr). Single-component .x (count==0) stays on the addressable build_addr path
		// below. Read-only here; swizzle-as-lvalue (`v.xy = …`) is not yet supported.
		if (se->swizzle_count > 0) {
			bool  via_ptr = is_type_pointer(se->expr->tav.type);
			Type *src_t   = base_type(via_ptr ? type_deref(se->expr->tav.type) : se->expr->tav.type);
			Type *elem    = src_t->kind == Type_SimdVector ? src_t->SimdVector.elem : src_t->Array.elem;
			i64       esz = type_size_of(elem);
			X64OpSize osz = x64_op_size_of(elem);
			if (via_ptr) {
				x64Value pv = x64_build_expr(p, se->expr);
				x64_value_to_reg(p, pv, X64Reg_RCX);            // RCX = &array
			} else {
				x64_emit_lea(&p->asm_, X64Reg_RCX, x64_build_addr(p, se->expr).mem);
			}
			i32 off = x64_alloc_local(p, x64_type_size(tav.type), x64_type_align(tav.type));
			for (u8 i = 0; i < se->swizzle_count; i++) {
				u8 idx = (se->swizzle_indices >> (i*2)) & 3;
				x64_emit_mov_rm(&p->asm_, osz, X64Reg_RAX, x64_mem(X64Reg_RCX, (i32)(idx*esz)));
				x64_emit_mov_mr(&p->asm_, osz, x64_rbp_mem(off + (i32)(i*esz)), X64Reg_RAX);
			}
			return x64v_mem(x64_typed(tav.type), x64_rbp_mem(off));
		}
		// bit_field member read: shift/mask the backing integer (no addressable byte location).
		{
			Type *ft = nullptr, *bk = nullptr; i64 boff = 0, bsize = 0, byoff = 0;
			if (x64_bit_field_member_info(expr, &ft, &bk, &boff, &bsize, &byoff)) {
				return x64_bit_field_load(p, expr, ft, bk, boff, bsize, byoff);
			}
		}
		// Struct/union field access or other non-import selector
		x64Addr addr = x64_build_addr(p, expr);
		return x64_load_addr(p, addr);
	} case_end;

	// ── IndexExpr (arr[i]) ────────────────────────────────────────────────
	case_ast_node(ie, IndexExpr, expr); {
		Type *ibt = base_type(type_deref(ie->expr->tav.type));
		if (ibt != nullptr && ibt->kind == Type_Map) {
			return x64_build_map_index_load(p, ie->expr, ie->index, tav.type);
		}
		// `soa[idx]` whole-element read → gather each component array's [idx] into an AoS element.
		if (ibt != nullptr && ibt->kind == Type_Struct && ibt->Struct.soa_kind != StructSoa_None) {
			return x64_soa_index_element(p, expr, x64_typed(tav.type), /*store*/false, x64v_none());
		}
		x64Addr addr = x64_build_addr(p, expr);
		return x64_load_addr(p, addr);
	} case_end;

	// ── CallExpr ─────────────────────────────────────────────────────────
	case_ast_node(ce, CallExpr, expr); {
		return x64_build_call_expr(p, expr);
	} case_end;

	// `recv->method(args)` (vtable/interface dispatch): the checker rewrote it into a normal
	// CallExpr (`se->call`) whose proc is the `recv.method` field selector and whose arg[0] is
	// `recv` — so just build that call (mirrors lb_build_expr SelectorCallExpr). NOTE: a
	// side-effecting receiver (`f()->m()`) is built twice (proc field + arg0); LLVM caches via
	// StateFlag_SelectorCallExpr. Fine for the usual variable receiver.
	case_ast_node(se, SelectorCallExpr, expr); {
		return x64_build_expr(p, se->call);
	} case_end;

	case_ast_node(te, TernaryIfExpr, expr); {
		Type *type = tav.type ? x64_typed(tav.type) : t_int;
		// Trivial (side-effect-free, scalar) operands → branchless select (mirrors lb_build_expr's
		// lb_is_expr_trivial → lb_emit_select fast path); build+spill both, then select. Non-trivial
		// operands fall through to the lazy branch below (only the taken side is evaluated).
		if (lb_is_expr_trivial(te->x) && lb_is_expr_trivial(te->y)) {
			x64Value cond = x64_spill_value(p, x64_build_expr(p, te->cond), t_bool);
			x64Value xv = x64_spill_value(p, x64_emit_conv(p, x64_build_expr(p, te->x), te->x->tav.type, type), type);
			x64Value yv = x64_spill_value(p, x64_emit_conv(p, x64_build_expr(p, te->y), te->y->tav.type, type), type);
			return x64_emit_select(p, cond, xv, yv);
		}

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
		return x64_build_slice_expr(p, expr);
	} case_end;

	// ── CompoundLit ({field: val} or {v0, v1, ...}) ──────────────────────
	// Anonymous proc literal used as a VALUE (e.g. a `slice.stable_sort_by(…, proc(){…})`
	// comparator). Generate the proc ON-DEMAND and return its address — mirrors lb_build_expr
	// ProcLit → lb_generate_anonymous_proc_lit. Without this it fell through to none → a NULL
	// function pointer → DEP execution at address 0 when called.
	case_ast_node(pl, ProcLit, expr); {
		return x64_generate_anonymous_proc_lit(p, expr);
	} case_end;

	case_ast_node(cl, CompoundLit, expr); {
		Type *type = tav.type ? x64_typed(tav.type) : nullptr;
		if (type == nullptr) return x64v_none();
		return x64_build_compound_lit(p, expr, type);
	} case_end;

	// ── TypeAssertion (expr.(T)  /  expr.?) ──────────────────────────────
	case_ast_node(ta, TypeAssertion, expr); {
		return x64_build_type_assertion(p, expr);
	} case_end;

	// ── OrReturnExpr ─────────────────────────────────────────────────────────
	case_ast_node(oe, OrReturnExpr, expr); {
		return x64_emit_or_return(p, expr);
	} case_end;

	// ── OrElseExpr (x or_else y) ─────────────────────────────────────────────
	case_ast_node(oe, OrElseExpr, expr); {
		return x64_emit_or_else(p, expr);
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

	// `x when cond else y`: cond is compile-time; build the taken branch (mirrors lb_build_expr's
	// TernaryWhenExpr). Was UNHANDLED → fell to default → returned 0 when the branch is a runtime value.
	case_ast_node(te, TernaryWhenExpr, expr); {
		TypeAndValue ctav = type_and_value_of_expr(te->cond);
		GB_ASSERT(ctav.mode == Addressing_Constant && ctav.value.kind == ExactValue_Bool);
		return x64_build_expr(p, ctav.value.value_bool ? te->x : te->y);
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

gb_internal x64Value x64_build_call_expr(x64Procedure *p, Ast *expr) {
	expr = unparen_expr(expr);
	ast_node(ce, CallExpr, expr);
	TypeAndValue tav = expr->tav;
		// Type-conversion call form (int(x), f64(x), etc.) — actually convert, same as
		// cast(T)x (mirrors lb_emit_conv via x64_emit_conv). Returning the operand untouched
		// skipped width/sign/float conversions and left the result mistyped.
		if (ce->proc->tav.mode == Addressing_Type) {
			if (ce->args.count >= 1) {
				x64Value cv = x64_build_expr(p, ce->args[0]);
				return x64_emit_conv(p, cv, ce->args[0]->tav.type, tav.type);
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
			// SIMD vector builtins (#simd[N]T) — mirrors lb_build_builtin_proc's simd-range dispatch.
			if (be != nullptr && be->kind == Entity_Builtin &&
			    BuiltinProc__simd_begin < be->Builtin.id && be->Builtin.id < BuiltinProc__simd_end) {
				return x64_build_builtin_simd_proc(p, expr, be->Builtin.id);
			}
			// Scalar compiler intrinsics (count_*, overflow_*, saturating_*, sqrt, byte_swap, ...).
			if (be != nullptr && be->kind == Entity_Builtin && x64_is_simple_intrinsic(be->Builtin.id)) {
				return x64_build_intrinsic(p, expr, be->Builtin.id);
			}
			// Value-producing builtins (complex/quaternion, real/imag/conj, swizzle, unreachable).
			if (be != nullptr && be->kind == Entity_Builtin && x64_is_value_builtin(be->Builtin.id)) {
				return x64_build_value_builtin(p, expr, be->Builtin.id);
			}
			// `intrinsics.constant_utf16_cstring("…")` (e.g. win.L). Encode the constant string to
			// UTF-16 (surrogate pairs for astral), NUL-terminate, intern as .rdata, return a pointer.
			// Mirrors lb's BuiltinProc_constant_utf16_cstring. Was returning none → 0 → null
			// CreateWindowExW class name → null HWND → the whole D3D11 swapchain null chain.
			if (be != nullptr && be->kind == Entity_Builtin && be->Builtin.id == BuiltinProc_constant_utf16_cstring) {
				String s = ce->args[0]->tav.value.value_string;
				u16  *buf = gb_alloc_array(temporary_allocator(), u16, s.len + 2);
				isize n = 0;
				u8 const *text = s.text;
				isize     len  = s.len;
				while (len > 0) {
					Rune  r = 0;
					isize w = gb_utf8_decode(text, len, &r);
					text += w; len -= w;
					if ((0 <= r && r < 0xd800) || (0xe000 <= r && r < 0x10000)) {
						buf[n++] = (u16)r;
					} else if (0x10000 <= r && r <= 0x10ffff) {
						Rune rr = r - 0x10000;
						buf[n++] = (u16)(0xd800 + ((rr >> 10) & 0x3ff));
						buf[n++] = (u16)(0xdc00 + (rr & 0x3ff));
					} else {
						buf[n++] = 0xfffd;
					}
				}
				buf[n++] = 0; // NUL terminator (cstring16)
				String sym = x64_const_intern_string(p->module, make_string((u8 const *)buf, n * 2));
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
				return x64v_reg(x64_typed(expr->tav.type), X64Reg_RAX);
			}
			// Map type intrinsics → address of a generated global / proc. Used INSIDE the runtime
			// map procs (map_total_allocation_size calls type_map_cell_info(Map_Hash)). Mirrors
			// lb_gen_map_info_ptr / lb_gen_map_cell_info_ptr / lb_{hasher,equal}_proc_for_type.
			if (be != nullptr && be->kind == Entity_Builtin && ce->args.count >= 1 &&
			    (be->Builtin.id == BuiltinProc_type_map_info ||
			     be->Builtin.id == BuiltinProc_type_map_cell_info ||
			     be->Builtin.id == BuiltinProc_type_hasher_proc ||
			     be->Builtin.id == BuiltinProc_type_equal_proc)) {
				Type *at = ce->args[0]->tav.type;
				String sym = {};
				switch (be->Builtin.id) {
				case BuiltinProc_type_map_info:      sym = x64_gen_map_info_ptr(p->module, at); break;
				case BuiltinProc_type_map_cell_info: sym = x64_gen_map_cell_info_ptr(p->module, at); break;
				case BuiltinProc_type_hasher_proc:   sym = x64_synth_proc_name(p->module, X64Synth_Hasher, at); x64_enqueue_synth(p->module, X64Synth_Hasher, at); break;
				case BuiltinProc_type_equal_proc:    sym = x64_synth_proc_name(p->module, X64Synth_Equal,  at); x64_enqueue_synth(p->module, X64Synth_Equal,  at); break;
				}
				i32 off = x64_alloc_local(p, 8, 8);
				x64_emit_lea_sym(&p->asm_, X64Reg_RAX, sym);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
				return x64v_mem(tav.type ? tav.type : t_rawptr, x64_rbp_mem(off));
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
				// `type_info_of(id)` → runtime.__type_info_of(id) (mirrors LLVM). Compile-time
				// `type_info_of(T)` passes T's canonical typeid (== Type_Info.id) through the SAME
				// lookup — robust vs leaing the slot symbol (an un-emitted kind would be unresolved).
				// Was unimplemented → nil → every reflect.* on a static type crashed.
				{
					AstPackage *rt_pkg = p->module->gen->info->runtime_package;
					Entity *tio = (rt_pkg != nullptr)
					    ? scope_lookup_current(rt_pkg->scope,
					          string_interner_insert(str_lit("__type_info_of")))
					    : nullptr;
					if (tio != nullptr && tio->kind == Entity_Procedure) {
						// Evaluate the typeid arg and spill it before the call setup.
						x64Value idv;
						if (arg0->tav.mode == Addressing_Type) {
							u64 h = (arg0->tav.type != nullptr) ? type_hash_canonical_type(default_type(arg0->tav.type)) : 0;
							idv = x64v_imm(t_typeid, (i64)h);
						} else {
							idv = x64_build_expr(p, arg0);
						}
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
					if (at->kind == Type_Map) {
						x64Value r = (be->Builtin.id == BuiltinProc_cap) ? x64_map_cap(p, av) : x64_map_len(p, av);
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, r.mem);
						return x64v_reg(t_int, X64Reg_RAX);
					}
					if (is_type_slice(at) || is_type_string(at)) {
						off = 8; // cap(slice) == len
					} else if (is_type_dynamic_array(at)) {
						off = (be->Builtin.id == BuiltinProc_cap) ? 16 : 8;
					} else if (at->kind == Type_FixedCapacityDynamicArray) {
						// cap is the compile-time N; len is the int field after the data array.
						if (be->Builtin.id == BuiltinProc_cap) {
							x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)at->FixedCapacityDynamicArray.capacity);
							return x64v_reg(t_int, X64Reg_RAX);
						}
						off = (i32)x64_fca_len_offset(at);
					} else if (at->kind == Type_Struct && at->Struct.soa_kind != StructSoa_None) {
						// #soa: fixed → compile-time soa_count; slice/dynamic → the __$len field
						// AFTER the per-member [^] pointers (cap, dynamic only, follows __$len).
						// Mirrors lb_soa_struct_len; without it len(#soa) fell to none → garbage,
						// which propagates a huge length into reflect.struct_fields_zipped users.
						if (at->Struct.soa_kind == StructSoa_Fixed) {
							x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, (i64)at->Struct.soa_count);
							return x64v_reg(t_int, X64Reg_RAX);
						}
						// SOA component count = struct field count, OR array length for an array element
						// (`#soa[dynamic][2]int` → 2 component arrays); __$len follows them, __$cap follows __$len.
						Type *ebt = base_type(at->Struct.soa_elem);
						int nmem = 0;
						if (ebt != nullptr) {
							if (ebt->kind == Type_Struct)     nmem = (int)ebt->Struct.fields.count;
							else if (ebt->kind == Type_Array) nmem = (int)ebt->Array.count;
						}
						type_set_offsets(at);
						int fi = nmem;
						if (be->Builtin.id == BuiltinProc_cap && at->Struct.soa_kind == StructSoa_Dynamic) fi = nmem + 1;
						if (fi < (int)at->Struct.fields.count) off = (i32)at->Struct.offsets[fi];
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
				return x64_emit_clamp(p, rt, x64v_mem(rt, x64_rbp_mem(x_off)),
				                      x64v_mem(rt, x64_rbp_mem(lo_off)), x64v_mem(rt, x64_rbp_mem(hi_off)));
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
				// f16: moved as a raw 2-byte integer (x64_is_float excludes size-2), so abs = clear
				// the logical sign bit. LE (f16/f16le): bit 15 → mask 0x7FFF. BE (f16be): the bytes
				// are swapped, so the sign lands at bit 7 → mask 0xFF7F.
				if (x64_is_f16(rt)) {
					Type *rtb = base_type(rt);
					bool be = (rtb != nullptr && rtb->kind == Type_Basic && (rtb->Basic.flags & BasicFlag_EndianBig) != 0);
					i32 off = x64_alloc_local(p, 2, 2);
					x64_store_value(p, x64addr(x64_rbp_mem(off), rt), xv);
					x64_emit_mov_rm(&p->asm_, X64OpSize_16, X64Reg_RAX, x64_rbp_mem(off));
					x64_emit_mov_ri(&p->asm_, X64OpSize_32, X64Reg_RCX, (i64)(be ? 0xFF7F : 0x7FFF));
					x64_emit_and_rr(&p->asm_, X64OpSize_16, X64Reg_RAX, X64Reg_RCX);
					x64_emit_mov_mr(&p->asm_, X64OpSize_16, x64_rbp_mem(off), X64Reg_RAX);
					return x64v_mem(rt, x64_rbp_mem(off));
				}
				// 128-bit signed abs: the single-register path below only touches the low 64 bits, so
				// abs(i128) returned garbage high bits (u128(abs(i128)) in strconv → fmt of any negative
				// i128 printed the raw two's-complement magnitude, e.g. -5 as -(2^128-5); the cbor
				// test_decode_negative -2^64 mismatch). Negate both halves (0-x with borrow) then select
				// on the sign of the HIGH word.
				if (type_size_of(rt) == 16 && x64_is_integer(rt)) {
					i32 vo  = x64_i128_operand(p, xv, rt, true);
					i32 off = x64_alloc_local(p, 16, 16);
					x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RAX, X64Reg_RAX);
					x64_emit_xor_rr(&p->asm_, X64OpSize_32, X64Reg_RDX, X64Reg_RDX);
					x64_emit_sub_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(vo));     // RAX = -x.lo
					x64_emit_sbb_rm(&p->asm_, X64OpSize_64, X64Reg_RDX, x64_rbp_mem(vo + 8)); // RDX = -x.hi
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RCX, x64_rbp_mem(vo));      // RCX = x.lo
					x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8,  x64_rbp_mem(vo + 8));  // R8  = x.hi
					x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_R8, X64Reg_R8);            // sign of x
					x64_emit_cmov_rr(&p->asm_, X64Cc_S, X64OpSize_64, X64Reg_RCX, X64Reg_RAX); // x<0 → -x.lo
					x64_emit_cmov_rr(&p->asm_, X64Cc_S, X64OpSize_64, X64Reg_R8,  X64Reg_RDX); // x<0 → -x.hi
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off),     X64Reg_RCX);
					x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off + 8), X64Reg_R8);
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
				return x64_typeid(at);
			}
			// ptr_offset(ptr, n): ptr + n*size_of(elem). ptr_sub(p0,p1): (p0-p1)/size.
			if (be != nullptr && be->kind == Entity_Builtin &&
			    (be->Builtin.id == BuiltinProc_ptr_offset || be->Builtin.id == BuiltinProc_ptr_sub) &&
			    ce->args.count >= 2) {
				Type *pt = x64_typed(ce->args[0]->tav.type);
				if (be->Builtin.id == BuiltinProc_ptr_offset) {
					// Spill both operands first (x64 has no SSA — building idx would clobber a
					// register-held ptr), then ptr + idx*esz via the shared helper.
					x64Value ptr = x64_spill_value(p, x64_build_expr(p, ce->args[0]), pt);
					x64Value idx = x64_spill_value(p, x64_build_expr(p, ce->args[1]), x64_typed(ce->args[1]->tav.type));
					return x64_emit_ptr_offset(p, ptr, idx);
				}
				// ptr_sub: (p0 - p1) / size_of(elem) (signed).
				Type *pbt  = base_type(pt);
				Type *elem = pbt && pbt->kind == Type_MultiPointer ? pbt->MultiPointer.elem :
				             pbt && pbt->kind == Type_Pointer       ? pbt->Pointer.elem : nullptr;
				i64 esz = elem ? type_size_of(elem) : 1; if (esz <= 0) esz = 1;
				i32 a0 = x64_alloc_local(p, 8, 8);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[0]), X64Reg_RAX);
				x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(a0), X64Reg_RAX);
				x64_value_to_reg(p, x64_build_expr(p, ce->args[1]), X64Reg_RCX);
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
				if (at != nullptr && at->kind == Type_FixedCapacityDynamicArray) {
					// data is the inline array @0 → raw_data = &value.
					if (av.kind == x64Value_Mem) x64_emit_lea(&p->asm_, X64Reg_RAX, av.mem);
					else                         x64_value_to_reg(p, av, X64Reg_RAX);
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
			callee_type_raw = callee_ent->type;
			if (callee_ent->Procedure.is_foreign) {
				String fl = callee_ent->Procedure.link_name;
				if (fl.len >= 5 && gb_strncmp((char const *)fl.text, "llvm.", 5) == 0) {
					// LLVM math intrinsics → C-runtime symbol (sin/cos/pow/exp/fma/…). f16 variants
					// have no ucrt symbol → compute via f32 promotion; other unmapped llvm.* no-op (0).
					callee_sym = x64_map_llvm_math_intrinsic(fl);
					if (callee_sym.len == 0) {
						x64Value r = x64_try_llvm_f16_intrinsic(p, ce, fl, x64_typed(tav.type));
						if (r.kind != x64Value_None) return r;
						return x64v_none();
					}
				} else {
					callee_sym = x64_get_entity_name(callee_ent);
				}
			} else {
				callee_sym = x64_get_entity_name(callee_ent);
				// On-demand: a proc nothing depends on (min_dep_count == 0) — a #force_inline
				// runtime helper (bounds_check_error_loc, …) that LLVM inlines and never emits
				// standalone. We don't inline, so queue it; the driver compiles it into its owner
				// module after the parallel pass (exactly one external definition).
				if (callee_ent->min_dep_count.load(std::memory_order_relaxed) == 0) {
					x64_enqueue_oncall(p, callee_ent);
				}
			}
		} else {
			callee_type_raw = ce->proc->tav.type;
			indirect = true;
		}

		Type *ct = base_type(callee_type_raw);
		if (ct == nullptr || ct->kind != Type_Proc) return x64v_none();

		// #force_inline: emit the callee body here instead of a CALL (honours Odin's mandatory
		// inlining). Falls through to the normal call path when not inlinable.
		if (!indirect && callee_ent != nullptr) {
			x64Value inl;
			if (x64_try_inline_call(p, ce, callee_ent, ct, &inl)) return inl;
		}

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

			// Pass 1: positional args. Usually 1 positional → 1 param, but a multi-valued call
			// arg spreads across consecutive params — param_cursor tracks the param index.
			int param_cursor = 0;
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
							// Each element converts to the variadic elem type (mirrors LLVM
							// lb_emit_conv to elem): `..any` boxes the value into an {data,typeid}
							// any (or passes an already-`any` through); `..T` converts to T. The
							// store does the conversion (x64_store_value → x64_emit_conv).
							x64Value ev = x64_build_expr(p, positional[variadic_index + vi]);
							x64_store_value(p, x64addr(x64_rbp_mem(slot_off), et), ev);
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
					// C default-argument promotion (mirrors lb_emit_c_vararg): f16/f32 → f64, small
					// ints/bools → i32, bit_set → its backing int. Without it a `%f` vararg stays f32
					// (wrong width) and the callee reads garbage.
					x64Value av = x64_build_expr(p, arg);
					Type *at = (av.type != nullptr) ? av.type : arg->tav.type;
					if (at != nullptr) {
						Type *core = core_type(at);
						if (core != nullptr && core->kind == Type_BitSet) {
							Type *it = bit_set_to_int(core);
							av = x64_emit_transmute(p, av, it);
							at = it;
						}
						Type *promoted = c_vararg_promote_type(at);
						if (promoted != nullptr && !are_types_identical(at, promoted)) {
							av = x64_emit_conv(p, av, at, promoted);
						}
					}
					cvar[cvar_count++] = x64_stabilize_value(p, av);
					continue;
				}
				if (arg->tav.mode == Addressing_Type) {
					// Poly type param ($T) → Entity_TypeName, no runtime slot.
					if (param_cursor >= n_params) continue;
					if (params->variables[param_cursor]->kind == Entity_TypeName) { param_cursor++; continue; }
				}
				if (param_cursor >= n_params) continue;          // safety (non-variadic overflow)
				x64Value av = x64_build_expr(p, arg);
				Type *abt = (av.type != nullptr) ? base_type(av.type) : nullptr;
				// Spread a tuple-returning call across consecutive params only when the checker typed
				// THIS arg as a tuple. An #optional_ok / #optional_allocator_error call used in single
				// value position is still a Type_Tuple value but arg->tav.type is the first result alone
				// — the extra returns must be dropped, not spread (else they displace the next arg).
				Type *arg_bt = arg->tav.type ? base_type(arg->tav.type) : nullptr;
				bool spread = arg_bt != nullptr && arg_bt->kind == Type_Tuple &&
				              abt != nullptr && abt->kind == Type_Tuple && av.kind == x64Value_Mem;
				if (spread) {
					// Multi-valued call arg → spread its fields across consecutive params
					// (mirrors LLVM lb_add_values_to_array). e.g. f(a, g()) where g returns 2.
					for (isize fi = 0; fi < abt->Tuple.variables.count && param_cursor < n_params; fi++) {
						x64Addr fa = x64_emit_tuple_ep(p, av, (i32)fi); // field fi @ canonical type_offset_of
						pvals[param_cursor++] = x64_stabilize_value(p, x64v_mem(fa.type, fa.mem));
					}
				} else {
					if (abt != nullptr && abt->kind == Type_Tuple && av.kind == x64Value_Mem) { x64Addr _f0 = x64_emit_tuple_ep(p, av, 0); av = x64v_mem(_f0.type, _f0.mem); } /* single-value use of a tuple call: field 0, drop rest */ av = x64_stabilize_value(p, av);
					if (av.type == nullptr) av.type = params->variables[param_cursor]->type;
					pvals[param_cursor++] = av;
				}
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
				if (is_c_vararg && i == variadic_index) continue; // c_vararg ..any param: no default, no slot
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
					dv = x64_handle_param_value(p, pe->type, pe->Variable.param_value, expr);
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
				// c_vararg: the `..any` param itself takes NO ABI slot — the promoted C varargs
				// (appended below) occupy its place. Without this skip it consumed a register slot,
				// shifting every vararg one slot over (snprintf read garbage for %d/%f).
				if (is_c_vararg && i == variadic_index) continue;
				// Skip non-runtime params: Entity_TypeName ($T) AND Entity_Constant (polymorphic
					// `$x: T` value params). These take NO ABI slot. Was: only TypeName skipped → a
					// `$const` value param consumed a slot here but not in the callee → partial-return
					// pointer shifted → callee wrote result[0] through a NULL pointer. THE
					// bit_array.iterate_internal_ (`$ITERATE_SET_BITS: bool`, 2 returns) crash →
					// blick project-open via core:flags.
					if (pe->kind != Entity_Variable) continue;
				Type *pt2 = pe->type;
				if (pt2 != nullptr && type_size_of(pt2) == 0) continue; // zero-sized
				x64Value v = pvals[i];
				if (v.kind == x64Value_None) v = x64v_imm(pt2 ? pt2 : t_int, 0);
				if (v.type == nullptr) v.type = pt2;
				// Convert each arg to its param type (lb_emit_conv per arg); re-stabilize since
				// emit_conv may return a register a later arg's conversion would clobber.
				if (v.type != nullptr && pt2 != nullptr && !are_types_identical(v.type, pt2)) {
					if (v.by_ref) {
						// v.mem holds a POINTER to the value (indirect aggregate). A non-identity
						// conversion — union→any boxing, #subtype extraction — must run on the VALUE,
						// so materialize it: load the pointer, deref, then convert. (A layout-identical
						// retype re-tags and re-stabilizes back to by_ref, no value copy.) Was the blick
						// project-open crash: get_union_variant_raw_tag(clip.effects[i]^) passed the raw
						// union bytes as the `any` {data,id} instead of boxing → type_info_of(garbage id).
						x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, v.mem);
						x64Value deref = x64v_mem(v.type, x64_mem(X64Reg_RAX, 0));
						v = x64_stabilize_value(p, x64_emit_conv(p, deref, v.type, pt2));
					} else {
						v = x64_stabilize_value(p, x64_emit_conv(p, v, v.type, pt2));
					}
				}
				if (x64_arg_is_indirect(pt2)) {
					if (v.by_ref) {
						// v.mem already holds a POINTER to the value (no copy was made); pass it.
						args[slot++] = x64v_mem(alloc_type_pointer(pt2), v.mem);
					} else {
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
					}
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

			// @(deferred_*): register a deferred call at scope exit (mirrors lb_emit_call).
			// `in` passes the call's INPUTS, `out` its RESULTS, `in_out` both; `*_by_ptr` passes
			// each by address. Covers sync.guard (in) and TEMP_ALLOCATOR_GUARD's arena reset (out).
			if (callee_ent != nullptr && entity_has_deferred_procedure(callee_ent)) {
				DeferredProcedureKind dk = callee_ent->Procedure.deferred_procedure.kind;
				Entity *dproc = callee_ent->Procedure.deferred_procedure.entity;
				// `@(deferred_none=F)` (dk == DeferredProcedure_none) defers F with NO in/out args — it
				// must STILL be registered (only the context is passed). Was skipped → the deferred call
				// never ran (blick ui.parent_scope's `@(deferred_none=parent_pop)` leaked every parent →
				// stale parent_stack across the frame swap → OOB block_from_index crash).
				if (dproc != nullptr) {
					bool by_ptr   = dk == DeferredProcedure_in_by_ptr || dk == DeferredProcedure_out_by_ptr || dk == DeferredProcedure_in_out_by_ptr;
					bool want_in  = dk == DeferredProcedure_in  || dk == DeferredProcedure_in_by_ptr  || dk == DeferredProcedure_in_out || dk == DeferredProcedure_in_out_by_ptr;
					bool want_out = dk == DeferredProcedure_out || dk == DeferredProcedure_out_by_ptr || dk == DeferredProcedure_in_out || dk == DeferredProcedure_in_out_by_ptr;
					Type *dct  = base_type(dproc->type);
					bool  dctx = dct != nullptr && dct->kind == Type_Proc && dct->Proc.calling_convention == ProcCC_Odin;

					int first_explicit = needs_rbp ? 1 : 0;
					int last_explicit  = slot - (needs_ctx ? 1 : 0);
					int in_count  = want_in ? gb_max(last_explicit - first_explicit, 0) : 0;
					int rcount    = (want_out && ct->Proc.results != nullptr) ? (int)ct->Proc.results->Tuple.variables.count : 0;
					x64Value *dargs = gb_alloc_array(permanent_allocator(), x64Value, in_count + rcount + 1);
					int di = 0;

					// Inputs: the explicit ABI arg slots (already lowered; deferred_in's params
					// match the callee's). Spill so they survive to scope exit.
					if (want_in) {
						for (int ai = first_explicit; ai < last_explicit; ai++) {
							Type *at = args[ai].type ? args[ai].type : t_rawptr;
							x64Value sv = x64_spill_value(p, args[ai], at);
							if (by_ptr) {
								i32 poff = x64_alloc_local(p, 8, 8);
								x64_emit_lea(&p->asm_, X64Reg_RAX, sv.mem);
								x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(poff), X64Reg_RAX);
								sv = x64v_mem(alloc_type_pointer(at), x64_rbp_mem(poff));
							}
							dargs[di++] = sv;
						}
					}
					// Outputs: the call's result field(s), ABI-lowered against the result type
					// (large aggregates → by pointer; `*_by_ptr` forces a pointer).
					for (int ri = 0; ri < rcount; ri++) {
						Type *rt = ct->Proc.results->Tuple.variables[ri]->type;
						x64Value rv = call_result;
						if (rcount > 1) { X64Mem m = call_result.mem; m.disp += (i32)type_offset_of(ct->Proc.results, ri); rv = x64v_mem(rt, m); }
						x64Value sv = x64_spill_value(p, rv, rt);
						if (by_ptr || x64_arg_is_indirect(rt)) {
							i32 poff = x64_alloc_local(p, 8, 8);
							x64_emit_lea(&p->asm_, X64Reg_RAX, sv.mem);
							x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(poff), X64Reg_RAX);
							sv = x64v_mem(alloc_type_pointer(rt), x64_rbp_mem(poff));
						}
						dargs[di++] = sv;
					}
					if (dctx) dargs[di++] = x64_context_ptr_value(p);

					x64Procedure::DeferEntry de = {};
					de.call_proc = dproc;
					de.call_type = dproc->type;
					de.call_args = dargs;
					de.call_argc = di;
					array_add(&p->deferred, de);
				}
			}

			// #optional_ok / #optional_allocator_error used as a SINGLE value (`!m[k]`, `x := f()`,
			// `assertf(!get(...))`): the proc returns a tuple but expr->tav.type is just the first
			// result. Reduce to field 0 here so EVERY consumer (unary/binary/arg/decl) sees the value,
			// not the tuple. Was core:flags: `!bit_array.get(...)` read field 1 (ok=true) → `!true`=false
			// → the "pos already assigned" assert fired. (A genuine multi-value use keeps tav.type a
			// tuple — e.g. `a, b := f()` or a spread arg — so this doesn't fire there.)
			if (ct->Proc.result_count > 1 && tav.type != nullptr && !is_type_tuple(tav.type) &&
			    call_result.kind == x64Value_Mem) {
				x64Addr f0 = x64_emit_tuple_ep(p, call_result, 0);
				call_result = x64v_mem(f0.type, f0.mem);
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
		x64Value ind_result = x64_reconstruct_call_result(p, ct, npartial, last_rt, needs_rbp, ret_local_off, partial_off, ind_rv);
		if (ct->Proc.result_count > 1 && tav.type != nullptr && !is_type_tuple(tav.type) &&
		    ind_result.kind == x64Value_Mem) {
			x64Addr f0 = x64_emit_tuple_ep(p, ind_result, 0);
			ind_result = x64v_mem(f0.type, f0.mem);
		}
		return ind_result;
}
