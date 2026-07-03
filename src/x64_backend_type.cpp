// x64 debug backend — type layout utilities, entity naming, value helpers.

// Type queries

// Resolve untyped basic types to their typed equivalents. type_size_of/type_align_of
// assert is_type_typed; always pass through this first.
gb_internal Type *x64_typed(Type *t) {
	if (t == nullptr) return t_int;
	Type *bt = base_type(t);
	if (bt == nullptr || bt->kind != Type_Basic) return t;
	if (!(bt->Basic.flags & BasicFlag_Untyped)) return t;
	switch (bt->Basic.kind) {
	case Basic_UntypedBool:    return t_bool;
	case Basic_UntypedInteger: return t_int;
	case Basic_UntypedFloat:   return t_f64;
	case Basic_UntypedRune:    return t_rune;
	case Basic_UntypedString:  return t_string;
	case Basic_UntypedNil:     return t_rawptr;
	default:                   return t;
	}
}

gb_internal i64 x64_type_size(Type *t)  { return type_size_of(x64_typed(t));  }
gb_internal i64 x64_type_align(Type *t) { return type_align_of(x64_typed(t)); }

// Byte count → X64OpSize enum.
gb_internal X64OpSize x64_op_size_of(Type *t) {
	i64 sz = type_size_of(x64_typed(base_type(t)));
	if (sz <= 1) return X64OpSize_8;
	if (sz == 2) return X64OpSize_16;
	if (sz <= 4) return X64OpSize_32;
	return X64OpSize_64;
}

gb_internal bool x64_is_float(Type *t) {
	t = base_type(t);
	// ONLY true scalar floats (f16/f32/f64) — NOT complex. complex is a 2-float AGGREGATE; including
	// BasicFlag_Complex here made x64_store_value/x64_load_addr treat a 16-byte complex128 as a
	// scalar and copy it with a single movss/movsd (4/8 bytes) → truncated/garbage. quaternion was
	// never affected (BasicFlag_Quaternion, not Complex). Complex arith/ABI must use the aggregate path.
	return t->kind == Type_Basic && (t->Basic.flags & BasicFlag_Float) != 0;
}

gb_internal bool x64_is_double(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic && t->Basic.kind == Basic_f64;
}

gb_internal bool x64_is_f32(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic && t->Basic.kind == Basic_f32;
}

gb_internal bool x64_is_integer(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic &&
	       (t->Basic.flags & (BasicFlag_Integer | BasicFlag_Unsigned)) != 0;
}

gb_internal bool x64_is_bool(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic && (t->Basic.flags & BasicFlag_Boolean) != 0;
}

gb_internal bool x64_is_ptr(Type *t) {
	t = base_type(t);
	return t->kind == Type_Pointer || t->kind == Type_MultiPointer ||
	       t->kind == Type_Proc;
}

gb_internal bool x64_is_scalar(Type *t) {
	return x64_is_integer(t) || x64_is_bool(t) || x64_is_ptr(t) || x64_is_float(t);
}

// Win64 indirect-arg rule (lbAbiAmd64Win64::compute_arg_types + lbAbi386::non_struct):
// an aggregate is passed BY POINTER unless size is exactly 1/2/4/8; likewise a non-float
// scalar wider than 8 (i128). Zero-sized aggregates are dropped (elsewhere), not indirect.
gb_internal bool x64_arg_is_indirect(Type *t) {
	if (t == nullptr) return false;
	i64 sz = x64_type_size(t);
	if (sz == 0) return false;
	if (x64_is_scalar(t)) return sz > 8;            // i128 by pointer; scalars ≤8 in register
	return !(sz == 1 || sz == 2 || sz == 4 || sz == 8);
}

gb_internal bool x64_is_signed_integer(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic &&
	       (t->Basic.flags & BasicFlag_Integer) &&
	       !(t->Basic.flags & BasicFlag_Unsigned);
}

// Entity naming (mirrors lb_get_entity_name)

gb_internal String x64_get_entity_name(Entity *e) {
	GB_ASSERT(e != nullptr);
	if (e->kind == Entity_Procedure && e->Procedure.link_name.len != 0) {
		return e->Procedure.link_name;
	}
	if (e->kind == Entity_Variable && e->Variable.link_name.len != 0) {
		return e->Variable.link_name;
	}
	if (e->pkg == nullptr) return e->token.string;

	// Mangle in the arena-backed temp allocator (no CRT heap lock), then persist in
	// the permanent arena (cached on the entity).
	gbString w = string_canonical_entity_name(temporary_allocator(), e);
	String name = copy_string(permanent_allocator(),
	                          make_string(cast(u8 const *)w, gb_string_length(w)));
	if (e->kind == Entity_Procedure) e->Procedure.link_name = name;
	else if (e->kind == Entity_Variable) e->Variable.link_name = name;
	return name;
}

// Stack frame helpers

// Returns the (negative) RBP-relative offset; slot occupies [RBP+off .. RBP+off+size).
// Zero-sized types (empty struct, [0]T) are legal in Odin and, like LLVM empty
// aggregates, occupy NO frame space and are never dereferenced (every zero-sized
// load/store is a no-op — see x64_store_value/x64_load_addr); returning the current
// frame offset is fine since 0 bytes live there.
gb_internal i32 x64_alloc_local(x64Procedure *p, i64 size, i64 align) {
	if (align <= 0) align = 1;
	if (size  <= 0) return -(i32)p->local_size;
	i64 total = (i64)p->local_size + size;
	total = (total + (align - 1)) & ~(align - 1); // align up
	p->local_size = (i32)total;
	if (p->local_size > p->frame_max) p->frame_max = p->local_size; // peak for SUB RSP
	return -(i32)total;
}

// ABI parameter home: slot N is at [RBP + 16 + N*8].
gb_internal gb_inline i32 x64_param_rbp_off(int slot) {
	return 16 + slot * 8;
}

gb_internal gb_inline X64Mem x64_rbp_mem(i32 rbp_off) {
	return x64_mem(X64Reg_RBP, rbp_off);
}

// Entity address lookup

gb_internal bool x64_entity_is_local(x64Procedure *p, Entity *e) {
	return x64_var_get(&p->var_offsets, e) != nullptr;
}

gb_internal x64Addr x64_entity_addr(x64Procedure *p, Entity *e) {
	i32 *off = x64_var_get(&p->var_offsets, e);
	GB_ASSERT_MSG(off != nullptr, "x64_entity_addr: '%.*s' not registered",
	              LIT(e->token.string));
	return x64addr(x64_rbp_mem(*off), e->type);
}

// Load &global into RAX. Ordinary: LEA [RIP+sym]. @(thread_local): Win64 TLS access
// sequence (resolves to THIS thread's copy — see x64_emit_tls_addr; clobbers R11 too).
gb_internal void x64_load_global_addr(x64Procedure *p, Entity *e) {
	bool is_tls = e != nullptr && e->kind == Entity_Variable &&
	              e->Variable.thread_local_model.len != 0;
	if (is_tls) {
		x64_emit_tls_addr(&p->asm_, x64_get_entity_name(e));
	} else {
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(e));
	}
	// Record the ref so the driver can lazily DEFINE this global in its owner module if
	// the eager pass skipped it (min_dep==0); mirrors lazy lb_find_value_from_entity.
	// Foreign globals are defined by their lib (reloc above already declares the extern).
	if (e != nullptr && e->kind == Entity_Variable && !e->Variable.is_foreign) {
		array_add(&p->module->onref_globals, e);
	}
}

// X64Mem for a global; loads &global into RAX, returns [RAX].
gb_internal X64Mem x64_global_mem(x64Procedure *p, Entity *e) {
	x64_load_global_addr(p, e);
	return x64_mem(X64Reg_RAX, 0);
}

// Value store / load

// Copy exactly `size` bytes src→dst via R10 in 8/4/2/1 chunks — handles odd tails
// (3/5/6/7/11/13/14/15) that Win64 by-pointer aggregates can have and that a naive
// 8-byte-stride loop would under-copy.
gb_internal void x64_copy_fixed(x64Procedure *p, X64Mem dst, X64Mem src, i64 size) {
	X64Assembler *a = &p->asm_;
	if (size <= 0) return;

	// Large aggregates: an unrolled mov chain explodes .text (a multi-MB by-value struct
	// becomes MBs of instructions), so above a threshold use REP MOVSB (O(1) code). It
	// clobbers RSI/RDI/RCX — preserve them to stay drop-in clobber-safe like the unrolled
	// path. DF is ABI-guaranteed clear.
	if (size > 256) {
		x64_emit_push_r(a, X64Reg_RCX);
		x64_emit_push_r(a, X64Reg_RSI);
		x64_emit_push_r(a, X64Reg_RDI);
		x64_emit_lea(a, X64Reg_RSI, src); // src/dst are RBP/RAX-based, not RSP — pushes don't shift them
		x64_emit_lea(a, X64Reg_RDI, dst);
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, size);
		x64_enc_b(a, 0xF3); // REP
		x64_enc_b(a, 0xA4); // MOVSB
		x64_emit_pop_r(a, X64Reg_RDI);
		x64_emit_pop_r(a, X64Reg_RSI);
		x64_emit_pop_r(a, X64Reg_RCX);
		return;
	}

	i32 b = 0;
	while (size - b >= 8) {
		x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R10, x64_mem(src.base, src.disp + b));
		x64_emit_mov_mr(a, X64OpSize_64, x64_mem(dst.base, dst.disp + b), X64Reg_R10);
		b += 8;
	}
	if (size - b >= 4) {
		x64_emit_mov_rm(a, X64OpSize_32, X64Reg_R10, x64_mem(src.base, src.disp + b));
		x64_emit_mov_mr(a, X64OpSize_32, x64_mem(dst.base, dst.disp + b), X64Reg_R10);
		b += 4;
	}
	if (size - b >= 2) {
		x64_emit_mov_rm(a, X64OpSize_16, X64Reg_R10, x64_mem(src.base, src.disp + b));
		x64_emit_mov_mr(a, X64OpSize_16, x64_mem(dst.base, dst.disp + b), X64Reg_R10);
		b += 2;
	}
	if (size - b >= 1) {
		x64_emit_mov_rm(a, X64OpSize_8, X64Reg_R10, x64_mem(src.base, src.disp + b));
		x64_emit_mov_mr(a, X64OpSize_8, x64_mem(dst.base, dst.disp + b), X64Reg_R10);
	}
}

// Construct a union value at `dst` from a variant value `src` whose type is one of `union_type`'s
// variants: zero the union, store the variant value at offset 0, set the discriminant tag. Mirrors
// lb_emit_store_union_variant (+_tag). A variant SMALLER than variant_block_size (e.g. the 1-byte
// Allocator_Error in 8-byte os.Error) needs the zero so stale high bytes/tag don't make a nil
// variant read non-nil. #shared_nil: tag MUST be 0 when the value is nil (else `== nil`/or_return
// read a non-nil tag). The caller has already converted `src` to the chosen variant type.
gb_internal void x64_store_union_variant(x64Procedure *p, X64Mem dst, x64Value src, Type *union_type) {
	X64Assembler *a   = &p->asm_;
	Type         *ubt = base_type(union_type);
	i64           usz = x64_type_size(union_type);
	if (usz == 0) return;
	x64_zero_mem(p, dst, usz);
	if (src.type != nullptr && type_size_of(src.type) != 0) {
		x64_store_value(p, x64addr(dst, src.type), src); // value @ offset 0
	}
	if (is_type_union_maybe_pointer(ubt) || src.type == nullptr) return;
	i64 tag_off = (i64)ubt->Union.variant_block_size;
	i64 tag_sz  = union_tag_size(ubt);
	i64 tag_val = union_variant_index_checked(ubt, src.type);
	X64OpSize tsz2 = tag_sz <= 1 ? X64OpSize_8 :
	                 tag_sz == 2 ? X64OpSize_16 :
	                 tag_sz == 4 ? X64OpSize_32 : X64OpSize_64;
	X64Mem tagm = dst; tagm.disp += (i32)tag_off;
	i64 vsz = type_size_of(src.type);
	if (ubt->Union.kind == UnionType_shared_nil && vsz > 0 && vsz <= 8) {
		// tag = (value@0 == 0) ? 0 : tag_val
		X64OpSize vosz = x64_op_size_of(src.type);
		x64_emit_mov_rm(a, vosz, X64Reg_RAX, dst);             // value@0
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RCX, tag_val); // mov: no flags
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RDX, 0);       // mov (NOT xor) — keep ZF
		x64_emit_test_rr(a, vosz, X64Reg_RAX, X64Reg_RAX);     // ZF = (value == 0)
		x64_emit_cmov_rr(a, X64Cc_E, X64OpSize_64, X64Reg_RCX, X64Reg_RDX);
		x64_emit_mov_mr(a, tsz2, tagm, X64Reg_RCX);
	} else {
		x64_emit_mov_mi(a, tsz2, tagm, (i32)tag_val);
	}
}

// Box `src` (of type `src_type`) into an `any` {data: rawptr @0, id: typeid @8} at `dst`: spill the
// value to a local, point data at it, set id = typeid(default_type(src_type)). The to-`any` case of
// lb_emit_conv (used by x64_emit_conv and the variadic `..any` arg packing).
gb_internal void x64_box_any(x64Procedure *p, X64Mem dst, x64Value src, Type *src_type) {
	X64Assembler *a  = &p->asm_;
	Type *at = (src_type != nullptr) ? default_type(src_type) : t_any;
	at = x64_typed(at);
	i64 asz = type_size_of(at); if (asz <= 0) asz = 1;
	i64 aal = type_align_of(at); if (aal <= 0) aal = 1;
	i32 val_off = x64_alloc_local(p, asz, aal);
	x64_store_value(p, x64addr(x64_rbp_mem(val_off), at), src);
	x64_emit_lea(a, X64Reg_RAX, x64_rbp_mem(val_off));
	x64_emit_mov_mr(a, X64OpSize_64, dst, X64Reg_RAX);          // data @0
	X64Mem idm = dst; idm.disp += 8;
	x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, (i64)type_hash_canonical_type(at));
	x64_emit_mov_mr(a, X64OpSize_64, idm, X64Reg_RAX);          // id @8
}

gb_internal void x64_store_value(x64Procedure *p, x64Addr dst, x64Value src) {
	X64Assembler *a  = &p->asm_;
	Type         *t  = dst.type;
	if (t != nullptr && x64_type_size(t) == 0) return; // zero-sized: nothing to store

	// Convert src to the destination type FIRST, then store (mirrors lb_addr_store → lb_emit_conv
	// → lb_emit_store). All conversion lives in x64_emit_conv; this is then a dumb store. No-op
	// when the types already match (the common case).
	if (src.kind != x64Value_None && src.type != nullptr && t != nullptr &&
	    !are_types_identical(src.type, t)) {
		src = x64_emit_conv(p, src, src.type, t);
	}

	X64OpSize sz = x64_op_size_of(t);
	switch (src.kind) {
	case x64Value_Imm: {
		i64 v = src.imm;
		// An Imm into an AGGREGATE dst (>8 bytes) only happens for a nil/zero store (e.g.
		// `u = nil` on a union, `s = nil` on a slice/map/dynarray) where untyped nil built to an
		// 8-byte 0. An 8-byte mov would leave the rest STALE — e.g. a union's tag (after the
		// variant block) keeping its previous variant → `== nil` reads non-nil. Zero it fully.
		if (x64_type_size(t) > 8) {
			x64_zero_mem(p, dst.mem, x64_type_size(t));
		} else if (sz == X64OpSize_64 && (v < 0 || v > 0x7fffffffLL)) {
			// Need full 64-bit: load into RAX then store
			x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, v);
			x64_emit_mov_mr(a, X64OpSize_64, dst.mem, X64Reg_RAX);
		} else {
			x64_emit_mov_mi(a, sz, dst.mem, (i32)v);
		}
		break;
	}
	case x64Value_Reg:
		x64_emit_mov_mr(a, sz, dst.mem, src.reg);
		break;
	case x64Value_XmmReg:
		if (x64_is_double(t)) x64_emit_movsd_mr(a, dst.mem, src.xmm);
		else                   x64_emit_movss_mr(a, dst.mem, src.xmm);
		break;
	case x64Value_Mem: {
		i64 total = x64_type_size(t);
		if (x64_is_float(t)) {
			if (x64_is_double(t)) {
				x64_emit_movsd_rm(a, X64XmmReg_XMM0, src.mem);
				x64_emit_movsd_mr(a, dst.mem, X64XmmReg_XMM0);
			} else {
				x64_emit_movss_rm(a, X64XmmReg_XMM0, src.mem);
				x64_emit_movss_mr(a, dst.mem, X64XmmReg_XMM0);
			}
		} else {
			x64_copy_fixed(p, dst.mem, src.mem, total);
		}
		break;
	}
	default:
		// Backstop: a None source carries no value; storing it would leave the dst STALE.
		// `nil` now builds to a real 0 (see x64_build_expr) so the common case skips here;
		// this just zero-fills any other stray None. That stale-store was the
		// create_multi_logger bug (success-path `nil` left `al` stale → spurious or_return
		// error on later calls).
		if (t != nullptr) x64_zero_mem(p, dst.mem, x64_type_size(t));
		break;
	}
}

// Load a value from `addr` into RAX (integer) or XMM0 (float).
// For multi-word values (>8 bytes) returns x64v_mem — callers must handle in-memory.
gb_internal x64Value x64_load_addr(x64Procedure *p, x64Addr addr) {
	X64Assembler *a  = &p->asm_;
	Type         *t  = addr.type;

	if (t != nullptr && x64_type_size(t) == 0) return x64v_none(); // zero-sized: no value

	if (x64_is_float(t)) {
		if (x64_is_double(t)) x64_emit_movsd_rm(a, X64XmmReg_XMM0, addr.mem);
		else                   x64_emit_movss_rm(a, X64XmmReg_XMM0, addr.mem);
		return x64v_xmm(t, X64XmmReg_XMM0);
	}

	i64 tsz = t ? type_size_of(x64_typed(t)) : 8;
	if (tsz > 8) {
		return x64v_mem(t, addr.mem); // multi-word: keep in memory, don't truncate to a reg
	}

	X64OpSize sz = x64_op_size_of(t);
	Type *bt = base_type(t);
	bool signed_int = (bt->kind == Type_Basic) &&
	                  (bt->Basic.flags & BasicFlag_Integer) &&
	                  !(bt->Basic.flags & BasicFlag_Unsigned);

	if (sz == X64OpSize_64 || sz == X64OpSize_32) {
		x64_emit_mov_rm(a, sz, X64Reg_RAX, addr.mem);
	} else if (signed_int) {
		x64_emit_movsx_rm(a, sz, X64Reg_RAX, addr.mem);
	} else {
		x64_emit_movzx_rm(a, sz, X64Reg_RAX, addr.mem);
	}
	return x64v_reg(t, X64Reg_RAX);
}

// Materialise helpers

gb_internal X64Reg x64_value_to_reg(x64Procedure *p, x64Value v, X64Reg into) {
	X64Assembler *a  = &p->asm_;
	Type         *t  = v.type;
	X64OpSize     sz = t ? x64_op_size_of(t) : X64OpSize_64;
	switch (v.kind) {
	case x64Value_Reg:
		if (v.reg != into) x64_emit_mov_rr(a, X64OpSize_64, into, v.reg);
		return into;
	case x64Value_Imm:
		x64_emit_mov_ri(a, X64OpSize_64, into, v.imm);
		return into;
	case x64Value_Mem:
		if (t && x64_is_signed_integer(t) && sz < X64OpSize_64) {
			x64_emit_movsx_rm(a, sz, into, v.mem);
		} else {
			x64_emit_mov_rm(a, sz, into, v.mem);
		}
		return into;
	case x64Value_XmmReg: {
		// Bitwise extract float bits into integer register via stack.
		x64_emit_sub_ri(a, X64OpSize_64, X64Reg_RSP, 8);
		if (v.type && x64_is_double(v.type)) x64_emit_movsd_mr(a, x64_mem(X64Reg_RSP, 0), v.xmm);
		else                                  x64_emit_movss_mr(a, x64_mem(X64Reg_RSP, 0), v.xmm);
		x64_emit_mov_rm(a, X64OpSize_64, into, x64_mem(X64Reg_RSP, 0));
		x64_emit_add_ri(a, X64OpSize_64, X64Reg_RSP, 8);
		return into;
	}
	case x64Value_None:
	default:
		// Empty value — zero the register rather than crash.
		x64_emit_xor_rr(a, X64OpSize_32, into, into);
		return into;
	}
}

gb_internal X64XmmReg x64_value_to_xmm(x64Procedure *p, x64Value v, X64XmmReg into) {
	X64Assembler *a = &p->asm_;
	switch (v.kind) {
	case x64Value_XmmReg:
		if (v.xmm != into) {
			if (x64_is_double(v.type)) x64_emit_movsd_rr(a, into, v.xmm);
			else                        x64_emit_movss_rr(a, into, v.xmm);
		}
		return into;
	case x64Value_Mem:
		if (x64_is_double(v.type)) x64_emit_movsd_rm(a, into, v.mem);
		else                        x64_emit_movss_rm(a, into, v.mem);
		return into;
	case x64Value_Reg:
	case x64Value_Imm: {
		// Integer bits → XMM via stack.
		x64_value_to_reg(p, v, X64Reg_RAX);
		x64_emit_push_r(a, X64Reg_RAX);
		if (v.type && x64_is_double(v.type)) x64_emit_movsd_rm(a, into, x64_mem(X64Reg_RSP, 0));
		else                                  x64_emit_movss_rm(a, into, x64_mem(X64Reg_RSP, 0));
		x64_emit_pop_r(a, X64Reg_RAX);
		return into;
	}
	case x64Value_None:
	default:
		// XOR-zero the XMM register rather than crash.
		x64_emit_xorpd(a, into, into);
		return into;
	}
}
