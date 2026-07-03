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

// True only for SSE-backed floats (f32/f64). f16 is EXCLUDED: it's 2 bytes, and the SSE path
// (movss/movsd) would move 4/8 bytes — a `movss` store on a 2-byte f16 slot overwrites the adjacent
// stack slot (THE `animate(.., alpha: f16)` crash: alpha's store clobbered the `flag` param next to
// it → garbage enum index → OOB write). f16 is moved as a raw 2-byte value via the integer path
// (see x64_is_scalar); f16<->f32 arithmetic/conversion (F16C) is not yet supported.
gb_internal bool x64_is_f16(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic && (t->Basic.flags & BasicFlag_Float) != 0 && type_size_of(t) == 2;
}
gb_internal bool x64_is_float(Type *t) {
	t = base_type(t);
	// ONLY true scalar SSE floats (f32/f64) — NOT f16 (see x64_is_f16) and NOT complex. complex is a
	// 2-float AGGREGATE; including BasicFlag_Complex here made x64_store_value/x64_load_addr treat a
	// 16-byte complex128 as a scalar and copy it with a single movss/movsd → truncated/garbage.
	// quaternion was never affected (BasicFlag_Quaternion). Complex arith/ABI uses the aggregate path.
	return t->kind == Type_Basic && (t->Basic.flags & BasicFlag_Float) != 0 && type_size_of(t) != 2;
}

// "double" = any 8-byte SSE float (f64 AND f64le/f64be) — the move/op dispatch keys on this to pick
// movsd vs movss. Matching only Basic_f64 made f64le/f64be fall to the 4-byte movss path → an 8-byte
// endian-float was truncated to f32 (e.g. f64le(-1.) stored as 0xbf800000). f16 is excluded by size.
gb_internal bool x64_is_double(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic && (t->Basic.flags & BasicFlag_Float) != 0 && type_size_of(t) == 8;
}

gb_internal bool x64_is_f32(Type *t) {
	t = base_type(t);
	return t->kind == Type_Basic && (t->Basic.flags & BasicFlag_Float) != 0 && type_size_of(t) == 4;
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
	// f16 is a scalar moved as a 2-byte integer (raw bits) — NOT via SSE (see x64_is_float).
	return x64_is_integer(t) || x64_is_bool(t) || x64_is_ptr(t) || x64_is_float(t) || x64_is_f16(t);
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
	// An enum's signedness is its underlying integer's. Without this, a signed enum (the default;
	// any enum with a negative value) compared unsigned & loaded zero-extended → e.g. Ordering.Less
	// (-1) read as a huge value → broken `<`/`>=` (was THE slice.sort / smoothsort Ordering bug).
	if (t->kind == Type_Enum) t = base_type(t->Enum.base_type);
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

// True if a value of this type contains NO pointers anywhere (scalars, enums, bit_sets,
// and fixed aggregates of those). Such a value can never alias stack scratch, so storing it
// into a local and then reclaiming the initializer's temps is safe — unlike slices/strings/
// pointers/maps/unions, whose data can point INTO the reclaimed scratch. Conservative:
// anything not provably pointer-free returns false.
gb_internal bool x64_type_is_pointer_free(Type *t) {
	if (t == nullptr) return false;
	t = base_type(t);
	switch (t->kind) {
	case Type_Basic:
		switch (t->Basic.kind) {
		case Basic_string: case Basic_cstring: case Basic_rawptr: case Basic_any:
			return false;
		}
		return true; // integers/floats/bools/runes/complex/quaternion/uintptr/typeid
	case Type_Enum:
	case Type_BitSet:
		return true;
	case Type_Array:           return x64_type_is_pointer_free(t->Array.elem);
	case Type_EnumeratedArray: return x64_type_is_pointer_free(t->EnumeratedArray.elem);
	case Type_Matrix:          return x64_type_is_pointer_free(t->Matrix.elem);
	case Type_SimdVector:      return x64_type_is_pointer_free(t->SimdVector.elem);
	case Type_BitField:        return x64_type_is_pointer_free(t->BitField.backing_type);
	case Type_Struct:
		for_array(i, t->Struct.fields) if (!x64_type_is_pointer_free(t->Struct.fields[i]->type)) return false;
		return true;
	case Type_Tuple:
		for_array(i, t->Tuple.variables) if (!x64_type_is_pointer_free(t->Tuple.variables[i]->type)) return false;
		return true;
	}
	return false; // Pointer/MultiPointer/Slice/DynamicArray/Map/Proc/Union/SoaPointer/…
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
	// Large indirect-ABI param: the slot holds the incoming pointer, not the data —
	// deref through it (like a ^T param). See x64_proc_begin / indirect_params.
	for (isize i = 0; i < p->indirect_params.count; i++) {
		if (p->indirect_params[i] == e) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(*off));
			return x64addr(x64_mem(X64Reg_RAX, 0), e->type);
		}
	}
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

// Tag address + width for a tagged union (mirrors lb_emit_union_tag_ptr). Used on both the store side
// (x64_store_union_variant) and the load side (x64_emit_union_tag_value).
gb_internal X64UnionTag x64_emit_union_tag_ptr(Type *ubt, X64Mem union_mem) {
	i64 tag_sz = union_tag_size(ubt);
	X64OpSize opsz = tag_sz <= 1 ? X64OpSize_8  : tag_sz == 2 ? X64OpSize_16 :
	                 tag_sz == 4 ? X64OpSize_32 : X64OpSize_64;
	X64Mem m = union_mem; m.disp += (i32)ubt->Union.variant_block_size;
	return {m, opsz};
}

// Write a tagged union's discriminant for variant `src_type` into the union at `dst` (mirrors
// lb_emit_store_union_variant_tag). #shared_nil: tag MUST be 0 when the value is nil (else `==
// nil`/or_return read a non-nil tag) → cmov on (value@0 == 0). maybe-pointer unions have no tag.
gb_internal void x64_emit_store_union_variant_tag(x64Procedure *p, X64Mem dst, Type *src_type, Type *union_type) {
	X64Assembler *a   = &p->asm_;
	Type         *ubt = base_type(union_type);
	if (is_type_union_maybe_pointer(ubt) || src_type == nullptr) return;
	i64 tag_val = union_variant_index_checked(ubt, src_type);
	X64UnionTag tag = x64_emit_union_tag_ptr(ubt, dst);
	X64OpSize tsz2 = tag.opsz;
	X64Mem tagm = tag.mem;
	i64 vsz = type_size_of(src_type);
	if (ubt->Union.kind == UnionType_shared_nil && vsz > 0 && vsz <= 8) {
		// tag = (value@0 == 0) ? 0 : tag_val
		X64OpSize vosz = x64_op_size_of(src_type);
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

// Construct a union value at `dst` from a variant value `src` whose type is one of `union_type`'s
// variants: zero the union, store the variant value at offset 0, set the discriminant tag. Mirrors
// lb_emit_store_union_variant. A variant SMALLER than variant_block_size (e.g. the 1-byte
// Allocator_Error in 8-byte os.Error) needs the zero so stale high bytes/tag don't make a nil
// variant read non-nil. The caller has already converted `src` to the chosen variant type.
gb_internal void x64_store_union_variant(x64Procedure *p, X64Mem dst, x64Value src, Type *union_type) {
	i64 usz = x64_type_size(union_type);
	if (usz == 0) return;
	x64_zero_mem(p, dst, usz);
	if (src.type != nullptr && type_size_of(src.type) != 0) {
		x64_store_value(p, x64addr(dst, src.type), src); // value @ offset 0
	}
	x64_emit_store_union_variant_tag(p, dst, src.type, union_type);
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
	if (p->is_startup) {
		// Global/static `any = v` initializer: the boxed value must live in a STATIC global — a
		// stack temp dies when __$startup_runtime returns, leaving any.data dangling. Mirrors the
		// is_startup &CompoundLit path. Store the value into the global, then data @0 = &global.
		String sym = x64_add_global_generated(p->module, at);
		i32 tmp = x64_alloc_local(p, asz, aal);
		x64_store_value(p, x64addr(x64_rbp_mem(tmp), at), src);
		x64_emit_lea_sym(a, X64Reg_RAX, sym);
		x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(tmp), asz);
		x64_emit_lea_sym(a, X64Reg_RAX, sym);                   // re-lea (copy clobbered RAX)
		x64_emit_mov_mr(a, X64OpSize_64, dst, X64Reg_RAX);     // data @0 = &global
		X64Mem idm0 = dst; idm0.disp += 8;
		x64_emit_mov_ri(a, X64OpSize_64, X64Reg_RAX, (i64)type_hash_canonical_type(at));
		x64_emit_mov_mr(a, X64OpSize_64, idm0, X64Reg_RAX);    // id @8
		return;
	}
	i32 val_off = x64_alloc_local(p, asz, aal);
	// The `any` holds a POINTER to this boxed value, so the temp must outlive the statement —
	// mark scope-lived (named_seq) so the per-statement temp reclaimer never reuses its slot.
	// Without this, `x = f32(1.1)` after other stmts left any.data dangling → garbage on read.
	p->named_seq++;
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
	// A zero-sized variant (empty struct, e.g. `union{Empty,…} = Empty{}`) builds to None: the
	// general "src has a value" guard would skip conversion, so the union's discriminant tag is
	// never written → the union reads as nil (`switch v in u` matches nothing). An empty variant
	// still needs tag construction, so let the union-boxing conv run for a None src too. Was THE
	// blick shutdown hang: Message_Quit is `struct{}` → boxed into the message union its tag stayed
	// 0 → the app pump never matched Message_Quit → the quit never propagated → join hung forever.
	bool needs_conv = src.type != nullptr && t != nullptr && !are_types_identical(src.type, t);
	bool zero_variant_box = needs_conv && src.kind == x64Value_None &&
	                        base_type(t)->kind == Type_Union && union_is_variant_of(base_type(t), src.type);
	if (needs_conv && (src.kind != x64Value_None || zero_variant_box)) {
		// x64_emit_conv may build the converted value with rep stos/movs (variant→union, array,
		// struct conversions) which CLOBBER volatile registers — INCLUDING one holding dst.mem.base.
		// Pin a volatile dst base across the conversion, then reload it. Was the blick project-open
		// crash: `panel.state = media_variant` held &panel.state in RCX while the variant→union
		// conversion's rep stos used RCX as the count (→ 0) → the store wrote to addr 0x8 → AV.
		bool pin_dst = !dst.mem.rip_rel && dst.mem.base != X64Reg_RBP && dst.mem.base != X64Reg_NONE;
		i32 dpin = 0;
		if (pin_dst) {
			dpin = x64_alloc_local(p, 8, 8);
			x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(dpin), dst.mem.base);
		}
		src = x64_emit_conv(p, src, src.type, t);
		if (pin_dst) {
			X64Reg base_reg = (src.kind == x64Value_Reg && src.reg == X64Reg_RCX) ? X64Reg_R11 : X64Reg_RCX;
			x64_emit_mov_rm(a, X64OpSize_64, base_reg, x64_rbp_mem(dpin));
			dst.mem.base = base_reg;
		}
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

	// Fill the WHOLE 64-bit register so 64-bit consumers (cvtsi2ss, 64-bit arith, calls) see the right
	// value. A signed sub-64 load MUST sign-extend (movsx/movsxd) — a plain `mov r32` zero-extends, so a
	// signed i32 of -1 became +0xFFFFFFFF and `cvtsi2ss(64)` produced 2^32 (the _to_f32 of negative ints
	// bug). Mirrors x64_value_to_reg's Mem path exactly (was a divergence between the two loaders).
	if (sz == X64OpSize_64) {
		x64_emit_mov_rm(a, sz, X64Reg_RAX, addr.mem);
	} else if (signed_int) {
		x64_emit_movsx_rm(a, sz, X64Reg_RAX, addr.mem); // sz==32 → MOVSXD, 8/16 → MOVSX
	} else if (sz == X64OpSize_32) {
		x64_emit_mov_rm(a, X64OpSize_32, X64Reg_RAX, addr.mem); // 32-bit mov already zero-extends
	} else {
		x64_emit_movzx_rm(a, sz, X64Reg_RAX, addr.mem);
	}
	return x64v_reg(t, X64Reg_RAX);
}

// Tuple field pointer / value by index (mirror lb_emit_tuple_ep / lb_emit_tuple_ev). `tuple` is the
// tuple's in-memory buffer; field i lives at the CANONICAL type_offset_of(tuple_type, i). Using this
// instead of the hand-rolled align_formula loops that were duplicated at each call site removes a
// divergence class (a loop disagreeing with type_offset_of would silently read the wrong field).
gb_internal x64Addr x64_emit_tuple_ep(x64Procedure *p, x64Value tuple, i32 index) {
	(void)p;
	Type *tt = base_type(tuple.type);
	Type *ft = tt->Tuple.variables[index]->type;
	X64Mem m = tuple.mem; m.disp += (i32)type_offset_of(tt, index);
	return x64addr(m, ft);
}
gb_internal x64Value x64_emit_tuple_ev(x64Procedure *p, x64Value tuple, i32 index) {
	return x64_load_addr(p, x64_emit_tuple_ep(p, tuple, index));
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
		// A sub-8-byte load must fill the WHOLE register: `mov al`/`mov ax` keep the high bits,
		// leaking a prior value (e.g. a call's leftover) into them. Sign-extend signed scalars;
		// zero-extend the rest (`mov r32` already zero-extends to 64).
		if (t != nullptr && sz < X64OpSize_64) {
			if (x64_is_signed_integer(t))       x64_emit_movsx_rm(a, sz, into, v.mem);
			else if (sz == X64OpSize_32)        x64_emit_mov_rm  (a, X64OpSize_32, into, v.mem);
			else                                x64_emit_movzx_rm(a, sz, into, v.mem);
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
