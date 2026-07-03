// x64 debug backend — STATIC constant emission.
//
// Mirrors lb_const_value, but lowers (ExactValue, Type) into raw bytes + COFF
// relocations in a section (.rdata / .data) so const globals/`::` constants are
// static data rather than runtime-constructed.
//
// Win64 only: ptr_size == int_size == 8, so string/slice headers are {ptr@0, len@8}.

gb_internal void x64_const_value(x64Module *m, CoffSection *sec, u32 off, Type *type,
                                 ExactValue value, Type *value_type);

// ── low-level helpers ────────────────────────────────────────────────────────

// Overwrite `n` bytes of an ALREADY-RESERVED region of `sec` at byte `off`.
gb_internal void x64_const_patch(CoffSection *sec, u32 off, void const *src, isize n) {
	if (n <= 0) return;
	GB_ASSERT((isize)off + n <= sec->data.count);
	gb_memmove(sec->data.data + off, src, n);
}

// Define (or upgrade an UNDEF ref to) symbol `name` at section+off. All data symbols
// are EXTERNAL (like x64_emit_global_variable) so cross-.obj relocations resolve.
gb_internal void x64_const_define_symbol(x64Module *m, String name, i16 section_number, u32 off) {
	u32 *existing = string_map_get(&m->coff.sym_map, name);
	if (existing != nullptr) {
		CoffSymEntry &se = m->coff.syms[*existing];
		se.value          = off;
		se.section_number = section_number;
		se.type           = COFF_SYM_TYPE_NULL;
		se.storage_class  = COFF_SYM_CLASS_EXTERNAL;
		return;
	}
	u32 sym_idx = (u32)m->coff.syms.count;
	CoffSymEntry se = {};
	se.name           = name;
	se.value          = off;
	se.section_number = section_number;
	se.type           = COFF_SYM_TYPE_NULL;
	se.storage_class  = COFF_SYM_CLASS_EXTERNAL;
	array_add(&m->coff.syms, se);
	string_map_set(&m->coff.sym_map, name, sym_idx);
}

// Unique, module-qualified name for compiler-generated constant backing data.
gb_internal String x64_const_new_name(x64Module *m, char const *prefix) {
	gbString gs = gb_string_make(m->alloc, prefix);
	gs = gb_string_append_length(gs, m->pkg->name.text, m->pkg->name.len);
	if (m->file != nullptr) gs = gb_string_append_fmt(gs, "_%d", (int)m->file->id);
	gs = gb_string_append_fmt(gs, "$%u", m->const_data_count++);
	return make_string(cast(u8 const *)gs, gb_string_length(gs));
}

// Two's-complement bytes of a BigInt into `out[0..sz)`, honouring the type's
// endianness. Mirrors lb_big_int_to_llvm.
gb_internal void x64_const_int_bytes(Type *type, BigInt const *a, u8 *out, isize sz_in) {
	gb_zero_size(out, sz_in);
	if (big_int_is_zero(a)) return;

	size_t sz = (size_t)sz_in;
	BigInt val = {};
	big_int_init(&val, a);
	if (big_int_is_neg(&val)) {
		mp_incr(&val);
	}

	// Buffer sized to the type (arbitrary width — mirrors lb_big_int_to_llvm). Small (≤32B = i256,
	// i128/u128, etc.) stays on the stack; wider (large bit_sets backed by an int) heap-temp.
	u64 stack64[4] = {}; // 32 bytes
	u8 *rop = cast(u8 *)stack64;
	if (sz_in > (isize)gb_size_of(stack64)) {
		rop = cast(u8 *)gb_alloc(temporary_allocator(), sz_in);
		gb_zero_size(rop, sz_in);
	}

	size_t written = 0;
	mp_err err = mp_pack(rop, sz, &written, MP_LSB_FIRST, 1, MP_LITTLE_ENDIAN, 0, &val);
	GB_ASSERT(err == MP_OKAY);

	if (!is_type_endian_little(type)) {
		for (size_t i = 0; i < sz/2; i++) {
			u8 t = rop[i]; rop[i] = rop[sz-1-i]; rop[sz-1-i] = t;
		}
	}
	if (big_int_is_neg(a)) {
		for (size_t i = 0; i < sz; i++) rop[i] = ~rop[i];
	}
	gb_memmove(out, rop, sz_in);
}

gb_internal void x64_const_float_bytes(CoffSection *sec, u32 off, i64 esz, f64 fv) {
	if (esz == 8) { f64 d = fv;          x64_const_patch(sec, off, &d, 8); }
	else if (esz == 4) { f32 f = (f32)fv; x64_const_patch(sec, off, &f, 4); }
	else if (esz == 2) { u16 h = f32_to_f16((f32)fv); x64_const_patch(sec, off, &h, 2); }
}

// Intern a string literal's BYTES into .rdata (one global per unique content,
// mirrors LLVM's m->const_strings). Returns the symbol naming byte[0].
gb_internal String x64_const_intern_string(x64Module *m, String s) {
	String *cached = string_map_get(&m->const_strings, s);
	if (cached != nullptr) return *cached;

	// Align to 2: a `cstring16` (win.L / constant_utf16_cstring) is a u16 array, and Win32 APIs
	// capture wide strings with a WCHAR (2-byte) ProbeForRead — an odd address raises
	// STATUS_DATATYPE_MISALIGNMENT kernel-side → ERROR_NOACCESS (998). 1-byte packing let wide
	// string constants land on odd addresses (the chaotic RegisterClassW/CreateWindowExW failures).
	coff_section_align(m->rdata, 2);
	u32 boff = (u32)coff_section_len(m->rdata);
	if (s.len > 0) coff_section_write(m->rdata, s.text, s.len);
	coff_section_write_u8(m->rdata, 0); // NUL (harmless for `string`, required for `cstring`)

	String name = x64_const_new_name(m, "__$cstr$");
	x64_const_define_symbol(m, name, 2 /*.rdata*/, boff);
	string_map_set(&m->const_strings, s, name);
	return name;
}

// Reserve `n` zero bytes at the end of `sec`, returning the start offset.
gb_internal u32 x64_const_reserve(CoffSection *sec, i64 align, i64 n) {
	coff_section_align(sec, (isize)gb_max(align, (i64)1));
	u32 off = (u32)coff_section_len(sec);
	for (i64 i = 0; i < n; i++) coff_section_write_u8(sec, 0);
	return off;
}

// ── element iteration over a compound literal ────────────────────────────────

// The constant value carried by a compound-literal element expression.
gb_internal ExactValue x64_const_elem_value(Ast *elem, Type **out_type) {
	Ast *e = elem;
	if (e != nullptr && e->kind == Ast_FieldValue) e = e->FieldValue.value;
	TypeAndValue tav = type_and_value_of_expr(e);
	if (out_type) *out_type = tav.type;
	return tav.value;
}

// Can x64_const_value FULLY emit a static constant of this type? If not, the caller
// must fall back to runtime init (else we'd silently emit zeros). Conservative:
// anything not explicitly supported returns false.
gb_internal bool x64_const_type_supported(Type *type) {
	Type *bt = base_type(core_type(type));
	if (bt == nullptr) return false;
	switch (bt->kind) {
	case Type_Basic:
		return true;
	case Type_Pointer:
	case Type_MultiPointer:
	case Type_Proc:
	case Type_Enum:
	case Type_BitSet:
		return true;
	case Type_Array:          return x64_const_type_supported(bt->Array.elem);
	case Type_EnumeratedArray:return x64_const_type_supported(bt->EnumeratedArray.elem);
	case Type_SimdVector:     return x64_const_type_supported(bt->SimdVector.elem);
	case Type_Slice:          return x64_const_type_supported(bt->Slice.elem);
	case Type_Struct:
		if (bt->Struct.is_raw_union) return false; // TODO: raw unions
		if (bt->Struct.soa_kind != StructSoa_None) return false; // TODO: #soa
		for_array(i, bt->Struct.fields) {
			if (!x64_const_type_supported(bt->Struct.fields[i]->type)) return false;
		}
		return true;
	// TODO(x64-const): Type_Matrix (stride), Type_Map, Type_DynamicArray (fixed-cap),
	// Type_Union (tagged), Type_Tuple. These fall back to runtime init.
	default:
		return false;
	}
}

// ── main converter ───────────────────────────────────────────────────────────

gb_internal void x64_const_value(x64Module *m, CoffSection *sec, u32 off, Type *type,
                                 ExactValue value, Type *value_type) {
	type = default_type(type);
	Type *bt = core_type(type);
	i64 sz = type_size_of(type);
	if (sz <= 0) return;

	// static `any = <const>` (e.g. `@(static) v: any = 3`): box the value into a hidden .data
	// global, then write any{data → it @0, id @8}. Without this the int bytes were written raw
	// into the 16-byte any slot → id=0 → every `v.(T)` missed. Mirrors x64_box_any for statics.
	if (is_type_any(bt) && value.kind != ExactValue_Invalid) {
		Type *vt = nullptr;
		if (value_type != nullptr && !is_type_untyped(value_type) && !is_type_any(base_type(value_type))) {
			vt = default_type(value_type);
		}
		if (vt == nullptr) {
			switch (value.kind) {
			case ExactValue_Integer: vt = t_int;    break;
			case ExactValue_Float:   vt = t_f64;    break;
			case ExactValue_Bool:    vt = t_bool;   break;
			case ExactValue_String:  vt = t_string; break;
			default: break;
			}
		}
		if (vt != nullptr) {
			i64 vsz = type_size_of(vt); if (vsz <= 0) vsz = 1;
			i64 val_al = type_align_of(vt); if (val_al <= 0) val_al = 1;
			u32 voff = x64_const_reserve(m->data, val_al, vsz);
			x64_const_value(m, m->data, voff, vt, value, value_type);
			gbString gs = gb_string_make(m->alloc, "__$anybox$");
			gs = gb_string_append_length(gs, m->pkg->name.text, m->pkg->name.len);
			if (m->file != nullptr) gs = gb_string_append_fmt(gs, "$%d", (int)m->file->id);
			gs = gb_string_append_fmt(gs, "$%u", m->gen_global_count++);
			String vsym = make_string(cast(u8 const *)gs, gb_string_length(gs));
			x64_const_define_symbol(m, vsym, 3 /*.data*/, voff);
			coff_reloc_add(sec, off, vsym, COFF_REL_ADDR64);             // any.data @0
			u64 idh = type_hash_canonical_type(vt);
			x64_const_patch(sec, off + 8, &idh, 8);                      // any.id @8
		}
		return;
	}

	value = convert_exact_value_for_type(value, type);

	switch (value.kind) {
	case ExactValue_Invalid:
		return; // leave the reserved zeros (nil / zero value)

	case ExactValue_Bool: {
		u64 v = value.value_bool ? 1 : 0;
		x64_const_patch(sec, off, &v, gb_min(sz, (i64)8));
		return;
	}

	case ExactValue_Integer: {
		if (is_type_pointer(bt) || is_type_multi_pointer(bt)) {
			i64 v = big_int_to_i64(&value.value_integer);
			x64_const_patch(sec, off, &v, 8);
			return;
		}
		u8 stackbuf[32] = {};
		u8 *buf = stackbuf;
		if (sz > (i64)gb_size_of(stackbuf)) {
			buf = cast(u8 *)gb_alloc(temporary_allocator(), sz);
			gb_zero_size(buf, sz);
		}
		x64_const_int_bytes(type, &value.value_integer, buf, sz);
		x64_const_patch(sec, off, buf, sz);
		return;
	}

	case ExactValue_Float:
		x64_const_float_bytes(sec, off, sz, value.value_float);
		return;

	case ExactValue_Pointer: {
		i64 v = value.value_pointer;
		x64_const_patch(sec, off, &v, 8);
		return;
	}

	case ExactValue_Complex: {
		Complex128 *c = value.value_complex;
		i64 esz = sz / 2;
		x64_const_float_bytes(sec, off,         esz, c ? c->real : 0.0);
		x64_const_float_bytes(sec, off + (u32)esz, esz, c ? c->imag : 0.0);
		return;
	}

	case ExactValue_Quaternion: {
		Quaternion256 *q = value.value_quaternion;
		i64 esz = sz / 4; // layout: imag, jmag, kmag, real (@QuaternionLayout)
		x64_const_float_bytes(sec, off + 0*(u32)esz, esz, q ? q->imag : 0.0);
		x64_const_float_bytes(sec, off + 1*(u32)esz, esz, q ? q->jmag : 0.0);
		x64_const_float_bytes(sec, off + 2*(u32)esz, esz, q ? q->kmag : 0.0);
		x64_const_float_bytes(sec, off + 3*(u32)esz, esz, q ? q->real : 0.0);
		return;
	}

	case ExactValue_String: {
		String s = value.value_string;
		if (is_type_cstring(bt)) {
			if (s.len == 0) return; // null pointer
			coff_reloc_add(sec, off, x64_const_intern_string(m, s), COFF_REL_ADDR64);
			return;
		}
		if (is_type_cstring16(bt)) {
			// wide C string pointer (8 bytes, NOT a {data,len} header): UTF-16 encode + NUL-terminate,
			// intern, emit ONLY the pointer. Mirrors the constant_utf16_cstring builtin. Without this,
			// a `cstring16` const global fell to the 16-byte string path → wrote len past its 8-byte
			// reservation (sys/windows package globals). Was the x64_const_patch off+n>data.count assert.
			u16  *buf = gb_alloc_array(temporary_allocator(), u16, s.len + 2);
			isize n = 0; u8 const *text = s.text; isize len = s.len;
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
			coff_reloc_add(sec, off, x64_const_intern_string(m, make_string((u8 const *)buf, n * 2)), COFF_REL_ADDR64);
			return;
		}
		// string / []T are {data, len}; data via ADDR64 reloc to interned bytes. For a non-u8
		// slice (#load to []u32 etc.) len is the ELEMENT count = bytes / elem size (mirrors
		// lb_find_or_add_entity_string_byte_slice_with_type).
		if (s.len > 0) coff_reloc_add(sec, off, x64_const_intern_string(m, s), COFF_REL_ADDR64);
		i64 len = s.len;
		if (is_type_slice(bt)) { i64 esz = type_size_of(bt->Slice.elem); if (esz > 1) len /= esz; }
		x64_const_patch(sec, off + 8, &len, 8);
		return;
	}

	case ExactValue_Procedure: {
		Ast *pa = (value.value_procedure != nullptr) ? unparen_expr(value.value_procedure) : nullptr;
		Entity *pe = (pa != nullptr) ? entity_of_node(pa) : nullptr;
		if (pe != nullptr && pe->kind == Entity_Procedure) {
			coff_reloc_add(sec, off, x64_get_entity_name(pe), COFF_REL_ADDR64);
		} else if (pa != nullptr && pa->kind == Ast_ProcLit) {
			// Anonymous `proc(){…}` literal as a compile-time-const global initializer (e.g. a
			// package-global dispatch var like bufio._read_writer_procedure). Generate the anon proc
			// + reloc the global to it. Was left 0 → the global proc pointer stayed nil (io.query
			// returned an empty Stream_Mode_Set because s.procedure was nil).
			Entity *ae = x64_anon_proc_entity(m, pa);
			if (ae != nullptr) coff_reloc_add(sec, off, x64_get_entity_name(ae), COFF_REL_ADDR64);
		}
		return;
	}

	case ExactValue_Typeid: {
		// Canonical type hash — the SAME value as the typeid_of builtin, the constant
		// typeid path in x64_build_expr, and the emitted Type_Info.id, so
		// type_info_of(it) finds the table entry. (Was left 0 → type_info_of → nil.)
		Type *tt = value.value_typeid;
		u64 h = (tt != nullptr) ? type_hash_canonical_type(default_type(tt)) : 0;
		x64_const_patch(sec, off, &h, gb_min(sz, (i64)8));
		return;
	}

	case ExactValue_Compound:
		break; // handled below

	default:
		return;
	}

	// ── ExactValue_Compound ──────────────────────────────────────────────────
	Ast *lit = value.value_compound;
	if (lit == nullptr) return;
	ast_node(cl, CompoundLit, lit);

	// Slices: backing array in .rdata + {ptr, len}.
	if (is_type_slice(bt)) {
		if (cl->elems.count == 0) return; // nil slice
		Type *et = bt->Slice.elem;
		i64 esz = type_size_of(et); if (esz <= 0) esz = 1;
		i64 eal = type_align_of(et); if (eal <= 0) eal = 1;
		i64 n   = cl->elems.count;
		u32 boff = x64_const_reserve(m->rdata, eal, n * esz);
		for (i64 i = 0; i < n; i++) {
			Ast *elem = cl->elems[i];
			if (elem == nullptr) continue;
			Type *evt = nullptr;
			ExactValue ev = x64_const_elem_value(elem, &evt);
			x64_const_value(m, m->rdata, boff + (u32)(i*esz), et, ev, evt);
		}
		String bsym = x64_const_new_name(m, "__$csl$");
		x64_const_define_symbol(m, bsym, 2 /*.rdata*/, boff);
		coff_reloc_add(sec, off, bsym, COFF_REL_ADDR64);
		i64 len = n;
		x64_const_patch(sec, off + 8, &len, 8);
		return;
	}

	// Structs (positional or field-value).
	if (bt->kind == Type_Struct) {
		type_set_offsets(bt);
		bool named = cl->elems.count > 0 && cl->elems[0] != nullptr &&
		             cl->elems[0]->kind == Ast_FieldValue;
		if (named) {
			for_array(ei, cl->elems) {
				Ast *elem = cl->elems[ei];
				if (elem == nullptr || elem->kind != Ast_FieldValue) continue;
				Ast *fn = elem->FieldValue.field;
				if (fn == nullptr || fn->kind != Ast_Ident) continue;
				String fname = fn->Ident.token.string;
				for_array(fi, bt->Struct.fields) {
					Entity *fe = bt->Struct.fields[fi];
					if (fe->token.string != fname) continue;
					Type *evt = nullptr;
					ExactValue ev = x64_const_elem_value(elem, &evt);
					x64_const_value(m, sec, off + (u32)bt->Struct.offsets[fi], fe->type, ev, evt);
					break;
				}
			}
		} else {
			isize n = gb_min((isize)cl->elems.count, (isize)bt->Struct.fields.count);
			for (isize ei = 0; ei < n; ei++) {
				Ast *elem = cl->elems[ei];
				if (elem == nullptr) continue;
				Entity *fe = bt->Struct.fields[ei];
				Type *evt = nullptr;
				ExactValue ev = x64_const_elem_value(elem, &evt);
				x64_const_value(m, sec, off + (u32)bt->Struct.offsets[ei], fe->type, ev, evt);
			}
		}
		return;
	}

	// Fixed arrays, enumerated arrays, matrices, SIMD vectors — element-indexed.
	{
		Type *et = nullptr;
		i64   ecount = 0;
		if (bt->kind == Type_Array)            { et = bt->Array.elem;            ecount = bt->Array.count; }
		else if (bt->kind == Type_EnumeratedArray) { et = bt->EnumeratedArray.elem; ecount = bt->EnumeratedArray.count; }
		else if (bt->kind == Type_SimdVector)  { et = bt->SimdVector.elem;        ecount = bt->SimdVector.count; }
		// NOTE: matrices are intentionally NOT handled here — their layout has row/col
		// stride padding (matrix_row_major_index_to_offset), so naive i*esz indexing
		// would be wrong. TODO(x64-const): static matrix constants.

		if (et != nullptr) {
			i64 esz = type_size_of(et); if (esz <= 0) esz = 1;
			bool named = cl->elems.count > 0 && cl->elems[0] != nullptr &&
			             cl->elems[0]->kind == Ast_FieldValue;
			if (named) {
				for_array(ei, cl->elems) {
					Ast *elem = cl->elems[ei];
					if (elem == nullptr || elem->kind != Ast_FieldValue) continue;
					Ast *idx = elem->FieldValue.field;
					if (idx == nullptr) continue;
					// Key is a single integer (`0x80 = v`) or a RANGE (`lo..=hi = v` / `lo..<hi = v`,
					// e.g. utf8.accept_sizes); ranges fill [lo, hi) (mirrors lb_const_value).
					i64 lo, hi;
					if (is_ast_range(idx)) {
						ast_node(ie, BinaryExpr, idx);
						lo = exact_value_to_i64(ie->left->tav.value);
						hi = exact_value_to_i64(ie->right->tav.value);
						if (ie->op.kind != Token_RangeHalf) hi += 1; // inclusive ..=
					} else {
						TypeAndValue itav = type_and_value_of_expr(idx);
						if (itav.value.kind != ExactValue_Integer) continue;
						lo = big_int_to_i64(&itav.value.value_integer);
						hi = lo + 1;
					}
					// Enumerated-array keys are enum VALUES; storage index = value - min.
					if (bt->kind == Type_EnumeratedArray && bt->EnumeratedArray.min_value) {
						i64 mn = exact_value_to_i64(*bt->EnumeratedArray.min_value);
						lo -= mn; hi -= mn;
					}
					Type *evt = nullptr;
					ExactValue ev = x64_const_elem_value(elem, &evt);
					for (i64 i = lo; i < hi; i++) {
						if (i < 0 || i >= ecount) continue;
						x64_const_value(m, sec, off + (u32)(i*esz), et, ev, evt);
					}
				}
			} else {
				isize n = gb_min((isize)cl->elems.count, (isize)ecount);
				for (isize i = 0; i < n; i++) {
					Ast *elem = cl->elems[i];
					if (elem == nullptr) continue;
					Type *evt = nullptr;
					ExactValue ev = x64_const_elem_value(elem, &evt);
					x64_const_value(m, sec, off + (u32)(i*esz), et, ev, evt);
				}
			}
			return;
		}
	}

	// Bit sets: OR (1 << (element - lower)). Accumulate in a u64 (covers bit_sets up to
	// 64 bits — the common case; >64-bit bit_sets are a TODO).
	if (bt->kind == Type_BitSet) {
		u64 acc = 0;
		i64 lower = bt->BitSet.lower;
		for_array(ei, cl->elems) {
			Ast *elem = cl->elems[ei];
			if (elem == nullptr) continue;
			TypeAndValue tav = type_and_value_of_expr(elem);
			if (tav.value.kind != ExactValue_Integer) continue;
			i64 bit = big_int_to_i64(&tav.value.value_integer) - lower;
			if (bit < 0 || bit >= 64) continue;
			acc |= (u64)1 << (u64)bit;
		}
		x64_const_patch(sec, off, &acc, gb_min(sz, (i64)8));
		return;
	}

	// Unhandled compound (e.g. SOA, fixed-capacity dynamic arrays): leave zeroed.
	// TODO(x64-const): SOA layouts, #soa, raw unions, maps.
}
