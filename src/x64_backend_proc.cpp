// x64 debug backend — procedure prologue / epilogue / call helpers.

static const X64Reg    X64_INT_ARG_REGS[4] = { X64Reg_RCX, X64Reg_RDX, X64Reg_R8, X64Reg_R9 };
static const X64XmmReg X64_XMM_ARG_REGS[4] = { X64XmmReg_XMM0, X64XmmReg_XMM1, X64XmmReg_XMM2, X64XmmReg_XMM3 };

// Indirect-ABI params at or below this size are copied into a frame-local on entry;
// larger ones are accessed through the incoming pointer (no copy) to avoid huge stack copies.
#define X64_INDIRECT_PARAM_COPY_MAX 4096

gb_internal bool x64_arg_is_float(Type *t) { return x64_is_float(t); }

// ─────────────────────────────────────────────────────────────────────────────
// CodeView line-number helpers
// ─────────────────────────────────────────────────────────────────────────────

// Get (or create) the byte offset of `file_id`'s entry within the module's
// DEBUG_S_FILECHKSMS table, registering the path in the string table as needed.
gb_internal u32 x64_cv_file_offset(x64Module *m, i32 file_id) {
	for (isize i = 0; i < m->cv_file_ids.count; i++) {
		if (m->cv_file_ids[i] == file_id) return m->cv_file_offs[i];
	}

	// String table: a leading NUL occupies offset 0 (CV convention).
	if (m->cv_strtab.count == 0) array_add(&m->cv_strtab, (u8)0);
	u32    str_off = (u32)m->cv_strtab.count;
	String path    = get_file_path_string(file_id);
	// CodeView source paths need backslashes; Odin uses forward slashes.
	for (isize i = 0; i < path.len; i++) {
		u8 c = (u8)path.text[i];
		if (c == '/') c = '\\';
		array_add(&m->cv_strtab, c);
	}
	array_add(&m->cv_strtab, (u8)0);

	// File checksum entry: offFileName(4) + cbChecksum(1) + checksumKind(1),
	// then padded to a 4-byte boundary. Kind 0 = None (no checksum bytes).
	u32 chk_off = (u32)m->cv_filechksms.count;
	u8  ent[6];
	gb_memmove(ent, &str_off, 4);
	ent[4] = 0; // cbChecksum
	ent[5] = 0; // checksumKind = None
	for (int i = 0; i < 6; i++) array_add(&m->cv_filechksms, ent[i]);
	while ((m->cv_filechksms.count % 4) != 0) array_add(&m->cv_filechksms, (u8)0);

	array_add(&m->cv_file_ids,  file_id);
	array_add(&m->cv_file_offs, chk_off);
	return chk_off;
}

// Record a source-line marker at the current code offset for `node`. Called per
// statement; dedups consecutive same-line markers and entries at the same offset.
gb_internal void x64_record_line(x64Procedure *p, Ast *node) {
	if (p->module->debug_s == nullptr) return; // no line tables outside -debug
	if (node == nullptr) return;
	TokenPos pos = ast_token(node).pos;
	u32 code_off = (u32)p->asm_.code.count;

	// S_INLINESITE: also record the callee's real line+file for the active inline site
	// (the binary annotations are built from these).
	if (p->cur_inline_site >= 0 && pos.line > 0 && pos.file_id > 0) {
		x64Procedure::InlineSiteRec *sr = &p->inline_sites[p->cur_inline_site];
		x64Procedure::InlineSiteLine il; il.offset = code_off; il.line = pos.line; il.file_id = pos.file_id;
		if (sr->lines.count > 0 && sr->lines[sr->lines.count - 1].offset == il.offset) {
			sr->lines[sr->lines.count - 1] = il; // newest at same offset wins
		} else if (sr->lines.count == 0 ||
		           sr->lines[sr->lines.count - 1].line != il.line ||
		           sr->lines[sr->lines.count - 1].file_id != il.file_id) {
			array_add(&sr->lines, il);
		}
	}

	i32 line = pos.line;
	i32 fid  = pos.file_id;
	if (p->inline_frames.count > 0) {
		// Inlined code → primary table gets the call-site line (caller's file); the inline frame's
		// own callee lines come from the S_INLINESITE annotations above. This pairing (primary=
		// call-site, site=callee) is what surfaces a nested inline frame in the debugger.
		fid  = p->file_id;
		line = p->inline_call_line;
	} else {
		if (fid != p->file_id) return; // non-inlined: one source file per proc (drops generic noise)
	}
	if (line <= 0) return;
	if (line == p->cur_line && fid == p->cur_file_id) return;
	p->cur_line = line;
	p->cur_file_id = fid;

	x64Procedure::LineEntry le;
	le.offset  = code_off;
	le.line    = (u32)line;
	le.file_id = fid;
	if (p->lines.count > 0 && p->lines[p->lines.count - 1].offset == le.offset) {
		// no code since last marker — newer entry wins
		p->lines[p->lines.count - 1].line    = le.line;
		p->lines[p->lines.count - 1].file_id = le.file_id;
		return;
	}
	array_add(&p->lines, le);
}

// Map a type to a CodeView built-in type index for S_REGREL32 locals. Aggregates
// fall back to u64 (first 8 bytes) — enough to inspect scalars/pointers/ids.
gb_internal u32 x64_cv_type_index(Type *t) {
	if (t == nullptr) return 0x0023u; // T_UQUAD
	if (x64_is_ptr(t)) return 0x0603u; // T_64PVOID
	if (x64_is_float(t)) return (type_size_of(x64_typed(t)) == 4) ? 0x0040u : 0x0041u; // T_REAL32/64
	if (x64_is_bool(t)) return 0x0030u; // T_BOOL08
	i64  sz  = type_size_of(x64_typed(t));
	bool sgn = x64_is_signed_integer(t);
	switch (sz) {
	case 1: return sgn ? 0x0010u : 0x0020u; // T_CHAR / T_UCHAR
	case 2: return sgn ? 0x0011u : 0x0021u; // T_SHORT / T_USHORT
	case 4: return sgn ? 0x0074u : 0x0075u; // T_INT4 / T_UINT4
	case 8: return sgn ? 0x0013u : 0x0023u; // T_QUAD / T_UQUAD
	}
	return 0x0023u; // aggregate: first 8 bytes as u64
}

// True for types mapping directly to a CodeView built-in index (no .debug$T record).
gb_internal bool x64_cv_is_builtin_scalar(Type *t) {
	if (t == nullptr) return true;
	if (x64_is_float(t) || x64_is_bool(t)) return true;
	Type *bt = base_type(t);
	if (bt->kind != Type_Basic) return false;
	u32 f = bt->Basic.flags;
	if (f & (BasicFlag_Integer | BasicFlag_Unsigned)) return true;
	if (bt->Basic.kind == Basic_rune) return true;
	return false;
}

// .debug$T helpers ───────────────────────────────────────────────────────────
gb_internal void x64_cv_numeric(CoffSection *s, i64 v) {
	if (v >= 0 && v < 0x8000) {
		coff_section_write_u16(s, (u16)v);
	} else {
		coff_section_write_u16(s, 0x8004u); // LF_ULONG
		coff_section_write_u32(s, (u32)v);
	}
}
gb_internal void x64_cv_align4_pad(CoffSection *s) {
	isize cur = coff_section_len(s) % 4;
	if (cur == 0) return;
	isize pad = 4 - cur;
	for (isize i = 0; i < pad; i++) coff_section_write_u8(s, (u8)(0xF0 + (pad - i))); // LF_PAD
}
gb_internal void x64_cv_finish_type(CoffSection *s, u32 len_pos) {
	x64_cv_align4_pad(s);
	u16 length = (u16)(coff_section_len(s) - (len_pos + 2));
	s->data[len_pos+0] = (u8)( length       & 0xFFu);
	s->data[len_pos+1] = (u8)((length >> 8) & 0xFFu);
}

// Emit (or look up) a CodeView .debug$T type record for `t` and return its index.
gb_internal u32 x64_cv_type(x64Module *m, Type *t) {
	if (m->debug_t == nullptr) return 0; // no type records outside -debug
	if (t == nullptr) return 0x0003u; // T_VOID
	if (x64_cv_is_builtin_scalar(t)) return x64_cv_type_index(t);

	Type *bt = base_type(t);
	if (bt == nullptr) return 0x0023u;
	if (bt->kind == Type_Basic) {
		switch (bt->Basic.kind) {
		case Basic_rawptr: case Basic_cstring: case Basic_cstring16: return 0x0603u; // void*
		case Basic_typeid: return 0x0023u; // u64
		default: break;
		}
	}
	if (bt->kind == Type_Proc) return 0x0603u; // void* (no signature modelling)
	// Enum → its backing integer (e.g. `enum byte` → u8), so the debugger reads the right WIDTH
	// (not a generic 8-byte blob that pulls in adjacent stack bytes). No enumerator names.
	if (bt->kind == Type_Enum) return x64_cv_type(m, bt->Enum.base_type);

	u32 *cached = map_get(&m->cv_types, t);
	if (cached) return *cached;

	CoffSection *T = m->debug_t;

	if (bt->kind == Type_Pointer || bt->kind == Type_MultiPointer) {
		Type *elem = (bt->kind == Type_Pointer) ? bt->Pointer.elem : bt->MultiPointer.elem;
		u32 ut = x64_cv_type(m, elem);
		u32 idx = m->cv_next_type++;
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1002u);     // LF_POINTER
		coff_section_write_u32(T, ut);
		// attr: ptrtype=CV_PTR_64(0x0C, bits0-4) | size=8(bits13-18 → 8<<13=0x10000)
		coff_section_write_u32(T, 0x0001000Cu);
		x64_cv_finish_type(T, lp);
		map_set(&m->cv_types, t, idx);
		return idx;
	}
	if (bt->kind == Type_Array) {
		u32 et = x64_cv_type(m, bt->Array.elem);
		u32 idx = m->cv_next_type++;
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1503u);     // LF_ARRAY
		coff_section_write_u32(T, et);
		coff_section_write_u32(T, 0x0023u);     // index type u64
		x64_cv_numeric(T, type_size_of(t));
		coff_section_write_u8(T, 0);            // empty name
		x64_cv_finish_type(T, lp);
		map_set(&m->cv_types, t, idx);
		return idx;
	}

	// Struct-like: real structs, slices, strings, dynamic arrays, any.
	struct CVMember { Type *type; i64 off; String name; };
	CVMember mem[80];
	int nmem = 0;
	String tname = {};
	if (bt->kind == Type_Struct) {
		type_set_offsets(bt);
		for_array(i, bt->Struct.fields) {
			if (nmem >= 80) break;
			Entity *f = bt->Struct.fields[i];
			mem[nmem].type = f->type;
			mem[nmem].off  = bt->Struct.offsets[i];
			mem[nmem].name = f->token.string;
			nmem++;
		}
		gbString gs = type_to_string(t);
		tname = copy_string(permanent_allocator(), make_string((u8 const *)gs, gb_string_length(gs)));
	} else if (bt->kind == Type_Slice) {
		mem[0].type = alloc_type_pointer(bt->Slice.elem); mem[0].off = 0;  mem[0].name = str_lit("data");
		mem[1].type = t_int;                              mem[1].off = 8;  mem[1].name = str_lit("len");
		nmem = 2; tname = str_lit("slice");
	} else if (bt->kind == Type_DynamicArray) {
		mem[0].type = alloc_type_pointer(bt->DynamicArray.elem); mem[0].off = 0;  mem[0].name = str_lit("data");
		mem[1].type = t_int;                                     mem[1].off = 8;  mem[1].name = str_lit("len");
		mem[2].type = t_int;                                     mem[2].off = 16; mem[2].name = str_lit("cap");
		nmem = 3; tname = str_lit("dynamic_array");
	} else if (is_type_string(t)) {
		mem[0].type = alloc_type_pointer(t_u8); mem[0].off = 0; mem[0].name = str_lit("data");
		mem[1].type = t_int;                    mem[1].off = 8; mem[1].name = str_lit("len");
		nmem = 2; tname = str_lit("string");
	} else if (is_type_any(t)) {
		mem[0].type = t_rawptr; mem[0].off = 0; mem[0].name = str_lit("data");
		mem[1].type = t_typeid; mem[1].off = 8; mem[1].name = str_lit("id");
		nmem = 2; tname = str_lit("any");
	} else if (bt->kind == Type_FixedCapacityDynamicArray) {
		mem[0].type = alloc_type_array(bt->FixedCapacityDynamicArray.elem, bt->FixedCapacityDynamicArray.capacity);
		mem[0].off  = 0; mem[0].name = str_lit("data");
		mem[1].type = t_int; mem[1].off = x64_fca_len_offset(t); mem[1].name = str_lit("len");
		nmem = 2; tname = str_lit("fixed_capacity_dynamic_array");
	} else {
		return 0x0023u; // unmodelled aggregate: show first 8 bytes
	}

	// Forward-ref struct first so self-referential pointers resolve by name.
	u32 fwd = m->cv_next_type++;
	{
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1505u); // LF_STRUCTURE
		coff_section_write_u16(T, 0);       // count
		coff_section_write_u16(T, 0x80u);   // property: forward ref
		coff_section_write_u32(T, 0);       // field list
		coff_section_write_u32(T, 0);       // derived
		coff_section_write_u32(T, 0);       // vshape
		x64_cv_numeric(T, 0);              // size
		coff_section_write(T, tname.text, tname.len); coff_section_write_u8(T, 0);
		x64_cv_finish_type(T, lp);
	}
	map_set(&m->cv_types, t, fwd);

	u32 mty[80];
	for (int i = 0; i < nmem; i++) mty[i] = x64_cv_type(m, mem[i].type);

	u32 fl = m->cv_next_type++;
	{
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1203u); // LF_FIELDLIST
		for (int i = 0; i < nmem; i++) {
			coff_section_write_u16(T, 0x150Du); // LF_MEMBER
			coff_section_write_u16(T, 0x0003u); // public
			coff_section_write_u32(T, mty[i]);
			x64_cv_numeric(T, mem[i].off);
			coff_section_write(T, mem[i].name.text, mem[i].name.len); coff_section_write_u8(T, 0);
			x64_cv_align4_pad(T);
		}
		x64_cv_finish_type(T, lp);
	}

	u32 full = m->cv_next_type++;
	{
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1505u); // LF_STRUCTURE
		coff_section_write_u16(T, (u16)nmem);
		coff_section_write_u16(T, 0);       // property
		coff_section_write_u32(T, fl);      // field list
		coff_section_write_u32(T, 0);       // derived
		coff_section_write_u32(T, 0);       // vshape
		x64_cv_numeric(T, type_size_of(t));
		coff_section_write(T, tname.text, tname.len); coff_section_write_u8(T, 0);
		x64_cv_finish_type(T, lp);
	}
	map_set(&m->cv_types, t, full); // direct uses get the full definition
	return full;
}

// Is a SINGLE value of type `rt` returned via a hidden pointer (vs RAX/XMM0)?
// Win64 return ABI (mirrors LLVM lbArg_Indirect): register-sized scalar → RAX/XMM0;
// aggregates → hidden pointer unless size is 1/2/4/8. Zero-sized returns nothing.
gb_internal bool x64_single_value_by_pointer(Type *rt) {
	if (rt == nullptr) return false;
	i64 sz = type_size_of(rt);
	if (sz == 0) return false;                      // direct empty aggregate
	if (x64_is_scalar(rt) && sz <= 8) return false; // RAX / XMM0
	return !(sz == 1 || sz == 2 || sz == 4 || sz == 8);
}

// Number of results returned via HIDDEN POINTER ARGS: the first N-1 of an N-result
// tuple (mirrors LLVM split returns / lb_abi_modify_return_is_tuple). 0 for single/void.
gb_internal int x64_num_partial_returns(Type *proc_type) {
	Type *pt = base_type(proc_type);
	if (pt == nullptr || pt->kind != Type_Proc) return 0;
	if (pt->Proc.result_count > 1) return (int)pt->Proc.result_count - 1;
	return 0;
}

// The "real" return value's type: the LAST result for a multi-result proc, the sole
// result for single-result, or null for void (mirrors LLVM; the rest are pointer outputs).
gb_internal Type *x64_last_result_type(Type *proc_type) {
	Type *pt = base_type(proc_type);
	if (pt == nullptr || pt->kind != Type_Proc) return nullptr;
	if (pt->Proc.result_count == 0 || pt->Proc.results == nullptr) return nullptr;
	auto &vars = pt->Proc.results->Tuple.variables;
	return vars[vars.count-1]->type;
}

// For N>1 results only the LAST determines whether a hidden sret pointer (param slot 0)
// is needed; the first N-1 use their own hidden pointer args (see x64_num_partial_returns).
gb_internal bool x64_returns_by_pointer(Type *proc_type) {
	Type *pt = base_type(proc_type);
	if (pt == nullptr || pt->kind != Type_Proc) return false;
	if (pt->Proc.result_count == 0 || pt->Proc.results == nullptr) return false;
	return x64_single_value_by_pointer(x64_last_result_type(pt));
}

// ─────────────────────────────────────────────────────────────────────────────
// Prologue
// ─────────────────────────────────────────────────────────────────────────────

gb_internal void x64_proc_begin(x64Procedure *p) {
	X64Assembler *a   = &p->asm_;
	Type         *pt  = p->type;
	GB_ASSERT(pt->kind == Type_Proc);

	p->returns_by_pointer = x64_returns_by_pointer(pt);
	p->has_context = (pt->Proc.calling_convention == ProcCC_Odin);

	// Count ABI slots: [ret_ptr?] [explicit_params...] [partial_ret_ptrs...] [context?]
	// (partial-return ptrs = first N-1 results of a multi-result proc; see x64_num_partial_returns.)
	int slot = p->returns_by_pointer ? 1 : 0;
	int first_explicit = slot;

	if (pt->Proc.params != nullptr) {
		TypeTuple *params = &pt->Proc.params->Tuple;
		for_array(i, params->variables) {
			Entity *e = params->variables[i];
			if (e->kind != Entity_Variable) continue;
			if (e->flags & EntityFlag_CVarArg) continue;
			if (x64_type_size(e->type) == 0) continue; // zero-sized: no ABI slot
			slot++;
		}
	}

	p->num_partial_rets = x64_num_partial_returns(pt);
	if (p->num_partial_rets > 0) {
		p->first_partial_ret_slot = slot;
		slot += p->num_partial_rets;
	} else {
		p->first_partial_ret_slot = -1;
	}

	if (p->has_context) {
		p->context_slot = slot++;
	} else {
		p->context_slot = -1;
	}
	p->total_param_slots = slot;

	// ── Prologue ────────────────────────────────────────────────────────
	x64_emit_push_r(a, X64Reg_RBP);
	x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RBP, X64Reg_RSP);

	// Stack allocation. Reserve a fixed 13-byte region so proc_end can patch it to
	// either `SUB RSP, imm32` (+NOP pad) for a small frame, or
	// `MOV EAX,imm32; CALL __chkstk; SUB RSP,RAX` for a frame larger than one page.
	// Win64 REQUIRES probing the guard pages for >4KB frames (__chkstk preserves the
	// arg registers RCX/RDX/R8/R9); without it a deep push skips a guard page and faults.
	p->prologue_alloc_off = (isize)a->code.count;
	x64_emit_sub_ri(a, X64OpSize_64, X64Reg_RSP, 0x1000); // 7-byte imm32 placeholder
	p->sub_rsp_patch = (isize)(a->code.count - 4);         // imm32 sits at -4
	for (int i = 0; i < 6; i++) x64_enc_b(a, 0x90u);       // NOP pad → 13-byte region

	// ── Home register parameters into shadow slots ───────────────────────
	int explicit_slot = first_explicit;

	if (p->returns_by_pointer && p->total_param_slots >= 1) {
		// Slot 0: hidden return pointer in RCX
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(x64_param_rbp_off(0)), X64Reg_RCX);
	}

	if (pt->Proc.params != nullptr) {
		TypeTuple *params = &pt->Proc.params->Tuple;
		for_array(i, params->variables) {
			Entity *e = params->variables[i];
			if (e->kind != Entity_Variable) continue;
			if (e->flags & EntityFlag_CVarArg) continue;

			// Zero-sized param (empty struct/[0]T): no data, no ABI slot. Map it to a
			// non-dereferenced offset so references resolve (loads are no-ops).
			if (x64_type_size(e->type) == 0) {
				x64_var_set(&p->var_offsets, e, x64_alloc_local(p, 0, 1));
				continue;
			}

			int s = explicit_slot++;
			i32 off = x64_param_rbp_off(s);

			if (s < 4) {
				// register param — home into shadow slot
				if (x64_arg_is_float(e->type)) {
					if (x64_is_double(e->type)) x64_emit_movsd_mr(a, x64_rbp_mem(off), X64_XMM_ARG_REGS[s]);
					else                         x64_emit_movss_mr(a, x64_rbp_mem(off), X64_XMM_ARG_REGS[s]);
					x64_var_set(&p->var_offsets, e, off);
				} else {
					i64 esz = x64_type_size(e->type);
					x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(off), X64_INT_ARG_REGS[s]);
					if (x64_arg_is_indirect(e->type)) {
						// Win64: indirect params arrive as a pointer (homed to [off]).
						if (esz > X64_INDIRECT_PARAM_COPY_MAX) {
							// Too big to copy onto the stack — keep the pointer slot and
							// deref through it on access (mirrors LLVM byval-immutable).
							x64_var_set(&p->var_offsets, e, off);
							array_add(&p->indirect_params, e);
						} else {
							// Small: copy the data into a local so accesses work directly.
							i64 eal = type_align_of(e->type);
							i32 data_off = x64_alloc_local(p, esz, eal);
							x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(off));
							x64_copy_fixed(p, x64_rbp_mem(data_off), x64_mem(X64Reg_RAX, 0), esz);
							x64_var_set(&p->var_offsets, e, data_off);
						}
					} else {
						x64_var_set(&p->var_offsets, e, off);
					}
				}
			} else {
				// Stack parameter (slot ≥4) lives at [RBP + off].
				i64 esz = x64_type_size(e->type);
				if (x64_arg_is_indirect(e->type)) {
					// Win64: indirect aggregate passed by pointer — the stack slot
					// holds the pointer.
					if (esz > X64_INDIRECT_PARAM_COPY_MAX) {
						// Too big to copy — deref through the pointer slot on access.
						x64_var_set(&p->var_offsets, e, off);
						array_add(&p->indirect_params, e);
					} else {
						// Small: deref + copy the data into a local.
						i64 eal = type_align_of(e->type);
						i32 data_off = x64_alloc_local(p, esz, eal);
						x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(off));
						x64_copy_fixed(p, x64_rbp_mem(data_off), x64_mem(X64Reg_RAX, 0), esz);
						x64_var_set(&p->var_offsets, e, data_off);
					}
				} else {
					x64_var_set(&p->var_offsets, e, off);
				}
			}
		}
	}

	// Home the partial-return pointers (split returns): each is just a pointer that,
	// if it arrives in a register (slot < 4), must be spilled to its shadow slot so
	// x64_emit_named_returns / ReturnStmt can load it to write the result back.
	for (int k = 0; k < p->num_partial_rets; k++) {
		int s = p->first_partial_ret_slot + k;
		if (s < 4) {
			x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(x64_param_rbp_off(s)), X64_INT_ARG_REGS[s]);
		}
	}

	// Context pointer is in the last slot (if within reg range)
	if (p->has_context && p->context_slot < 4) {
		int s = p->context_slot;
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(x64_param_rbp_off(s)),
		                X64_INT_ARG_REGS[s]);
	}

	// Named returns: zero-init stack locals, written back in x64_emit_named_returns.
	if (pt->Proc.results != nullptr) {
		TypeTuple *results = &pt->Proc.results->Tuple;
		for_array(i, results->variables) {
			Entity *e = results->variables[i];
			if (e->kind != Entity_Variable) continue;
			// Unnamed result (`-> Allocator_Error`): only given a slot in a -debug build
			// (named "result"/"result_N" in CodeView, see x64_proc_end) so it's inspectable.
			if (e->token.string.len == 0 && p->module->debug_s == nullptr) continue;
			i64 sz    = x64_type_size(e->type);
			i64 align = x64_type_align(e->type);
			i32 off   = x64_alloc_local(p, sz, align);
			x64_var_set(&p->var_offsets, e, off);
			// zero-init (x64_zero_mem caps unrolling, REP STOSB for large results)
			x64_zero_mem(p, x64_rbp_mem(off), sz);
			// Named return with a DEFAULT value (`-> (pow: u32 = 1)`): apply it
			// (mirrors LLVM lb_build_proc_body's result-default store).
			ParameterValue const &pv = e->Variable.param_value;
			if (pv.kind != ParameterValue_Invalid &&
			    pv.kind != ParameterValue_Location &&    // not valid for a result default (LLVM asserts)
			    pv.kind != ParameterValue_Expression) {
				x64_store_value(p, x64addr(x64_rbp_mem(off), e->type),
				                x64_handle_param_value(p, e->type, pv, nullptr));
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Write named return locals back to caller's return area / registers
// ─────────────────────────────────────────────────────────────────────────────

gb_internal void x64_emit_named_returns(x64Procedure *p) {
	Type     *pt      = p->type;
	TypeTuple *results = (pt->Proc.results != nullptr) ? &pt->Proc.results->Tuple : nullptr;
	if (results == nullptr) return;

	bool has_named = false;
	for_array(i, results->variables) {
		Entity *e = results->variables[i];
		if (e->kind == Entity_Variable && e->token.string.len > 0 &&
		    x64_var_get(&p->var_offsets, e) != nullptr) {
			has_named = true; break;
		}
	}
	if (!has_named) return;

	X64Assembler *a = &p->asm_;
	int nres = (int)results->variables.count;

	// Split returns (mirrors LLVM): results[0..N-2] copy through their hidden pointer args;
	// result[N-1] is the REAL return — sret (param slot 0) if by pointer, else RAX/XMM0.
	for (int i = 0; i < nres; i++) {
		Entity *e = results->variables[i];
		if (e->kind != Entity_Variable) continue;
		i32 *loff = (e->token.string.len > 0) ? x64_var_get(&p->var_offsets, e) : nullptr;
		i64  sz   = x64_type_size(e->type);
		bool is_last = (i == nres - 1);

		if (!is_last) {
			// Partial return: dst pointer is the hidden arg slot; copy the local to it.
			if (loff == nullptr || sz == 0) continue;
			int s = p->first_partial_ret_slot + i;
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(s)));
			x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(*loff), sz);
			continue;
		}

		// Last (real) result.
		if (loff == nullptr || sz == 0) continue;
		if (p->returns_by_pointer) {
			x64_emit_mov_rm(a, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
			x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(*loff), sz);
		} else if (x64_is_float(e->type)) {
			if (x64_is_double(e->type)) x64_emit_movsd_rm(a, X64XmmReg_XMM0, x64_rbp_mem(*loff));
			else                         x64_emit_movss_rm(a, X64XmmReg_XMM0, x64_rbp_mem(*loff));
		} else {
			x64_value_to_reg(p, x64v_mem(e->type, x64_rbp_mem(*loff)), X64Reg_RAX);
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Epilogue (shared by ReturnStmt and proc_end)
// ─────────────────────────────────────────────────────────────────────────────

gb_internal void x64_proc_emit_epilogue(x64Procedure *p) {
	X64Assembler *a = &p->asm_;
	x64_emit_mov_rr(a, X64OpSize_64, X64Reg_RSP, X64Reg_RBP);
	x64_emit_pop_r(a, X64Reg_RBP);
}

// ─────────────────────────────────────────────────────────────────────────────
// CodeView S_INLINESITE (#force_inline inline frames)
// ─────────────────────────────────────────────────────────────────────────────

// CodeView compressed unsigned int (CodeViewRecordIO::writeEncodedInteger): 1/2/4 bytes,
// big-endian with 0x80 / 0xC0 continuation bits.
gb_internal void x64_cv_annot_uint(Array<u8> *o, u32 v) {
	if (v < 0x80u) {
		array_add(o, (u8)v);
	} else if (v < 0x4000u) {
		array_add(o, (u8)((v >> 8) | 0x80u));
		array_add(o, (u8)(v & 0xFFu));
	} else {
		array_add(o, (u8)((v >> 24) | 0xC0u));
		array_add(o, (u8)((v >> 16) & 0xFFu));
		array_add(o, (u8)((v >> 8) & 0xFFu));
		array_add(o, (u8)(v & 0xFFu));
	}
}
// Signed → zig-zag (positive: v<<1; negative: (-v<<1)|1), then the unsigned encoding.
gb_internal void x64_cv_annot_sint(Array<u8> *o, i32 v) {
	x64_cv_annot_uint(o, (v >= 0) ? ((u32)v << 1) : (((u32)(-v) << 1) | 1u));
}

// Binary-annotation bytestream for an inline site: maps code offsets (from function start) to the
// callee's source lines, relative to the inlinee's declaration line+file. Opcodes: 3=ChangeCodeOffset,
// 4=ChangeCodeLength, 5=ChangeFile, 6=ChangeLineOffset (separate ops, never the packed form — simpler
// & safe). The interpreter starts at codeOffset=0, line=decl_line, file=decl_file.
gb_internal void x64_cv_build_annotations(x64Module *m, Array<u8> *o, x64Procedure::InlineSiteRec *site) {
	if (site->lines.count == 0) {
		// No body lines recorded: position at code_start, span to code_end (avoids claiming the
		// whole function prefix as inlined). Degenerate — a real #force_inline body has lines.
		if (site->code_start > 0)               { x64_cv_annot_uint(o, 3u); x64_cv_annot_uint(o, site->code_start); }
		if (site->code_end > site->code_start)  { x64_cv_annot_uint(o, 4u); x64_cv_annot_uint(o, site->code_end - site->code_start); }
		return;
	}
	u32 last_off  = 0;
	i32 last_line = site->decl_line;
	u32 last_file = x64_cv_file_offset(m, site->decl_file_id);
	for (isize i = 0; i < site->lines.count; i++) {
		x64Procedure::InlineSiteLine *L = &site->lines[i];
		u32 file = x64_cv_file_offset(m, L->file_id);
		if (file != last_file) { x64_cv_annot_uint(o, 5u); x64_cv_annot_uint(o, file); last_file = file; }
		u32 code_delta = L->offset - last_off;
		i32 line_delta = L->line  - last_line;
		// Line BEFORE code: the marker created when the code offset advances must already carry the
		// new line (mirrors the packed ChangeCodeOffsetAndLineOffset / LLVM's emission order).
		if (line_delta != 0) { x64_cv_annot_uint(o, 6u); x64_cv_annot_sint(o, line_delta); }
		if (code_delta != 0) { x64_cv_annot_uint(o, 3u); x64_cv_annot_uint(o, code_delta); }
		last_off = L->offset; last_line = L->line;
	}
	if (site->code_end > last_off) { x64_cv_annot_uint(o, 4u); x64_cv_annot_uint(o, site->code_end - last_off); }
}

// Emit (once, cached) an LF_FUNC_ID id-record for `callee` → its .debug$T index (referenced by
// S_INLINESITE.inlinee). Needs an LF_PROCEDURE; a shared void() suffices (the debugger uses the id's
// NAME for the inline frame). ID + type records share the obj .debug$T index space; the linker splits
// TPI/IPI by leaf kind.
gb_internal u32 x64_cv_func_id(x64Module *m, Entity *callee) {
	if (m->debug_t == nullptr) return 0;
	u32 *cached = map_get(&m->cv_func_ids, callee);
	if (cached) return *cached;

	CoffSection *T = m->debug_t;
	if (m->cv_empty_arglist == 0) {
		u32 idx = m->cv_next_type++;
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1201u);  // LF_ARGLIST
		coff_section_write_u32(T, 0);        // count = 0
		x64_cv_finish_type(T, lp);
		m->cv_empty_arglist = idx;
	}
	if (m->cv_void_proc_type == 0) {
		u32 idx = m->cv_next_type++;
		u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
		coff_section_write_u16(T, 0x1008u);  // LF_PROCEDURE
		coff_section_write_u32(T, 0x0003u);  // return type = T_VOID
		coff_section_write_u8(T, 0);         // calling convention = NEAR_C
		coff_section_write_u8(T, 0);         // funcattr
		coff_section_write_u16(T, 0);        // parm count
		coff_section_write_u32(T, m->cv_empty_arglist);
		x64_cv_finish_type(T, lp);
		m->cv_void_proc_type = idx;
	}

	String name = callee->token.string;
	u32 idx = m->cv_next_type++;
	u32 lp = (u32)coff_section_len(T); coff_section_write_u16(T, 0);
	coff_section_write_u16(T, 0x1601u);   // LF_FUNC_ID
	coff_section_write_u32(T, 0);         // scopeId = 0
	coff_section_write_u32(T, m->cv_void_proc_type);
	coff_section_write(T, name.text, name.len);
	coff_section_write_u8(T, 0);
	x64_cv_finish_type(T, lp);
	map_set(&m->cv_func_ids, callee, idx);
	array_add(&m->cv_inlinees, callee); // → DEBUG_S_INLINEELINES at finalize
	return idx;
}

// One S_REGREL32 (RBP-relative local) — shared by the GPROC flat locals and inline-site scopes.
gb_internal void x64_cv_emit_regrel32(x64Module *m, String name, i32 off, u32 cvt) {
	isize content = 2 + 4 + 4 + 2 + name.len + 1;        // rectype+offset+typind+reg+name
	isize total   = (2 + content + 3) & ~(isize)3;
	coff_section_write_u16(m->debug_s, (u16)(total - 2)); // reclen
	coff_section_write_u16(m->debug_s, 0x1111u);          // S_REGREL32
	coff_section_write_u32(m->debug_s, (u32)off);
	coff_section_write_u32(m->debug_s, cvt);
	coff_section_write_u16(m->debug_s, 334u);             // CV_AMD64_RBP
	coff_section_write(m->debug_s, name.text, name.len);
	coff_section_write_u8(m->debug_s, 0);
	x64_cv_pad4(m->debug_s);
}

// Recursively emit S_INLINESITE … (scoped locals) … (child sites) … S_INLINESITE_END for `idx` and
// every site whose parent is `idx`. pParent/pEnd left 0 — the linker recomputes scope linkage from
// the balanced record nesting (same as the GPROC, which also writes 0).
gb_internal void x64_cv_emit_inline_site(x64Procedure *p, i32 idx) {
	x64Module *m = p->module;
	x64Procedure::InlineSiteRec *site = &p->inline_sites[idx];

	u32 inlinee = x64_cv_func_id(m, site->callee);

	Array<u8> annot; array_init(&annot, p->alloc, 0, 32);
	x64_cv_build_annotations(m, &annot, site);
	// Pad the ANNOTATION stream to 4 bytes with 0x00 (BA_OP_Invalid) — the annotation parser reads
	// trailing record bytes as opcodes, so the record's alignment padding must be Invalid (0), NOT
	// the LF_PAD 0xF1-0xF3 used elsewhere (those decode as bogus opcodes). The fixed header is 16B
	// (mult of 4), so a 4-aligned annot makes the whole record 4-aligned without x64_cv_pad4.
	while ((annot.count % 4) != 0) array_add(&annot, (u8)0);

	// S_INLINESITE: rectype(2)+pParent(4)+pEnd(4)+inlinee(4)+annotations.
	isize content = 2 + 4 + 4 + 4 + annot.count; // already a multiple of 4
	isize total   = 2 + content;
	coff_section_write_u16(m->debug_s, (u16)(total - 2)); // reclen
	coff_section_write_u16(m->debug_s, 0x114Du);          // S_INLINESITE
	coff_section_write_u32(m->debug_s, 0);                // pParent (linker recomputes)
	coff_section_write_u32(m->debug_s, 0);                // pEnd    (linker recomputes)
	coff_section_write_u32(m->debug_s, inlinee);
	if (annot.count > 0) coff_section_write(m->debug_s, annot.data, annot.count);

	// Scoped locals (per-site offsets → correct even when the same callee inlines at many sites).
	for (isize i = 0; i < site->locals.count; i++) {
		x64Procedure::InlineSiteLocal *L = &site->locals[i];
		u32 cvt = L->is_ptr ? 0x0603u : x64_cv_type(m, L->type); // by-ptr param: show as pointer
		x64_cv_emit_regrel32(m, L->name, L->offset, cvt);
	}

	// Child inline sites (nested #force_inline within this body).
	for (i32 c = 0; c < (i32)p->inline_sites.count; c++) {
		if (p->inline_sites[c].parent == idx) x64_cv_emit_inline_site(p, c);
	}

	// S_INLINESITE_END
	coff_section_write_u16(m->debug_s, 2u);
	coff_section_write_u16(m->debug_s, 0x114Eu);
}

// ─────────────────────────────────────────────────────────────────────────────
// Finalise: patch frame size, write COFF
// ─────────────────────────────────────────────────────────────────────────────

gb_internal void x64_proc_end(x64Procedure *p) {
	X64Assembler *a = &p->asm_;

	x64_emit_ret(a);

	// frame = locals + shadow space (32) + outgoing stack args, 16-byte aligned. The outgoing
	// area is the peak over all calls (max_outgoing_bytes), floored at 64 so the alloca path's
	// fixed rsp+96 (shadow 32 + 64) assumption holds.
	i32 outgoing = gb_max(p->max_outgoing_bytes, 64);
	i32 frame = ((p->frame_max + 32 + outgoing + 15) & ~15);
	if (frame < 0x1000) {
		// Under one page: plain SUB RSP is safe (a deep push still hits & commits
		// the guard page). NOP pad follows.
		x64_enc_patch_d(a, p->sub_rsp_patch, frame);
	} else {
		// >1 page: probe via __chkstk (Win64 ABI). Rewrite the reserved 13 bytes to
		// MOV EAX,frame ; CALL __chkstk ; SUB RSP,RAX; rel32 filled by a REL32 reloc.
		u8 *q = a->code.data + p->prologue_alloc_off;
		q[0]  = 0xB8;                                                     // mov eax, imm32
		q[1]  = (u8)( frame        & 0xFF); q[2] = (u8)((frame >> 8)  & 0xFF);
		q[3]  = (u8)((frame >> 16) & 0xFF); q[4] = (u8)((frame >> 24) & 0xFF);
		q[5]  = 0xE8;                                                     // call rel32
		q[6]  = q[7] = q[8] = q[9] = 0;                                   // rel32 (reloc fills)
		q[10] = 0x48; q[11] = 0x29; q[12] = 0xC4;                         // sub rsp, rax
		X64RelocEntry r = {};
		r.code_offset = p->prologue_alloc_off + 6;
		r.type        = X64Reloc_REL32;
		r.sym_name    = str_lit("__chkstk");
		array_add(&a->relocs, r);
	}

	x64Module *m       = p->module;
	u32        base_off = (u32)coff_section_len(m->text);

	// All procs must be EXTERNAL: even file-private (#+private) procs can be
	// referenced as UNDEF externals from other .obj files before their definition
	// is seen, and STATIC symbols are never searched when resolving those refs.
	// EXCEPTION: synthetic map hasher/equal procs (is_static) are referenced only
	// within their own module → STATIC, so parallel modules don't clash on the name.
	coff_sym_add_proc(&m->coff, p->link_name, 1 /*text is sec 1*/, base_off, !p->is_static);

	coff_section_write(m->text, a->code.data, a->code.count);
	coff_apply_x64_relocs(m->text, a, &m->coff, base_off);

	// ── Win64 unwind info (.xdata UNWIND_INFO + .pdata RUNTIME_FUNCTION) ──────
	// Required for the debugger/OS to walk the stack past the current frame.
	// Prologue is fixed: push rbp(1) ; mov rbp,rsp(3) ; sub rsp,imm32(7).
	if (m->xdata && m->pdata) {
		u32 proc_sz   = (u32)a->code.count;
		u32 xdata_off = (u32)coff_section_len(m->xdata);

		bool alloc_small = (frame <= 128);
		coff_section_write_u8(m->xdata, 0x01);                  // Version=1, Flags=0
		coff_section_write_u8(m->xdata, 11);                    // SizeOfProlog
		coff_section_write_u8(m->xdata, alloc_small ? 3u : 4u); // CountOfCodes
		coff_section_write_u8(m->xdata, 0x05);                  // FrameReg=RBP(5), FrameOff=0
		// Unwind codes, descending prolog offset:
		if (alloc_small) {
			coff_section_write_u8(m->xdata, 11);                              // ALLOC_SMALL @11
			coff_section_write_u8(m->xdata, (u8)((((frame/8) - 1) << 4) | 2));
		} else {
			coff_section_write_u8(m->xdata, 11);                              // ALLOC_LARGE @11
			coff_section_write_u8(m->xdata, 0x01);
			coff_section_write_u16(m->xdata, (u16)(frame/8));                 // size/8
		}
		coff_section_write_u8(m->xdata, 4);    coff_section_write_u8(m->xdata, 0x03); // SET_FPREG @4
		coff_section_write_u8(m->xdata, 1);    coff_section_write_u8(m->xdata, 0x50); // PUSH_NONVOL RBP @1
		if (alloc_small) coff_section_write_u16(m->xdata, 0); // pad to even slot count

		// RUNTIME_FUNCTION: {BeginAddress, EndAddress, UnwindInfoAddress} (RVAs).
		u32 poff = (u32)coff_section_len(m->pdata);
		coff_section_write_u32(m->pdata, 0);          // Begin = func
		coff_reloc_add(m->pdata, poff, p->link_name, COFF_REL_ADDR32NB);
		coff_section_write_u32(m->pdata, proc_sz);    // End = func + size (addend)
		coff_reloc_add(m->pdata, poff + 4, p->link_name, COFF_REL_ADDR32NB);
		coff_section_write_u32(m->pdata, xdata_off);  // Unwind = xdata + off (addend)
		coff_reloc_add_idx(m->pdata, poff + 8, m->xdata->sym_idx, COFF_REL_ADDR32NB);
	}

	// ── CodeView .debug$S — S_GPROC32 function name record ──────────────────
	// Gives the debugger a name ↔ address mapping so crash addresses show up
	// as function names in the call stack.
	if (m->debug_s) {
		u32   proc_size = (u32)a->code.count;
		isize name_len  = p->link_name.len;

		// S_GPROC32 (0x1110): every CV symbol record is 4-byte aligned, padding
		// (0xF1..0xF3) inside the record and counted in reclen. Fixed part is 35
		// bytes; raw_size = reclen(2) + 38 + name(name_len+1).
		isize raw_size  = 40 + name_len;
		isize pad_count = (4 - raw_size % 4) % 4;
		isize gproc_size = raw_size + pad_count;

		(void)gproc_size;

		// DEBUG_S_SYMBOLS subsection: S_GPROC32 + S_FRAMEPROC + S_REGREL32 locals +
		// S_END. Variable-length, so placeholder length patched after emitting.
		coff_section_write_u32(m->debug_s, 0xF1u);              // DEBUG_S_SYMBOLS
		u32 sub_len_pos  = (u32)coff_section_len(m->debug_s);
		coff_section_write_u32(m->debug_s, 0);                  // length (patched below)
		u32 sub_data_pos = (u32)coff_section_len(m->debug_s);

		// S_GPROC32:
		coff_section_write_u16(m->debug_s, (u16)(38 + name_len + pad_count)); // reclen
		coff_section_write_u16(m->debug_s, 0x1110u);  // rectype = S_GPROC32
		coff_section_write_u32(m->debug_s, 0);        // pParent
		coff_section_write_u32(m->debug_s, 0);        // pEnd
		coff_section_write_u32(m->debug_s, 0);        // pNext
		coff_section_write_u32(m->debug_s, proc_size); // len
		coff_section_write_u32(m->debug_s, 0);        // DbgStart
		coff_section_write_u32(m->debug_s, proc_size); // DbgEnd
		coff_section_write_u32(m->debug_s, 0);        // typind

		// off: section-relative offset — filled by SECREL reloc at link time
		u32 off_pos = (u32)coff_section_len(m->debug_s);
		coff_section_write_u32(m->debug_s, 0);
		coff_reloc_add(m->debug_s, off_pos, p->link_name, COFF_REL_SECREL);

		// seg: section index — filled by SECTION reloc at link time
		u32 seg_pos = (u32)coff_section_len(m->debug_s);
		coff_section_write_u16(m->debug_s, 0);
		coff_reloc_add(m->debug_s, seg_pos, p->link_name, COFF_REL_SECTION);

		coff_section_write_u8(m->debug_s, 0);  // flags
		coff_section_write(m->debug_s, p->link_name.text, name_len); // name
		coff_section_write_u8(m->debug_s, 0);  // null terminator
		x64_cv_pad4(m->debug_s);

		// S_FRAMEPROC (0x1012): declare an RBP-based frame so the debugger
		// interprets S_REGREL32 offsets relative to RBP.
		coff_section_write_u16(m->debug_s, 30u);          // reclen (28 content + 2 pad)
		coff_section_write_u16(m->debug_s, 0x1012u);      // rectype = S_FRAMEPROC
		coff_section_write_u32(m->debug_s, (u32)frame);   // cbFrame
		coff_section_write_u32(m->debug_s, 0);            // cbPad
		coff_section_write_u32(m->debug_s, 0);            // offPad
		coff_section_write_u32(m->debug_s, 0);            // cbSaveRegs
		coff_section_write_u32(m->debug_s, 0);            // offExHdlr
		coff_section_write_u16(m->debug_s, 0);            // sectExHdlr
		// flags: encodedLocalBasePointer=2 (RBP)<<14 | encodedParamBasePointer=2 (RBP)<<16
		coff_section_write_u32(m->debug_s, 0x00028000u);
		x64_cv_pad4(m->debug_s);

		// Results tuple — used to give unnamed returns a synthetic debugger name below.
		TypeTuple *res_tuple = (p->type != nullptr && p->type->kind == Type_Proc &&
		                        p->type->Proc.results != nullptr) ? &p->type->Proc.results->Tuple : nullptr;

		// S_REGREL32 (0x1111) for each named local/param: name @ [RBP + offset].
		for (i32 vi = 0; vi < p->var_offsets.count; vi++) {
			Entity *ve   = p->var_offsets.keys[vi];
			i32     voff = p->var_offsets.vals[vi];
			if (ve == nullptr) continue;
			// Inline-site locals are emitted inside their S_INLINESITE scope (below); skip them here
			// so they aren't duplicated (a callee inlined N times would otherwise show only its last
			// slot in this flat list). Counts are tiny → linear scan.
			bool is_inlined_local = false;
			for (isize s = 0; s < p->inline_sites.count && !is_inlined_local; s++) {
				for (isize li = 0; li < p->inline_sites[s].locals.count; li++) {
					if (p->inline_sites[s].locals[li].entity == ve) { is_inlined_local = true; break; }
				}
			}
			if (is_inlined_local) continue;
			String vn = ve->token.string;
			char   synthbuf[24];
			if (vn.len == 0) {
				// Unnamed result → synthetic name ("return", or "return_N" for multi-result).
				// `return` is a reserved word so it can never collide with a real local. Only
				// unnamed RESULTS are var_set without a name (see x64_proc_begin); skip any
				// other nameless slot.
				int ridx = -1;
				if (res_tuple != nullptr) {
					for_array(ri, res_tuple->variables) {
						if (res_tuple->variables[ri] == ve) { ridx = (int)ri; break; }
					}
				}
				if (ridx < 0) continue;
				if (res_tuple->variables.count == 1) {
					vn = str_lit("return");
				} else {
					gb_snprintf(synthbuf, gb_size_of(synthbuf), "return_%d", ridx);
					vn = make_string((u8 const *)synthbuf, gb_strlen(synthbuf));
				}
			}
			u32 cvt = x64_cv_type(m, ve->type);
			// Large indirect param: the slot holds the incoming POINTER, not the data —
			// describe it as a pointer so the debugger shows a valid address to deref
			// (not the pointer bytes misread as the aggregate).
			for (isize ii = 0; ii < p->indirect_params.count; ii++) {
				if (p->indirect_params[ii] == ve) { cvt = 0x0603u; break; } // T_64PVOID
			}
			// content: rectype(2)+offset(4)+typind(4)+reg(2)+name(len+1)
			isize content = 2 + 4 + 4 + 2 + vn.len + 1;
			isize total   = (2 + content + 3) & ~(isize)3;
			coff_section_write_u16(m->debug_s, (u16)(total - 2)); // reclen
			coff_section_write_u16(m->debug_s, 0x1111u);          // rectype = S_REGREL32
			coff_section_write_u32(m->debug_s, (u32)voff);        // offset from reg
			coff_section_write_u32(m->debug_s, cvt);              // typind
			coff_section_write_u16(m->debug_s, 334u);             // reg = CV_AMD64_RBP
			coff_section_write(m->debug_s, vn.text, vn.len);
			coff_section_write_u8(m->debug_s, 0);
			x64_cv_pad4(m->debug_s);
		}

		// S_INLINESITE trees for #force_inline calls (each a nested, step-into-able frame whose
		// own lines come from binary annotations; the primary line table keeps the call-site line).
		for (i32 s = 0; s < (i32)p->inline_sites.count; s++) {
			if (p->inline_sites[s].parent == -1) x64_cv_emit_inline_site(p, s);
		}

		// S_END:
		coff_section_write_u16(m->debug_s, 2u);       // reclen
		coff_section_write_u16(m->debug_s, 0x0006u);  // rectype = S_END

		// Patch the subsection length now that all records are written.
		u32 sub_len = (u32)coff_section_len(m->debug_s) - sub_data_pos;
		m->debug_s->data[sub_len_pos + 0] = (u8)( sub_len        & 0xFFu);
		m->debug_s->data[sub_len_pos + 1] = (u8)((sub_len >> 8)  & 0xFFu);
		m->debug_s->data[sub_len_pos + 2] = (u8)((sub_len >> 16) & 0xFFu);
		m->debug_s->data[sub_len_pos + 3] = (u8)((sub_len >> 24) & 0xFFu);

		// ── DEBUG_S_LINES subsection — code offset → source line mapping ──────
		// Entries are grouped into one CV file block per run of the same file_id; inlined
		// #force_inline code carries the callee's file, so the table can span multiple files
		// (lets the debugger step INTO inlined bodies showing their own source).
		if (p->lines.count > 0) {
			isize n = p->lines.count;
			// Payload: CV_DebugSLinesHeader_t(12) + per file block (FileBlockHeader(12) + lines*8).
			u32 cb = 12;
			for (isize i = 0; i < n;) {
				i32 f = p->lines[i].file_id; isize j = i;
				while (j < n && p->lines[j].file_id == f) j++;
				cb += 12 + (u32)(j - i) * 8;
				i = j;
			}

			coff_section_write_u32(m->debug_s, 0xF2u);  // DEBUG_S_LINES
			coff_section_write_u32(m->debug_s, cb);

			// CV_DebugSLinesHeader_t: offCon(4,SECREL) + segCon(2,SECTION) +
			//                         flags(2) + cbCon(4)
			u32 off_pos2 = (u32)coff_section_len(m->debug_s);
			coff_section_write_u32(m->debug_s, 0);
			coff_reloc_add(m->debug_s, off_pos2, p->link_name, COFF_REL_SECREL);

			u32 seg_pos2 = (u32)coff_section_len(m->debug_s);
			coff_section_write_u16(m->debug_s, 0);
			coff_reloc_add(m->debug_s, seg_pos2, p->link_name, COFF_REL_SECTION);

			coff_section_write_u16(m->debug_s, 0);          // flags (0 = no columns)
			coff_section_write_u32(m->debug_s, proc_size);  // cbCon

			// One CV_DebugSLinesFileBlockHeader_t per file run: offFile(4) + nLines(4) + cbBlock(4).
			for (isize i = 0; i < n;) {
				i32 f = p->lines[i].file_id; isize j = i;
				while (j < n && p->lines[j].file_id == f) j++;
				u32 cnt  = (u32)(j - i);
				u32 foff = x64_cv_file_offset(m, f);
				coff_section_write_u32(m->debug_s, foff);
				coff_section_write_u32(m->debug_s, cnt);
				coff_section_write_u32(m->debug_s, 12 + cnt * 8); // cbBlock
				// CV_Line_t records: offset(4) + flags(4); flags = line | fStatement.
				for (isize k = i; k < j; k++) {
					coff_section_write_u32(m->debug_s, p->lines[k].offset);
					coff_section_write_u32(m->debug_s,
					    (p->lines[k].line & 0x00FFFFFFu) | 0x80000000u);
				}
				i = j;
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Call emission
// ─────────────────────────────────────────────────────────────────────────────

gb_internal x64Value x64_emit_call(x64Procedure *p,
                                    String        callee_name,
                                    Type         *callee_type_raw,
                                    x64Value     *args,
                                    int           arg_count)
{
	X64Assembler *a  = &p->asm_;
	Type         *ct = base_type(callee_type_raw);
	GB_ASSERT(ct->kind == Type_Proc);

	// Reserve enough outgoing stack-arg space for THIS call (args beyond the 4 register slots);
	// x64_proc_end sizes the frame's outgoing area from this peak.
	if (arg_count > 4) {
		i32 ob = (arg_count - 4) * 8;
		if (ob > p->max_outgoing_bytes) p->max_outgoing_bytes = ob;
	}

	// Stack args (slots 4+), reverse order
	for (int i = gb_max(arg_count - 1, 3); i >= 4; i--) {
		x64Value v = args[i];
		X64Mem stack_slot = x64_mem(X64Reg_RSP, 32 + (i - 4) * 8);
		if (x64_arg_is_float(v.type)) {
			x64_value_to_xmm(p, v, X64XmmReg_XMM0);
			if (x64_is_double(v.type)) x64_emit_movsd_mr(a, stack_slot, X64XmmReg_XMM0);
			else                        x64_emit_movss_mr(a, stack_slot, X64XmmReg_XMM0);
		} else {
			x64_value_to_reg(p, v, X64Reg_RAX);
			x64_emit_mov_mr(a, X64OpSize_64, stack_slot, X64Reg_RAX);
		}
	}

	// Register args (slots 0-3), reverse order to avoid clobbering: fill R9/XMM3 first.
	// Win64 variadic ABI: an FP arg in a register slot of a c_vararg callee must ALSO be
	// placed in the corresponding GP register (the callee reads `...` args from GP).
	bool variadic_abi = ct->Proc.c_vararg;
	int reg_count = gb_min(arg_count, 4);
	for (int i = reg_count - 1; i >= 0; i--) {
		x64Value v = args[i];
		if (x64_arg_is_float(v.type)) {
			x64_value_to_xmm(p, v, X64_XMM_ARG_REGS[i]);
			if (variadic_abi) x64_value_to_reg(p, v, X64_INT_ARG_REGS[i]); // duplicate FP bits into GP
		} else {
			x64_value_to_reg(p, v, X64_INT_ARG_REGS[i]);
		}
	}

	x64_emit_call_sym(a, callee_name);

	Type *ret_type = nullptr;
	if (ct->Proc.results != nullptr && ct->Proc.result_count == 1) {
		ret_type = ct->Proc.results->Tuple.variables[0]->type;
	}
	if (x64_returns_by_pointer(callee_type_raw) || ct->Proc.result_count != 1) {
		return x64v_none();
	}
	if (ret_type && type_size_of(ret_type) == 0) return x64v_none(); // zero-sized: no value
	if (ret_type && x64_is_float(ret_type)) return x64v_xmm(ret_type, X64XmmReg_XMM0);
	if (ret_type) return x64v_reg(ret_type, X64Reg_RAX);
	return x64v_none();
}

// ─────────────────────────────────────────────────────────────────────────────
// Defer / loop helpers
// ─────────────────────────────────────────────────────────────────────────────

// Emit the deferred entries at indices [base, count) in reverse order, WITHOUT
// popping them. Used at non-local exits (return/break/continue) where all (or a
// range of) active defers must run but the entries stay live for other paths.
gb_internal void x64_run_deferred_from(x64Procedure *p, isize base) {
	for (isize i = p->deferred.count - 1; i >= base; i--) {
		x64Procedure::DeferEntry &de = p->deferred[i];
		if (de.stmt != nullptr) {
			x64_build_stmt(p, de.stmt);
		} else if (de.call_proc != nullptr) {
			x64_emit_call(p, x64_get_entity_name(de.call_proc),
			              de.call_type, de.call_args, de.call_argc);
		}
	}
}

gb_internal void x64_run_deferred(x64Procedure *p) {
	x64_run_deferred_from(p, 0);
}

// Normal scope exit: run this scope's defers (>= base) then pop them so they don't
// leak to sibling scopes/the epilogue. Defers are scoped to their enclosing block
// (a `defer` in a switch case runs only when that case executes).
gb_internal void x64_scope_end(x64Procedure *p, isize base) {
	x64_run_deferred_from(p, base);
	p->deferred.count = base;
}

gb_internal void x64_push_loop(x64Procedure *p, isize lbl_break, isize lbl_continue, Ast *label) {
	x64Procedure::LoopInfo li = {};
	li.lbl_break    = lbl_break;
	li.lbl_continue = lbl_continue;
	li.ast_label    = label;
	li.defer_base   = p->deferred.count; // branch unwind floor for break/continue
	array_add(&p->loops, li);
}

gb_internal void x64_pop_loop(x64Procedure *p) {
	GB_ASSERT(p->loops.count > 0);
	p->loops.count--;
}
