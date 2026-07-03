#include "x64_encode.hpp"

// ===========================================================================
// Memory operand constructors
// ===========================================================================

gb_internal X64Mem x64_mem(X64Reg base, i32 disp) {
	X64Mem m = {};
	m.base  = base;
	m.index = X64Reg_NONE;
	m.scale = 1;
	m.disp  = disp;
	return m;
}

gb_internal X64Mem x64_mem_idx(X64Reg base, X64Reg index, u8 scale, i32 disp) {
	X64Mem m = {};
	m.base  = base;
	m.index = index;
	m.scale = scale;
	m.disp  = disp;
	return m;
}

gb_internal X64Mem x64_mem_rip(i32 disp) {
	X64Mem m = {};
	m.base    = X64Reg_NONE;
	m.index   = X64Reg_NONE;
	m.scale   = 1;
	m.disp    = disp;
	m.rip_rel = true;
	return m;
}

gb_internal X64Mem x64_mem_abs(i32 disp) {
	X64Mem m = {};
	m.base  = X64Reg_NONE;
	m.index = X64Reg_NONE;
	m.scale = 1;
	m.disp  = disp;
	return m;
}

// ===========================================================================
// Assembler lifecycle
// ===========================================================================

gb_internal void x64_asm_init(X64Assembler *a, gbAllocator allocator) {
	a->allocator = allocator;
	array_init(&a->code,   allocator, 0, 256);
	array_init(&a->labels, allocator, 0, 16);
	array_init(&a->fixups, allocator, 0, 16);
	array_init(&a->relocs, allocator, 0, 16);
}

gb_internal void x64_asm_free(X64Assembler *a) {
	array_free(&a->code);
	array_free(&a->labels);
	array_free(&a->fixups);
	array_free(&a->relocs);
}

gb_internal isize x64_asm_len(X64Assembler *a) {
	return a->code.count;
}

// ===========================================================================
// Labels
// ===========================================================================

gb_internal isize x64_label_alloc(X64Assembler *a) {
	isize idx = a->labels.count;
	array_add(&a->labels, cast(isize)X64_LABEL_UNSET);
	return idx;
}

gb_internal void x64_label_bind(X64Assembler *a, isize label) {
	GB_ASSERT(label >= 0 && label < a->labels.count);
	GB_ASSERT_MSG(a->labels[label] == X64_LABEL_UNSET, "label already bound");
	isize target = a->code.count;
	a->labels[label] = target;

	// Patch all pending fixups that reference this label.
	for_array(i, a->fixups) {
		X64Fixup *fx = &a->fixups[i];
		if (fx->label != label) continue;
		isize patch_at = fx->patch_at;
		i32 rel = cast(i32)(target - (patch_at + 4));
		u8 *p = a->code.data + patch_at;
		p[0] = cast(u8)( rel        & 0xFF);
		p[1] = cast(u8)((rel >>  8) & 0xFF);
		p[2] = cast(u8)((rel >> 16) & 0xFF);
		p[3] = cast(u8)((rel >> 24) & 0xFF);
		fx->label = X64_LABEL_UNSET; // mark consumed
	}
}

gb_internal bool x64_label_bound(X64Assembler *a, isize label) {
	GB_ASSERT(label >= 0 && label < a->labels.count);
	return a->labels[label] != X64_LABEL_UNSET;
}

// ===========================================================================
// Internal encoding helpers
// ===========================================================================

// Raw byte / word / dword / qword writers

gb_internal void x64_enc_b(X64Assembler *a, u8 v) {
	array_add(&a->code, v);
}
gb_internal void x64_enc_w(X64Assembler *a, u16 v) {
	array_add(&a->code, cast(u8)( v       & 0xFF));
	array_add(&a->code, cast(u8)((v >> 8) & 0xFF));
}
gb_internal void x64_enc_d(X64Assembler *a, u32 v) {
	array_add(&a->code, cast(u8)( v        & 0xFF));
	array_add(&a->code, cast(u8)((v >>  8) & 0xFF));
	array_add(&a->code, cast(u8)((v >> 16) & 0xFF));
	array_add(&a->code, cast(u8)((v >> 24) & 0xFF));
}
gb_internal void x64_enc_q(X64Assembler *a, u64 v) {
	x64_enc_d(a, cast(u32)( v        & 0xFFFFFFFFull));
	x64_enc_d(a, cast(u32)((v >> 32) & 0xFFFFFFFFull));
}

// Patch a previously emitted i32 at a given offset
gb_internal void x64_enc_patch_d(X64Assembler *a, isize offset, i32 v) {
	GB_ASSERT(offset + 4 <= a->code.count);
	u8 *p = a->code.data + offset;
	p[0] = cast(u8)( v        & 0xFF);
	p[1] = cast(u8)((v >>  8) & 0xFF);
	p[2] = cast(u8)((v >> 16) & 0xFF);
	p[3] = cast(u8)((v >> 24) & 0xFF);
}

// SIB scale → 2-bit encoding
gb_internal u8 x64_enc_scale_bits(u8 scale) {
	switch (scale) {
	case 1: return 0;
	case 2: return 1;
	case 4: return 2;
	case 8: return 3;
	}
	GB_PANIC("invalid SIB scale");
	return 0;
}

// REX prefix
//   W   = 64-bit operand override
//   reg = register in ModRM.reg / opcode reg  (for REX.R / REX.B resp.)
//   rm  = register in ModRM.r/m or SIB base   (for REX.B)
//   idx = SIB index register                  (for REX.X)
//
// Only emits the byte when at least one flag is required.  For 8-bit ops,
// emits REX 0x40 when reg or rm is RSP/RBP/RSI/RDI so the CPU uses
// SPL/BPL/SIL/DIL rather than AH/CH/DH/BH.
gb_internal void x64_enc_rex(X64Assembler *a, X64OpSize sz,
                              X64Reg reg, X64Reg rm, X64Reg idx) {
	bool W = (sz == X64OpSize_64);
	bool R = (reg != X64Reg_NONE) && (reg >= 8);
	bool X = (idx != X64Reg_NONE) && (idx >= 8);
	bool B = (rm  != X64Reg_NONE) && (rm  >= 8);

	bool need_8bit = false;
	if (sz == X64OpSize_8) {
		if (reg != X64Reg_NONE && (reg & 0xF) >= 4 && (reg & 0xF) <= 7) need_8bit = true;
		if (rm  != X64Reg_NONE && (rm  & 0xF) >= 4 && (rm  & 0xF) <= 7) need_8bit = true;
	}

	u8 rex = cast(u8)(0x40u | (W ? 0x08u : 0u) | (R ? 0x04u : 0u)
	                        | (X ? 0x02u : 0u) | (B ? 0x01u : 0u));
	if (rex != 0x40u || need_8bit) {
		x64_enc_b(a, rex);
	}
}

// ModRM byte + optional SIB + displacement for a memory operand.
// reg_field: the low 3 bits placed in ModRM.reg (already masked by caller).
gb_internal void x64_enc_modrm_mem(X64Assembler *a, u8 reg_field, X64Mem mem) {
	if (mem.rip_rel) {
		// [RIP + disp32]: mod=00, r/m=101
		x64_enc_b(a, cast(u8)(0x00u | (reg_field << 3) | 5u));
		x64_enc_d(a, cast(u32)cast(i32)mem.disp);
		return;
	}

	if (mem.base == X64Reg_NONE) {
		// No base: use SIB with base=101 (no-base in mod=00).
		// mod=00, r/m=100 (SIB follows)
		x64_enc_b(a, cast(u8)(0x00u | (reg_field << 3) | 4u));
		u8 scale_bits = x64_enc_scale_bits(mem.scale);
		u8 index_lo   = (mem.index == X64Reg_NONE) ? 4u : cast(u8)(mem.index & 7u);
		x64_enc_b(a, cast(u8)((scale_bits << 6) | (index_lo << 3) | 5u));
		x64_enc_d(a, cast(u32)cast(i32)mem.disp);
		return;
	}

	u8 base_lo  = cast(u8)(mem.base & 7u);
	bool need_sib = (base_lo == 4u) || (mem.index != X64Reg_NONE); // base=RSP/R12, or SIB index

	// Determine mod.  Base RBP/R13 (base_lo==5) with disp==0 would collide with
	// RIP-relative (mod=00, r/m=101); force mod=01 with disp8=0 instead.
	u8 mod;
	if (mem.disp == 0 && base_lo != 5u) {
		mod = 0u;
	} else if (mem.disp >= -128 && mem.disp <= 127) {
		mod = 1u;
	} else {
		mod = 2u;
	}

	if (need_sib) {
		x64_enc_b(a, cast(u8)((mod << 6) | (reg_field << 3) | 4u));
		u8 scale_bits = x64_enc_scale_bits(mem.scale);
		u8 index_lo   = (mem.index == X64Reg_NONE) ? 4u : cast(u8)(mem.index & 7u);
		x64_enc_b(a, cast(u8)((scale_bits << 6) | (index_lo << 3) | base_lo));
	} else {
		x64_enc_b(a, cast(u8)((mod << 6) | (reg_field << 3) | base_lo));
	}

	if      (mod == 1u) x64_enc_b(a, cast(u8)cast(i8)mem.disp);
	else if (mod == 2u) x64_enc_d(a, cast(u32)cast(i32)mem.disp);
}

// Emit the standard 2-operand instruction layout for reg-reg:
//   [66] [REX] op8/op ModRM(11, reg, rm)
gb_internal void x64_enc_op_rr(X64Assembler *a, X64OpSize sz,
                                u8 op8, u8 op,
                                X64Reg reg, X64Reg rm) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, reg, rm, X64Reg_NONE);
	x64_enc_b(a, (sz == X64OpSize_8) ? op8 : op);
	x64_enc_b(a, cast(u8)(0xC0u | ((reg & 7u) << 3) | (rm & 7u)));
}

// Emit layout for reg-mem (reg is in ModRM.reg, memory is r/m):
//   [66] [REX] op8/op ModRM [SIB] [disp]
gb_internal void x64_enc_op_rm(X64Assembler *a, X64OpSize sz,
                                u8 op8, u8 op,
                                X64Reg reg, X64Mem mem) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	X64Reg base = mem.rip_rel ? X64Reg_NONE : mem.base;
	x64_enc_rex(a, sz, reg, base, mem.index);
	x64_enc_b(a, (sz == X64OpSize_8) ? op8 : op);
	x64_enc_modrm_mem(a, cast(u8)(reg & 7u), mem);
}

// Emit layout where r/m is dst (memory or register destination) and a
// fixed /digit extends the opcode:
//   [66] [REX] op8/op ModRM(mod, /digit, rm)
gb_internal void x64_enc_op_digit_r(X64Assembler *a, X64OpSize sz,
                                     u8 op8, u8 op, u8 digit,
                                     X64Reg rm) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, X64Reg_NONE, rm, X64Reg_NONE);
	x64_enc_b(a, (sz == X64OpSize_8) ? op8 : op);
	x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (rm & 7u)));
}

gb_internal void x64_enc_op_digit_m(X64Assembler *a, X64OpSize sz,
                                     u8 op8, u8 op, u8 digit,
                                     X64Mem mem) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	X64Reg base = mem.rip_rel ? X64Reg_NONE : mem.base;
	x64_enc_rex(a, sz, X64Reg_NONE, base, mem.index);
	x64_enc_b(a, (sz == X64OpSize_8) ? op8 : op);
	x64_enc_modrm_mem(a, digit, mem);
}

// Emit imm8/imm32 for ALU immediate forms, choosing the compact imm8 form
// when the value fits.  Returns false when the opcode has no imm8 variant
// (caller must use full imm32 in that case).
// digit = ModRM.reg extension (/0 .. /7)
gb_internal void x64_enc_alu_imm_r(X64Assembler *a, X64OpSize sz,
                                    u8 digit, X64Reg dst, i32 imm) {
	bool fits8 = (sz != X64OpSize_8) && (imm >= -128 && imm <= 127);
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, X64Reg_NONE, dst, X64Reg_NONE);

	if (sz == X64OpSize_8) {
		x64_enc_b(a, 0x80u);
		x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (dst & 7u)));
		x64_enc_b(a, cast(u8)(cast(i8)imm));
	} else if (fits8) {
		x64_enc_b(a, 0x83u);
		x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (dst & 7u)));
		x64_enc_b(a, cast(u8)(cast(i8)imm));
	} else {
		// 81 /digit imm32 (general form; skip the RAX-only 05+digit*8 short form).
		x64_enc_b(a, 0x81u);
		x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (dst & 7u)));
		if (sz == X64OpSize_16) {
			x64_enc_w(a, cast(u16)(cast(i16)imm));
		} else {
			x64_enc_d(a, cast(u32)imm);
		}
	}
}

gb_internal void x64_enc_alu_imm_m(X64Assembler *a, X64OpSize sz,
                                    u8 digit, X64Mem dst, i32 imm) {
	bool fits8 = (sz != X64OpSize_8) && (imm >= -128 && imm <= 127);
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	x64_enc_rex(a, sz, X64Reg_NONE, base, dst.index);

	if (sz == X64OpSize_8) {
		x64_enc_b(a, 0x80u);
		x64_enc_modrm_mem(a, digit, dst);
		x64_enc_b(a, cast(u8)(cast(i8)imm));
	} else if (fits8) {
		x64_enc_b(a, 0x83u);
		x64_enc_modrm_mem(a, digit, dst);
		x64_enc_b(a, cast(u8)(cast(i8)imm));
	} else {
		x64_enc_b(a, 0x81u);
		x64_enc_modrm_mem(a, digit, dst);
		if (sz == X64OpSize_16) x64_enc_w(a, cast(u16)(cast(i16)imm));
		else                     x64_enc_d(a, cast(u32)imm);
	}
}

// Helper: emit a REL32 forward branch and record a fixup.
gb_internal void x64_enc_emit_branch_fixup(X64Assembler *a, isize label) {
	if (x64_label_bound(a, label)) {
		// Already bound: compute displacement directly
		i32 rel = cast(i32)(a->labels[label] - (a->code.count + 4));
		x64_enc_d(a, cast(u32)rel);
	} else {
		// Forward branch: emit placeholder, register fixup
		isize patch_at = a->code.count;
		x64_enc_d(a, 0u);
		X64Fixup fx = {};
		fx.patch_at = patch_at;
		fx.label    = label;
		array_add(&a->fixups, fx);
	}
}

// Helper: record a symbol relocation at the current code offset and emit a
// zero placeholder dword.
gb_internal void x64_enc_reloc_sym(X64Assembler *a, X64RelocType type, String sym_name) {
	X64RelocEntry r = {};
	r.code_offset = a->code.count;
	r.type        = type;
	r.sym_name    = sym_name;
	array_add(&a->relocs, r);
	x64_enc_d(a, 0u); // placeholder; linker fills in
}

// SSE2 instruction helper: [prefix] [REX] 0F op ModRM [SIB] [disp]
// xmm registers use the same 4-bit encoding as GP registers.
gb_internal void x64_enc_sse_rr(X64Assembler *a, u8 prefix, u8 op,
                                 X64XmmReg dst, X64XmmReg src) {
	if (prefix) x64_enc_b(a, prefix); // packed ops (movups/movaps) have no mandatory prefix
	bool R = (dst >= 8);
	bool B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, op);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_enc_sse_rm(X64Assembler *a, u8 prefix, u8 op,
                                 X64XmmReg dst, X64Mem src) {
	if (prefix) x64_enc_b(a, prefix); // packed ops (movups/movaps) have no mandatory prefix
	X64Reg base = src.rip_rel ? X64Reg_NONE : src.base;
	bool R = (dst >= 8);
	bool X = (src.index != X64Reg_NONE) && (src.index >= 8);
	bool B = (base      != X64Reg_NONE) && (base      >= 8);
	if (R || X || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u)
	                                              | (X ? 0x02u : 0u)
	                                              | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, op);
	x64_enc_modrm_mem(a, cast(u8)(dst & 7u), src);
}

gb_internal void x64_enc_sse_mr(X64Assembler *a, u8 prefix, u8 op,
                                 X64Mem dst, X64XmmReg src) {
	if (prefix) x64_enc_b(a, prefix); // packed ops (movups/movaps) have no mandatory prefix
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	bool R = (src >= 8);
	bool X = (dst.index != X64Reg_NONE) && (dst.index >= 8);
	bool B = (base      != X64Reg_NONE) && (base      >= 8);
	if (R || X || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u)
	                                              | (X ? 0x02u : 0u)
	                                              | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, op);
	x64_enc_modrm_mem(a, cast(u8)(src & 7u), dst);
}

// 3-byte opcode SSE (66 0F 38 op /r) xmm,xmm — SSE4.1/4.2 packed ops (pmulld, pcmpeqq, pminsd, …).
gb_internal void x64_enc_sse38_rr(X64Assembler *a, u8 prefix, u8 op,
                                  X64XmmReg dst, X64XmmReg src) {
	if (prefix) x64_enc_b(a, prefix);
	bool R = (dst >= 8);
	bool B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x38u); x64_enc_b(a, op);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

// ---- AVX/AVX2 VEX-prefix encoders ----------------------------------------------------------------
// VEX fields are stored INVERTED (1 = "not extended"). We always emit the 3-byte (C4) form — it can
// encode every map/W combination, so we skip the 2-byte (C5) shortcut (a pure size optimization, TODO).
//   pp: 0=none 1=66 2=F3 3=F2     mm (map): 1=0F 2=0F38 3=0F3A     L: false=128(xmm) true=256(ymm)
//   reg/rm = ModRM operands; vvvv = the (non-destructive) 2nd source register — pass X64XmmReg_XMM0 for
//   ops with no 2nd source (its inverted encoding 1111 is exactly the spec's "unused" value).
gb_internal void x64_enc_vex3(X64Assembler *a, bool R, bool X, bool B, u8 mm, bool W, u8 vvvv, bool L, u8 pp) {
	x64_enc_b(a, 0xC4u);
	x64_enc_b(a, cast(u8)(((R?0u:1u)<<7) | ((X?0u:1u)<<6) | ((B?0u:1u)<<5) | (mm & 0x1Fu)));
	x64_enc_b(a, cast(u8)(((W?1u:0u)<<7) | (((~cast(u32)vvvv) & 0xFu)<<3) | ((L?1u:0u)<<2) | (pp & 3u)));
}

gb_internal void x64_emit_vex_rr(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L,
                                 X64XmmReg reg, X64XmmReg vvvv, X64XmmReg rm) {
	x64_enc_vex3(a, reg>=8, false, rm>=8, mm, W, cast(u8)vvvv, L, pp);
	x64_enc_b(a, op);
	x64_enc_b(a, cast(u8)(0xC0u | ((reg & 7u) << 3) | (rm & 7u)));
}
gb_internal void x64_emit_vex_rm(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L,
                                 X64XmmReg reg, X64XmmReg vvvv, X64Mem rm) {
	X64Reg base = rm.rip_rel ? X64Reg_NONE : rm.base;
	bool Xb = (rm.index != X64Reg_NONE) && (rm.index >= 8);
	bool Bb = (base != X64Reg_NONE) && (base >= 8);
	x64_enc_vex3(a, reg>=8, Xb, Bb, mm, W, cast(u8)vvvv, L, pp);
	x64_enc_b(a, op);
	x64_enc_modrm_mem(a, cast(u8)(reg & 7u), rm);
}
gb_internal void x64_emit_vex_mr(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L,
                                 X64Mem rm, X64XmmReg vvvv, X64XmmReg reg) {
	X64Reg base = rm.rip_rel ? X64Reg_NONE : rm.base;
	bool Xb = (rm.index != X64Reg_NONE) && (rm.index >= 8);
	bool Bb = (base != X64Reg_NONE) && (base >= 8);
	x64_enc_vex3(a, reg>=8, Xb, Bb, mm, W, cast(u8)vvvv, L, pp);
	x64_enc_b(a, op);
	x64_enc_modrm_mem(a, cast(u8)(reg & 7u), rm);
}
// imm8-suffixed reg-reg form (vroundps/pd, vpermq/pd, vshufps, vpblendd, vblendv* — the latter encode
// a register selector in imm8[7:4]).
gb_internal void x64_emit_vex_rr_i(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L,
                                   X64XmmReg reg, X64XmmReg vvvv, X64XmmReg rm, u8 imm) {
	x64_emit_vex_rr(a, pp, mm, op, W, L, reg, vvvv, rm);
	x64_enc_b(a, imm);
}

// VSIB gather form: ModRM.reg = dst, SIB index = a VECTOR register (xmm/ymm), base = GP. Used by
// vpgatherdd/vgatherdps and friends. mod/disp follow the standard rules; index high bit → VEX.X.
gb_internal void x64_emit_vex_gather(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L,
                                     X64XmmReg dst, X64XmmReg mask, X64Reg base, X64XmmReg vindex, u8 scale) {
	x64_enc_vex3(a, dst>=8, vindex>=8, base>=8, mm, W, cast(u8)mask, L, pp);
	x64_enc_b(a, op);
	// mod=00 (no disp) with SIB; base RBP/R13 (lo==5) needs mod=01+disp8=0, like x64_enc_modrm_mem.
	u8 base_lo = cast(u8)(base & 7u);
	u8 mod = (base_lo == 5u) ? 1u : 0u;
	x64_enc_b(a, cast(u8)((mod << 6) | ((dst & 7u) << 3) | 4u)); // r/m=100 → SIB
	u8 scale_bits = scale==8?3u : scale==4?2u : scale==2?1u : 0u;
	x64_enc_b(a, cast(u8)((scale_bits << 6) | ((vindex & 7u) << 3) | base_lo));
	if (mod == 1u) x64_enc_b(a, 0u);
}

// VZEROUPPER (C5 F8 77): zero bits 255:128 of all YMM regs, preserving the low 128 (XMM). Emitted on
// exit from AVX2 SIMD codegen so the boundary with the rest of the backend's legacy-SSE instructions
// (scalar float math, movups struct copies) doesn't hit the SSE↔AVX transition stall.
gb_internal void x64_emit_vzeroupper(X64Assembler *a) {
	x64_enc_b(a, 0xC5u); x64_enc_b(a, 0xF8u); x64_enc_b(a, 0x77u);
}

// SSE2 int<->float conversion (has W bit from GP register width)
gb_internal void x64_enc_sse_cvt_xmm_r(X64Assembler *a, u8 prefix, u8 op,
                                         X64XmmReg dst, X64Reg src, X64OpSize src_sz) {
	x64_enc_b(a, prefix);
	bool W = (src_sz == X64OpSize_64);
	bool R = (dst >= 8);
	bool B = (src != X64Reg_NONE) && (src >= 8);
	x64_enc_b(a, cast(u8)(0x40u | (W ? 0x08u : 0u) | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, op);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_enc_sse_cvt_r_xmm(X64Assembler *a, u8 prefix, u8 op,
                                         X64Reg dst, X64OpSize dst_sz, X64XmmReg src) {
	x64_enc_b(a, prefix);
	bool W = (dst_sz == X64OpSize_64);
	bool R = (dst != X64Reg_NONE) && (dst >= 8);
	bool B = (src >= 8);
	x64_enc_b(a, cast(u8)(0x40u | (W ? 0x08u : 0u) | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, op);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

// ===========================================================================
// Alignment / misc
// ===========================================================================

gb_internal void x64_emit_align(X64Assembler *a, isize align, u8 fill) {
	isize cur = a->code.count;
	isize rem = cur % align;
	if (rem == 0) return;
	isize pad = align - rem;
	for (isize i = 0; i < pad; i++) x64_enc_b(a, fill);
}

gb_internal void x64_emit_nop(X64Assembler *a) {
	x64_enc_b(a, 0x90u);
}

gb_internal void x64_emit_int3(X64Assembler *a) {
	x64_enc_b(a, 0xCCu);
}

// ===========================================================================
// Stack
// ===========================================================================

gb_internal void x64_emit_push_r(X64Assembler *a, X64Reg r) {
	if (r >= 8) x64_enc_b(a, 0x41u); // REX.B
	x64_enc_b(a, cast(u8)(0x50u + (r & 7u)));
}

gb_internal void x64_emit_pop_r(X64Assembler *a, X64Reg r) {
	if (r >= 8) x64_enc_b(a, 0x41u);
	x64_enc_b(a, cast(u8)(0x58u + (r & 7u)));
}

gb_internal void x64_emit_push_i8(X64Assembler *a, i8 imm) {
	x64_enc_b(a, 0x6Au);
	x64_enc_b(a, cast(u8)imm);
}

gb_internal void x64_emit_push_i32(X64Assembler *a, i32 imm) {
	x64_enc_b(a, 0x68u);
	x64_enc_d(a, cast(u32)imm);
}

// ===========================================================================
// Control flow
// ===========================================================================

gb_internal void x64_emit_ret(X64Assembler *a) {
	x64_enc_b(a, 0xC3u);
}

gb_internal void x64_emit_ret_n(X64Assembler *a, u16 pop_bytes) {
	x64_enc_b(a, 0xC2u);
	x64_enc_w(a, pop_bytes);
}

gb_internal void x64_emit_jmp(X64Assembler *a, isize label) {
	x64_enc_b(a, 0xE9u); // JMP rel32
	x64_enc_emit_branch_fixup(a, label);
}

gb_internal void x64_emit_jcc(X64Assembler *a, X64Cc cc, isize label) {
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, cast(u8)(0x80u + cast(u8)cc)); // Jcc rel32
	x64_enc_emit_branch_fixup(a, label);
}

gb_internal void x64_emit_jmp_r(X64Assembler *a, X64Reg r) {
	if (r >= 8) x64_enc_b(a, 0x41u);
	x64_enc_b(a, 0xFFu);
	x64_enc_b(a, cast(u8)(0xE0u + (r & 7u))); // ModRM: mod=11, /4, r/m=r
}

gb_internal void x64_emit_jmp_m(X64Assembler *a, X64Mem m) {
	x64_enc_op_digit_m(a, X64OpSize_64, 0xFFu, 0xFFu, 4u, m);
}

gb_internal void x64_emit_call_r(X64Assembler *a, X64Reg r) {
	if (r >= 8) x64_enc_b(a, 0x41u);
	x64_enc_b(a, 0xFFu);
	x64_enc_b(a, cast(u8)(0xD0u + (r & 7u))); // ModRM: mod=11, /2, r/m=r
}

gb_internal void x64_emit_call_m(X64Assembler *a, X64Mem m) {
	x64_enc_op_digit_m(a, X64OpSize_64, 0xFFu, 0xFFu, 2u, m);
}

gb_internal void x64_emit_call_sym(X64Assembler *a, String sym_name) {
	x64_enc_b(a, 0xE8u); // CALL rel32
	x64_enc_reloc_sym(a, X64Reloc_REL32, sym_name);
}

gb_internal void x64_emit_jmp_sym(X64Assembler *a, String sym_name) {
	x64_enc_b(a, 0xE9u); // JMP rel32
	x64_enc_reloc_sym(a, X64Reloc_REL32, sym_name);
}

// ===========================================================================
// Data movement
// ===========================================================================

gb_internal void x64_emit_mov_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	// MOV r/m, r  (89 /r for >=16-bit; 88 /r for 8-bit)
	// dst goes in r/m, src goes in reg field
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, src, dst, X64Reg_NONE);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0x88u : 0x89u);
	x64_enc_b(a, cast(u8)(0xC0u | ((src & 7u) << 3) | (dst & 7u)));
}

gb_internal void x64_emit_mov_rm(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Mem src) {
	// MOV r, r/m  (8B /r for >=16-bit; 8A /r for 8-bit)
	x64_enc_op_rm(a, sz, 0x8Au, 0x8Bu, dst, src);
}

gb_internal void x64_emit_mov_mr(X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src) {
	// MOV r/m, r  (89 /r; 88 /r for 8-bit)
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	x64_enc_rex(a, sz, src, base, dst.index);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0x88u : 0x89u);
	x64_enc_modrm_mem(a, cast(u8)(src & 7u), dst);
}

gb_internal void x64_emit_mov_ri(X64Assembler *a, X64OpSize sz, X64Reg dst, i64 imm) {
	switch (sz) {
	case X64OpSize_8:
		if (dst >= 8 || (dst >= 4 && dst <= 7)) {
			x64_enc_b(a, cast(u8)(0x40u | ((dst >= 8) ? 0x01u : 0u))); // REX.B or REX
		}
		x64_enc_b(a, cast(u8)(0xB0u + (dst & 7u)));
		x64_enc_b(a, cast(u8)(imm & 0xFF));
		break;
	case X64OpSize_16:
		x64_enc_b(a, 0x66u);
		if (dst >= 8) x64_enc_b(a, 0x41u); // REX.B
		x64_enc_b(a, cast(u8)(0xB8u + (dst & 7u)));
		x64_enc_w(a, cast(u16)(imm & 0xFFFF));
		break;
	case X64OpSize_32:
		// MOV r32, imm32 — zero-extends to 64-bit
		if (dst >= 8) x64_enc_b(a, 0x41u); // REX.B
		x64_enc_b(a, cast(u8)(0xB8u + (dst & 7u)));
		x64_enc_d(a, cast(u32)(imm & 0xFFFFFFFF));
		break;
	case X64OpSize_64:
		if (imm >= 0 && imm <= 0xFFFFFFFFLL) {
			// Use 32-bit zero-extending MOV (smaller encoding)
			if (dst >= 8) x64_enc_b(a, 0x41u);
			x64_enc_b(a, cast(u8)(0xB8u + (dst & 7u)));
			x64_enc_d(a, cast(u32)imm);
		} else if (imm >= -2147483648LL && imm <= 2147483647LL) {
			// Sign-extended 32-bit: REX.W C7 /0 imm32
			x64_enc_rex(a, X64OpSize_64, X64Reg_NONE, dst, X64Reg_NONE);
			x64_enc_b(a, 0xC7u);
			x64_enc_b(a, cast(u8)(0xC0u | (dst & 7u)));
			x64_enc_d(a, cast(u32)(i32)imm);
		} else {
			// Full 64-bit immediate: REX.W B8+rd imm64
			x64_enc_b(a, cast(u8)(0x48u | ((dst >= 8) ? 0x01u : 0u)));
			x64_enc_b(a, cast(u8)(0xB8u + (dst & 7u)));
			x64_enc_q(a, cast(u64)imm);
		}
		break;
	}
}

gb_internal void x64_emit_mov_mi(X64Assembler *a, X64OpSize sz, X64Mem dst, i32 imm) {
	// MOV r/m, imm  (C7 /0 for >=16-bit; C6 /0 for 8-bit)
	x64_enc_op_digit_m(a, sz, 0xC6u, 0xC7u, 0u, dst);
	if (sz == X64OpSize_8)       x64_enc_b(a, cast(u8)(i8)imm);
	else if (sz == X64OpSize_16) x64_enc_w(a, cast(u16)(i16)imm);
	else                         x64_enc_d(a, cast(u32)imm);
}

gb_internal void x64_emit_movzx_rr(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Reg src) {
	// MOVZX r64, r8/16  (0F B6 / 0F B7); always emitted 64-bit for simplicity.
	x64_enc_rex(a, X64OpSize_64, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, (src_sz == X64OpSize_8) ? 0xB6u : 0xB7u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_movzx_rm(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Mem src) {
	X64Reg base = src.rip_rel ? X64Reg_NONE : src.base;
	x64_enc_rex(a, X64OpSize_64, dst, base, src.index);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, (src_sz == X64OpSize_8) ? 0xB6u : 0xB7u);
	x64_enc_modrm_mem(a, cast(u8)(dst & 7u), src);
}

gb_internal void x64_emit_movsx_rr(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Reg src) {
	if (src_sz == X64OpSize_32) {
		// MOVSXD r64, r/m32  (REX.W 63 /r)
		x64_enc_rex(a, X64OpSize_64, dst, src, X64Reg_NONE);
		x64_enc_b(a, 0x63u);
	} else {
		// MOVSX r64, r8/r16  (REX.W 0F BE /r or 0F BF /r)
		x64_enc_rex(a, X64OpSize_64, dst, src, X64Reg_NONE);
		x64_enc_b(a, 0x0Fu);
		x64_enc_b(a, (src_sz == X64OpSize_8) ? 0xBEu : 0xBFu);
	}
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_movsx_rm(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Mem src) {
	X64Reg base = src.rip_rel ? X64Reg_NONE : src.base;
	if (src_sz == X64OpSize_32) {
		x64_enc_rex(a, X64OpSize_64, dst, base, src.index);
		x64_enc_b(a, 0x63u);
	} else {
		x64_enc_rex(a, X64OpSize_64, dst, base, src.index);
		x64_enc_b(a, 0x0Fu);
		x64_enc_b(a, (src_sz == X64OpSize_8) ? 0xBEu : 0xBFu);
	}
	x64_enc_modrm_mem(a, cast(u8)(dst & 7u), src);
}

gb_internal void x64_emit_lea(X64Assembler *a, X64Reg dst, X64Mem src) {
	// LEA r64, m  (REX.W 8D /r)
	X64Reg base = src.rip_rel ? X64Reg_NONE : src.base;
	x64_enc_rex(a, X64OpSize_64, dst, base, src.index);
	x64_enc_b(a, 0x8Du);
	x64_enc_modrm_mem(a, cast(u8)(dst & 7u), src);
}

gb_internal void x64_emit_lea_sym(X64Assembler *a, X64Reg dst, String sym_name) {
	// LEA r64, [RIP + sym]  — REX.W 8D /r, then REL32 relocation
	x64_enc_rex(a, X64OpSize_64, dst, X64Reg_NONE, X64Reg_NONE);
	x64_enc_b(a, 0x8Du);
	// ModRM: mod=00, reg=dst, r/m=101 (RIP-relative)
	x64_enc_b(a, cast(u8)(0x00u | ((dst & 7u) << 3) | 5u));
	x64_enc_reloc_sym(a, X64Reloc_REL32, sym_name);
}

// Windows x64 thread-local address (the standard MSVC sequence). Computes the address of
// `sym` (a variable in the `.tls$` section) for the CURRENT thread into RAX, using R11 as
// scratch:
//     mov  eax, dword [rip + _tls_index]   ; module's TLS slot index   (REL32 -> _tls_index)
//     mov  r11, qword gs:[0x58]            ; TEB->ThreadLocalStoragePointer
//     mov  rax, qword [r11 + rax*8]        ; base of this module's TLS block for this thread
//     lea  rax, [rax + sym]                ; &sym                       (SECREL -> sym)
// The SECREL relocation resolves to sym's offset within the merged `.tls` section, which is
// exactly its offset within each thread's copied TLS block.
gb_internal void x64_emit_tls_addr(X64Assembler *a, String sym_name) {
	// mov eax, dword [rip + _tls_index]   (8B /r, ModRM mod=00 reg=EAX rm=101 RIP-rel)
	x64_enc_b(a, 0x8Bu); x64_enc_b(a, 0x05u);
	x64_enc_reloc_sym(a, X64Reloc_REL32, str_lit("_tls_index"));
	// mov r11, qword gs:[0x58]   (65 GS, 4C REX.WR, 8B, ModRM=1C SIB=25 disp32=0x58)
	x64_enc_b(a, 0x65u); x64_enc_b(a, 0x4Cu); x64_enc_b(a, 0x8Bu);
	x64_enc_b(a, 0x1Cu); x64_enc_b(a, 0x25u); x64_enc_d(a, 0x58u);
	// mov rax, qword [r11 + rax*8]   (49 REX.WB, 8B, ModRM=04 SIB=C3: scale=8 index=RAX base=R11)
	x64_enc_b(a, 0x49u); x64_enc_b(a, 0x8Bu); x64_enc_b(a, 0x04u); x64_enc_b(a, 0xC3u);
	// lea rax, [rax + sym]   (48 REX.W, 8D, ModRM=80: mod=10 reg=RAX rm=RAX, disp32 = SECREL)
	x64_enc_b(a, 0x48u); x64_enc_b(a, 0x8Du); x64_enc_b(a, 0x80u);
	x64_enc_reloc_sym(a, X64Reloc_SECREL, sym_name);
}

gb_internal void x64_emit_xchg_rr(X64Assembler *a, X64OpSize sz, X64Reg a_, X64Reg b) {
	// Special short form for XCHG rAX, r  (90+r) when one operand is RAX
	if (sz == X64OpSize_64 && (a_ == X64Reg_RAX || b == X64Reg_RAX)) {
		X64Reg other = (a_ == X64Reg_RAX) ? b : a_;
		x64_enc_rex(a, X64OpSize_64, X64Reg_NONE, other, X64Reg_NONE);
		x64_enc_b(a, cast(u8)(0x90u + (other & 7u)));
		return;
	}
	x64_enc_op_rr(a, sz, 0x86u, 0x87u, a_, b);
}

gb_internal void x64_emit_cmov_rr(X64Assembler *a, X64Cc cc, X64OpSize sz, X64Reg dst, X64Reg src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, cast(u8)(0x40u + cast(u8)cc)); // CMOVcc
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_cmov_rm(X64Assembler *a, X64Cc cc, X64OpSize sz, X64Reg dst, X64Mem src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	X64Reg base = src.rip_rel ? X64Reg_NONE : src.base;
	x64_enc_rex(a, sz, dst, base, src.index);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, cast(u8)(0x40u + cast(u8)cc));
	x64_enc_modrm_mem(a, cast(u8)(dst & 7u), src);
}

// ===========================================================================
// Integer ALU
// ALU opcode table (base opcode = multiple of 8 starting at 0x00):
//   ADD=0x00  OR=0x08  ADC=0x10  SBB=0x18  AND=0x20  SUB=0x28  XOR=0x30  CMP=0x38
// For each:
//   base+0 = r/m8, r8       (mr8)
//   base+1 = r/m, r         (mr)
//   base+2 = r8, r/m8       (rm8)
//   base+3 = r, r/m         (rm)
// Immediate digit (/0..7) matches the base >> 3.
// ===========================================================================

#define X64_ALU_IMPL(name, base_op, imm_digit) \
gb_internal void x64_emit_##name##_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) { \
	/* r/m, r form: dst is r/m, src goes in reg field */ \
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u); \
	x64_enc_rex(a, sz, src, dst, X64Reg_NONE); \
	x64_enc_b(a, (sz == X64OpSize_8) ? cast(u8)(base_op) : cast(u8)(base_op + 1)); \
	x64_enc_b(a, cast(u8)(0xC0u | ((src & 7u) << 3) | (dst & 7u))); \
} \
gb_internal void x64_emit_##name##_rm(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Mem src) { \
	/* r, r/m form: dst in reg field, src is memory */ \
	x64_enc_op_rm(a, sz, cast(u8)(base_op + 2), cast(u8)(base_op + 3), dst, src); \
} \
gb_internal void x64_emit_##name##_mr(X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src) { \
	/* r/m, r form: dst is memory */ \
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u); \
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base; \
	x64_enc_rex(a, sz, src, base, dst.index); \
	x64_enc_b(a, (sz == X64OpSize_8) ? cast(u8)(base_op) : cast(u8)(base_op + 1)); \
	x64_enc_modrm_mem(a, cast(u8)(src & 7u), dst); \
} \
gb_internal void x64_emit_##name##_ri(X64Assembler *a, X64OpSize sz, X64Reg dst, i32 imm) { \
	x64_enc_alu_imm_r(a, sz, (imm_digit), dst, imm); \
} \
gb_internal void x64_emit_##name##_mi(X64Assembler *a, X64OpSize sz, X64Mem dst, i32 imm) { \
	x64_enc_alu_imm_m(a, sz, (imm_digit), dst, imm); \
}

X64_ALU_IMPL(add, 0x00, 0)
X64_ALU_IMPL(or,  0x08, 1)
X64_ALU_IMPL(adc, 0x10, 2)
X64_ALU_IMPL(sbb, 0x18, 3)
X64_ALU_IMPL(and, 0x20, 4)
X64_ALU_IMPL(sub, 0x28, 5)
X64_ALU_IMPL(xor, 0x30, 6)
X64_ALU_IMPL(cmp, 0x38, 7)

#undef X64_ALU_IMPL

// SHLD/SHRD r/m, r, CL  (0F A5 /r ; 0F AD /r). reg field = src, r/m = dst.
gb_internal void x64_emit_shld_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, src, dst, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xA5u);
	x64_enc_b(a, cast(u8)(0xC0u | ((src & 7u) << 3) | (dst & 7u)));
}
gb_internal void x64_emit_shrd_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, src, dst, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xADu);
	x64_enc_b(a, cast(u8)(0xC0u | ((src & 7u) << 3) | (dst & 7u)));
}

gb_internal void x64_emit_test_rr(X64Assembler *a, X64OpSize sz, X64Reg lhs, X64Reg rhs) {
	// TEST r/m, r  (85 /r; 84 /r for 8-bit) — lhs is r/m, rhs is reg
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, rhs, lhs, X64Reg_NONE);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0x84u : 0x85u);
	x64_enc_b(a, cast(u8)(0xC0u | ((rhs & 7u) << 3) | (lhs & 7u)));
}

gb_internal void x64_emit_test_ri(X64Assembler *a, X64OpSize sz, X64Reg lhs, i32 imm) {
	// TEST r/m, imm  (F7 /0 for >=16-bit; F6 /0 for 8-bit)
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, X64Reg_NONE, lhs, X64Reg_NONE);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0xF6u : 0xF7u);
	x64_enc_b(a, cast(u8)(0xC0u | (lhs & 7u))); // /0
	if (sz == X64OpSize_8)       x64_enc_b(a, cast(u8)(i8)imm);
	else if (sz == X64OpSize_16) x64_enc_w(a, cast(u16)(i16)imm);
	else                         x64_enc_d(a, cast(u32)imm);
}

gb_internal void x64_emit_test_mi(X64Assembler *a, X64OpSize sz, X64Mem lhs, i32 imm) {
	x64_enc_op_digit_m(a, sz, 0xF6u, 0xF7u, 0u, lhs);
	if (sz == X64OpSize_8)       x64_enc_b(a, cast(u8)(i8)imm);
	else if (sz == X64OpSize_16) x64_enc_w(a, cast(u16)(i16)imm);
	else                         x64_enc_d(a, cast(u32)imm);
}

// Unary

gb_internal void x64_emit_neg_r(X64Assembler *a, X64OpSize sz, X64Reg r) {
	x64_enc_op_digit_r(a, sz, 0xF6u, 0xF7u, 3u, r);
}
gb_internal void x64_emit_not_r(X64Assembler *a, X64OpSize sz, X64Reg r) {
	x64_enc_op_digit_r(a, sz, 0xF6u, 0xF7u, 2u, r);
}
gb_internal void x64_emit_neg_m(X64Assembler *a, X64OpSize sz, X64Mem m) {
	x64_enc_op_digit_m(a, sz, 0xF6u, 0xF7u, 3u, m);
}
gb_internal void x64_emit_not_m(X64Assembler *a, X64OpSize sz, X64Mem m) {
	x64_enc_op_digit_m(a, sz, 0xF6u, 0xF7u, 2u, m);
}
gb_internal void x64_emit_inc_r(X64Assembler *a, X64OpSize sz, X64Reg r) {
	x64_enc_op_digit_r(a, sz, 0xFEu, 0xFFu, 0u, r);
}
gb_internal void x64_emit_dec_r(X64Assembler *a, X64OpSize sz, X64Reg r) {
	x64_enc_op_digit_r(a, sz, 0xFEu, 0xFFu, 1u, r);
}

// Multiply

gb_internal void x64_emit_imul_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	// IMUL r, r/m  (0F AF /r)
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xAFu);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_imul_rm(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Mem src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	X64Reg base = src.rip_rel ? X64Reg_NONE : src.base;
	x64_enc_rex(a, sz, dst, base, src.index);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xAFu);
	x64_enc_modrm_mem(a, cast(u8)(dst & 7u), src);
}

gb_internal void x64_emit_imul_rri(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src, i32 imm) {
	bool fits8 = (imm >= -128 && imm <= 127);
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, fits8 ? 0x6Bu : 0x69u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
	if (fits8) x64_enc_b(a, cast(u8)(i8)imm);
	else       x64_enc_d(a, cast(u32)imm);
}

gb_internal void x64_emit_imul_r(X64Assembler *a, X64OpSize sz, X64Reg src) {
	x64_enc_op_digit_r(a, sz, 0xF6u, 0xF7u, 5u, src);
}

gb_internal void x64_emit_mul_r(X64Assembler *a, X64OpSize sz, X64Reg src) {
	x64_enc_op_digit_r(a, sz, 0xF6u, 0xF7u, 4u, src);
}

gb_internal void x64_emit_idiv_r(X64Assembler *a, X64OpSize sz, X64Reg src) {
	x64_enc_op_digit_r(a, sz, 0xF6u, 0xF7u, 7u, src);
}

gb_internal void x64_emit_div_r(X64Assembler *a, X64OpSize sz, X64Reg src) {
	x64_enc_op_digit_r(a, sz, 0xF6u, 0xF7u, 6u, src);
}

gb_internal void x64_emit_cqo(X64Assembler *a) {
	x64_enc_b(a, 0x48u); x64_enc_b(a, 0x99u); // REX.W CQO
}
gb_internal void x64_emit_cdq(X64Assembler *a) {
	x64_enc_b(a, 0x99u); // CDQ
}
gb_internal void x64_emit_cwde(X64Assembler *a) {
	x64_enc_b(a, 0x98u); // CWDE
}

// Shifts

gb_internal void x64_enc_shift_ri(X64Assembler *a, X64OpSize sz, u8 digit, X64Reg dst, u8 count) {
	if (count == 1) {
		// D1 /digit form (shift by 1, no imm byte)
		if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
		x64_enc_rex(a, sz, X64Reg_NONE, dst, X64Reg_NONE);
		x64_enc_b(a, (sz == X64OpSize_8) ? 0xD0u : 0xD1u);
		x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (dst & 7u)));
	} else {
		// C1 /digit imm8
		if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
		x64_enc_rex(a, sz, X64Reg_NONE, dst, X64Reg_NONE);
		x64_enc_b(a, (sz == X64OpSize_8) ? 0xC0u : 0xC1u);
		x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (dst & 7u)));
		x64_enc_b(a, count);
	}
}

gb_internal void x64_enc_shift_rcl(X64Assembler *a, X64OpSize sz, u8 digit, X64Reg dst) {
	// D3 /digit (shift by CL)
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, X64Reg_NONE, dst, X64Reg_NONE);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0xD2u : 0xD3u);
	x64_enc_b(a, cast(u8)(0xC0u | (digit << 3) | (dst & 7u)));
}

gb_internal void x64_emit_shl_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count) { x64_enc_shift_ri(a, sz, 4u, dst, count); }
gb_internal void x64_emit_shr_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count) { x64_enc_shift_ri(a, sz, 5u, dst, count); }
gb_internal void x64_emit_sar_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count) { x64_enc_shift_ri(a, sz, 7u, dst, count); }
gb_internal void x64_emit_rol_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count) { x64_enc_shift_ri(a, sz, 0u, dst, count); }
gb_internal void x64_emit_ror_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count) { x64_enc_shift_ri(a, sz, 1u, dst, count); }
gb_internal void x64_emit_shl_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst)           { x64_enc_shift_rcl(a, sz, 4u, dst); }
gb_internal void x64_emit_shr_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst)           { x64_enc_shift_rcl(a, sz, 5u, dst); }
gb_internal void x64_emit_sar_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst)           { x64_enc_shift_rcl(a, sz, 7u, dst); }

// Bit scan / count

gb_internal void x64_emit_bsf_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xBCu);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_bsr_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xBDu);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_tzcnt_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	// F3 [66] [REX] 0F BC /r
	x64_enc_b(a, 0xF3u);
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xBCu);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_lzcnt_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	// F3 [66] [REX] 0F BD /r
	x64_enc_b(a, 0xF3u);
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xBDu);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_popcnt_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src) {
	// F3 [66] [REX] 0F B8 /r
	x64_enc_b(a, 0xF3u);
	if (sz == X64OpSize_16) x64_enc_b(a, 0x66u);
	x64_enc_rex(a, sz, dst, src, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xB8u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_bswap_r(X64Assembler *a, X64OpSize sz, X64Reg r) {
	// [REX] 0F C8+r
	x64_enc_rex(a, sz, X64Reg_NONE, r, X64Reg_NONE);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, cast(u8)(0xC8u + (r & 7u)));
}

// SETcc

gb_internal void x64_emit_setcc_r(X64Assembler *a, X64Cc cc, X64Reg dst) {
	// [REX] 0F 90+cc /0  (8-bit result, no size prefix)
	if (dst >= 8 || (dst >= 4 && dst <= 7)) {
		x64_enc_b(a, cast(u8)(0x40u | ((dst >= 8) ? 0x01u : 0u)));
	}
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, cast(u8)(0x90u + cast(u8)cc));
	x64_enc_b(a, cast(u8)(0xC0u | (dst & 7u)));
}

gb_internal void x64_emit_setcc_m(X64Assembler *a, X64Cc cc, X64Mem dst) {
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	x64_enc_rex(a, X64OpSize_8, X64Reg_NONE, base, dst.index);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, cast(u8)(0x90u + cast(u8)cc));
	x64_enc_modrm_mem(a, 0u, dst);
}

// ===========================================================================
// SSE2 scalar floating-point
// ===========================================================================

gb_internal void x64_emit_movsd_rr(X64Assembler *a, X64XmmReg dst, X64XmmReg src) { x64_enc_sse_rr(a, 0xF2u, 0x10u, dst, src); }
gb_internal void x64_emit_movsd_rm(X64Assembler *a, X64XmmReg dst, X64Mem src)    { x64_enc_sse_rm(a, 0xF2u, 0x10u, dst, src); }
gb_internal void x64_emit_movsd_mr(X64Assembler *a, X64Mem dst, X64XmmReg src)    { x64_enc_sse_mr(a, 0xF2u, 0x11u, dst, src); }
gb_internal void x64_emit_movss_rr(X64Assembler *a, X64XmmReg dst, X64XmmReg src) { x64_enc_sse_rr(a, 0xF3u, 0x10u, dst, src); }
gb_internal void x64_emit_movss_rm(X64Assembler *a, X64XmmReg dst, X64Mem src)    { x64_enc_sse_rm(a, 0xF3u, 0x10u, dst, src); }
gb_internal void x64_emit_movss_mr(X64Assembler *a, X64Mem dst, X64XmmReg src)    { x64_enc_sse_mr(a, 0xF3u, 0x11u, dst, src); }

// Unaligned 128-bit packed move (load/store a whole #simd vector). No mandatory prefix: 0F 10 / 0F 11.
gb_internal void x64_emit_movups_rm(X64Assembler *a, X64XmmReg dst, X64Mem src)   { x64_enc_sse_rm(a, 0x00u, 0x10u, dst, src); }
gb_internal void x64_emit_movups_mr(X64Assembler *a, X64Mem dst, X64XmmReg src)   { x64_enc_sse_mr(a, 0x00u, 0x11u, dst, src); }

// ROUNDPS/ROUNDPD xmm1, xmm2, imm8 (SSE4.1): 66 0F 3A 08/09 /r ib. imm8 picks the rounding mode
// (0x09 floor, 0x0A ceil, 0x0B trunc, 0x0C nearbyint; all | 0x08 to suppress the precision exception).
gb_internal void x64_emit_round_packed(X64Assembler *a, u8 op, X64XmmReg dst, X64XmmReg src, u8 imm) {
	x64_enc_b(a, 0x66u);
	bool R = (dst >= 8);
	bool B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x3Au); x64_enc_b(a, op);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
	x64_enc_b(a, imm);
}
gb_internal void x64_emit_roundps(X64Assembler *a, X64XmmReg dst, X64XmmReg src, u8 imm) { x64_emit_round_packed(a, 0x08u, dst, src, imm); }
gb_internal void x64_emit_roundpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src, u8 imm) { x64_emit_round_packed(a, 0x09u, dst, src, imm); }

// --- Packed (whole-XMM) SSE ops for #simd vectors: every lane at once ---
// Float single (no prefix) / double (0x66).
gb_internal void x64_emit_addps(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x00u, 0x58u, d, s); }
gb_internal void x64_emit_subps(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x00u, 0x5Cu, d, s); }
gb_internal void x64_emit_mulps(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x00u, 0x59u, d, s); }
gb_internal void x64_emit_divps(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x00u, 0x5Eu, d, s); }
gb_internal void x64_emit_minps(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x00u, 0x5Du, d, s); }
gb_internal void x64_emit_maxps(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x00u, 0x5Fu, d, s); }
gb_internal void x64_emit_addpd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x58u, d, s); }
gb_internal void x64_emit_subpd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x5Cu, d, s); }
gb_internal void x64_emit_mulpd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x59u, d, s); }
gb_internal void x64_emit_divpd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x5Eu, d, s); }
gb_internal void x64_emit_minpd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x5Du, d, s); }
gb_internal void x64_emit_maxpd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x5Fu, d, s); }
// Integer (0x66): add/sub by lane width, bitwise (width-agnostic), eq/gt compares, 16/32-bit multiply.
gb_internal void x64_emit_paddb(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xFCu, d, s); }
gb_internal void x64_emit_paddw(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xFDu, d, s); }
gb_internal void x64_emit_paddd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xFEu, d, s); }
gb_internal void x64_emit_paddq(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xD4u, d, s); }
gb_internal void x64_emit_psubb(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xF8u, d, s); }
gb_internal void x64_emit_psubw(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xF9u, d, s); }
gb_internal void x64_emit_psubd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xFAu, d, s); }
gb_internal void x64_emit_psubq(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xFBu, d, s); }
gb_internal void x64_emit_pand (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xDBu, d, s); }
gb_internal void x64_emit_por  (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xEBu, d, s); }
gb_internal void x64_emit_pxor (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xEFu, d, s); }
gb_internal void x64_emit_pandn(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xDFu, d, s); } // (~d) & s
gb_internal void x64_emit_pcmpeqb(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x74u, d, s); }
gb_internal void x64_emit_pcmpeqw(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x75u, d, s); }
gb_internal void x64_emit_pcmpeqd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x76u, d, s); }
gb_internal void x64_emit_pcmpgtb(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x64u, d, s); }
gb_internal void x64_emit_pcmpgtw(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x65u, d, s); }
gb_internal void x64_emit_pcmpgtd(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0x66u, d, s); }
gb_internal void x64_emit_pmullw (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xD5u, d, s); }
// 3-byte (66 0F 38) SSE4.1/4.2.
gb_internal void x64_emit_pmulld (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x40u, d, s); }
gb_internal void x64_emit_pcmpeqq(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x29u, d, s); }
gb_internal void x64_emit_pcmpgtq(X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x37u, d, s); }
gb_internal void x64_emit_pminsb (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x38u, d, s); }
gb_internal void x64_emit_pminsd (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x39u, d, s); }
gb_internal void x64_emit_pmaxsb (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x3Cu, d, s); }
gb_internal void x64_emit_pmaxsd (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x3Du, d, s); }
gb_internal void x64_emit_pminuw (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x3Au, d, s); }
gb_internal void x64_emit_pminud (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x3Bu, d, s); }
gb_internal void x64_emit_pmaxuw (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x3Eu, d, s); }
gb_internal void x64_emit_pmaxud (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse38_rr(a, 0x66u, 0x3Fu, d, s); }
gb_internal void x64_emit_pminub (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xDAu, d, s); }
gb_internal void x64_emit_pmaxub (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xDEu, d, s); }
gb_internal void x64_emit_pminsw (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xEAu, d, s); }
gb_internal void x64_emit_pmaxsw (X64Assembler *a, X64XmmReg d, X64XmmReg s) { x64_enc_sse_rr(a, 0x66u, 0xEEu, d, s); }

// CMPPS/CMPPD xmm1, xmm2, imm8 (packed compare → per-lane all-ones/all-zeros mask): 0F C2 /r ib.
gb_internal void x64_emit_cmpps(X64Assembler *a, X64XmmReg d, X64XmmReg s, u8 imm) {
	bool R = (d >= 8), B = (s >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xC2u);
	x64_enc_b(a, cast(u8)(0xC0u | ((d & 7u) << 3) | (s & 7u)));
	x64_enc_b(a, imm);
}
gb_internal void x64_emit_cmppd(X64Assembler *a, X64XmmReg d, X64XmmReg s, u8 imm) {
	x64_enc_b(a, 0x66u);
	bool R = (d >= 8), B = (s >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xC2u);
	x64_enc_b(a, cast(u8)(0xC0u | ((d & 7u) << 3) | (s & 7u)));
	x64_enc_b(a, imm);
}
// PSRLDQ xmm, imm8 — shift the WHOLE 128-bit register right by imm bytes (zero-fill). 66 0F 73 /3 ib.
// Used to fold lanes in a horizontal reduction.
gb_internal void x64_emit_psrldq(X64Assembler *a, X64XmmReg d, u8 imm) {
	x64_enc_b(a, 0x66u);
	if (d >= 8) x64_enc_b(a, 0x41u); // REX.B (r/m is the xmm)
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x73u);
	x64_enc_b(a, cast(u8)(0xC0u | (3u << 3) | (d & 7u))); // /3
	x64_enc_b(a, imm);
}

gb_internal void x64_emit_addsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF2u, 0x58u, dst, src); }
gb_internal void x64_emit_subsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF2u, 0x5Cu, dst, src); }
gb_internal void x64_emit_mulsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF2u, 0x59u, dst, src); }
gb_internal void x64_emit_divsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF2u, 0x5Eu, dst, src); }
gb_internal void x64_emit_sqrtsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src) { x64_enc_sse_rr(a, 0xF2u, 0x51u, dst, src); }

gb_internal void x64_emit_addss(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF3u, 0x58u, dst, src); }
gb_internal void x64_emit_subss(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF3u, 0x5Cu, dst, src); }
gb_internal void x64_emit_mulss(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF3u, 0x59u, dst, src); }
gb_internal void x64_emit_divss(X64Assembler *a, X64XmmReg dst, X64XmmReg src)  { x64_enc_sse_rr(a, 0xF3u, 0x5Eu, dst, src); }
gb_internal void x64_emit_sqrtss(X64Assembler *a, X64XmmReg dst, X64XmmReg src) { x64_enc_sse_rr(a, 0xF3u, 0x51u, dst, src); }

gb_internal void x64_emit_ucomisd(X64Assembler *a, X64XmmReg lhs, X64XmmReg rhs) {
	// 66 [REX] 0F 2E /r
	x64_enc_b(a, 0x66u);
	bool R = (lhs >= 8), B = (rhs >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x2Eu);
	x64_enc_b(a, cast(u8)(0xC0u | ((lhs & 7u) << 3) | (rhs & 7u)));
}

gb_internal void x64_emit_ucomiss(X64Assembler *a, X64XmmReg lhs, X64XmmReg rhs) {
	// [REX] 0F 2E /r  (no 66 prefix)
	bool R = (lhs >= 8), B = (rhs >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x2Eu);
	x64_enc_b(a, cast(u8)(0xC0u | ((lhs & 7u) << 3) | (rhs & 7u)));
}

gb_internal void x64_emit_xorpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src) {
	// 66 [REX] 0F 57 /r
	x64_enc_b(a, 0x66u);
	bool R = (dst >= 8), B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x57u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}
gb_internal void x64_emit_xorps(X64Assembler *a, X64XmmReg dst, X64XmmReg src) {
	bool R = (dst >= 8), B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x57u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}
gb_internal void x64_emit_andpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src) {
	x64_enc_b(a, 0x66u);
	bool R = (dst >= 8), B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x54u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}
gb_internal void x64_emit_andps(X64Assembler *a, X64XmmReg dst, X64XmmReg src) {
	bool R = (dst >= 8), B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x54u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}
gb_internal void x64_emit_orpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src) {
	x64_enc_b(a, 0x66u);
	bool R = (dst >= 8), B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x56u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}
gb_internal void x64_emit_orps(X64Assembler *a, X64XmmReg dst, X64XmmReg src) {
	bool R = (dst >= 8), B = (src >= 8);
	if (R || B) x64_enc_b(a, cast(u8)(0x40u | (R ? 0x04u : 0u) | (B ? 0x01u : 0u)));
	x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x56u);
	x64_enc_b(a, cast(u8)(0xC0u | ((dst & 7u) << 3) | (src & 7u)));
}

gb_internal void x64_emit_cvtsi2sd(X64Assembler *a, X64XmmReg dst, X64Reg src, X64OpSize src_sz) {
	x64_enc_sse_cvt_xmm_r(a, 0xF2u, 0x2Au, dst, src, src_sz);
}
gb_internal void x64_emit_cvtsi2ss(X64Assembler *a, X64XmmReg dst, X64Reg src, X64OpSize src_sz) {
	x64_enc_sse_cvt_xmm_r(a, 0xF3u, 0x2Au, dst, src, src_sz);
}
gb_internal void x64_emit_cvttsd2si(X64Assembler *a, X64Reg dst, X64OpSize dst_sz, X64XmmReg src) {
	x64_enc_sse_cvt_r_xmm(a, 0xF2u, 0x2Cu, dst, dst_sz, src);
}
gb_internal void x64_emit_cvttss2si(X64Assembler *a, X64Reg dst, X64OpSize dst_sz, X64XmmReg src) {
	x64_enc_sse_cvt_r_xmm(a, 0xF3u, 0x2Cu, dst, dst_sz, src);
}
gb_internal void x64_emit_cvtsd2ss(X64Assembler *a, X64XmmReg dst, X64XmmReg src) { x64_enc_sse_rr(a, 0xF2u, 0x5Au, dst, src); }
gb_internal void x64_emit_cvtss2sd(X64Assembler *a, X64XmmReg dst, X64XmmReg src) { x64_enc_sse_rr(a, 0xF3u, 0x5Au, dst, src); }

// ===========================================================================
// Atomics
// ===========================================================================

gb_internal void x64_emit_lock_xadd_mr(X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src) {
	x64_enc_b(a, 0xF0u); // LOCK
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	x64_enc_rex(a, sz, src, base, dst.index);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0xC0u : 0xC1u);
	x64_enc_modrm_mem(a, cast(u8)(src & 7u), dst);
}

gb_internal void x64_emit_lock_cmpxchg_mr(X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src) {
	x64_enc_b(a, 0xF0u); // LOCK
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	x64_enc_rex(a, sz, src, base, dst.index);
	x64_enc_b(a, 0x0Fu);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0xB0u : 0xB1u);
	x64_enc_modrm_mem(a, cast(u8)(src & 7u), dst);
}

gb_internal void x64_emit_lock_add_mi(X64Assembler *a, X64OpSize sz, X64Mem dst, i32 imm) {
	x64_enc_b(a, 0xF0u); // LOCK
	x64_enc_alu_imm_m(a, sz, 0u, dst, imm);
}

gb_internal void x64_emit_lock_xchg_mr(X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src) {
	// XCHG has an implied LOCK when a memory operand is involved
	X64Reg base = dst.rip_rel ? X64Reg_NONE : dst.base;
	x64_enc_rex(a, sz, src, base, dst.index);
	x64_enc_b(a, (sz == X64OpSize_8) ? 0x86u : 0x87u);
	x64_enc_modrm_mem(a, cast(u8)(src & 7u), dst);
}

gb_internal void x64_emit_mfence(X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xAEu); x64_enc_b(a, 0xF0u); }
gb_internal void x64_emit_sfence(X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xAEu); x64_enc_b(a, 0xF8u); }
gb_internal void x64_emit_lfence(X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xAEu); x64_enc_b(a, 0xE8u); }

gb_internal void x64_emit_pause (X64Assembler *a) { x64_enc_b(a, 0xF3u); x64_enc_b(a, 0x90u); }                    // cpu_relax
gb_internal void x64_emit_ud2   (X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x0Bu); }                    // trap
gb_internal void x64_emit_rdtsc (X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x31u); }                    // EDX:EAX = TSC
gb_internal void x64_emit_cpuid (X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0xA2u); }                    // leaf EAX, sub ECX
gb_internal void x64_emit_xgetbv(X64Assembler *a) { x64_enc_b(a, 0x0Fu); x64_enc_b(a, 0x01u); x64_enc_b(a, 0xD0u); } // EDX:EAX = XCR[ECX]
