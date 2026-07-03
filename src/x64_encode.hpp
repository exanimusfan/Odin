// x64 instruction encoder for the Odin fast debug backend. Target: x86-64, Windows x64 ABI.
//
// Instruction functions append machine code to X64Assembler::code. Forward branches
// resolve lazily via a fixup list patched at label bind; external symbol references
// record X64RelocEntry items the COFF writer turns into COFF relocations.


enum X64Reg : u8 {
	X64Reg_RAX = 0,  X64Reg_RCX = 1,  X64Reg_RDX = 2,  X64Reg_RBX = 3,
	X64Reg_RSP = 4,  X64Reg_RBP = 5,  X64Reg_RSI = 6,  X64Reg_RDI = 7,
	X64Reg_R8  = 8,  X64Reg_R9  = 9,  X64Reg_R10 = 10, X64Reg_R11 = 11,
	X64Reg_R12 = 12, X64Reg_R13 = 13, X64Reg_R14 = 14, X64Reg_R15 = 15,
	X64Reg_NONE = 0xFF,
};

enum X64XmmReg : u8 {
	X64XmmReg_XMM0  =  0, X64XmmReg_XMM1  =  1, X64XmmReg_XMM2  =  2, X64XmmReg_XMM3  =  3,
	X64XmmReg_XMM4  =  4, X64XmmReg_XMM5  =  5, X64XmmReg_XMM6  =  6, X64XmmReg_XMM7  =  7,
	X64XmmReg_XMM8  =  8, X64XmmReg_XMM9  =  9, X64XmmReg_XMM10 = 10, X64XmmReg_XMM11 = 11,
	X64XmmReg_XMM12 = 12, X64XmmReg_XMM13 = 13, X64XmmReg_XMM14 = 14, X64XmmReg_XMM15 = 15,
};

// ---------------------------------------------------------------------------
// Operand sizes
// ---------------------------------------------------------------------------

enum X64OpSize : u8 {
	X64OpSize_8  = 1,
	X64OpSize_16 = 2,
	X64OpSize_32 = 4,
	X64OpSize_64 = 8,
};

// ---------------------------------------------------------------------------
// Memory addressing
// ---------------------------------------------------------------------------

struct X64Mem {
	X64Reg base;    // X64Reg_NONE = no base (disp32 or RIP-relative)
	X64Reg index;   // X64Reg_NONE = no index
	u8     scale;   // 1, 2, 4, or 8 (ignored if index == NONE)
	i32    disp;
	bool   rip_rel; // [RIP + disp]; base/index/scale are ignored
};

gb_internal X64Mem x64_mem(X64Reg base, i32 disp = 0);
gb_internal X64Mem x64_mem_idx(X64Reg base, X64Reg index, u8 scale, i32 disp = 0);
gb_internal X64Mem x64_mem_rip(i32 disp = 0);
gb_internal X64Mem x64_mem_abs(i32 disp);        // [disp32], no base/index

// ---------------------------------------------------------------------------
// Condition codes  (lower 4 bits of Jcc / SETcc opcode byte)
// ---------------------------------------------------------------------------

enum X64Cc : u8 {
	X64Cc_O   = 0x0,
	X64Cc_NO  = 0x1,
	X64Cc_B   = 0x2,   // below / carry (unsigned <)
	X64Cc_AE  = 0x3,   // above-or-equal / no-carry (unsigned >=)
	X64Cc_E   = 0x4,   // equal / zero
	X64Cc_NE  = 0x5,   // not-equal / not-zero
	X64Cc_BE  = 0x6,   // below-or-equal (unsigned <=)
	X64Cc_A   = 0x7,   // above (unsigned >)
	X64Cc_S   = 0x8,   // sign
	X64Cc_NS  = 0x9,   // no-sign
	X64Cc_P   = 0xA,   // parity-even
	X64Cc_NP  = 0xB,   // parity-odd / no-parity
	X64Cc_L   = 0xC,   // less (signed <)
	X64Cc_GE  = 0xD,   // greater-or-equal (signed >=)
	X64Cc_LE  = 0xE,   // less-or-equal (signed <=)
	X64Cc_G   = 0xF,   // greater (signed >)
	// Aliases
	X64Cc_Z   = X64Cc_E,
	X64Cc_NZ  = X64Cc_NE,
	X64Cc_C   = X64Cc_B,
	X64Cc_NC  = X64Cc_AE,
	X64Cc_NAE = X64Cc_B,
	X64Cc_NB  = X64Cc_AE,
	X64Cc_NA  = X64Cc_BE,
	X64Cc_NBE = X64Cc_A,
	X64Cc_NL  = X64Cc_GE,
	X64Cc_NLE = X64Cc_G,
	X64Cc_NG  = X64Cc_LE,
	X64Cc_NGE = X64Cc_L,
};

// ---------------------------------------------------------------------------
// Labels and fixups
// ---------------------------------------------------------------------------

// Index into X64Assembler::labels; X64_LABEL_UNSET means unbound.
#define X64_LABEL_UNSET ((isize)-1)

struct X64Fixup {
	isize patch_at; // byte offset in code where the i32 displacement lives
	isize label;    // index into X64Assembler::labels
};

// ---------------------------------------------------------------------------
// Symbol relocations
// ---------------------------------------------------------------------------

// These map 1-to-1 to Windows COFF AMD64 relocation type values.
enum X64RelocType : u16 {
	X64Reloc_ADDR64   = 0x0001, // 64-bit absolute VA
	X64Reloc_ADDR32   = 0x0002, // 32-bit absolute VA
	X64Reloc_ADDR32NB = 0x0003, // 32-bit VA without image base
	X64Reloc_REL32    = 0x0004, // 32-bit PC-relative (CALL/JMP/LEA-RIP)
	X64Reloc_REL32_1  = 0x0005,
	X64Reloc_REL32_2  = 0x0006,
	X64Reloc_REL32_3  = 0x0007,
	X64Reloc_REL32_4  = 0x0008,
	X64Reloc_REL32_5  = 0x0009,
	X64Reloc_SECREL   = 0x000B, // section-relative (CodeView debug info)
};

struct X64RelocEntry {
	isize        code_offset; // byte offset in this section's code buffer
	X64RelocType type;
	String       sym_name;    // resolved by the COFF writer
};

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

struct X64Assembler {
	gbAllocator        allocator;
	Array<u8>          code;    // raw machine-code bytes
	Array<isize>       labels;  // label targets; X64_LABEL_UNSET = unbound
	Array<X64Fixup>    fixups;  // pending forward-branch fixups
	Array<X64RelocEntry> relocs; // symbol relocations to hand off to COFF writer
};

gb_internal void  x64_asm_init(X64Assembler *a, gbAllocator allocator);
gb_internal void  x64_asm_free(X64Assembler *a);
gb_internal isize x64_asm_len (X64Assembler *a);

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

gb_internal isize x64_label_alloc(X64Assembler *a);
gb_internal void  x64_label_bind (X64Assembler *a, isize label); // bind at current offset
gb_internal bool  x64_label_bound(X64Assembler *a, isize label);

// ---------------------------------------------------------------------------
// Alignment / misc
// ---------------------------------------------------------------------------

gb_internal void x64_emit_align(X64Assembler *a, isize align, u8 fill = 0x90);
gb_internal void x64_emit_nop  (X64Assembler *a);
gb_internal void x64_emit_int3 (X64Assembler *a); // software breakpoint / debug trap

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

gb_internal void x64_emit_push_r  (X64Assembler *a, X64Reg r);
gb_internal void x64_emit_pop_r   (X64Assembler *a, X64Reg r);
gb_internal void x64_emit_push_i8 (X64Assembler *a, i8  imm);
gb_internal void x64_emit_push_i32(X64Assembler *a, i32 imm);

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

gb_internal void x64_emit_ret   (X64Assembler *a);
gb_internal void x64_emit_ret_n (X64Assembler *a, u16 pop_bytes);
gb_internal void x64_emit_jmp   (X64Assembler *a, isize label);
gb_internal void x64_emit_jcc   (X64Assembler *a, X64Cc cc, isize label);
gb_internal void x64_emit_jmp_r (X64Assembler *a, X64Reg r);
gb_internal void x64_emit_jmp_m (X64Assembler *a, X64Mem m);
gb_internal void x64_emit_call_r(X64Assembler *a, X64Reg r);
gb_internal void x64_emit_call_m(X64Assembler *a, X64Mem m);
// CALL near with REL32 relocation against a named symbol
gb_internal void x64_emit_call_sym(X64Assembler *a, String sym_name);
// JMP near with REL32 relocation (tail-call / inter-section jump)
gb_internal void x64_emit_jmp_sym (X64Assembler *a, String sym_name);

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

gb_internal void x64_emit_mov_rr (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_mov_rm (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Mem src);
gb_internal void x64_emit_mov_mr (X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src);
gb_internal void x64_emit_mov_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, i64 imm);
gb_internal void x64_emit_mov_mi (X64Assembler *a, X64OpSize sz, X64Mem dst, i32 imm);
// dst (64-bit) = zero_extend(src_sz operand)
gb_internal void x64_emit_movzx_rr(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_movzx_rm(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Mem src);
// dst (64-bit) = sign_extend(src_sz operand)
gb_internal void x64_emit_movsx_rr(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_movsx_rm(X64Assembler *a, X64OpSize src_sz, X64Reg dst, X64Mem src);
gb_internal void x64_emit_lea    (X64Assembler *a, X64Reg dst, X64Mem src);
// LEA r, [RIP + sym]  — emits REL32 relocation so global data can be addressed
gb_internal void x64_emit_lea_sym(X64Assembler *a, X64Reg dst, String sym_name);
// &sym for this thread into RAX; clobbers RAX and R11. REL32 reloc to `_tls_index`, SECREL to `sym`.
gb_internal void x64_emit_tls_addr(X64Assembler *a, String sym_name);
gb_internal void x64_emit_xchg_rr(X64Assembler *a, X64OpSize sz, X64Reg a_, X64Reg b);
gb_internal void x64_emit_cmov_rr(X64Assembler *a, X64Cc cc, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_cmov_rm(X64Assembler *a, X64Cc cc, X64OpSize sz, X64Reg dst, X64Mem src);

// ---------------------------------------------------------------------------
// Integer ALU  (add / sub / and / or / xor / cmp share the same encoding shape)
// ---------------------------------------------------------------------------
//
// _rr : dst op= src        (both registers)
// _rm : dst op= [mem]      (reg destination, memory source)
// _mr : [mem] op= src      (memory destination, register source)
// _ri : dst op= imm        (register destination, immediate source)
// _mi : [mem] op= imm      (memory destination, immediate source)

#define X64_ALU_DECLS(op) \
	gb_internal void x64_emit_##op##_rr(X64Assembler *a, X64OpSize sz, X64Reg  dst, X64Reg src); \
	gb_internal void x64_emit_##op##_rm(X64Assembler *a, X64OpSize sz, X64Reg  dst, X64Mem src); \
	gb_internal void x64_emit_##op##_mr(X64Assembler *a, X64OpSize sz, X64Mem  dst, X64Reg src); \
	gb_internal void x64_emit_##op##_ri(X64Assembler *a, X64OpSize sz, X64Reg  dst, i32    imm); \
	gb_internal void x64_emit_##op##_mi(X64Assembler *a, X64OpSize sz, X64Mem  dst, i32    imm)

X64_ALU_DECLS(add);
X64_ALU_DECLS(adc); // add-with-carry (128-bit lo:hi arithmetic)
X64_ALU_DECLS(sub);
X64_ALU_DECLS(sbb); // sub-with-borrow (128-bit lo:hi arithmetic)
X64_ALU_DECLS(and);
X64_ALU_DECLS(or);
X64_ALU_DECLS(xor);
X64_ALU_DECLS(cmp); // sets flags only, does not write result

#undef X64_ALU_DECLS

// Double-precision shifts (for 128-bit lo:hi shifting). `dst` is r/m, `src` is the reg whose bits fill
// in. shld: dst = (dst << cl) | (src >> (n-cl)); shrd: dst = (dst >> cl) | (src << (n-cl)). CL form only.
gb_internal void x64_emit_shld_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_shrd_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);

gb_internal void x64_emit_test_rr(X64Assembler *a, X64OpSize sz, X64Reg lhs, X64Reg rhs);
gb_internal void x64_emit_test_ri(X64Assembler *a, X64OpSize sz, X64Reg lhs, i32    imm);
gb_internal void x64_emit_test_mi(X64Assembler *a, X64OpSize sz, X64Mem lhs, i32    imm);

gb_internal void x64_emit_neg_r(X64Assembler *a, X64OpSize sz, X64Reg r);
gb_internal void x64_emit_not_r(X64Assembler *a, X64OpSize sz, X64Reg r);
gb_internal void x64_emit_neg_m(X64Assembler *a, X64OpSize sz, X64Mem m);
gb_internal void x64_emit_not_m(X64Assembler *a, X64OpSize sz, X64Mem m);
gb_internal void x64_emit_inc_r(X64Assembler *a, X64OpSize sz, X64Reg r);
gb_internal void x64_emit_dec_r(X64Assembler *a, X64OpSize sz, X64Reg r);

gb_internal void x64_emit_imul_rr (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_imul_rm (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Mem src);
gb_internal void x64_emit_imul_rri(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src, i32 imm);
gb_internal void x64_emit_imul_r  (X64Assembler *a, X64OpSize sz, X64Reg src); // RDX:RAX = RAX * src (signed)
gb_internal void x64_emit_mul_r   (X64Assembler *a, X64OpSize sz, X64Reg src); // RDX:RAX = RAX * src (unsigned)

// Divide  (quotient → RAX, remainder → RDX; caller must sign/zero-extend RAX first)
gb_internal void x64_emit_idiv_r(X64Assembler *a, X64OpSize sz, X64Reg src); // signed
gb_internal void x64_emit_div_r (X64Assembler *a, X64OpSize sz, X64Reg src); // unsigned

// Sign-extension before divide
gb_internal void x64_emit_cqo (X64Assembler *a); // RAX → RDX:RAX (64-bit)
gb_internal void x64_emit_cdq (X64Assembler *a); // EAX → EDX:EAX (32-bit)
gb_internal void x64_emit_cwde(X64Assembler *a); // AX  → EAX     (16→32)

// Shifts
gb_internal void x64_emit_shl_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count);
gb_internal void x64_emit_shr_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count);
gb_internal void x64_emit_sar_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count);
gb_internal void x64_emit_rol_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count);
gb_internal void x64_emit_ror_ri (X64Assembler *a, X64OpSize sz, X64Reg dst, u8 count);
gb_internal void x64_emit_shl_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst);
gb_internal void x64_emit_shr_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst);
gb_internal void x64_emit_sar_rcl(X64Assembler *a, X64OpSize sz, X64Reg dst);

// Bit scan / count
gb_internal void x64_emit_bsf_rr   (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_bsr_rr   (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_tzcnt_rr (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src); // BMI1
gb_internal void x64_emit_lzcnt_rr (X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_popcnt_rr(X64Assembler *a, X64OpSize sz, X64Reg dst, X64Reg src);
gb_internal void x64_emit_bswap_r  (X64Assembler *a, X64OpSize sz, X64Reg r);

// SETcc  (result is always 8-bit; caller zero-extends if needed)
gb_internal void x64_emit_setcc_r(X64Assembler *a, X64Cc cc, X64Reg dst);
gb_internal void x64_emit_setcc_m(X64Assembler *a, X64Cc cc, X64Mem dst);

// ---------------------------------------------------------------------------
// SSE2 scalar floating-point
// ---------------------------------------------------------------------------

gb_internal void x64_emit_movsd_rr(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_movsd_rm(X64Assembler *a, X64XmmReg dst, X64Mem src);
gb_internal void x64_emit_movsd_mr(X64Assembler *a, X64Mem dst, X64XmmReg src);
gb_internal void x64_emit_movss_rr(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_movss_rm(X64Assembler *a, X64XmmReg dst, X64Mem src);
gb_internal void x64_emit_movss_mr(X64Assembler *a, X64Mem dst, X64XmmReg src);

// AVX/AVX2 VEX-encoded generic emitters (pp: 0=none 1=66 2=F3 3=F2; mm: 1=0F 2=0F38 3=0F3A;
// L: false=128/xmm true=256/ymm; vvvv = 2nd source, pass X64XmmReg_XMM0 when the op has none).
gb_internal void x64_emit_vex_rr(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L, X64XmmReg reg, X64XmmReg vvvv, X64XmmReg rm);
gb_internal void x64_emit_vex_rm(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L, X64XmmReg reg, X64XmmReg vvvv, X64Mem rm);
gb_internal void x64_emit_vex_mr(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L, X64Mem rm, X64XmmReg vvvv, X64XmmReg reg);
gb_internal void x64_emit_vex_rr_i(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L, X64XmmReg reg, X64XmmReg vvvv, X64XmmReg rm, u8 imm);
gb_internal void x64_emit_vex_gather(X64Assembler *a, u8 pp, u8 mm, u8 op, bool W, bool L, X64XmmReg dst, X64XmmReg mask, X64Reg base, X64XmmReg vindex, u8 scale);
gb_internal void x64_emit_vzeroupper(X64Assembler *a);

// Unaligned 128-bit packed move + packed round (SSE4.1) — for #simd vector ops
gb_internal void x64_emit_movups_rm(X64Assembler *a, X64XmmReg dst, X64Mem src);
gb_internal void x64_emit_movups_mr(X64Assembler *a, X64Mem dst, X64XmmReg src);
gb_internal void x64_emit_roundps(X64Assembler *a, X64XmmReg dst, X64XmmReg src, u8 imm);
gb_internal void x64_emit_roundpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src, u8 imm);

// Packed (whole-XMM) ops for #simd vectors
gb_internal void x64_emit_addps(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_subps(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_mulps(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_divps(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_minps(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_maxps(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_addpd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_subpd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_mulpd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_divpd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_minpd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_maxpd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_paddb(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_paddw(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_paddd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_paddq(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_psubb(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_psubw(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_psubd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_psubq(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pand (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_por  (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pxor (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pandn(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpeqb(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpeqw(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpeqd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpgtb(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpgtw(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpgtd(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmullw (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmulld (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpeqq(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pcmpgtq(X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pminsb (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pminsd (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmaxsb (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmaxsd (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pminuw (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pminud (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmaxuw (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmaxud (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pminub (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmaxub (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pminsw (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_pmaxsw (X64Assembler *a, X64XmmReg d, X64XmmReg s);
gb_internal void x64_emit_cmpps(X64Assembler *a, X64XmmReg d, X64XmmReg s, u8 imm);
gb_internal void x64_emit_cmppd(X64Assembler *a, X64XmmReg d, X64XmmReg s, u8 imm);
gb_internal void x64_emit_psrldq(X64Assembler *a, X64XmmReg d, u8 imm);

// Arithmetic (scalar double)
gb_internal void x64_emit_addsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_subsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_mulsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_divsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_sqrtsd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);

// Arithmetic (scalar single)
gb_internal void x64_emit_addss(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_subss(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_mulss(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_divss(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_sqrtss(X64Assembler *a, X64XmmReg dst, X64XmmReg src);

// Comparison (ordered, sets ZF/PF/CF)
gb_internal void x64_emit_ucomisd(X64Assembler *a, X64XmmReg lhs, X64XmmReg rhs);
gb_internal void x64_emit_ucomiss(X64Assembler *a, X64XmmReg lhs, X64XmmReg rhs);

// Bitwise (often used to negate / abs)
gb_internal void x64_emit_xorpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_xorps(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_andpd(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_andps(X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_orpd (X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_orps (X64Assembler *a, X64XmmReg dst, X64XmmReg src);

gb_internal void x64_emit_cvtsi2sd (X64Assembler *a, X64XmmReg dst, X64Reg src, X64OpSize src_sz);
gb_internal void x64_emit_cvtsi2ss (X64Assembler *a, X64XmmReg dst, X64Reg src, X64OpSize src_sz);
gb_internal void x64_emit_cvttsd2si(X64Assembler *a, X64Reg dst, X64OpSize dst_sz, X64XmmReg src);
gb_internal void x64_emit_cvttss2si(X64Assembler *a, X64Reg dst, X64OpSize dst_sz, X64XmmReg src);
gb_internal void x64_emit_cvtsd2ss (X64Assembler *a, X64XmmReg dst, X64XmmReg src);
gb_internal void x64_emit_cvtss2sd (X64Assembler *a, X64XmmReg dst, X64XmmReg src);

// ---------------------------------------------------------------------------
// Atomics
// ---------------------------------------------------------------------------

gb_internal void x64_emit_lock_xadd_mr    (X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src);
gb_internal void x64_emit_lock_cmpxchg_mr (X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src);
gb_internal void x64_emit_lock_add_mi     (X64Assembler *a, X64OpSize sz, X64Mem dst, i32 imm);
gb_internal void x64_emit_lock_xchg_mr    (X64Assembler *a, X64OpSize sz, X64Mem dst, X64Reg src);
gb_internal void x64_emit_mfence(X64Assembler *a);
gb_internal void x64_emit_sfence(X64Assembler *a);
gb_internal void x64_emit_lfence(X64Assembler *a);

// No-operand / fixed-encoding instructions
gb_internal void x64_emit_pause (X64Assembler *a); // F3 90      — spin-loop hint (cpu_relax)
gb_internal void x64_emit_ud2   (X64Assembler *a); // 0F 0B      — undefined instruction (trap)
gb_internal void x64_emit_rdtsc (X64Assembler *a); // 0F 31      — EDX:EAX = timestamp counter
gb_internal void x64_emit_cpuid (X64Assembler *a); // 0F A2      — CPUID(EAX leaf, ECX subleaf) -> EAX/EBX/ECX/EDX
gb_internal void x64_emit_xgetbv(X64Assembler *a); // 0F 01 D0   — EDX:EAX = XCR[ECX]
