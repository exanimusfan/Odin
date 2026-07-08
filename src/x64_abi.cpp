// x64_abi.cpp — calling-convention layer (mirrors llvm_abi.cpp's role for lb_).
//
// ALL knowledge of where arguments, returns, and incoming parameters PHYSICALLY live
// is in this file. The rest of the backend deals only in ABI SLOT indices: the flat
// [sret?][explicit params][partial-ret ptrs][context?] argument sequence that
// x64_build_call_expr / x64_emit_runtime_call construct and x64_abi_home_params homes.
// A slot's value is the arg itself (direct) or a pointer to it (x64_arg_is_indirect).
//
// Two implementations share the seam, selected by x64_abi_win64 (from the TARGET os):
//   Win64 (windows): 4 unified register slots, caller-provided shadow space, __chkstk,
//     aggregates not sized 1/2/4/8 passed by pointer (x64_arg_is_indirect).
//   SysV  (darwin/linux): full eightbyte classification (x64_sysv_classify) — ≤16-byte
//     values in 1–2 GP/XMM registers (RDI..R9 + XMM0..XMM7, SEPARATE counters,
//     all-or-nothing take), MEMORY class as by-value stack copies at [RSP+0] (no
//     shadow space; register params get callee-allocated frame homes), returns ≤16B
//     in RAX:RDX / XMM0:XMM1 pairs (>16B via sret), AL = #XMM regs for C varargs,
//     plain SUB RSP (no stack probe). Known deviations from strict SysV: f16 travels
//     as a GP scalar (backend-wide convention), raw unions classify INTEGER, an
//     all-padding eightbyte defaults to SSE.

// x64_abi_win64 (defined in x64_backend.hpp so earlier-compiled files see it) is
// selected at generator init from build_context.metrics.os (the TARGET, not the host).
gb_internal void x64_abi_init_target(void) {
	x64_abi_win64 = (build_context.metrics.os == TargetOs_windows);
}

// Win64: 4 unified register slots (a slot is GP or XMM, never both).
static const X64Reg    X64_INT_ARG_REGS[4] = { X64Reg_RCX, X64Reg_RDX, X64Reg_R8, X64Reg_R9 };
static const X64XmmReg X64_XMM_ARG_REGS[4] = { X64XmmReg_XMM0, X64XmmReg_XMM1, X64XmmReg_XMM2, X64XmmReg_XMM3 };

// SysV: 6 GP + 8 XMM argument registers, independent counters.
static const X64Reg    X64_SYSV_INT_ARG_REGS[6] = {
	X64Reg_RDI, X64Reg_RSI, X64Reg_RDX, X64Reg_RCX, X64Reg_R8, X64Reg_R9,
};
static const X64XmmReg X64_SYSV_XMM_ARG_REGS[8] = {
	X64XmmReg_XMM0, X64XmmReg_XMM1, X64XmmReg_XMM2, X64XmmReg_XMM3,
	X64XmmReg_XMM4, X64XmmReg_XMM5, X64XmmReg_XMM6, X64XmmReg_XMM7,
};

// ── SysV eightbyte classification (System V AMD64 ABI §3.2.3) ────────────────
// A ≤16-byte value is split into 1–2 eightbytes, each classed INTEGER or SSE by the
// scalar fields covering its bytes (INTEGER dominates on overlap). >16 bytes (or a
// register shortfall) → MEMORY: passed as a by-value stack copy, returned via sret.
// Simplifications vs the full algorithm: no x87 (Odin has no f80), raw unions are
// conservatively INTEGER, and an all-padding eightbyte defaults to SSE.

enum { X64_SYSV_NONE = 0, X64_SYSV_SSE = 1, X64_SYSV_INT = 2 };

static void x64_sysv_mark_bytes(i64 off, i64 sz, u8 c, u8 cls[2]) {
	if (sz <= 0) return;
	i64 b0 = off / 8, b1 = (off + sz - 1) / 8;
	for (i64 b = gb_max(b0, (i64)0); b <= b1 && b < 2; b++) {
		if (cls[b] < c) cls[b] = c; // INT(2) dominates SSE(1) dominates NONE(0)
	}
}

static void x64_sysv_mark(Type *t, i64 off, u8 cls[2]) {
	t = core_type(t);
	i64 sz = type_size_of(t);
	if (sz <= 0) return;
	switch (t->kind) {
	case Type_Basic:
		// f16 is EXCLUDED from SSE: the backend moves f16 scalars as 2-byte ints through
		// GP registers (see x64_is_scalar/x64_is_float), so classifying it SSE would make
		// the classifier disagree with the value loaders. Deviates from strict SysV
		// (_Float16 → SSE) only at C boundaries passing f16 by value — rare.
		if ((is_type_float(t) && sz != 2) || is_type_complex(t) || is_type_quaternion(t)) {
			x64_sysv_mark_bytes(off, sz, X64_SYSV_SSE, cls);
		} else {
			// ints/bools/rune/uintptr/rawptr/cstring/typeid/string/any/f16 → INTEGER
			x64_sysv_mark_bytes(off, sz, X64_SYSV_INT, cls);
		}
		return;
	case Type_Array: {
		i64 esz = type_size_of(t->Array.elem);
		for (i64 i = 0; i < t->Array.count; i++) x64_sysv_mark(t->Array.elem, off + i*esz, cls);
		return;
	}
	case Type_EnumeratedArray: {
		i64 esz = type_size_of(t->EnumeratedArray.elem);
		for (i64 i = 0; i < t->EnumeratedArray.count; i++) x64_sysv_mark(t->EnumeratedArray.elem, off + i*esz, cls);
		return;
	}
	case Type_Struct:
		if (t->Struct.is_raw_union) {
			x64_sysv_mark_bytes(off, sz, X64_SYSV_INT, cls); // conservative variant merge
			return;
		}
		for_array(i, t->Struct.fields) {
			x64_sysv_mark(t->Struct.fields[i]->type, off + type_offset_of(t, i), cls);
		}
		return;
	case Type_Matrix:
		x64_sysv_mark_bytes(off, sz, (u8)(is_type_float(core_type(t->Matrix.elem)) ? X64_SYSV_SSE : X64_SYSV_INT), cls);
		return;
	case Type_SimdVector:
		x64_sysv_mark_bytes(off, sz, (u8)(is_type_float(core_type(t->SimdVector.elem)) ? X64_SYSV_SSE : X64_SYSV_INT), cls);
		return;
	default:
		// pointers, procs, slices, unions, maps, bit_sets, bit_fields, ... → INTEGER
		x64_sysv_mark_bytes(off, sz, X64_SYSV_INT, cls);
		return;
	}
}

// Classify `t` for SysV passing: fills cls[0..n8) and returns n8 (1 or 2 eightbytes),
// or 0 for MEMORY class (size > 16). Zero-sized types never reach here (no ABI slot).
gb_internal int x64_sysv_classify(Type *t, u8 cls[2]) {
	cls[0] = cls[1] = X64_SYSV_NONE;
	i64 sz = type_size_of(t);
	if (sz <= 0 || sz > 16) return 0;
	x64_sysv_mark(t, 0, cls);
	int n8 = (int)((sz + 7) / 8);
	for (int i = 0; i < n8; i++) {
		if (cls[i] == X64_SYSV_NONE) cls[i] = X64_SYSV_SSE; // all-padding eightbyte
	}
	return n8;
}

// Running SysV argument cursor — caller and callee walk the flat slot list with the
// same logic so they agree on every slot's physical location.
struct x64SysVCursor {
	int gp;        // next GP arg register index (X64_SYSV_INT_ARG_REGS)
	int xmm;       // next XMM arg register index
	i32 stack_off; // next stack-arg byte offset (8-aligned, more for over-aligned args)
};

// Reserve registers for one classified value, ALL-OR-NOTHING (SysV: if any eightbyte
// doesn't fit its register class, the whole value goes to memory and no registers are
// consumed). On success fills reg_idx[k] with each eightbyte's register index.
static bool x64_sysv_take_regs(x64SysVCursor *cur, u8 const cls[2], int n8, int reg_idx[2]) {
	int ngp = 0, nsse = 0;
	for (int k = 0; k < n8; k++) {
		if (cls[k] == X64_SYSV_INT) ngp++; else nsse++;
	}
	if (cur->gp + ngp > 6 || cur->xmm + nsse > 8) return false;
	for (int k = 0; k < n8; k++) {
		reg_idx[k] = (cls[k] == X64_SYSV_INT) ? cur->gp++ : cur->xmm++;
	}
	return true;
}

// Advance the cursor past a MEMORY-class / register-exhausted stack arg of `sz` bytes,
// returning its byte offset in the stack-arg area. Args are 8-aligned, over-aligned
// types (16-byte structs with #align) keep their alignment.
static i32 x64_sysv_take_stack(x64SysVCursor *cur, i64 sz, i64 al) {
	i64 a = gb_max(al, (i64)8);
	cur->stack_off = (i32)((cur->stack_off + a - 1) & ~(a - 1));
	i32 off = cur->stack_off;
	cur->stack_off += (i32)((sz + 7) & ~7);
	return off;
}

// Caller-reserved spill area for the register slots (Win64 shadow space).
#define X64_ABI_SHADOW_SPACE 32
// Backend floor for the outgoing stack-arg area — the alloca fast path assumes a
// fixed RSP + X64_ABI_SHADOW_SPACE + X64_ABI_MIN_OUTGOING call area below it.
#define X64_ABI_MIN_OUTGOING 64

// Indirect-ABI params at or below this size are copied into a frame-local on entry;
// larger ones are accessed through the incoming pointer (no copy) to avoid huge stack copies.
#define X64_INDIRECT_PARAM_COPY_MAX 4096

gb_internal bool x64_arg_is_float(Type *t) { return x64_is_float(t); }

// Does this arg's ABI SLOT hold a POINTER to the value (vs the value itself)?
// Win64 (lbAbiAmd64Win64::compute_arg_types + lbAbi386::non_struct): an aggregate is
// passed BY POINTER unless size is exactly 1/2/4/8; likewise a non-float scalar wider
// than 8 (i128). SysV: NEVER — every value slot is direct; the physical form (register
// eightbytes vs by-value stack copy) is decided inside this file by classification.
// Zero-sized aggregates are dropped (elsewhere), not indirect.
gb_internal bool x64_arg_is_indirect(Type *t) {
	if (t == nullptr) return false;
	if (!x64_abi_win64) return false;
	i64 sz = x64_type_size(t);
	if (sz == 0) return false;
	if (x64_is_scalar(t)) return sz > 8;            // i128 by pointer; scalars ≤8 in register
	return !(sz == 1 || sz == 2 || sz == 4 || sz == 8);
}

// Lower ONE aggregate value (already in memory at `m`) to its ABI arg slot for a
// hand-built runtime-helper call: Win64 indirect → spill &m into a pointer slot;
// SysV → the value itself (register pairs / stack copies are applied later by
// x64_abi_emit_call_args). Scalars and direct Win64 aggregates also take the value path.
gb_internal x64Value x64_abi_value_or_addr_slot(x64Procedure *p, Type *t, X64Mem m) {
	if (x64_arg_is_indirect(t)) {
		i32 po = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, m);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(po), X64Reg_RAX);
		return x64v_mem(t_rawptr, x64_rbp_mem(po));
	}
	return x64v_mem(t, m);
}

// Should x64_stabilize_value keep only the ADDRESS of this value (by_ref) instead of
// snapshotting it? True when the ABI never needs a frame copy of the VALUE on the
// caller side: Win64 indirect args (a pointer is passed) and SysV MEMORY-class args
// (x64_abi_emit_call_args copies straight from the source into the outgoing area).
gb_internal bool x64_arg_stabilize_by_ref(Type *t) {
	if (t == nullptr) return false;
	if (x64_abi_win64) return x64_arg_is_indirect(t);
	if (x64_is_scalar(t)) return false;
	u8 cls[2];
	return type_size_of(t) > 0 && x64_sysv_classify(t, cls) == 0; // MEMORY class
}

// Is a SINGLE value of type `rt` returned via a hidden pointer (vs return registers)?
// Win64 (mirrors LLVM lbArg_Indirect): register-sized scalar → RAX/XMM0; aggregates →
// hidden pointer unless size is 1/2/4/8. SysV: only MEMORY class (>16 bytes) — ≤16-byte
// values (incl. i128 and 2-eightbyte aggregates) come back in RAX:RDX / XMM0:XMM1 pairs.
// Zero-sized returns nothing.
gb_internal bool x64_single_value_by_pointer(Type *rt) {
	if (rt == nullptr) return false;
	i64 sz = type_size_of(rt);
	if (sz == 0) return false;                      // direct empty aggregate
	if (!x64_abi_win64) {
		u8 cls[2];
		return x64_sysv_classify(rt, cls) == 0;
	}
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

// RBP-relative home of incoming ABI slot `slot`. Win64: EVERY slot has a caller-provided
// home (shadow space for slots 0-3, the stack arg itself beyond) at RBP+16+slot*8.
// SysV: looked up in the per-proc table x64_abi_home_params filled (register slots are
// callee-allocated frame locals; stack slots are the incoming args at RBP+16+k*8).
gb_internal gb_inline i32 x64_param_home_off(x64Procedure *p, int slot) {
	if (x64_abi_win64) return 16 + slot * 8;
	GB_ASSERT(p->abi_slot_home != nullptr && slot >= 0 && slot < p->total_param_slots);
	return p->abi_slot_home[slot];
}

// SysV: assign the next slot a physical home and return its RBP offset, emitting the
// register→frame spills when it arrives in registers (SysV callers provide no home).
// `t` is the slot's value type (null for pointer slots: sret / partial-ret / context —
// classified as one INTEGER eightbyte). Register values land in a callee-allocated
// local padded to 8·n8 (spills are always full eightbytes); MEMORY-class values map
// straight onto the caller's by-value stack copy (which the callee owns per SysV).
static i32 x64_sysv_home_slot(x64Procedure *p, x64SysVCursor *cur, Type *t) {
	X64Assembler *a = &p->asm_;

	u8  cls[2] = { X64_SYSV_INT, X64_SYSV_NONE };
	int n8 = 1;
	i64 sz = 8, al = 8;
	if (t != nullptr) {
		sz = type_size_of(t);
		al = type_align_of(t);
		n8 = x64_sysv_classify(t, cls);
	}

	int ridx[2];
	if (n8 > 0 && x64_sysv_take_regs(cur, cls, n8, ridx)) {
		i32 off = x64_alloc_local(p, 8 * n8, gb_max(al, (i64)8));
		for (int k = 0; k < n8; k++) {
			if (cls[k] == X64_SYSV_INT) {
				x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(off + 8*k), X64_SYSV_INT_ARG_REGS[ridx[k]]);
			} else {
				x64_emit_movsd_mr(a, x64_rbp_mem(off + 8*k), X64_SYSV_XMM_ARG_REGS[ridx[k]]);
			}
		}
		return off;
	}

	return 16 + x64_sysv_take_stack(cur, sz, al); // incoming stack arg / by-value copy
}

// SysV homing: walk the flat slot list with the register cursor, spilling register
// slots into callee-allocated frame locals (SysV callers provide NO home for them)
// and mapping stack slots to the incoming args at RBP+16+k*8. Fills abi_slot_home,
// which x64_param_home_off serves lookups from. Slot-count fields were already set
// by x64_abi_home_params' shared counting pass.
static void x64_abi_home_params_sysv(x64Procedure *p) {
	Type         *pt = p->type;
	x64SysVCursor cur = {};

	p->abi_slot_home = gb_alloc_array(p->alloc, i32, gb_max(p->total_param_slots, 1));

	int slot = 0;
	if (p->returns_by_pointer) {
		p->abi_slot_home[slot++] = x64_sysv_home_slot(p, &cur, nullptr); // sret ptr (RDI)
	}

	if (pt->Proc.params != nullptr) {
		TypeTuple *params = &pt->Proc.params->Tuple;
		for_array(i, params->variables) {
			Entity *e = params->variables[i];
			if (e->kind != Entity_Variable) continue;
			if (e->flags & EntityFlag_CVarArg) continue;

			// Zero-sized param: no data, no ABI slot (same as Win64).
			if (x64_type_size(e->type) == 0) {
				x64_var_set(&p->var_offsets, e, x64_alloc_local(p, 0, 1));
				continue;
			}

			// SysV has NO by-pointer params: register values were spilled into a frame
			// local, MEMORY-class values sit in the caller's by-value stack copy (owned
			// by the callee per SysV) — either way `off` addresses the DATA directly.
			i32 off = x64_sysv_home_slot(p, &cur, e->type);
			p->abi_slot_home[slot++] = off;
			x64_var_set(&p->var_offsets, e, off);
		}
	}

	// Partial-return pointers and the context pointer are plain GP slots.
	for (int k = 0; k < p->num_partial_rets; k++) {
		p->abi_slot_home[slot++] = x64_sysv_home_slot(p, &cur, nullptr);
	}
	if (p->has_context) {
		p->abi_slot_home[slot++] = x64_sysv_home_slot(p, &cur, nullptr);
	}
	GB_ASSERT(slot == p->total_param_slots);
}

// Assign incoming ABI slots and home register parameters into their slots' homes.
// Called by x64_proc_begin right after the frame prologue. Sets returns_by_pointer /
// has_context / context_slot / first_partial_ret_slot / total_param_slots and the
// var_offsets entry for every explicit parameter.
gb_internal void x64_abi_home_params(x64Procedure *p) {
	X64Assembler *a  = &p->asm_;
	Type         *pt = p->type;

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

	if (!x64_abi_win64) {
		x64_abi_home_params_sysv(p);
		return;
	}

	// ── Win64: home register parameters into shadow slots ────────────────
	int explicit_slot = first_explicit;

	if (p->returns_by_pointer && p->total_param_slots >= 1) {
		// Slot 0: hidden return pointer in RCX
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(x64_param_home_off(p, 0)), X64Reg_RCX);
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
			i32 off = x64_param_home_off(p, s);

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
			x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(x64_param_home_off(p, s)), X64_INT_ARG_REGS[s]);
		}
	}

	// Context pointer is in the last slot (if within reg range)
	if (p->has_context && p->context_slot < 4) {
		int s = p->context_slot;
		x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(x64_param_home_off(p, s)),
		                X64_INT_ARG_REGS[s]);
	}
}

// SysV caller side: classify every slot up front with the same cursor logic the callee
// uses (eightbyte classes, all-or-nothing register take), place MEMORY-class /
// overflow args as by-value copies in the outgoing area (at [RSP+0] — no shadow
// space), then fill registers in reverse — including 2-eightbyte GP/XMM pairs.
// `c_vararg`: SysV variadic ABI wants AL = number of XMM registers used — set LAST so
// no later value load can scratch RAX.
static void x64_abi_emit_call_args_sysv(x64Procedure *p, x64Value *args, int arg_count, bool c_vararg) {
	X64Assembler *a = &p->asm_;

	struct Loc {
		i64 sz;
		u8  n8;
		u8  cls[2];
		i8  ridx[2];
		bool in_reg;
		i32 stack_off;
	};
	Loc *L = gb_alloc_array(temporary_allocator(), Loc, gb_max(arg_count, 1));

	// Pass 1: classify (must match x64_abi_home_params_sysv's walk exactly).
	x64SysVCursor cur = {};
	for (int i = 0; i < arg_count; i++) {
		Type *t = args[i].type;
		i64 sz = (t != nullptr) ? type_size_of(t) : 8;
		i64 al = (t != nullptr) ? type_align_of(t) : 8;
		L[i] = {};
		L[i].sz = sz;
		if (sz <= 0) continue; // zero-sized: no placement (callee skips it too)

		u8  cls[2] = { X64_SYSV_INT, X64_SYSV_NONE };
		int n8 = 1;
		if (t != nullptr) n8 = x64_sysv_classify(t, cls);

		// A multi-eightbyte or MEMORY-class value that isn't in memory (e.g. an Imm
		// default) must be materialized NOW — passes 2/3 need a memory source for it,
		// and store_value's scratch registers must not run once argument registers are
		// loaded. Single-eightbyte non-Mem values (incl. ENUM/bit_set immediates, which
		// are not x64_is_scalar) stay as-is: the register/stack passes load them with
		// the ordinary value loaders, preserving the immediate's value.
		if (n8 != 1 && args[i].kind != x64Value_Mem) {
			i32 mo = x64_alloc_local(p, sz, gb_max(al, (i64)8));
			x64_store_value(p, x64addr(x64_rbp_mem(mo), t), args[i]);
			args[i] = x64v_mem(t, x64_rbp_mem(mo));
		}

		int ridx[2] = {0, 0};
		L[i].n8     = (u8)gb_max(n8, 1);
		L[i].cls[0] = cls[0];
		L[i].cls[1] = cls[1];
		L[i].in_reg = (n8 > 0) && x64_sysv_take_regs(&cur, cls, n8, ridx);
		if (L[i].in_reg) {
			L[i].ridx[0] = (i8)ridx[0];
			L[i].ridx[1] = (i8)ridx[1];
		} else {
			L[i].stack_off = x64_sysv_take_stack(&cur, sz, al);
		}
	}
	if (cur.stack_off > p->max_outgoing_bytes) p->max_outgoing_bytes = cur.stack_off;

	// Pass 2: stack args (RAX/R10/R11/XMM0 are free as scratch until the register pass).
	for (int i = arg_count - 1; i >= 0; i--) {
		if (L[i].sz <= 0 || L[i].in_reg) continue;
		x64Value v = args[i];
		X64Mem stack_slot = x64_mem(X64Reg_RSP, L[i].stack_off);
		if (v.kind == x64Value_Mem) {
			X64Mem src = v.mem;
			if (v.by_ref) { // slot holds a POINTER to the value — copy from behind it
				x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R11, v.mem);
				src = x64_mem(X64Reg_R11, 0);
			}
			// copy_fixed's REP path push/pops (shifts RSP) → give it a stable RAX-based dst.
			x64_emit_lea(a, X64Reg_RAX, stack_slot);
			x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), src, L[i].sz);
		} else if (v.type != nullptr && x64_arg_is_float(v.type)) {
			x64_value_to_xmm(p, v, X64XmmReg_XMM0);
			if (x64_is_double(v.type)) x64_emit_movsd_mr(a, stack_slot, X64XmmReg_XMM0);
			else                       x64_emit_movss_mr(a, stack_slot, X64XmmReg_XMM0);
		} else {
			x64_value_to_reg(p, v, X64Reg_RAX);
			x64_emit_mov_mr(a, X64OpSize_64, stack_slot, X64Reg_RAX);
		}
	}

	// Pass 3: register args, reverse order to avoid clobbering earlier loads.
	for (int i = arg_count - 1; i >= 0; i--) {
		if (L[i].sz <= 0 || !L[i].in_reg) continue;
		x64Value v = args[i];

		// Single-eightbyte scalars — and ANY single-eightbyte non-memory value (enum /
		// bit_set immediates are not x64_is_scalar but are plain integer payloads):
		// the existing loaders handle imm/reg/mem + widths.
		bool single = (L[i].n8 == 1) && !v.by_ref;
		if (single && (v.kind != x64Value_Mem || (v.type != nullptr && x64_is_scalar(v.type)))) {
			if (L[i].cls[0] == X64_SYSV_SSE) x64_value_to_xmm(p, v, X64_SYSV_XMM_ARG_REGS[L[i].ridx[0]]);
			else                             x64_value_to_reg(p, v, X64_SYSV_INT_ARG_REGS[L[i].ridx[0]]);
			continue;
		}

		// Aggregates (1–2 eightbytes): load from memory through a stable R11 base
		// (v.mem may be RIP-relative or based on a transient register). Full-8-byte
		// loads may overread a short tail — harmless, always within mapped frame/data.
		GB_ASSERT_MSG(v.kind == x64Value_Mem, "sysv: register-class aggregate arg must be a memory value");
		if (v.by_ref) x64_emit_mov_rm(a, X64OpSize_64, X64Reg_R11, v.mem);
		else          x64_emit_lea(a, X64Reg_R11, v.mem);
		for (int k = 0; k < (int)L[i].n8; k++) {
			if (L[i].cls[k] == X64_SYSV_INT) {
				x64_emit_mov_rm(a, X64OpSize_64, X64_SYSV_INT_ARG_REGS[L[i].ridx[k]], x64_mem(X64Reg_R11, 8*k));
			} else {
				x64_emit_movsd_rm(a, X64_SYSV_XMM_ARG_REGS[L[i].ridx[k]], x64_mem(X64Reg_R11, 8*k));
			}
		}
	}

	// SysV variadic: AL = upper bound of XMM registers holding args.
	if (c_vararg) {
		x64_emit_mov_ri(a, X64OpSize_32, X64Reg_RAX, cur.xmm);
	}
}

// Place the already-lowered slot values into their physical locations for an outgoing
// call (registers + outgoing stack area) and reserve the frame's outgoing bytes.
// `c_vararg`: Win64 variadic ABI — an FP arg in a register slot of a c_vararg callee
// must ALSO be placed in the corresponding GP register (the callee reads `...` from GP).
gb_internal void x64_abi_emit_call_args(x64Procedure *p, x64Value *args, int arg_count, bool c_vararg) {
	X64Assembler *a = &p->asm_;

	if (!x64_abi_win64) {
		x64_abi_emit_call_args_sysv(p, args, arg_count, c_vararg);
		return;
	}

	// Reserve enough outgoing stack-arg space for THIS call (args beyond the 4 register
	// slots); x64_proc_end sizes the frame's outgoing area from this peak.
	if (arg_count > 4) {
		i32 ob = (arg_count - 4) * 8;
		if (ob > p->max_outgoing_bytes) p->max_outgoing_bytes = ob;
	}

	// Stack args (slots 4+), reverse order
	for (int i = gb_max(arg_count - 1, 3); i >= 4; i--) {
		x64Value v = args[i];
		X64Mem stack_slot = x64_mem(X64Reg_RSP, X64_ABI_SHADOW_SPACE + (i - 4) * 8);
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
	int reg_count = gb_min(arg_count, 4);
	for (int i = reg_count - 1; i >= 0; i--) {
		x64Value v = args[i];
		if (x64_arg_is_float(v.type)) {
			x64_value_to_xmm(p, v, X64_XMM_ARG_REGS[i]);
			if (c_vararg) x64_value_to_reg(p, v, X64_INT_ARG_REGS[i]); // duplicate FP bits into GP
		} else {
			x64_value_to_reg(p, v, X64_INT_ARG_REGS[i]);
		}
	}
}

// ── Return values ────────────────────────────────────────────────────────────
// SysV return registers per eightbyte class, in order: INTEGER → RAX, RDX; SSE →
// XMM0, XMM1 (mixed pairs use one of each: {i64,f64} → RAX + XMM0).
static const X64Reg    X64_SYSV_INT_RET_REGS[2] = { X64Reg_RAX, X64Reg_RDX };
static const X64XmmReg X64_SYSV_XMM_RET_REGS[2] = { X64XmmReg_XMM0, X64XmmReg_XMM1 };

// Callee side: load the SysV return register(s) from the rt-typed value at `m`.
// Goes through a stable R11 base (m may be based on a soon-clobbered register).
// Full-8-byte tail loads may overread — harmless, always within mapped frame/data.
static void x64_sysv_ret_from_mem(x64Procedure *p, Type *rt, X64Mem m) {
	X64Assembler *a = &p->asm_;
	u8 cls[2]; int n8 = x64_sysv_classify(rt, cls);
	GB_ASSERT_MSG(n8 > 0, "sysv: MEMORY-class return must go through sret");
	x64_emit_lea(a, X64Reg_R11, m);
	int igp = 0, isse = 0;
	for (int k = 0; k < n8; k++) {
		if (cls[k] == X64_SYSV_INT) x64_emit_mov_rm(a, X64OpSize_64, X64_SYSV_INT_RET_REGS[igp++], x64_mem(X64Reg_R11, 8*k));
		else                        x64_emit_movsd_rm(a, X64_SYSV_XMM_RET_REGS[isse++], x64_mem(X64Reg_R11, 8*k));
	}
}

// Caller side: spill the SysV return register(s) of a just-emitted call into the
// rt-sized local at dst_off. Eightbyte stores are always full 8 bytes, so a short
// tail (12-byte struct) spills into a padded temp first, then copies the exact size
// — a direct store would overwrite the neighbouring local.
static void x64_sysv_spill_ret_regs(x64Procedure *p, Type *rt, i32 dst_off) {
	X64Assembler *a = &p->asm_;
	i64 sz = type_size_of(rt);
	u8 cls[2]; int n8 = x64_sysv_classify(rt, cls);
	GB_ASSERT_MSG(n8 > 0, "sysv: MEMORY-class return must go through sret");
	i32 tmp = (sz == 8*n8) ? dst_off : x64_alloc_local(p, 8*n8, 8);
	int igp = 0, isse = 0;
	for (int k = 0; k < n8; k++) {
		if (cls[k] == X64_SYSV_INT) x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(tmp + 8*k), X64_SYSV_INT_RET_REGS[igp++]);
		else                        x64_emit_movsd_mr(a, x64_rbp_mem(tmp + 8*k), X64_SYSV_XMM_RET_REGS[isse++]);
	}
	if (tmp != dst_off) x64_copy_fixed(p, x64_rbp_mem(dst_off), x64_rbp_mem(tmp), sz);
}

// The value a just-emitted call left in the return register(s), or none for
// void / sret / multi-result (those are reassembled by x64_reconstruct_call_result).
// SysV aggregate/i128 results come back in register PAIRS, which no x64Value can
// carry — they are spilled to a fresh padded local here, immediately after the call,
// and returned as a Mem value.
gb_internal x64Value x64_abi_direct_result(x64Procedure *p, Type *callee_type_raw) {
	Type *ct = base_type(callee_type_raw);
	Type *ret_type = nullptr;
	if (ct->Proc.results != nullptr && ct->Proc.result_count == 1) {
		ret_type = ct->Proc.results->Tuple.variables[0]->type;
	}
	if (x64_returns_by_pointer(callee_type_raw) || ct->Proc.result_count != 1) {
		return x64v_none();
	}
	if (ret_type == nullptr) return x64v_none();
	i64 sz = type_size_of(ret_type);
	if (sz == 0) return x64v_none(); // zero-sized: no value
	if (!x64_abi_win64 && !(x64_is_scalar(ret_type) && sz <= 8)) {
		u8 cls[2]; int n8 = x64_sysv_classify(ret_type, cls);
		GB_ASSERT(n8 > 0);
		i32 off = x64_alloc_local(p, 8*n8, gb_max(type_align_of(ret_type), (i64)8));
		x64_sysv_spill_ret_regs(p, ret_type, off); // padded local: direct full-8 stores fit
		return x64v_mem(ret_type, x64_rbp_mem(off));
	}
	if (x64_is_float(ret_type)) return x64v_xmm(ret_type, X64XmmReg_XMM0);
	return x64v_reg(ret_type, X64Reg_RAX);
}

// Place a computed value in the return register(s) for a direct (non-sret) result.
gb_internal void x64_abi_emit_return_value(x64Procedure *p, x64Value v, Type *rt) {
	if (!x64_abi_win64 && !(x64_is_scalar(rt) && type_size_of(rt) <= 8)) {
		// SysV pair return: load the return registers from memory. A non-Mem value
		// (e.g. an 8-byte aggregate currently in a register) is spilled first.
		if (v.kind != x64Value_Mem) {
			i64 sz = type_size_of(rt); if (sz <= 0) sz = 8;
			i64 al = type_align_of(rt); if (al <= 0) al = 8;
			i32 so = x64_alloc_local(p, sz, al);
			x64_store_value(p, x64addr(x64_rbp_mem(so), rt), v);
			v = x64v_mem(rt, x64_rbp_mem(so));
		}
		X64Mem m = v.mem;
		if (v.by_ref) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R11, v.mem);
			m = x64_mem(X64Reg_R11, 0);
		}
		x64_sysv_ret_from_mem(p, rt, m);
		return;
	}
	if (x64_is_float(rt)) x64_value_to_xmm(p, v, X64XmmReg_XMM0);
	else                  x64_value_to_reg(p, v, X64Reg_RAX);
}

// Load a direct (non-sret) result from its RBP-relative local into the return register(s).
gb_internal void x64_abi_emit_return_from_local(x64Procedure *p, Type *rt, i32 off) {
	if (!x64_abi_win64 && !(x64_is_scalar(rt) && type_size_of(rt) <= 8)) {
		x64_sysv_ret_from_mem(p, rt, x64_rbp_mem(off)); // SysV pair return
		return;
	}
	if (x64_is_float(rt)) {
		if (x64_is_double(rt)) x64_emit_movsd_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(off));
		else                    x64_emit_movss_rm(&p->asm_, X64XmmReg_XMM0, x64_rbp_mem(off));
	} else {
		x64_value_to_reg(p, x64v_mem(rt, x64_rbp_mem(off)), X64Reg_RAX);
	}
}

// Spill the return register(s) of a just-emitted call's direct result to an RBP local.
gb_internal void x64_abi_spill_direct_result(x64Procedure *p, Type *rt, i32 dst_off) {
	i64 sz = type_size_of(rt);
	if (sz <= 0) return;
	if (!x64_abi_win64 && !(x64_is_scalar(rt) && sz <= 8)) {
		x64_sysv_spill_ret_regs(p, rt, dst_off); // SysV pair return
		return;
	}
	if (x64_is_float(rt)) {
		if (x64_is_double(rt)) x64_emit_movsd_mr(&p->asm_, x64_rbp_mem(dst_off), X64XmmReg_XMM0);
		else                    x64_emit_movss_mr(&p->asm_, x64_rbp_mem(dst_off), X64XmmReg_XMM0);
	} else {
		X64OpSize os = (sz >= 8) ? X64OpSize_64 : (sz >= 4) ? X64OpSize_32
		             : (sz >= 2) ? X64OpSize_16 : X64OpSize_8;
		x64_emit_mov_mr(&p->asm_, os, x64_rbp_mem(dst_off), X64Reg_RAX);
	}
}
