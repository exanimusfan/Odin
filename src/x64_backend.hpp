// x64 fast debug backend — core data structures.
// Compiled only on Windows x64 (guarded in main.cpp).

#if !defined(X64_BACKEND_HPP)
#define X64_BACKEND_HPP

// ─────────────────────────────────────────────────────────────────────────────
// Value: the result of evaluating an expression
// ─────────────────────────────────────────────────────────────────────────────

enum x64ValueKind : u8 {
	x64Value_None   = 0,
	x64Value_Reg,     // integer / pointer in an integer register
	x64Value_XmmReg,  // float in an XMM register
	x64Value_Imm,     // integer immediate (fits i64)
	x64Value_Mem,     // value already lives at a computed memory address
};

struct x64Value {
	x64ValueKind kind;
	Type        *type;
	union {
		X64Reg    reg;
		X64XmmReg xmm;
		i64       imm;
		X64Mem    mem;
	};
	// Set only by x64_stabilize_value for a large indirect-ABI aggregate arg: `mem` holds a
	// POINTER to the value (the value itself is left in place, not copied). Consumed only by the
	// call-arg indirect lowering, which passes that pointer directly. False everywhere else.
	bool by_ref;
};

gb_internal gb_inline x64Value x64v_none(void) {
	return {};
}
gb_internal gb_inline x64Value x64v_imm(Type *t, i64 v) {
	x64Value r = {}; r.kind = x64Value_Imm; r.type = t; r.imm = v; return r;
}
gb_internal gb_inline x64Value x64v_reg(Type *t, X64Reg r_) {
	x64Value r = {}; r.kind = x64Value_Reg; r.type = t; r.reg = r_; return r;
}
gb_internal gb_inline x64Value x64v_xmm(Type *t, X64XmmReg x) {
	x64Value r = {}; r.kind = x64Value_XmmReg; r.type = t; r.xmm = x; return r;
}
gb_internal gb_inline x64Value x64v_mem(Type *t, X64Mem m) {
	x64Value r = {}; r.kind = x64Value_Mem; r.type = t; r.mem = m; return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Addr: materialised lvalue — the effective address of a named location
// ─────────────────────────────────────────────────────────────────────────────

struct x64Addr {
	X64Mem mem;
	Type  *type;
};

gb_internal gb_inline x64Addr x64addr(X64Mem mem, Type *t) {
	x64Addr a = {}; a.mem = mem; a.type = t; return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module: one COFF .obj file per AstPackage
// ─────────────────────────────────────────────────────────────────────────────

struct x64Generator;

// A compiler-generated procedure with no Entity/AST body — emitted by hand via x64
// emit primitives (mirrors LLVM's lb_hasher_proc_for_type / lb_equal_proc_generate_body).
// Queued (not generated inline) to avoid re-entering codegen on the shared temp arena.
enum X64SynthKind { X64Synth_Hasher, X64Synth_Equal };
struct X64SynthProc {
	X64SynthKind kind;
	Type        *type; // the key/value type the hasher/equal is FOR
};

struct x64Module {
	// Per-module bump arena backing ALL of this module's allocations. custom_arena=true
	// so the compile worker and the write worker (sequenced by thread_pool_wait) can both
	// allocate without the per-thread assertion. Never freed individually (reclaimed at
	// process exit) so the hot path avoids the global CRT heap lock that was serializing
	// the parallel workers. See x64-perf-coding-style.
	Arena         arena;
	gbAllocator   alloc;
	x64Generator *gen;
	AstPackage   *pkg;
	AstFile      *file; // non-null for per-file modules (runtime split); else whole package

	CoffWriter     coff;
	CoffSection   *text;
	CoffSection   *rdata;
	CoffSection   *data;
	CoffSection   *bss;
	CoffSection   *debug_s; // .debug$S — CodeView function name records
	CoffSection   *debug_t; // .debug$T — CodeView type records (LF_*)
	CoffSection   *pdata;   // .pdata — RUNTIME_FUNCTION entries (stack unwinding)
	CoffSection   *xdata;   // .xdata — UNWIND_INFO
	CoffSection   *tls;     // .tls$ — thread-local storage template (created lazily)
	i16            tls_secnum; // 1-based COFF section number of `tls` (0 until created)
	CoffSection   *drectve; // .drectve — linker directives (/INCLUDE, /EXPORT), created lazily

	// darwin TLV (created lazily, SysV targets only; mapped by the Mach-O writer to
	// __thread_data / __thread_bss / __thread_vars):
	CoffSection   *tdata;   // .tdata — const-initialized thread-local templates
	CoffSection   *tbss;    // .tbss  — zero-initialized thread-local storage
	CoffSection   *tlv;     // .tlv   — TLVDescriptor {getter, key, &init} per variable
	i16            tdata_secnum, tbss_secnum, tlv_secnum;

	// darwin DWARF (-debug): per-proc source-line records collected at proc_end and
	// serialized by x64_dwarf_finalize into .dwabb/.dwinf/.dwlin (mapped by the
	// Mach-O writer to __DWARF,__debug_abbrev/__debug_info/__debug_line; lldb reads
	// them from the .o files through ld64's debug map / dsymutil). Line entry code
	// offsets are PROC-relative; addresses bind via ADDR64 relocs to the proc symbol.
	struct DwFunc {
		String link_name;
		u32    code_size;
		i32    line_lo, line_hi; // index range [lo, hi) into dw_lines
		i32    decl_line;        // proc's source line (DW_AT_decl_line)
		i32    var_lo, var_hi;   // index range [lo, hi) into dw_vars
	};
	struct DwLine { u32 offset; u32 line; i32 file_id; };
	// A local/param: name @ [RBP + rbp_off], of Odin type `type` (resolved to a
	// DWARF type index at finalize). (SysV addresses MEMORY-class params in place, so
	// there are no by-pointer slots to special-case, unlike Win64/CodeView.)
	struct DwVar { String name; i32 rbp_off; Type *type; bool is_param; };
	Array<DwFunc> dw_funcs;
	Array<DwLine> dw_lines;
	Array<DwVar>  dw_vars;
	// Interned DWARF type descriptors → assigned .debug_info offset at finalize. kind:
	//   0 base type   — encoding + size + name
	//   1 pointer     — inner = pointee type index (0 = void*)
	//   2 generic word— unsigned u8[8] fallback for un-modelled aggregates
	//   3 struct      — name + size, members [mem_lo, mem_hi) in dw_members
	//   4 array       — inner = element type index, size = element count
	//   5 enum        — name + size, inner = underlying int type, enumerators [mem_lo, mem_hi) in dw_enums
	struct DwType { u8 kind; u8 encoding; u32 size; u32 inner; String name; i32 mem_lo, mem_hi; };
	struct DwMember { String name; u32 type; u32 offset; };
	struct DwEnum { String name; i64 value; }; // one enumerator (value may be negative / 64-bit)
	Array<DwType>       dw_types;
	Array<DwMember>     dw_members;
	Array<DwEnum>       dw_enums;
	PtrMap<Type *, u32> dw_type_cache; // Odin Type* → index into dw_types (+1; 0 = none)

	// CodeView type records: Type* → CV type index (>= 0x1000); builtins < 0x1000.
	PtrMap<Type *, u32> cv_types;
	u32 cv_next_type;
	// CodeView id records for #force_inline frames (S_INLINESITE.inlinee): callee → LF_FUNC_ID.
	PtrMap<Entity *, u32> cv_func_ids;
	Array<Entity *>       cv_inlinees;  // unique inlined callees → one DEBUG_S_INLINEELINES at finalize
	u32 cv_empty_arglist;  // LF_ARGLIST (0 args), 0 = not yet emitted
	u32 cv_void_proc_type; // LF_PROCEDURE (void()), 0 = not yet emitted

	// Float constant pool (deduplicated by bit pattern)
	Array<u64> rdata_f64_bits;
	Array<u32> rdata_f64_offs;
	Array<u32> rdata_f32_bits;
	Array<u32> rdata_f32_offs;

	// String constant counter — used to generate unique rdata symbol names
	u32 rdata_str_count;

	// Interning map for STATIC string-literal data (content → .rdata symbol name),
	// mirrors LLVM's m->const_strings: one rodata global per unique string.
	StringMap<String> const_strings;
	// Unique-name counter for x64_const_value backing globals (strings, slice backing
	// arrays, &CONST aggregates).
	u32 const_data_count;

	// Counter for compiler-generated anonymous globals (e.g. the static backing a
	// `&CompoundLit` in the startup runtime: `INT_ZERO := &Int{}`).
	u32 gen_global_count;

	// Root procedures to compile for THIS module (token-pos sorted, deterministic).
	// Compiled on a single worker so all writes to this module's COFF happen on one
	// thread (mirrors LLVM's per-module parallel codegen).
	Array<Entity *> compile_roots;

	// Per-module proc worklist (mirrors LLVM's m->procedures_to_generate). MPSC: any worker
	// may enqueue (a referenced min_dep==0 oncall proc goes to its OWNER module's queue; a
	// nested proc goes to its enclosing module's queue), but only this module's worker dequeues
	// — so every entry is compiled into THIS module by one thread. Drained to a fixpoint after
	// the roots pass. See x64-no-on-demand, x64-nested-proc-defer.
	MPSCQueue<Entity *> proc_queue;

	// Compiler-generated map hasher/equal procs to emit (see X64SynthProc). Drained to a
	// fixpoint alongside proc_queue; a struct/array hasher enqueues its field hashers here.
	// Dedup is by checking the proc symbol is already DEFINED (like x64_compile_procedure).
	MPSCQueue<X64SynthProc> synth_queue;
	// type → emitted Map_Info / Map_Cell_Info backing-global symbol name (emit once per type).
	PtrMap<Type *, String> map_info_map;
	PtrMap<Type *, String> map_cell_info_map;

	// On-demand worklist for GLOBALS referenced/read in this module's code. Mirrors LLVM's
	// lazy lb_find_value_from_entity: a global first referenced only by an on-demand
	// (min_dep==0) proc was skipped by the eager pass; record it and define it in its
	// owner module after the proc closure.
	Array<Entity *> onref_globals;

	// Phase 4 (parallel COFF serialize+write) results, set by the write worker.
	String obj_path;   // output .obj path ({} if the module was empty/skipped)
	bool   obj_failed; // serialization/write failed

	// CodeView line-number support (emitted into .debug$S).
	//   cv_strtab        — DEBUG_S_STRINGTABLE payload (offset 0 is a leading NUL)
	//   cv_filechksms    — DEBUG_S_FILECHKSMS payload (entries reference cv_strtab)
	//   cv_file_ids/offs — file_id → byte offset into cv_filechksms (parallel arrays)
	Array<u8>  cv_strtab;
	Array<u8>  cv_filechksms;
	Array<i32> cv_file_ids;
	Array<u32> cv_file_offs;
};

// ─────────────────────────────────────────────────────────────────────────────
// x64VarMap: arena-backed entity→RBP-offset map for a proc's locals/params.
//
// N is small, so a linear scan over a contiguous arena array beats a heap PtrMap:
// no CRT heap lock on the per-proc hot path, and cache-friendly. Reclaimed wholesale
// when the proc's scratch arena temp guard resets.
// ─────────────────────────────────────────────────────────────────────────────

struct x64VarMap {
	Entity     **keys;
	i32         *vals;
	i32          count;
	i32          cap;
	gbAllocator  alloc;
};

gb_internal gb_inline void x64_var_map_init(x64VarMap *vm, gbAllocator a, i32 cap) {
	vm->alloc = a;
	vm->count = 0;
	vm->cap   = cap;
	vm->keys  = cap ? gb_alloc_array(a, Entity *, cap) : nullptr;
	vm->vals  = cap ? gb_alloc_array(a, i32,      cap) : nullptr;
}

// Returns a pointer to the stored offset (consume immediately — a later set may
// grow/relocate the arrays), or nullptr if the entity is not present.
gb_internal gb_inline i32 *x64_var_get(x64VarMap *vm, Entity *e) {
	for (i32 i = 0; i < vm->count; i++) {
		if (vm->keys[i] == e) return &vm->vals[i];
	}
	return nullptr;
}

gb_internal gb_inline void x64_var_set(x64VarMap *vm, Entity *e, i32 off) {
	for (i32 i = 0; i < vm->count; i++) {
		if (vm->keys[i] == e) { vm->vals[i] = off; return; }
	}
	if (vm->count == vm->cap) {
		i32 new_cap = vm->cap ? vm->cap*2 : 16;
		Entity **nk = gb_alloc_array(vm->alloc, Entity *, new_cap);
		i32     *nv = gb_alloc_array(vm->alloc, i32,      new_cap);
		gb_memcopy(nk, vm->keys, cast(usize)vm->count*gb_size_of(Entity *));
		gb_memcopy(nv, vm->vals, cast(usize)vm->count*gb_size_of(i32));
		vm->keys = nk; vm->vals = nv; vm->cap = new_cap;
	}
	vm->keys[vm->count] = e;
	vm->vals[vm->count] = off;
	vm->count++;
}

// ─────────────────────────────────────────────────────────────────────────────
// x64RegCache: local register cache. Register-width NAMED locals are mirrored
// in callee-saved registers so repeated reads (and reads after a write-through
// store) hit a register instead of the stack slot. Memory stays authoritative —
// every write also stores to the slot — so debug info and aliasing stay
// correct. See x64_backend_regcache.cpp for the coherence rules.
// ─────────────────────────────────────────────────────────────────────────────

enum { X64_RC_GP = 5 }; // RBX, R12..R15 (callee-saved GP the rest of the backend never scratches)

struct x64RegSlot {
	i32   off;      // RBP offset of the named local held here (0 = empty; RBP+0 is never a local)
	Type *type;     // the local's type (drives load/store width)
	u32   age;      // LRU clock stamp
	bool  survives; // local's address is provably never taken → entry survives calls
};

// Arena-backed Entity* set (same rationale as x64VarMap: N is small, arena beats heap).
struct x64EntSet {
	Entity     **keys;
	i32          count;
	i32          cap;
	gbAllocator  alloc;
};

struct x64RegCache {
	x64RegSlot gp[X64_RC_GP];   // parallels x64_rc_gp[] (RBX,R12..R15)
	u8   pins[X64_RC_GP];       // in-flight operand holds: a pinned reg is never re-filled/evicted,
	                            // even after its ENTRY is invalidated (the x64Value still references
	                            // the register). Counted: `x + x` pins the same slot twice.
	i32  save_off[X64_RC_GP];   // RBP offset of each reg's spill slot (0 = not yet allocated)
	u32  clock;                 // LRU tick
	u32  merge_stamp;           // asm merge_epoch at which gp[] was last coherent (lazy flush)
	u32  call_stamp;            // asm call_epoch  at which non-surviving entries were last coherent
	u32  seq_stamp;             // proc named_seq: a new named local may REUSE a dropped local's
	                            // frame offset (block-scope slot reclaim) with different escape
	                            // status — drop everything when it changes
	i32  pend_off;              // one-shot write-through handshake, see x64_rc_note_store/x64_rc_write
	bool pend_survives;
	u8   used_gp;               // bitmask of gp[] ever populated (drives prologue save/restore + unwind)
	bool enabled;               // per-proc master switch
	bool scan_ok;               // escape prescan ran and understood the whole body (gates `survives`)
	x64EntSet escaped;          // locals whose address may be materialized (from the prescan)
};

#define X64_MAX_INLINE_DEPTH 8 // manual #force_inline nesting cap (x64_try_inline_call + escape prescan)

// ─────────────────────────────────────────────────────────────────────────────
// Procedure: per-procedure compilation state
// ─────────────────────────────────────────────────────────────────────────────

struct x64Procedure {
	x64Module *module;
	Entity    *entity;
	Type      *type;       // base_type(entity->type), always Type_Proc
	ProcInfo  *proc_info;
	String     link_name;
	gbAllocator alloc;     // per-proc scratch arena (== the arena backing all p->* arrays)

	X64Assembler asm_;

	// Frame layout — locals grow downward from RBP
	i32   local_size;         // current locals high-watermark (the allocation cursor)
	i32   frame_max;          // peak local_size ever reached → drives the prologue SUB RSP.
	                          // Lets callers reset local_size to reuse stack slots (e.g.
	                          // the per-global startup init loop) without under-reserving.
	i32   escape_floor;       // slots below this hold address-escaped temporaries (`&T{}` whose
	                          // pointer outlives the block) → statement- and block-scope reclaim
	                          // must never drop local_size below it (LLVM hoists these allocas to
	                          // function entry; x64 emulates by permanently reserving the slot).
	i32   max_outgoing_bytes; // peak outgoing stack-arg bytes over all calls (args beyond the
	                          // 4 register slots, ×8). Sizes the reserved outgoing area in the
	                          // frame; a hardcoded 64 under-reserved for calls with >12 args.
	isize sub_rsp_patch;      // byte offset of the imm32 inside the SUB RSP insn
	isize prologue_alloc_off; // start of the 13-byte stack-allocation region (patched
	                          // at proc_end to SUB RSP, or a __chkstk probe if >1 page)

	// Is the first incoming slot a hidden return-ptr?
	bool returns_by_pointer;

	// Compiler-generated (no Entity) map hasher/equal proc — emitted as a STATIC,
	// file-local symbol (mirrors LLVM internal linkage). Referenced only within its own
	// module (by the Map_Info global), so no cross-obj UNDEF ref needs it external.
	bool is_static;

	// Set while emitting global-variable initializers (the startup runtime body):
	// makes `&CompoundLit` allocate a STATIC global instead of a stack local so the
	// address escapes correctly (mirrors LLVM's lbProcedure::is_startup).
	bool is_startup;

	// Per-statement/expression state (mirrors lbProcedure::state_flags): accumulated through
	// x64_build_stmt/x64_build_expr from each node's `#no_bounds_check`/`#bounds_check` directive,
	// inheriting from the parent. Honoured by x64_bounds_check_disabled.
	u16 state_flags;

	// `fallthrough` target: the next case clause's BODY label for the switch case currently being
	// built (-1 outside any case). Saved/restored around each switch so nested switches don't leak.
	isize fallthrough_lbl;

	bool has_context; // proc uses the Odin context (ProcCC_Odin)
	i32  context_slot; // incoming ABI slot of the context pointer (-1 if none)
	// For NON-Odin procs that touch `context` (e.g. the `proc "c"` entry point that
	// runs `context = default_context()` then __entry_point): RBP offset of a generated
	// local Context, initialized once by runtime.__init_context. 0 = not yet created.
	// Mirrors lb_find_or_generate_context_ptr's generated-local path.
	i32  gen_context_off;

	// Current "scoped context" override: RBP offset of a local Context created by a
	// `context.field = X` / `context = X` assignment (mirrors LLVM lb_addr_store's
	// lbAddr_Context path: copy the current context into a fresh local, make it current,
	// store the field). 0 = none (use the param / generated context). Saved & restored
	// around block scopes so the modification is visible to the rest of the scope and
	// its callees but not the caller.
	i32  ctx_override_off;

	// Total incoming ABI parameter slots (ret_ptr + explicit + partial-rets + context).
	i32  total_param_slots;

	// SysV only: RBP-relative home of each incoming ABI slot, filled by
	// x64_abi_home_params (register slots get callee-allocated frame locals — SysV has
	// no caller-provided shadow space; stack slots point at the incoming stack arg).
	// Win64 leaves this null and uses the fixed RBP+16+slot*8 formula.
	i32 *abi_slot_home;

	// Split returns (mirrors LLVM): for an N-result proc the first N-1 results are
	// returned through hidden pointer args placed AFTER the explicit params and
	// BEFORE the context. These record where they sit and how many there are.
	i32  first_partial_ret_slot; // ABI slot of partial-return ptr #0 (-1 if none)
	i32  num_partial_rets;       // N-1 for N>1 results, else 0

	// entity → RBP-relative offset
	// Params: positive offsets (slot homes assigned by x64_abi_home_params)
	// Locals: negative offsets (alloc'd with x64_alloc_local)
	x64VarMap var_offsets;

	// Large indirect-ABI params (Win64 by-pointer aggregates) we DON'T copy into a
	// frame-local — their var_offsets slot holds the incoming POINTER and accesses
	// deref through it (mirrors LLVM's byval-immutable params). Avoids copying huge
	// by-value structs onto the stack (stack overflow). Small indirect params are
	// still copied (cheap), so only big ones land here.
	Array<Entity*> indirect_params;

	// Bumped whenever a SCOPE-LIVED slot is allocated: a named local (x64_alloc_var,
	// has a debug record) or a compiler-managed persistent local (the scoped/generated
	// `context` — x64_push_new_context / x64_ensure_local_context). The per-statement
	// temp-slot reclaimer (x64_build_stmt_list) only reuses a statement's stack slots
	// when this didn't change across it, so such slots are never reclaimed mid-scope;
	// only throwaway temps are.
	u32 named_seq;

	// Defer stack. Either a `defer <stmt>` (stmt != null) or a deferred procedure
	// call from a @(deferred_*) attribute (call_proc != null), e.g. sync.guard.
	struct DeferEntry {
		Ast      *stmt;
		Entity   *call_proc;
		Type     *call_type;
		x64Value *call_args;
		int       call_argc;
	};
	Array<DeferEntry> deferred;

	// Loop label stack (break/continue targets)
	struct LoopInfo {
		isize lbl_break;
		isize lbl_continue;
		Ast  *ast_label;  // may be null
		isize defer_base; // p->deferred.count at loop/switch entry (branch unwind floor)
	};
	Array<LoopInfo> loops;

	// CodeView line table: (code offset from proc start) → source line, in `file_id`'s file.
	// file_id varies within a proc when #force_inline bodies are attributed to the callee's
	// own source (emitted as separate DEBUG_S_LINES file blocks → step INTO the inlined code).
	struct LineEntry { u32 offset; u32 line; i32 file_id; };
	Array<LineEntry> lines;
	i32 cur_line;     // last recorded source line (for dedupe); -1 = none yet
	i32 cur_file_id;  // last recorded file (dedupe pairs with cur_line); -1 = none yet
	i32 file_id;      // source file this proc is declared in

	// Manual #force_inline: while emitting an inlined callee body, `return` stores into the
	// active frame's result slots and jumps to its join label instead of the real epilogue.
	// LLVM realises #force_inline via `alwaysinline` + the always-inliner pass (runs even at
	// -O0); we have no pass, so we inline at the call site here.
	struct InlineFrame {
		isize   join_label;     // return → jump here
		i32    *result_offs;    // result slot RBP offsets (nres)
		Type  **result_types;
		int     nres;
		isize   defer_base;     // inlined return runs defers down to here (not the fn's)
		Entity *entity;         // recursion guard
	};
	Array<InlineFrame> inline_frames;
	i32 inline_call_line; // outermost call-site line; the PRIMARY line table attributes inlined code
	                      // to it (so the caller frame shows where the inline was called)

	// CodeView S_INLINESITE: one record per inlined #force_inline call. The primary line table keeps
	// the call-site line; each site carries the CALLEE's own lines (binary annotations) + scoped
	// locals so the debugger shows a nested inline frame you can step into. `cur_inline_site` is the
	// active site during body emission (-1 = none); `parent` nests sites (a site inlined within a site).
	struct InlineSiteLine  { u32 offset; i32 line; i32 file_id; };
	struct InlineSiteLocal { Entity *entity; String name; i32 offset; Type *type; bool is_ptr; };
	struct InlineSiteRec {
		Entity *callee;
		i32     parent;       // index into inline_sites, -1 = directly under the GPROC
		i32     decl_file_id; // callee source file (base for ChangeFile)
		i32     decl_line;    // callee declaration line (base for line deltas)
		u32     code_start;   // code offset of the first inlined instruction (fallback if no lines)
		u32     code_end;     // code offset just past the site's last instruction
		Array<InlineSiteLine>  lines;
		Array<InlineSiteLocal> locals;
	};
	Array<InlineSiteRec> inline_sites;
	i32 cur_inline_site;

	// Local register cache. Cached locals live in callee-saved regs, so any that get
	// used are spilled once in the prologue and reloaded in every epilogue. The exact
	// set isn't known until the body is built, so a fixed-size NOP region is reserved at
	// each site and patched (with the real MOVs + UNWIND_CODE SAVE_NONVOLs) in proc_end.
	x64RegCache  regcache;
	isize        rc_save_region;      // code offset of the prologue spill region (-1 = none)
	Array<isize> rc_restore_regions;  // code offsets of each epilogue reload region
};

// ─────────────────────────────────────────────────────────────────────────────
// Generator: top-level; owns all modules and satisfies LinkerData
// ─────────────────────────────────────────────────────────────────────────────

struct x64Generator : LinkerData {
	CheckerInfo *info;
	gbAllocator  alloc;
	// Keyed by AstPackage* OR AstFile* (mirrors LLVM's gen->modules): one module
	// per package, plus per-file modules for base:runtime (and -module-per-file)
	// so the big runtime package parallelizes. Entity→module is file-first.
	PtrMap<void *, x64Module *> modules;
};

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations (implemented across the sub-files)
// ─────────────────────────────────────────────────────────────────────────────

// Type utilities
gb_internal i64        x64_type_size  (Type *t);
gb_internal i64        x64_type_align (Type *t);
gb_internal X64OpSize  x64_op_size_of (Type *t);
gb_internal bool       x64_is_float   (Type *t);
gb_internal bool       x64_is_double  (Type *t);
gb_internal bool       x64_is_integer (Type *t);
gb_internal bool       x64_is_bool    (Type *t);
gb_internal bool       x64_is_ptr     (Type *t);
gb_internal String     x64_get_entity_name(Entity *e);

// Procedure helpers
gb_internal i32        x64_alloc_local    (x64Procedure *p, i64 size, i64 align);
gb_internal x64Addr    x64_entity_addr    (x64Procedure *p, Entity *e);
gb_internal void       x64_store_value    (x64Procedure *p, x64Addr dst, x64Value src);
gb_internal x64Value   x64_load_addr      (x64Procedure *p, x64Addr addr);

// Local register cache — see x64_backend_regcache.cpp
gb_internal void       x64_rc_init            (x64Procedure *p);
gb_internal void       x64_rc_escape_prescan  (x64Procedure *p, Ast *body);      // fills regcache.escaped/scan_ok
gb_internal void       x64_rc_note_store      (x64Procedure *p, X64Mem dst, Type *t); // called from x64_store_value
gb_internal void       x64_rc_note_clobber    (x64Procedure *p, X64Mem dst, i64 size); // bulk writers (zero_mem/copy_fixed)
gb_internal void       x64_rc_write           (x64Procedure *p, X64Mem dst, Type *t, X64Reg src); // write-through refill
gb_internal x64Value   x64_rc_read            (x64Procedure *p, Entity *e, i32 off, Type *t);
gb_internal x64Value   x64_rc_operand         (x64Procedure *p, Entity *e, i32 off, Type *t); // read + pin
gb_internal void       x64_rc_unpin_value     (x64Procedure *p, x64Value v);
gb_internal bool       x64_rc_value_pinned    (x64Procedure *p, x64Value v);
gb_internal void       x64_rc_emit_saves      (x64Procedure *p);       // reserve prologue spill region
gb_internal void       x64_rc_reserve_restore (x64Procedure *p);       // reserve one epilogue reload region
gb_internal void       x64_rc_ensure_save_slots(x64Procedure *p);      // assign frame slots to used regs
gb_internal void       x64_rc_patch           (x64Procedure *p);       // fill the reserved regions
gb_internal int        x64_rc_used_count      (x64RegCache *rc);       // # cache regs actually used
// Conversion (mirrors lb_emit_conv): the single authority for value→type conversion. Every
// store/return/arg site routes through here so a conversion always actually converts.
gb_internal x64Value   x64_emit_conv      (x64Procedure *p, x64Value src, Type *from, Type *to);
// Bit reinterpret (mirrors lb_emit_transmute) — same size, no numeric conversion.
gb_internal x64Value   x64_emit_transmute (x64Procedure *p, x64Value value, Type *t);
// Binary arithmetic / comparison on pre-built operands (mirror lb_emit_arith / lb_emit_comp).
gb_internal x64Value   x64_emit_arith     (x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs, Type *type);
gb_internal x64Value   x64_emit_comp      (x64Procedure *p, TokenKind op, x64Value lhs, x64Value rhs);
// A type used as a value → its typeid (mirrors lb_typeid).
gb_internal x64Value   x64_typeid         (Type *type);
// Construct a union value (variant `src` → `union_type`) at `dst` (mirrors lb_emit_store_union_variant).
gb_internal void       x64_store_union_variant(x64Procedure *p, X64Mem dst, x64Value src, Type *union_type);
// Box `src` into an `any` {data, typeid} at `dst` (the to-`any` case of lb_emit_conv).
gb_internal void       x64_box_any        (x64Procedure *p, X64Mem dst, x64Value src, Type *src_type);

// Memory helpers (defined in stmt.cpp, used earlier in expr.cpp)
gb_internal void      x64_zero_mem(x64Procedure *p, X64Mem dst, i64 size);
gb_internal void      x64_copy_mem(x64Procedure *p, X64Mem dst, X64Mem src, i64 size);

// Global variable emission (defined in x64_backend.cpp, used in stmt.cpp for
// @(static) / @(thread_local) locals which are lowered to module globals).
gb_internal void      x64_emit_global_variable(x64Module *m, Entity *e);
gb_internal void      x64_emit_global_tls(x64Module *m, Entity *e, DeclInfo *d);
gb_internal void      x64_emit_global_static_value(x64Module *m, Entity *e, Ast *init_expr);
gb_internal bool      x64_global_has_const_init(DeclInfo *d, Entity *e);

// Anonymous EXTERNAL global (zero-init .bss) returning its link symbol. Used for
// `&CompoundLit` in the startup runtime.
gb_internal String    x64_add_global_generated(x64Module *m, Type *type);

gb_internal void      x64_cv_pad4(CoffSection *s); // CodeView .debug$S padding
// CodeView .debug$T type index for a type (emits LF_* records as needed).
gb_internal u32       x64_cv_type(x64Module *m, Type *t);

// ── Calling-convention layer (x64_abi.cpp) ──────────────────────────────────
// ALL knowledge of where args/returns/params physically live. Win64 and SysV
// (darwin/linux) implementations behind the same seam, selected by x64_abi_win64.
// Defined here (not x64_abi.cpp) so files compiled earlier in the unity build
// (x64_backend_type.cpp's TLS check) can read it. Set by x64_abi_init_target.
gb_internal bool x64_abi_win64 = true;
gb_internal bool      x64_arg_is_float(Type *t);
gb_internal bool      x64_arg_is_indirect(Type *t);          // arg SLOT holds a pointer (Win64 only)
gb_internal bool      x64_arg_stabilize_by_ref(Type *t);     // stabilize keeps address, not a copy
gb_internal bool      x64_single_value_by_pointer(Type *rt); // sret rule for one value
gb_internal bool      x64_returns_by_pointer(Type *proc_type);
gb_internal int       x64_num_partial_returns(Type *proc_type);
gb_internal Type     *x64_last_result_type(Type *proc_type);
gb_internal i32       x64_param_home_off(x64Procedure *p, int slot); // incoming slot's RBP home
gb_internal void      x64_abi_home_params(x64Procedure *p);          // slot assignment + homing
gb_internal void      x64_abi_emit_call_args(x64Procedure *p, x64Value *args, int arg_count, bool c_vararg);
gb_internal x64Value  x64_abi_direct_result(x64Procedure *p, Type *callee_type_raw); // value left in return reg(s); SysV pairs spilled to a local
gb_internal void      x64_abi_emit_return_value(x64Procedure *p, x64Value v, Type *rt);
gb_internal void      x64_abi_emit_return_from_local(x64Procedure *p, Type *rt, i32 off);
gb_internal void      x64_abi_spill_direct_result(x64Procedure *p, Type *rt, i32 dst_off);

// Loop/switch break+continue target lookup (defined in stmt.cpp, used by
// OrBranchExpr in expr.cpp which is compiled earlier in the unity build).
gb_internal x64Procedure::LoopInfo *x64_find_loop(x64Procedure *p, String label);

// Code generators
gb_internal void      x64_build_stmt(x64Procedure *p, Ast *stmt);
gb_internal void      x64_build_stmt_list(x64Procedure *p, Slice<Ast *> const &stmts);
// Per-statement builders (mirror LLVM's lb_build_*_stmt). Dispatched from x64_build_stmt; defined
// after it so they can recurse. Forward-declared here for that mutual recursion.
gb_internal void      x64_build_when_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_if_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_for_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_range_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_switch_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_type_switch_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_assign_stmt(x64Procedure *p, Ast *node);
gb_internal void      x64_build_value_decl(x64Procedure *p, Ast *node);
// Branch on `cond` to true_lbl/false_lbl with short-circuit recursion on &&/||/! (mirrors
// lb_build_cond). `true_is_fallthrough` = which target is the next instruction (so the leaf emits one
// conditional jump, no redundant jmp); the caller binds that label immediately after.
gb_internal void      x64_build_cond(x64Procedure *p, Ast *cond, isize true_lbl, isize false_lbl, bool true_is_fallthrough);
// `using`-promoted field address (mirrors lb_emit_deep_field_gep); used in x64_build_compound_lit
// (before its definition) and x64_build_addr.
gb_internal x64Addr   x64_emit_deep_field_gep(x64Procedure *p, X64Mem base_mem, Type *st, InternedString interned, bool allow_deref);
// Address + access width of a tagged union's discriminant (mirrors lb_emit_union_tag_ptr): tag lives
// at `variant_block_size`, width `union_tag_size`. Shared by x64_store_union_variant (store side) and
// x64_emit_union_tag_value (load side).
struct X64UnionTag { X64Mem mem; X64OpSize opsz; };
gb_internal X64UnionTag x64_emit_union_tag_ptr(Type *ubt, X64Mem union_mem);
// Load a tagged union's variant index (mirrors lb_emit_union_tag_value): tag @ variant_block_size,
// width union_tag_size, zero-extended into `dst`. `union_mem` is the union's base address. Shared by
// type-assert, type-switch, `union == nil`, and the map hasher.
gb_internal void      x64_emit_union_tag_value(x64Procedure *p, X64Mem union_mem, Type *ubt, X64Reg dst);
// Index bounds check (mirrors lb_emit_bounds_check); defined after x64_emit_conv, called from the
// IndexExpr/SliceExpr lvalue builders above it.
gb_internal void      x64_emit_bounds_check(x64Procedure *p, TokenPos pos, x64Value index, x64Value len);
gb_internal u16       x64_push_state_flags(x64Procedure *p, Ast *node); // fold node's #(no_)bounds_check into p->state_flags; returns prev
gb_internal bool      x64_try_inline_call(x64Procedure *p, AstCallExpr *ce, Entity *callee, Type *ct, x64Value *out);
gb_internal bool      x64_type_is_pointer_free(Type *t);
gb_internal x64Value  x64_build_expr(x64Procedure *p, Ast *expr);
gb_internal x64Addr   x64_build_addr(x64Procedure *p, Ast *expr);
gb_internal x64Value  x64_soa_index_element(x64Procedure *p, Ast *ie_expr, Type *elem_type, bool store, x64Value src); // #soa whole-element gather/scatter
gb_internal x64Value  x64_emit_logical_binary_expr(x64Procedure *p, TokenKind op, Ast *left, Ast *right, Type *type);
gb_internal i64       x64_fca_len_offset(Type *fca); // len-field offset of a [dynamic;N]E (NOT type_size-8 when E is over-aligned)
// CallExpr lowering, split out of x64_build_expr (mirrors lb_build_call_expr). Mutually
// recursive with x64_build_expr, hence the forward decl.
gb_internal x64Value  x64_build_call_expr(x64Procedure *p, Ast *expr);
// Materialise a default parameter / named-return-default value (mirrors lb_handle_param_value).
gb_internal x64Value  x64_handle_param_value(x64Procedure *p, Type *ptype, ParameterValue const &pv, Ast *call_expr);

gb_internal void      x64_compile_procedure (x64Module *m, Entity *e, Ast *body);
gb_internal void      x64_build_nested_proc (x64Procedure *p, Ast *proc_lit, Entity *e);
gb_internal void      x64_enqueue_oncall (x64Procedure *p, Entity *e);

// ── map[K]V support (defined in x64_backend_map.cpp) ─────────────────────────
// Runtime-call path (mirrors LLVM dynamic_map_calls): all ops route through the
// __dynamic_map_* runtime procs + a per-key-type Map_Info table.
gb_internal String   x64_gen_map_cell_info_ptr(x64Module *m, Type *type);  // → backing-global sym
gb_internal String   x64_gen_map_info_ptr     (x64Module *m, Type *map_type);
// hasher/equal synthetic-proc symbol name + enqueue for body emission (the type_*_proc intrinsics).
gb_internal String   x64_synth_proc_name(x64Module *m, X64SynthKind kind, Type *type);
gb_internal void     x64_enqueue_synth  (x64Module *m, X64SynthKind kind, Type *type);
gb_internal Entity  *x64_anon_proc_entity(x64Module *m, Ast *expr);
gb_internal x64Value x64_map_len  (x64Procedure *p, x64Value map_value);
gb_internal x64Value x64_map_cap  (x64Procedure *p, x64Value map_value);
gb_internal x64Value x64_map_data_uintptr(x64Procedure *p, x64Value map_value);
// get_ptr returns ^Map.value (nil if absent). map_ptr is ^Map (the map's address).
gb_internal x64Value x64_internal_dynamic_map_get_ptr(x64Procedure *p, x64Value map_ptr, Type *map_type, Ast *key_expr);
gb_internal void     x64_internal_dynamic_map_set(x64Procedure *p, x64Value map_ptr, Type *map_type, Ast *key_expr, x64Value value, Ast *node);
gb_internal void     x64_dynamic_map_reserve(x64Procedure *p, x64Value map_ptr, Type *map_type, i64 capacity);
// for-range helpers (called from stmt.cpp). cells_ptr is a uintptr value.
gb_internal x64Value x64_map_cell_index_static(x64Procedure *p, Type *elem_type, x64Value cells_ptr, x64Value index);
gb_internal x64Value x64_map_hash_is_valid(x64Procedure *p, x64Value hash);
// Drain m->synth_queue, emitting any not-yet-defined hasher/equal procs. Returns true
// if it generated at least one (so the fixpoint driver knows to loop again).
gb_internal bool     x64_generate_synth_procs(x64Module *m);
// `m[k]` read (result_type V or (V,bool) tuple) / `m[k] = v` write.
gb_internal x64Value x64_build_map_index_load (x64Procedure *p, Ast *map_expr, Ast *key_expr, Type *result_type);
gb_internal x64Value x64_build_map_index_ptr  (x64Procedure *p, Ast *map_expr, Ast *key_expr, Type *result_type); // &m[k]
gb_internal void     x64_build_map_index_store(x64Procedure *p, Ast *map_expr, Ast *key_expr, Ast *rhs_expr, Ast *node);
// Spilled ^map pointer for a map-typed lvalue expr (used by compound-assign `m[k] op= v`).
gb_internal x64Value x64_map_addr_of(x64Procedure *p, Ast *map_expr, Type *map_type);
// Runtime-call helper + a null Source_Code_Location arg (used by map and [dynamic] literals).
gb_internal x64Value x64_emit_runtime_call(x64Procedure *p, String name, x64Value *xargs, int n);
gb_internal x64Value x64_map_null_loc(x64Procedure *p);

// bit_field member access: detect `base.field` selecting a bit_field member, then read
// (shift/mask/sign-extend the backing) or write (read-modify-write the backing).
gb_internal bool     x64_bit_field_member_info(Ast *expr, Type **field_type, Type **backing_type, i64 *bit_offset, i64 *bit_size, i64 *byte_offset);
gb_internal x64Value x64_bit_field_load (x64Procedure *p, Ast *expr, Type *field_type, Type *backing_type, i64 bit_offset, i64 bit_size, i64 byte_offset);
gb_internal void     x64_bit_field_store(x64Procedure *p, Ast *expr, Ast *rhs_expr, Type *field_type, Type *backing_type, i64 bit_offset, i64 bit_size, i64 byte_offset);

// Entry point from main.cpp; returns the generator, or nullptr on failure.
gb_internal x64Generator *x64_generate_code(Checker *c);

#endif // X64_BACKEND_HPP
