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

	// CodeView type records: Type* → CV type index (>= 0x1000); builtins < 0x1000.
	PtrMap<Type *, u32> cv_types;
	u32 cv_next_type;

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
// Procedure: per-procedure compilation state
// ─────────────────────────────────────────────────────────────────────────────

struct x64Procedure {
	x64Module *module;
	Entity    *entity;
	Type      *type;       // base_type(entity->type), always Type_Proc
	ProcInfo  *proc_info;
	String     link_name;

	X64Assembler asm_;

	// Frame layout — locals grow downward from RBP
	i32   local_size;         // current locals high-watermark (the allocation cursor)
	i32   frame_max;          // peak local_size ever reached → drives the prologue SUB RSP.
	                          // Lets callers reset local_size to reuse stack slots (e.g.
	                          // the per-global startup init loop) without under-reserving.
	isize sub_rsp_patch;      // byte offset of the imm32 inside the SUB RSP insn
	isize prologue_alloc_off; // start of the 13-byte stack-allocation region (patched
	                          // at proc_end to SUB RSP, or a __chkstk probe if >1 page)

	// Is the first incoming slot a hidden return-ptr?
	bool returns_by_pointer;

	// Set while emitting global-variable initializers (the startup runtime body):
	// makes `&CompoundLit` allocate a STATIC global instead of a stack local so the
	// address escapes correctly (mirrors LLVM's lbProcedure::is_startup).
	bool is_startup;

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

	// Split returns (mirrors LLVM): for an N-result proc the first N-1 results are
	// returned through hidden pointer args placed AFTER the explicit params and
	// BEFORE the context. These record where they sit and how many there are.
	i32  first_partial_ret_slot; // ABI slot of partial-return ptr #0 (-1 if none)
	i32  num_partial_rets;       // N-1 for N>1 results, else 0

	// entity → RBP-relative offset
	// Params: positive offsets (param slot N lives at RBP + 16 + N*8)
	// Locals: negative offsets (alloc'd with x64_alloc_local)
	x64VarMap var_offsets;

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

	// CodeView line table: (code offset from proc start) → source line.
	struct LineEntry { u32 offset; u32 line; };
	Array<LineEntry> lines;
	i32 cur_line;   // last recorded source line (for dedupe); -1 = none yet
	i32 file_id;    // source file this proc is declared in
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
gb_internal i32        x64_param_rbp_off  (int slot); // → RBP + 16 + slot*8
gb_internal x64Addr    x64_entity_addr    (x64Procedure *p, Entity *e);
gb_internal void       x64_store_value    (x64Procedure *p, x64Addr dst, x64Value src);
gb_internal x64Value   x64_load_addr      (x64Procedure *p, x64Addr addr);
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

// Anonymous EXTERNAL global (zero-init .bss) returning its link symbol. Used for
// `&CompoundLit` in the startup runtime.
gb_internal String    x64_add_global_generated(x64Module *m, Type *type);

gb_internal void      x64_cv_pad4(CoffSection *s); // CodeView .debug$S padding
// CodeView .debug$T type index for a type (emits LF_* records as needed).
gb_internal u32       x64_cv_type(x64Module *m, Type *t);

// Does this proc type return its result via a hidden pointer? (Proc.return_by_pointer
// is never set for the x64 path; compute it ourselves.)
gb_internal bool      x64_returns_by_pointer(Type *proc_type);

// Loop/switch break+continue target lookup (defined in stmt.cpp, used by
// OrBranchExpr in expr.cpp which is compiled earlier in the unity build).
gb_internal x64Procedure::LoopInfo *x64_find_loop(x64Procedure *p, String label);

// Code generators
gb_internal void      x64_build_stmt(x64Procedure *p, Ast *stmt);
gb_internal x64Value  x64_build_expr(x64Procedure *p, Ast *expr);
gb_internal x64Addr   x64_build_addr(x64Procedure *p, Ast *expr);
// CallExpr lowering, split out of x64_build_expr (mirrors lb_build_call_expr). Mutually
// recursive with x64_build_expr, hence the forward decl.
gb_internal x64Value  x64_build_call_expr(x64Procedure *p, Ast *expr);

gb_internal void      x64_compile_procedure (x64Module *m, Entity *e, Ast *body);
gb_internal void      x64_build_nested_proc (x64Procedure *p, Ast *proc_lit, Entity *e);
gb_internal void      x64_enqueue_oncall (x64Procedure *p, Entity *e);

// Entry point from main.cpp; returns the generator, or nullptr on failure.
gb_internal x64Generator *x64_generate_code(Checker *c);

#endif // X64_BACKEND_HPP
