// x64_backend_map.cpp — map[K]V support.
//
// Mirrors the LLVM runtime-call path: every map op routes through the __dynamic_map_* runtime
// procs plus a per-key-type Map_Info {ks, vs, key_hasher, key_equal}. Hasher/equal procs are
// compiler-generated (no Odin AST), emitted by hand as STATIC file-local symbols (mirrors
// lb_hasher_proc_for_type / lb_equal_proc_generate_body).
//
// Unity build: AFTER x64_backend_expr.cpp, BEFORE x64_backend_stmt.cpp.

// ── small local helpers ─────────────────────────────────────────────────────

// Store an 8-byte value into a RBP local (via RAX).
gb_internal void x64_map_store_local(x64Procedure *p, i32 off, x64Value v) {
	x64_value_to_reg(p, v, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
}

// Spill an 8-byte value to a fresh RBP local and return a stable mem ref to it.
gb_internal x64Value x64_map_spill_u64(x64Procedure *p, x64Value v, Type *t) {
	i32 off = x64_alloc_local(p, 8, 8);
	x64_map_store_local(p, off, v);
	return x64v_mem(t, x64_rbp_mem(off));
}

// Call a runtime proc (enqueue if on-demand min_dep==0), lowering the full ABI: sret,
// partial-return pointers, context for "odin" cc. `xargs[0..n)` are the EXPLICIT params.
// Returns the single result (RAX), or none for multi-result procs.
gb_internal x64Value x64_emit_runtime_call(x64Procedure *p, String name, x64Value *xargs, int n) {
	AstPackage *rt = p->module->gen->info->runtime_package;
	Entity *e = (rt != nullptr) ? scope_lookup_current(rt->scope, string_interner_insert(name)) : nullptr;
	GB_ASSERT_MSG(e != nullptr && e->kind == Entity_Procedure, "x64: missing runtime proc %.*s", LIT(name));
	if (!e->Procedure.is_foreign && e->min_dep_count.load(std::memory_order_relaxed) == 0) {
		x64_enqueue_oncall(p, e);
	}

	Type *ct = base_type(e->type);
	bool needs_rbp = x64_returns_by_pointer(e->type);
	int  npartial  = x64_num_partial_returns(ct);
	bool needs_ctx = ct->Proc.calling_convention == ProcCC_Odin;

	int cap = (needs_rbp ? 1 : 0) + n + npartial + (needs_ctx ? 1 : 0);
	if (cap < 1) cap = 1;
	x64Value *args = gb_alloc_array(temporary_allocator(), x64Value, cap);
	int slot = 0;

	Type *last_rt = nullptr;
	if (ct->Proc.result_count >= 1) {
		last_rt = ct->Proc.results->Tuple.variables[ct->Proc.result_count-1]->type;
	}

	// sret pointer for the last result when returned by hidden pointer.
	if (needs_rbp) {
		i32 ro = x64_alloc_local(p, type_size_of(last_rt), type_align_of(last_rt));
		i32 rp = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(ro));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(rp), X64Reg_RAX);
		args[slot++] = x64v_mem(t_rawptr, x64_rbp_mem(rp));
	}
	for (int i = 0; i < n; i++) args[slot++] = xargs[i];
	// Partial-return pointers for results[0..N-2] (throwaway locals — ignored here).
	for (int i = 0; i < npartial; i++) {
		Type *pty = ct->Proc.results->Tuple.variables[i]->type;
		i64 psz = type_size_of(pty); if (psz <= 0) psz = 1;
		i64 pal = type_align_of(pty); if (pal <= 0) pal = 1;
		i32 po = x64_alloc_local(p, psz, pal);
		i32 pp = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(po));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(pp), X64Reg_RAX);
		args[slot++] = x64v_mem(t_rawptr, x64_rbp_mem(pp));
	}
	if (needs_ctx) args[slot++] = x64_context_ptr_value(p);

	return x64_emit_call(p, x64_get_entity_name(e), e->type, args, slot);
}

// Deterministic per-type symbol for a synthetic hasher/equal proc. Canonicalised (core_type
// for hashers, base_type for equal) so equivalent types share one proc and the Map_Info reloc
// agrees with the proc definition's name.
gb_internal String x64_synth_proc_name(x64Module *m, X64SynthKind kind, Type *type) {
	Type *t = (kind == X64Synth_Hasher) ? core_type(type) : base_type(type);
	u64 h = type_hash_canonical_type(t);
	gbString gs = gb_string_make(m->alloc, kind == X64Synth_Hasher ? "__$hasher$" : "__$equal$");
	gs = gb_string_append_fmt(gs, "%llu", cast(unsigned long long)h);
	return make_string(cast(u8 const *)gs, gb_string_length(gs));
}

gb_internal void x64_enqueue_synth(x64Module *m, X64SynthKind kind, Type *type) {
	X64SynthProc d = {kind, type};
	mpsc_enqueue(&m->synth_queue, d);
}

gb_internal void x64_synth_call_child_hasher_to_res(x64Procedure *p, Type *child, x64Value data, x64Value seed, i32 res_off);

// ── Raw_Map field extraction (mirrors lb_map_len / lb_map_cap / lb_map_data_uintptr) ─────
// `map_value` is the map struct IN MEMORY ({data: uintptr @0, len: int @8, ...}).

gb_internal x64Value x64_map_len(x64Procedure *p, x64Value map_value) {
	GB_ASSERT(map_value.kind == x64Value_Mem);
	X64Mem len_mem = map_value.mem; len_mem.disp += 8; // field 1
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, len_mem);
	return x64_map_spill_u64(p, x64v_reg(t_int, X64Reg_RAX), t_int);
}

gb_internal x64Value x64_map_data_uintptr(x64Procedure *p, x64Value map_value) {
	GB_ASSERT(map_value.kind == x64Value_Mem);
	// data & ~(MAP_CACHE_LINE_SIZE-1)  — strips the log2-capacity low bits.
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, map_value.mem); // field 0
	x64_emit_and_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, cast(i32)(~cast(i64)(MAP_CACHE_LINE_SIZE-1)));
	return x64_map_spill_u64(p, x64v_reg(t_uintptr, X64Reg_RAX), t_uintptr);
}

gb_internal x64Value x64_map_cap(x64Procedure *p, x64Value map_value) {
	GB_ASSERT(map_value.kind == x64Value_Mem);
	// cap = (data == 0) ? 0 : 1 << (data & (MAP_CACHE_LINE_SIZE-1))
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, map_value.mem); // data → RAX
	x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RAX);    // keep data in RDX
	x64_emit_mov_rr(&p->asm_, X64OpSize_64, X64Reg_RCX, X64Reg_RAX);
	x64_emit_and_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, cast(i32)(MAP_CACHE_LINE_SIZE-1)); // log2_cap → CL
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
	x64_emit_shl_rcl(&p->asm_, X64OpSize_64, X64Reg_RAX); // RAX = 1 << log2_cap
	// if data (RDX) == 0 → RAX = 0
	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RDX, X64Reg_RDX);
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, 0);
	x64_emit_cmov_rr(&p->asm_, X64Cc_E, X64OpSize_64, X64Reg_RAX, X64Reg_RCX); // data==0 → 0
	return x64_map_spill_u64(p, x64v_reg(t_int, X64Reg_RAX), t_int);
}

// ── Map_Cell_Info / Map_Info backing globals ─────────────────────────────────

// Emit (once per type) a private const Map_Cell_Info{size,align,cell_size,cell_len} into
// .rdata; returns its symbol name.
gb_internal String x64_gen_map_cell_info_ptr(x64Module *m, Type *type) {
	String *found = map_get(&m->map_cell_info_map, type);
	if (found != nullptr) return *found;

	i64 cell_size = 0, cell_len = 0;
	map_cell_size_and_len(type, &cell_size, &cell_len);

	CoffSection *rd = m->rdata;
	coff_section_align(rd, 8);
	u32 off = (u32)coff_section_len(rd);
	u64 vals[4] = {
		cast(u64)type_size_of(type),
		cast(u64)type_align_of(type),
		cast(u64)cell_size,
		cast(u64)cell_len,
	};
	coff_section_write(rd, vals, 32);

	gbString gs = gb_string_make(m->alloc, "__$map_cell_info$");
	gs = gb_string_append_fmt(gs, "%llu", cast(unsigned long long)type_hash_canonical_type(type));
	String name = make_string(cast(u8 const *)gs, gb_string_length(gs));
	coff_sym_add_proc(&m->coff, name, 2 /*rdata*/, off, false /*static*/);

	map_set(&m->map_cell_info_map, type, name);
	return name;
}

// Emit (once per map type) a private const Map_Info{ks, vs, key_hasher, key_equal} into .rdata
// (4 pointer fields → ADDR64 relocs), enqueue the key hasher+equal, and return its symbol name.
gb_internal String x64_gen_map_info_ptr(x64Module *m, Type *map_type) {
	map_type = base_type(map_type);
	GB_ASSERT(map_type->kind == Type_Map);

	String *found = map_get(&m->map_info_map, map_type);
	if (found != nullptr) return *found;

	String ks_sym = x64_gen_map_cell_info_ptr(m, map_type->Map.key);
	String vs_sym = x64_gen_map_cell_info_ptr(m, map_type->Map.value);
	String hasher = x64_synth_proc_name(m, X64Synth_Hasher, map_type->Map.key);
	String equal  = x64_synth_proc_name(m, X64Synth_Equal,  map_type->Map.key);
	x64_enqueue_synth(m, X64Synth_Hasher, map_type->Map.key);
	x64_enqueue_synth(m, X64Synth_Equal,  map_type->Map.key);

	CoffSection *rd = m->rdata;
	coff_section_align(rd, 8);
	u32 off = (u32)coff_section_len(rd);
	u64 zero[4] = {0,0,0,0};
	coff_section_write(rd, zero, 32);
	coff_reloc_add(rd, off + 0,  ks_sym, COFF_REL_ADDR64);
	coff_reloc_add(rd, off + 8,  vs_sym, COFF_REL_ADDR64);
	coff_reloc_add(rd, off + 16, hasher, COFF_REL_ADDR64);
	coff_reloc_add(rd, off + 24, equal,  COFF_REL_ADDR64);

	gbString gs = gb_string_make(m->alloc, "__$map_info$");
	gs = gb_string_append_fmt(gs, "%llu", cast(unsigned long long)type_hash_canonical_type(map_type));
	String name = make_string(cast(u8 const *)gs, gb_string_length(gs));
	coff_sym_add_proc(&m->coff, name, 2 /*rdata*/, off, false /*static*/);

	map_set(&m->map_info_map, map_type, name);
	return name;
}

// rawptr to a zeroed Source_Code_Location for the `loc` arg of the "odin"-cc map procs. The
// runtime only reads it on the alloc-failure panic path. TODO: emit real source locations.
gb_internal x64Value x64_map_null_loc(x64Procedure *p) {
	i64 sz = type_size_of(t_source_code_location); if (sz <= 0) sz = 8;
	i64 al = type_align_of(t_source_code_location); if (al <= 0) al = 8;
	i32 off = x64_alloc_local(p, sz, al);
	x64_zero_mem(p, x64_rbp_mem(off), sz);
	i32 po = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(po), X64Reg_RAX);
	return x64v_mem(t_rawptr, x64_rbp_mem(po));
}

// Build the key value from `key_expr`, convert to the map's key type, spill it, and return a
// rawptr value to the spilled key (mirrors the key_ptr in lb_gen_map_key_hash).
gb_internal x64Value x64_map_key_ptr(x64Procedure *p, Ast *key_expr, Type *key_type) {
	x64Value k = x64_build_expr(p, key_expr);
	k = x64_emit_conv(p, k, k.type, key_type);
	i64 sz = type_size_of(key_type); if (sz <= 0) sz = 1;
	i64 al = type_align_of(key_type); if (al <= 0) al = 1;
	i32 off = x64_alloc_local(p, sz, al);
	x64_store_value(p, x64addr(x64_rbp_mem(off), key_type), k);
	i32 po = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(off));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(po), X64Reg_RAX);
	return x64v_mem(t_rawptr, x64_rbp_mem(po));
}

// hash = key_hasher(key_ptr, map_seed_from_map_data(data & ~63)). Mirrors lb_gen_map_key_hash;
// always goes through the hasher proc (const-hash path is disabled in LLVM too).
gb_internal x64Value x64_gen_map_key_hash(x64Procedure *p, x64Value map_value, x64Value key_ptr, Type *map_type) {
	Type *key_type = base_type(map_type)->Map.key;
	x64Value data = x64_map_data_uintptr(p, map_value);
	x64Value sa[1]; sa[0] = data;
	x64Value seed = x64_emit_runtime_call(p, str_lit("map_seed_from_map_data"), sa, 1);
	seed = x64_map_spill_u64(p, seed, t_uintptr);

	String hname = x64_synth_proc_name(p->module, X64Synth_Hasher, key_type);
	x64_enqueue_synth(p->module, X64Synth_Hasher, key_type);
	x64Value ha[2]; ha[0] = key_ptr; ha[1] = seed;
	x64Value hash = x64_emit_call(p, hname, t_hasher_proc, ha, 2);
	return x64_map_spill_u64(p, hash, t_uintptr);
}

// Load *map_ptr (the map struct) into a fresh RBP local and return a stable mem ref to it.
gb_internal x64Value x64_map_load_struct(x64Procedure *p, x64Value map_ptr, Type *map_type) {
	i64 sz = type_size_of(map_type); if (sz <= 0) sz = 24;
	i64 al = type_align_of(map_type); if (al <= 0) al = 8;
	i32 off = x64_alloc_local(p, sz, al);
	x64_value_to_reg(p, map_ptr, X64Reg_RAX);
	x64_copy_mem(p, x64_rbp_mem(off), x64_mem(X64Reg_RAX, 0), sz);
	return x64v_mem(map_type, x64_rbp_mem(off));
}

// ── get / set / reserve (mirror lb_internal_dynamic_map_get_ptr / _set / lb_dynamic_map_reserve) ──

gb_internal x64Value x64_internal_dynamic_map_get_ptr(x64Procedure *p, x64Value map_ptr, Type *map_type, Ast *key_expr) {
	Type *mt = base_type(map_type);
	map_ptr = x64_map_spill_u64(p, map_ptr, alloc_type_pointer(map_type));
	x64Value map_value = x64_map_load_struct(p, map_ptr, mt);

	x64Value key_ptr = x64_map_key_ptr(p, key_expr, mt->Map.key);
	x64Value hash    = x64_gen_map_key_hash(p, map_value, key_ptr, mt);
	String info      = x64_gen_map_info_ptr(p->module, mt);

	i32 info_off = x64_alloc_local(p, 8, 8);
	x64_emit_lea_sym(&p->asm_, X64Reg_RAX, info);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(info_off), X64Reg_RAX);

	x64Value a[4];
	a[0] = map_ptr;                                       // ^Raw_Map
	a[1] = x64v_mem(t_rawptr, x64_rbp_mem(info_off));     // ^Map_Info
	a[2] = hash;                                          // uintptr
	a[3] = key_ptr;                                       // rawptr
	x64Value ptr = x64_emit_runtime_call(p, str_lit("__dynamic_map_get"), a, 4);
	return x64_map_spill_u64(p, ptr, alloc_type_pointer(mt->Map.value));
}

gb_internal void x64_internal_dynamic_map_set(x64Procedure *p, x64Value map_ptr, Type *map_type, Ast *key_expr, x64Value value, Ast *node) {
	gb_unused(node);
	Type *mt = base_type(map_type);
	map_ptr = x64_map_spill_u64(p, map_ptr, alloc_type_pointer(map_type));
	x64Value map_value = x64_map_load_struct(p, map_ptr, mt);

	x64Value key_ptr = x64_map_key_ptr(p, key_expr, mt->Map.key);
	x64Value hash    = x64_gen_map_key_hash(p, map_value, key_ptr, mt);

	// value → map value type, spilled; pass its address.
	value = x64_emit_conv(p, value, value.type, mt->Map.value);
	i64 vsz = type_size_of(mt->Map.value); if (vsz <= 0) vsz = 1;
	i64 val_al = type_align_of(mt->Map.value); if (val_al <= 0) val_al = 1;
	i32 voff = x64_alloc_local(p, vsz, val_al);
	x64_store_value(p, x64addr(x64_rbp_mem(voff), mt->Map.value), value);
	i32 vptr = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(voff));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(vptr), X64Reg_RAX);

	String info = x64_gen_map_info_ptr(p->module, mt);
	i32 info_off = x64_alloc_local(p, 8, 8);
	x64_emit_lea_sym(&p->asm_, X64Reg_RAX, info);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(info_off), X64Reg_RAX);

	x64Value a[6];
	a[0] = map_ptr;
	a[1] = x64v_mem(t_rawptr, x64_rbp_mem(info_off));
	a[2] = hash;
	a[3] = key_ptr;
	a[4] = x64v_mem(t_rawptr, x64_rbp_mem(vptr));
	a[5] = x64_map_null_loc(p);
	x64_emit_runtime_call(p, str_lit("__dynamic_map_set"), a, 6);
}

gb_internal void x64_dynamic_map_reserve(x64Procedure *p, x64Value map_ptr, Type *map_type, i64 capacity) {
	Type *mt = base_type(map_type);
	map_ptr = x64_map_spill_u64(p, map_ptr, alloc_type_pointer(map_type));
	String info = x64_gen_map_info_ptr(p->module, mt);
	i32 info_off = x64_alloc_local(p, 8, 8);
	x64_emit_lea_sym(&p->asm_, X64Reg_RAX, info);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(info_off), X64Reg_RAX);

	x64Value a[4];
	a[0] = map_ptr;
	a[1] = x64v_mem(t_rawptr, x64_rbp_mem(info_off));
	a[2] = x64v_imm(t_uint, capacity);
	a[3] = x64_map_null_loc(p);
	x64_emit_runtime_call(p, str_lit("__dynamic_map_reserve"), a, 4);
}

// ── for-range cell addressing (mirror lb_map_cell_index_static / lb_map_hash_is_valid) ───
// cells_ptr is a uintptr value pointing at the start of an element-cell array; returns ^elem.

gb_internal x64Value x64_map_cell_index_static(x64Procedure *p, Type *elem_type, x64Value cells_ptr, x64Value index) {
	i64 cell_size = 0, cell_len = 0;
	i64 elem_sz = type_size_of(elem_type);
	map_cell_size_and_len(elem_type, &cell_size, &cell_len);

	cells_ptr = x64_map_spill_u64(p, cells_ptr, t_uintptr);
	index     = x64_map_spill_u64(p, index, t_uintptr);

	if (cell_size == cell_len*elem_sz) {
		// No padding: cells_ptr + index*elem_sz
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, index.mem);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, elem_sz);
		x64_emit_imul_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64_emit_add_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, cells_ptr.mem);
		return x64_map_spill_u64(p, x64v_reg(alloc_type_pointer(elem_type), X64Reg_RAX), alloc_type_pointer(elem_type));
	}

	// Padded cells: cell_index = index / cell_len ; data_index = index % cell_len
	//   elems_ptr = cells_ptr + cell_index*cell_size ; result = elems_ptr + data_index*elem_sz
	x64Value cell_index, data_index;
	if (is_power_of_two(cell_len)) {
		u64 log2_len = floor_log2(cast(u64)cell_len);
		x64Value sh = x64v_imm(t_uintptr, cast(i64)log2_len);
		cell_index = (log2_len == 0) ? index : x64_emit_arith(p, Token_Shr, index, sh, t_uintptr);
		cell_index = x64_map_spill_u64(p, cell_index, t_uintptr);
		data_index = x64_emit_arith(p, Token_And, index, x64v_imm(t_uintptr, cell_len-1), t_uintptr);
		data_index = x64_map_spill_u64(p, data_index, t_uintptr);
	} else {
		cell_index = x64_emit_arith(p, Token_Quo, index, x64v_imm(t_uintptr, cell_len), t_uintptr);
		cell_index = x64_map_spill_u64(p, cell_index, t_uintptr);
		data_index = x64_emit_arith(p, Token_Mod, index, x64v_imm(t_uintptr, cell_len), t_uintptr);
		data_index = x64_map_spill_u64(p, data_index, t_uintptr);
	}

	x64Value cell_off = x64_emit_arith(p, Token_Mul, cell_index, x64v_imm(t_uintptr, cell_size), t_uintptr);
	cell_off = x64_map_spill_u64(p, cell_off, t_uintptr);
	x64Value base = x64_emit_arith(p, Token_Add, cells_ptr, cell_off, t_uintptr);
	base = x64_map_spill_u64(p, base, t_uintptr);
	x64Value data_off = x64_emit_arith(p, Token_Mul, data_index, x64v_imm(t_uintptr, elem_sz), t_uintptr);
	data_off = x64_map_spill_u64(p, data_off, t_uintptr);
	x64Value res = x64_emit_arith(p, Token_Add, base, data_off, t_uintptr);
	return x64_map_spill_u64(p, res, alloc_type_pointer(elem_type));
}

gb_internal x64Value x64_map_hash_is_valid(x64Procedure *p, x64Value hash) {
	// (hash != 0) & ((hash >> (bits-1)) == 0)  — non-empty and non-tombstone.
	hash = x64_map_spill_u64(p, hash, t_uintptr);
	u64 top = cast(u64)(type_size_of(t_uintptr)*8 - 1);

	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, hash.mem);
	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
	x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RCX);        // CL = (hash != 0)
	x64_emit_shr_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, cast(u8)top); // RAX = hash >> (bits-1)
	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
	x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RAX);         // AL = (tombstone bit == 0)
	x64_emit_and_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RCX);
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	return x64_map_spill_u64(p, x64v_reg(t_bool, X64Reg_RAX), t_bool);
}

// ── synthetic hasher / equal proc bodies ─────────────────────────────────────

// Address of param `slot` (a rawptr) loaded into RAX; returns a pointer-in-local mem ref.
gb_internal x64Value x64_synth_ptr_param(x64Procedure *p, int slot) {
	return x64v_mem(t_rawptr, x64_rbp_mem(x64_param_rbp_off(slot)));
}

// Compute (param0 data ptr) + offset → a stable pointer local.
gb_internal x64Value x64_synth_field_ptr(x64Procedure *p, int data_slot, i64 offset) {
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(data_slot)));
	if (offset != 0) x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, cast(i32)offset);
	return x64_map_spill_u64(p, x64v_reg(t_rawptr, X64Reg_RAX), t_rawptr);
}

// Call a child hasher: seed = __$hasher$child(elem_ptr, seed); store back to seed_off.
gb_internal void x64_synth_call_child_hasher(x64Procedure *p, Type *child, x64Value elem_ptr, i32 seed_off) {
	String hn = x64_synth_proc_name(p->module, X64Synth_Hasher, child);
	x64_enqueue_synth(p->module, X64Synth_Hasher, child);
	x64Value a[2];
	a[0] = elem_ptr;
	a[1] = x64v_mem(t_uintptr, x64_rbp_mem(seed_off));
	x64Value r = x64_emit_call(p, hn, t_hasher_proc, a, 2);
	x64_map_store_local(p, seed_off, r);
}

// Emit the hasher body: result (uintptr) stored into res_off. Mirrors lb_hasher_proc_for_type.
gb_internal void x64_emit_hasher_body(x64Procedure *p, Type *type, i32 res_off) {
	Type *t = core_type(type);
	x64Value data = x64_synth_ptr_param(p, 0);
	x64Value seed = x64v_mem(t_uintptr, x64_rbp_mem(x64_param_rbp_off(1)));

	if (is_type_simple_compare(t)) {
		x64Value a[3]; a[0] = data; a[1] = seed; a[2] = x64v_imm(t_int, type_size_of(t));
		x64Value r = x64_emit_runtime_call(p, str_lit("default_hasher"), a, 3);
		x64_map_store_local(p, res_off, r);
		return;
	}
	if (is_type_cstring(t)) {
		x64Value a[2]; a[0] = data; a[1] = seed;
		x64_map_store_local(p, res_off, x64_emit_runtime_call(p, str_lit("default_hasher_cstring"), a, 2));
		return;
	}
	if (is_type_string(t)) {
		x64Value a[2]; a[0] = data; a[1] = seed;
		x64_map_store_local(p, res_off, x64_emit_runtime_call(p, str_lit("default_hasher_string"), a, 2));
		return;
	}
	if (is_type_float(t)) {
		// load *data into a local, widen to f64, default_hasher_f64(v, seed)
		i64 fsz = type_size_of(t); if (fsz <= 0) fsz = 4;
		i32 lo = x64_alloc_local(p, fsz, fsz);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
		x64_copy_mem(p, x64_rbp_mem(lo), x64_mem(X64Reg_RAX, 0), fsz);
		x64Value v = x64_emit_conv(p, x64v_mem(t, x64_rbp_mem(lo)), t, t_f64);
		v = x64_map_spill_u64(p, v, t_f64);
		x64Value a[2]; a[0] = v; a[1] = seed;
		x64_map_store_local(p, res_off, x64_emit_runtime_call(p, str_lit("default_hasher_f64"), a, 2));
		return;
	}
	if (is_type_struct(t)) {
		type_set_offsets(t);
		i32 seed_off = x64_alloc_local(p, 8, 8);
		x64_map_store_local(p, seed_off, seed);
		for_array(i, t->Struct.fields) {
			Entity *field = t->Struct.fields[i];
			i64 offset = t->Struct.offsets[i];
			x64Value fp = x64_synth_field_ptr(p, 0, offset);
			x64_synth_call_child_hasher(p, field->type, fp, seed_off);
		}
		x64_map_store_local(p, res_off, x64v_mem(t_uintptr, x64_rbp_mem(seed_off)));
		return;
	}
	if (t->kind == Type_Array || t->kind == Type_EnumeratedArray) {
		Type *elem = (t->kind == Type_Array) ? t->Array.elem : t->EnumeratedArray.elem;
		i64 count  = (t->kind == Type_Array) ? t->Array.count : t->EnumeratedArray.count;
		i64 esz    = type_size_of(elem);
		i32 seed_off = x64_alloc_local(p, 8, 8);
		x64_map_store_local(p, seed_off, seed);
		i32 idx_off = x64_alloc_local(p, 8, 8);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 0);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(idx_off), X64Reg_RAX);
		isize top = x64_label_alloc(&p->asm_);
		isize end = x64_label_alloc(&p->asm_);
		x64_label_bind(&p->asm_, top);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(idx_off));
		x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, cast(i32)count);
		x64_emit_jcc(&p->asm_, X64Cc_AE, end);
		// elem_ptr = data + idx*esz
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(idx_off));
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, esz);
		x64_emit_imul_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RCX);
		x64_emit_add_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
		x64Value ep = x64_map_spill_u64(p, x64v_reg(t_rawptr, X64Reg_RAX), t_rawptr);
		x64_synth_call_child_hasher(p, elem, ep, seed_off);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(idx_off));
		x64_emit_add_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(idx_off), X64Reg_RAX);
		x64_emit_jmp(&p->asm_, top);
		x64_label_bind(&p->asm_, end);
		x64_map_store_local(p, res_off, x64v_mem(t_uintptr, x64_rbp_mem(seed_off)));
		return;
	}
	if (t->kind == Type_Union) {
		if (is_type_union_maybe_pointer(t)) {
			x64_synth_call_child_hasher_to_res(p, t->Union.variants[0], data, seed, res_off);
			return;
		}
		// default result = seed (no variant set)
		x64_map_store_local(p, res_off, seed);
		isize end = x64_label_alloc(&p->asm_);
		for (Type *v : t->Union.variants) {
			i64 tag_val = union_variant_index_checked(t, v);
			isize next = x64_label_alloc(&p->asm_);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0))); // data ptr
			x64_emit_union_tag_value(p, x64_mem(X64Reg_RAX, 0), t, X64Reg_RCX); // tag
			x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_RCX, cast(i32)tag_val);
			x64_emit_jcc(&p->asm_, X64Cc_NE, next);
			x64_synth_call_child_hasher_to_res(p, v, data, seed, res_off);
			x64_emit_jmp(&p->asm_, end);
			x64_label_bind(&p->asm_, next);
		}
		x64_label_bind(&p->asm_, end);
		return;
	}

	GB_PANIC("x64: unhandled hasher type %s", type_to_string(type));
}

// variant hasher whose result goes straight to res_off (union cases).
gb_internal void x64_synth_call_child_hasher_to_res(x64Procedure *p, Type *child, x64Value data, x64Value seed, i32 res_off) {
	String hn = x64_synth_proc_name(p->module, X64Synth_Hasher, child);
	x64_enqueue_synth(p->module, X64Synth_Hasher, child);
	x64Value a[2]; a[0] = data; a[1] = seed;
	x64Value r = x64_emit_call(p, hn, t_hasher_proc, a, 2);
	x64_map_store_local(p, res_off, r);
}

// Emit the equal body: bool result (0/1) stored into res_off. Mirrors lb_equal_proc_generate_body.
gb_internal void x64_emit_equal_body(x64Procedure *p, Type *type, i32 res_off) {
	Type *t = base_type(type);
	isize end = x64_label_alloc(&p->asm_);

	// Fast path: identical pointers → equal.
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
	x64_emit_cmp_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(1)));
	isize diff = x64_label_alloc(&p->asm_);
	x64_emit_jcc(&p->asm_, X64Cc_NE, diff);
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
	x64_emit_jmp(&p->asm_, end);
	x64_label_bind(&p->asm_, diff);

	if (is_type_union(t) && !is_type_union_maybe_pointer(t)) {
		// Compare tags; differ → unequal, equal-and-nil → equal. Otherwise dispatch on the tag and
		// compare the active variant BY CONTENT (a flat byte compare would compare string headers).
		bool no_nil = t->Union.kind == UnionType_no_nil;
		i64 tag_sz = union_tag_size(t);
		X64OpSize tsz = tag_sz <= 1 ? X64OpSize_8 : (tag_sz == 2 ? X64OpSize_16 : (tag_sz == 4 ? X64OpSize_32 : X64OpSize_64));
		i32 tag_off = cast(i32)t->Union.variant_block_size;

		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
		if (tsz == X64OpSize_64) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_mem(X64Reg_RAX, tag_off));
		} else if (tsz == X64OpSize_32) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_32, X64Reg_R8, x64_mem(X64Reg_RAX, tag_off));
		} else {
			x64_emit_movzx_rm(&p->asm_, tsz, X64Reg_R8, x64_mem(X64Reg_RAX, tag_off));
		}
		i32 ltag = x64_alloc_local(p, 8, 8);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ltag), X64Reg_R8);

		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(1)));
		if (tsz == X64OpSize_64) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R9, x64_mem(X64Reg_RAX, tag_off));
		} else if (tsz == X64OpSize_32) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_32, X64Reg_R9, x64_mem(X64Reg_RAX, tag_off));
		} else {
			x64_emit_movzx_rm(&p->asm_, tsz, X64Reg_R9, x64_mem(X64Reg_RAX, tag_off));
		}
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ltag));
		x64_emit_cmp_rr(&p->asm_, X64OpSize_64, X64Reg_R8, X64Reg_R9);
		isize tags_ne = x64_label_alloc(&p->asm_);
		x64_emit_jcc(&p->asm_, X64Cc_NE, tags_ne);

		// tags equal → default equal (also the nil-tag answer).
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
		if (!no_nil) {
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ltag));
			x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_R8, X64Reg_R8);
			x64_emit_jcc(&p->asm_, X64Cc_E, end);
		}

		for_array(i, t->Union.variants) {
			Type *vt = t->Union.variants[i];
			i64 tagv = no_nil ? i : (i + 1);
			i64 vsz = type_size_of(vt); if (vsz <= 0) vsz = 1;
			i64 val = type_align_of(vt); if (val <= 0) val = 1;
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_R8, x64_rbp_mem(ltag));
			x64_emit_cmp_ri(&p->asm_, X64OpSize_64, X64Reg_R8, cast(i32)tagv);
			isize nxt = x64_label_alloc(&p->asm_);
			x64_emit_jcc(&p->asm_, X64Cc_NE, nxt);
			i32 lv = x64_alloc_local(p, vsz, val);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
			x64_copy_mem(p, x64_rbp_mem(lv), x64_mem(X64Reg_RAX, 0), vsz);
			i32 rv = x64_alloc_local(p, vsz, val);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(1)));
			x64_copy_mem(p, x64_rbp_mem(rv), x64_mem(X64Reg_RAX, 0), vsz);
			x64Value eq = x64_emit_comp(p, Token_CmpEq, x64v_mem(vt, x64_rbp_mem(lv)), x64v_mem(vt, x64_rbp_mem(rv)));
			x64_value_to_reg(p, eq, X64Reg_RAX);
			x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
			x64_emit_jmp(&p->asm_, end);
			x64_label_bind(&p->asm_, nxt);
		}
		x64_emit_jmp(&p->asm_, end);

		x64_label_bind(&p->asm_, tags_ne);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 0);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
		x64_label_bind(&p->asm_, end);
		return;
	}

	if (is_type_struct(t)) {
		type_set_offsets(t);
		// result = 1; for each field: if !equal → result = 0, jump end.
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 1);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
		for_array(i, t->Struct.fields) {
			Entity *field = t->Struct.fields[i];
			i64 off = t->Struct.offsets[i];
			Type *ft = field->type;
			i64 fsz = type_size_of(ft); if (fsz <= 0) fsz = 1;
			i64 fal = type_align_of(ft); if (fal <= 0) fal = 1;
			// load lhs.field, rhs.field into locals
			i32 la = x64_alloc_local(p, fsz, fal);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
			x64_copy_mem(p, x64_rbp_mem(la), x64_mem(X64Reg_RAX, cast(i32)off), fsz);
			i32 ra = x64_alloc_local(p, fsz, fal);
			x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(1)));
			x64_copy_mem(p, x64_rbp_mem(ra), x64_mem(X64Reg_RAX, cast(i32)off), fsz);
			x64Value eq = x64_emit_comp(p, Token_CmpEq, x64v_mem(ft, x64_rbp_mem(la)), x64v_mem(ft, x64_rbp_mem(ra)));
			x64_value_to_reg(p, eq, X64Reg_RAX);
			x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
			isize next = x64_label_alloc(&p->asm_);
			x64_emit_jcc(&p->asm_, X64Cc_NE, next); // equal → keep checking
			x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 0);
			x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
			x64_emit_jmp(&p->asm_, end);
			x64_label_bind(&p->asm_, next);
		}
		x64_label_bind(&p->asm_, end);
		return;
	}

	// Scalar / simple-aggregate: load *lhs and *rhs (T-sized), compare for equality.
	i64 sz = type_size_of(t); if (sz <= 0) sz = 1;
	i64 al = type_align_of(t); if (al <= 0) al = 1;
	i32 la = x64_alloc_local(p, sz, al);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(0)));
	x64_copy_mem(p, x64_rbp_mem(la), x64_mem(X64Reg_RAX, 0), sz);
	i32 ra = x64_alloc_local(p, sz, al);
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(x64_param_rbp_off(1)));
	x64_copy_mem(p, x64_rbp_mem(ra), x64_mem(X64Reg_RAX, 0), sz);
	x64Value eq = x64_emit_comp(p, Token_CmpEq, x64v_mem(t, x64_rbp_mem(la)), x64v_mem(t, x64_rbp_mem(ra)));
	x64_value_to_reg(p, eq, X64Reg_RAX);
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(res_off), X64Reg_RAX);
	x64_label_bind(&p->asm_, end);
}

// Set up + emit one synthetic proc (deduped by symbol-already-defined). Mirrors the scaffolding
// in x64_compile_procedure but with a hand-emitted body and no Entity/AST.
gb_internal void x64_emit_synth_proc(x64Module *m, X64SynthProc desc) {
	String name = x64_synth_proc_name(m, desc.kind, desc.type);
	u32 *existing = string_map_get(&m->coff.sym_map, name);
	if (existing != nullptr && m->coff.syms[*existing].section_number != COFF_SECT_UNDEF) {
		return; // already emitted in this module
	}

	Arena         *scratch = get_arena(ThreadArena_Temporary);
	ArenaTempGuard scratch_guard(scratch);
	gbAllocator    a = arena_allocator(scratch);

	x64Procedure *p = gb_alloc_item(a, x64Procedure);
	p->module    = m;
	p->alloc     = a;
	p->entity    = nullptr;
	p->type      = (desc.kind == X64Synth_Hasher) ? t_hasher_proc : t_equal_proc;
	p->link_name = name;
	p->is_static = true;
	x64_asm_init(&p->asm_, a);
	x64_var_map_init(&p->var_offsets, a, 4);
	array_init(&p->deferred, a, 0, 1);
	array_init(&p->loops,    a, 0, 1);
	array_init(&p->lines,    a, 0, 4);
	array_init(&p->indirect_params, a, 0, 1);
	array_init(&p->inline_frames,   a, 0, 1);
	array_init(&p->inline_sites,    a, 0, 1);
	p->cur_inline_site = -1;
	p->fallthrough_lbl = -1;
	p->cur_line = -1;
	p->cur_file_id = -1;
	p->file_id  = 0;

	x64_proc_begin(p);

	i32 res_off = x64_alloc_local(p, 8, 8);
	if (desc.kind == X64Synth_Hasher) x64_emit_hasher_body(p, desc.type, res_off);
	else                               x64_emit_equal_body (p, desc.type, res_off);

	// Single exit: result → RAX, epilogue, ret.
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_rbp_mem(res_off));
	x64_proc_emit_epilogue(p);
	x64_emit_ret(&p->asm_);
	x64_proc_end(p);
}

// ── high-level wiring (called from expr.cpp IndexExpr / stmt.cpp AssignStmt) ──────────────

// Spilled ^map pointer value for a map-typed lvalue expression `map_expr`.
gb_internal x64Value x64_map_addr_of(x64Procedure *p, Ast *map_expr, Type *map_type) {
	if (is_type_pointer(map_expr->tav.type)) {
		// map accessed through a pointer (`pm[k]`): the expr value IS the ^map.
		return x64_map_spill_u64(p, x64_build_expr(p, map_expr), alloc_type_pointer(map_type));
	}
	x64Addr ma = x64_build_addr(p, map_expr);
	i32 off = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, ma.mem);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
	return x64v_mem(alloc_type_pointer(map_type), x64_rbp_mem(off));
}

// `m[k]` read (rvalue). result_type is V (plain) or a (V, bool) tuple (comma-ok form).
// Mirrors lb_addr_load(lbAddr_Map).
gb_internal x64Value x64_build_map_index_load(x64Procedure *p, Ast *map_expr, Ast *key_expr, Type *result_type) {
	Type *mt = base_type(type_deref(map_expr->tav.type));
	GB_ASSERT(mt->kind == Type_Map);
	x64Value map_ptr = x64_map_addr_of(p, map_expr, mt);
	x64Value ptr = x64_internal_dynamic_map_get_ptr(p, map_ptr, mt, key_expr); // ^V (nil if absent)

	Type *lrt = mt->Map.lookup_result_type; // struct { value: V, ok: bool }
	i64 lrt_sz = type_size_of(lrt);
	i32 r_off = x64_alloc_local(p, lrt_sz, type_align_of(lrt));
	x64_zero_mem(p, x64_rbp_mem(r_off), lrt_sz);

	i64 val_off = type_offset_of(lrt, 0);
	i64 ok_off  = type_offset_of(lrt, 1);

	// ok = (ptr != nil); if ok, copy *ptr → value field.
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, ptr.mem);
	x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
	x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RCX);
	x64_emit_mov_mr(&p->asm_, X64OpSize_8, x64_rbp_mem(r_off + cast(i32)ok_off), X64Reg_RCX);
	isize done = x64_label_alloc(&p->asm_);
	x64_emit_jcc(&p->asm_, X64Cc_E, done);
	x64_copy_mem(p, x64_rbp_mem(r_off + cast(i32)val_off), x64_mem(X64Reg_RAX, 0), type_size_of(mt->Map.value));
	x64_label_bind(&p->asm_, done);

	if (is_type_tuple(result_type)) {
		return x64v_mem(result_type, x64_rbp_mem(r_off));
	}
	return x64v_mem(mt->Map.value, x64_rbp_mem(r_off + cast(i32)val_off));
}

// `&m[k]` — map element POINTER (^V, nil if absent). Comma-ok form `p, ok := &m[k]` → (^V, bool) tuple.
// Mirrors lb_internal_dynamic_map_get_ptr + the comma-ok wrap.
gb_internal x64Value x64_build_map_index_ptr(x64Procedure *p, Ast *map_expr, Ast *key_expr, Type *result_type) {
	Type *mt = base_type(type_deref(map_expr->tav.type));
	GB_ASSERT(mt->kind == Type_Map);
	x64Value map_ptr = x64_map_addr_of(p, map_expr, mt);
	x64Value ptr = x64_internal_dynamic_map_get_ptr(p, map_ptr, mt, key_expr); // ^V (nil if absent), spilled
	if (result_type != nullptr && is_type_tuple(result_type)) {
		// (^V, bool): field0 = ^V @0, field1 = ok @8 (^V is 8 bytes).
		i64 sz = type_size_of(result_type); if (sz <= 0) sz = 16;
		i32 off = x64_alloc_local(p, sz, type_align_of(result_type));
		x64_zero_mem(p, x64_rbp_mem(off), sz);
		x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, ptr.mem);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
		x64_emit_test_rr(&p->asm_, X64OpSize_64, X64Reg_RAX, X64Reg_RAX);
		x64_emit_setcc_r(&p->asm_, X64Cc_NE, X64Reg_RCX);
		x64_emit_mov_mr(&p->asm_, X64OpSize_8, x64_rbp_mem(off + 8), X64Reg_RCX);
		return x64v_mem(result_type, x64_rbp_mem(off));
	}
	ptr.type = alloc_type_pointer(mt->Map.value);
	return ptr;
}

// `m[k] = v` write. Mirrors lb_addr_store(lbAddr_Map).
gb_internal void x64_build_map_index_store(x64Procedure *p, Ast *map_expr, Ast *key_expr, Ast *rhs_expr, Ast *node) {
	Type *mt = base_type(type_deref(map_expr->tav.type));
	GB_ASSERT(mt->kind == Type_Map);
	x64Value rv = x64_build_expr(p, rhs_expr);
	rv = x64_spill_value(p, rv, rv.type ? rv.type : mt->Map.value);
	x64Value map_ptr = x64_map_addr_of(p, map_expr, mt);
	x64_internal_dynamic_map_set(p, map_ptr, mt, key_expr, rv, node);
}

// Drain m->synth_queue, emitting any not-yet-defined hasher/equal procs. A struct/array hasher
// enqueues its child hashers here (and runtime hashers into proc_queue), so the caller loops.
// Returns true if it generated ≥1 proc.
gb_internal bool x64_generate_synth_procs(x64Module *m) {
	bool did = false;
	for (X64SynthProc desc = {}; mpsc_dequeue(&m->synth_queue, &desc); /**/) {
		x64_emit_synth_proc(m, desc);
		did = true;
	}
	return did;
}
