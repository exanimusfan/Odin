// x64 fast debug backend — entry point and module management.
// Compiled only on Windows x64 (guarded in main.cpp).

#include "x64_backend.hpp"
#include "x64_backend_type.cpp"
#include "x64_backend_const.cpp"
#include "x64_backend_proc.cpp"
#include "x64_backend_expr.cpp"
#include "x64_backend_map.cpp"
#include "x64_backend_stmt.cpp"

// Pad .debug$S to a 4-byte boundary using CodeView LF_PAD bytes (the convention
// real CV emitters use inside symbol records).
gb_internal void x64_cv_pad4(CoffSection *s) {
	static const u8 lf_pad[3] = {0xF3u, 0xF2u, 0xF1u}; // 3,2,1-byte pads
	isize rem = coff_section_len(s) % 4;
	if (rem == 0) return;
	isize pad = 4 - rem;
	for (isize i = 0; i < pad; i++) coff_section_write_u8(s, lf_pad[3 - pad + i]);
}

// Emit the per-module CodeView lead-in: a DEBUG_S_SYMBOLS subsection containing
// S_OBJNAME + S_COMPILE3. lld-link uses these to register the object as a debug
// module; without them it keeps public symbol names but discards the module's
// CodeView symbol and line subsections (→ names but no source stepping).
gb_internal void x64_emit_cv_module_header(x64Module *m, String obj_name) {
	if (m->debug_s == nullptr) return;

	String ver = str_lit("Odin x64 debug backend");

	// Record sizes — each 4-byte aligned, padding counted in reclen.
	// S_OBJNAME: rectype(2)+signature(4)+name(len+1)
	isize obj_body  = 2 + 4 + (obj_name.len + 1);
	isize obj_total = 2 + obj_body;                       // + reclen field
	isize obj_pad   = (4 - obj_total % 4) % 4;
	obj_total += obj_pad;
	// S_COMPILE3: rectype(2)+flags(4)+machine(2)+8*u16(16)+ver(len+1)
	isize cmp_body  = 2 + 4 + 2 + 16 + (ver.len + 1);
	isize cmp_total = 2 + cmp_body;
	isize cmp_pad   = (4 - cmp_total % 4) % 4;
	cmp_total += cmp_pad;

	// DEBUG_S_SYMBOLS subsection header
	coff_section_write_u32(m->debug_s, 0xF1u);                       // DEBUG_S_SYMBOLS
	coff_section_write_u32(m->debug_s, (u32)(obj_total + cmp_total)); // payload bytes

	// S_OBJNAME (0x1101)
	coff_section_write_u16(m->debug_s, (u16)(obj_body + obj_pad)); // reclen
	coff_section_write_u16(m->debug_s, 0x1101u);                   // rectype
	coff_section_write_u32(m->debug_s, 0);                         // signature
	coff_section_write(m->debug_s, obj_name.text, obj_name.len);
	coff_section_write_u8(m->debug_s, 0);                          // nul terminator
	x64_cv_pad4(m->debug_s);

	// S_COMPILE3 (0x113C)
	coff_section_write_u16(m->debug_s, (u16)(cmp_body + cmp_pad)); // reclen
	coff_section_write_u16(m->debug_s, 0x113Cu);                   // rectype
	coff_section_write_u32(m->debug_s, 0);                         // flags (lang C=0)
	coff_section_write_u16(m->debug_s, 0x00D0u);                   // machine = CV_CFL_X64
	for (int i = 0; i < 8; i++) coff_section_write_u16(m->debug_s, 0); // FE/BE ver fields
	coff_section_write(m->debug_s, ver.text, ver.len);
	coff_section_write_u8(m->debug_s, 0);                          // nul terminator
	x64_cv_pad4(m->debug_s);
}

gb_internal x64Module *x64_module_create(x64Generator *gen, AstPackage *pkg, AstFile *file) {
	// The x64Module struct itself lives on the generator's heap (pointer-stable, held
	// in gen->modules), but everything it OWNS comes from its embedded bump arena so
	// the parallel compile/write workers never contend on the CRT heap lock.
	x64Module *m = gb_alloc_item(gen->alloc, x64Module);
	m->arena = {};
	m->arena.custom_arena = true; // cross-thread (compile→write) use; bypasses parent-thread assert
	gbAllocator a = arena_allocator(&m->arena);
	m->alloc = a;
	m->gen   = gen;
	m->pkg   = pkg;
	m->file  = file;

	coff_writer_init(&m->coff, a);

	// Always-present sections get stable numbers 1–6 regardless of debug, so the
	// optional CodeView sections can be appended (or omitted) without renumbering.
	m->text  = coff_section_add(&m->coff, str_lit(".text"),
	                             COFF_SCN_TEXT | COFF_SCN_ALIGN_16);
	m->rdata = coff_section_add(&m->coff, str_lit(".rdata"),
	                             COFF_SCN_RDATA | COFF_SCN_ALIGN_8);
	m->data  = coff_section_add(&m->coff, str_lit(".data"),
	                             COFF_SCN_DATA | COFF_SCN_ALIGN_8);
	m->bss   = coff_section_add(&m->coff, str_lit(".bss"),
	                             COFF_SCN_BSS | COFF_SCN_ALIGN_8);

	// .pdata / .xdata — Win64 stack-unwinding tables (sections 5, 6). NOT debug
	// info: required for OS/debugger stack walks, so emitted regardless of -debug.
	m->pdata = coff_section_add(&m->coff, str_lit(".pdata"),
	    COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ | COFF_SCN_ALIGN_4);
	m->xdata = coff_section_add(&m->coff, str_lit(".xdata"),
	    COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ | COFF_SCN_ALIGN_4);

	// Section symbols (text=1..xdata=6); xdata's is referenced by .pdata relocs.
	// debug$S/$T need no section sym.
	coff_sym_add_section(&m->coff, 1, m->text);
	coff_sym_add_section(&m->coff, 2, m->rdata);
	coff_sym_add_section(&m->coff, 3, m->data);
	coff_sym_add_section(&m->coff, 4, m->bss);
	coff_sym_add_section(&m->coff, 5, m->pdata);
	coff_sym_add_section(&m->coff, 6, m->xdata);

	map_init(&m->cv_types, 64);
	m->cv_next_type = 0x1000; // user type indices start here; below is built-in
	map_init(&m->cv_func_ids, 16);
	array_init(&m->cv_inlinees, a, 0, 8);
	m->cv_empty_arglist = 0;
	m->cv_void_proc_type = 0;

	// CodeView debug info (.debug$S symbols/lines + .debug$T types) — appended
	// last (sections 7, 8) and ONLY for -debug builds. Without -debug the backend
	// is a plain fast codegen path (e.g. for future runtime/bytecode use); these
	// stay null and every emission site guards on `m->debug_s`/`m->debug_t`.
	if (build_context.ODIN_DEBUG) {
		m->debug_s = coff_section_add(&m->coff,
		    str_lit(".debug$S"),
		    COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ | COFF_SCN_MEM_DISCARDABLE | COFF_SCN_ALIGN_1);
		u32 cv_sig = 4; // CV_SIGNATURE_C13 (required at the start of every .debug$S)
		coff_section_write(m->debug_s, &cv_sig, 4);

		x64_emit_cv_module_header(m, pkg->name);

		m->debug_t = coff_section_add(&m->coff, str_lit(".debug$T"),
		    COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ | COFF_SCN_MEM_DISCARDABLE | COFF_SCN_ALIGN_1);
		u32 cv_sig_t = 4;
		coff_section_write(m->debug_t, &cv_sig_t, 4);
	}

	array_init(&m->rdata_f64_bits, a, 0, 8);
	array_init(&m->rdata_f64_offs, a, 0, 8);
	array_init(&m->rdata_f32_bits, a, 0, 8);
	array_init(&m->rdata_f32_offs, a, 0, 8);

	array_init(&m->cv_strtab,     a, 0, 64);
	array_init(&m->cv_filechksms, a, 0, 32);
	array_init(&m->cv_file_ids,   a, 0, 8);
	array_init(&m->cv_file_offs,  a, 0, 8);
	array_init(&m->compile_roots, a, 0, 16);
	mpsc_init(&m->proc_queue, a);
	mpsc_init(&m->synth_queue, a);
	map_init(&m->map_info_map, 8);
	map_init(&m->map_cell_info_map, 8);
	array_init(&m->onref_globals, a, 0, 8);
	string_map_init(&m->const_strings, 16);

	return m;
}

// Append the DEBUG_S_FILECHKSMS and DEBUG_S_STRINGTABLE subsections that the
// per-procedure DEBUG_S_LINES records reference. Must run after all procedures
// in the module are compiled and before the COFF file is written.
gb_internal void x64_module_finalize_debug(x64Module *m) {
	if (m->debug_s == nullptr || m->cv_filechksms.count == 0) return;

	// DEBUG_S_INLINEELINES (0xF6) — each inlined callee (LF_FUNC_ID) → its source file + decl line,
	// the base the S_INLINESITE binary annotations are relative to. Emitted BEFORE F4 so a file first
	// seen here still lands in the checksum table. payload: signature(4) + n*(inlinee,file,line)(12).
	if (m->cv_inlinees.count > 0) {
		u32 cb = 4 + (u32)m->cv_inlinees.count * 12;
		coff_section_write_u32(m->debug_s, 0xF6u);
		coff_section_write_u32(m->debug_s, cb);
		coff_section_write_u32(m->debug_s, 0); // CV_INLINEE_SOURCE_LINE_SIGNATURE (normal)
		for (isize i = 0; i < m->cv_inlinees.count; i++) {
			Entity *callee = m->cv_inlinees[i];
			u32 *fid = map_get(&m->cv_func_ids, callee);
			DeclInfo *d = decl_info_of_entity(callee);
			Ast *pl = (d != nullptr) ? d->proc_lit : nullptr;
			TokenPos dp = (pl != nullptr) ? ast_token(pl).pos : callee->token.pos;
			coff_section_write_u32(m->debug_s, fid ? *fid : 0u);
			coff_section_write_u32(m->debug_s, x64_cv_file_offset(m, dp.file_id));
			coff_section_write_u32(m->debug_s, (u32)dp.line);
		}
		coff_section_align(m->debug_s, 4);
	}

	// DEBUG_S_FILECHKSMS (0xF4)
	coff_section_write_u32(m->debug_s, 0xF4u);
	coff_section_write_u32(m->debug_s, (u32)m->cv_filechksms.count);
	coff_section_write(m->debug_s, m->cv_filechksms.data, m->cv_filechksms.count);
	coff_section_align(m->debug_s, 4); // subsections must start 4-byte aligned

	// DEBUG_S_STRINGTABLE (0xF3)
	coff_section_write_u32(m->debug_s, 0xF3u);
	coff_section_write_u32(m->debug_s, (u32)m->cv_strtab.count);
	coff_section_write(m->debug_s, m->cv_strtab.data, m->cv_strtab.count);
	coff_section_align(m->debug_s, 4);
}

gb_internal void x64_compile_procedure(x64Module *m, Entity *e, Ast *body) {
	if (e == nullptr || e->kind != Entity_Procedure) return;
	if (body == nullptr) return; // foreign / abstract
	if (e->Procedure.is_foreign) return;

	Type *proc_type = base_type(e->type);
	if (proc_type == nullptr || proc_type->kind != Type_Proc) return;

	// Skip unspecialised polymorphic templates — their bodies have unresolved
	// type-parameter idents with null entity pointers.
	if (proc_type->Proc.is_polymorphic && !proc_type->Proc.is_poly_specialized) return;

	ProcCallingConvention cc = proc_type->Proc.calling_convention;
	switch (cc) {
	case ProcCC_Odin:
	case ProcCC_Contextless:
	case ProcCC_CDecl:
	case ProcCC_Win64:
	case ProcCC_StdCall: // "system" on Windows x64 == Win64 ABI
		break;
	default:
		return; // naked, inline asm, etc. unsupported
	}

	// Skip if already DEFINED here. sym_map also holds UNDEF external refs from
	// relocations, so test section_number != COFF_SECT_UNDEF (i.e. already compiled).
	String link_name = x64_get_entity_name(e);
	{
		u32 *existing_idx = string_map_get(&m->coff.sym_map, link_name);
		if (existing_idx != nullptr &&
		    m->coff.syms[*existing_idx].section_number != COFF_SECT_UNDEF) {
			return;
		}
	}

	// All per-proc scratch is bump-allocated from this thread's temporary arena and
	// freed in bulk by the guard at function exit. Per-thread → safe under parallel
	// per-module codegen. Anything that must persist (emitted code/relocs/symbols) is
	// copied into m->coff (the module's own arena) by x64_proc_end before the reset.
	Arena         *scratch = get_arena(ThreadArena_Temporary);
	ArenaTempGuard scratch_guard(scratch);
	gbAllocator    a = arena_allocator(scratch);

	x64Procedure *p = gb_alloc_item(a, x64Procedure);
	p->module    = m;
	p->alloc     = a;
	p->entity    = e;
	p->type      = proc_type;
	p->link_name = link_name;

	x64_asm_init(&p->asm_, a);
	x64_var_map_init(&p->var_offsets, a, 16); // arena-backed; reclaimed with the scratch guard
	array_init(&p->deferred, a, 0, 4);
	array_init(&p->loops,    a, 0, 4);
	array_init(&p->lines,    a, 0, 64);
	array_init(&p->indirect_params, a, 0, 2);
	array_init(&p->inline_frames,   a, 0, 2);
	array_init(&p->inline_sites,    a, 0, 2);
	p->cur_inline_site = -1;
	p->fallthrough_lbl = -1;
	p->cur_line = -1;
	p->cur_file_id = -1;
	// Line-table file from the BODY, not the entity token (mirrors LLVM's node->file()).
	// For a generic instantiation (e.g. make_slice[[]Logger]) the entity token resolves
	// to a DIFFERENT file than the body; wrong file → x64_record_line drops every body
	// line (file_id mismatch) → no DEBUG_S_LINES → debugger falls to disassembly.
	// (x64 never inlines, so a body is a single file — one file block suffices.)
	p->file_id  = ast_token(body).pos.file_id;
	if (p->file_id == 0) p->file_id = e->token.pos.file_id;

	// Attribute the prologue to the procedure's opening line so stepping into
	// the function lands on its body rather than the first statement's code.
	x64_record_line(p, body);

	x64_proc_begin(p);

	x64_build_stmt(p, body);

	// Safety epilogue for fall-through paths (in case of missing returns).
	x64_run_deferred(p);
	x64_emit_named_returns(p);
	x64_proc_emit_epilogue(p);
	x64_emit_ret(&p->asm_);

	x64_proc_end(p); // copies code/relocs/debug into m->coff
}

// Queue a nested proc for deferred compilation (reachability flows through the parent).
gb_internal void x64_build_nested_proc(x64Procedure *p, Ast *proc_lit, Entity *e) {
	if (e == nullptr || e->kind != Entity_Procedure) return;
	if (e->min_dep_count.load(std::memory_order_relaxed) == 0) return;
	if (proc_lit == nullptr || proc_lit->kind != Ast_ProcLit) return;
	if (proc_lit->ProcLit.body == nullptr) return;
	// DEFER into THIS (enclosing) module — mirrors LLVM lb_build_nested_proc (enqueue, not
	// inline). Compiling inline re-enters x64_compile_procedure on the SAME per-thread temp
	// arena (reset via ArenaTempGuard) mid-codegen, corrupting the enclosing proc. Enclosing
	// module (not the entity's owner) because a nested proc has no standalone module mapping.
	mpsc_enqueue(&p->module->proc_queue, e);
}

gb_internal GB_COMPARE_PROC(x64_proc_entity_cmp) {
	Entity *x = *cast(Entity **)a;
	Entity *y = *cast(Entity **)b;
	if (x == y) return 0;
	return token_pos_cmp(x->token.pos, y->token.pos);
}

// Emit a CodeView S_GDATA32 (0x110D) record so the debugger can inspect a global by
// name. Wrapped in its own DEBUG_S_SYMBOLS subsection; off/seg resolved via SECREL/
// SECTION relocs against the global's COFF symbol (`link_name`). Display name uses the
// simple source name (mirrors LLVM). `rectype` = S_GTHREAD32 (0x1113) for thread-locals
// — THREADSYM32 is byte-identical to DATASYM32; only the relocs target a `.tls$` symbol
// (resolved per-thread via the TLS array).
gb_internal void x64_emit_cv_global(x64Module *m, String link_name, Entity *e, u16 rectype = 0x110Du) {
	if (m->debug_s == nullptr || link_name.len == 0) return;
	String disp = e->token.string;
	if (disp.len == 0 || disp == str_lit("_")) return;
	u32 cvt = x64_cv_type(m, e->type);

	coff_section_write_u32(m->debug_s, 0xF1u);                 // DEBUG_S_SYMBOLS
	u32 sub_len_pos  = (u32)coff_section_len(m->debug_s);
	coff_section_write_u32(m->debug_s, 0);                     // length (patched)
	u32 sub_data_pos = (u32)coff_section_len(m->debug_s);

	// reclen counts everything after it (incl. padding); record is 4-aligned.
	isize content = 2 + 4 + 4 + 2 + disp.len + 1; // rectype+typind+off+seg+name+nul
	isize total   = (2 + content + 3) & ~(isize)3;
	coff_section_write_u16(m->debug_s, (u16)(total - 2)); // reclen
	coff_section_write_u16(m->debug_s, rectype);         // S_GDATA32 / S_GTHREAD32
	coff_section_write_u32(m->debug_s, cvt);             // typind

	u32 off_pos = (u32)coff_section_len(m->debug_s);
	coff_section_write_u32(m->debug_s, 0);               // off (SECREL)
	coff_reloc_add(m->debug_s, off_pos, link_name, COFF_REL_SECREL);

	u32 seg_pos = (u32)coff_section_len(m->debug_s);
	coff_section_write_u16(m->debug_s, 0);               // seg (SECTION)
	coff_reloc_add(m->debug_s, seg_pos, link_name, COFF_REL_SECTION);

	coff_section_write(m->debug_s, disp.text, disp.len);
	coff_section_write_u8(m->debug_s, 0);
	x64_cv_pad4(m->debug_s);

	u32 sub_len = (u32)coff_section_len(m->debug_s) - sub_data_pos;
	m->debug_s->data[sub_len_pos + 0] = (u8)( sub_len        & 0xFFu);
	m->debug_s->data[sub_len_pos + 1] = (u8)((sub_len >> 8)  & 0xFFu);
	m->debug_s->data[sub_len_pos + 2] = (u8)((sub_len >> 16) & 0xFFu);
	m->debug_s->data[sub_len_pos + 3] = (u8)((sub_len >> 24) & 0xFFu);
}

gb_internal void x64_emit_global_variable(x64Module *m, Entity *e) {
	// Handled by x64_emit_type_table
	if (e->pkg != nullptr &&
	    e->pkg->kind == Package_Runtime &&
	    e->token.string == str_lit("type_table")) {
		return;
	}

	if (e->Variable.is_foreign) {
		// Foreign globals are already defined elsewhere; add external sym ref.
		String name = x64_get_entity_name(e);
		coff_sym_find_or_add_extern(&m->coff, name);
		return;
	}

	String name = x64_get_entity_name(e);
	// Dedup: eager and on-demand passes can both reach the same global. UNDEF entries
	// (from an earlier call-site reloc) are upgraded below, so only skip when DEFINED.
	{
		u32 *existing = string_map_get(&m->coff.sym_map, name);
		if (existing != nullptr && m->coff.syms[*existing].section_number != COFF_SECT_UNDEF) {
			return;
		}
	}
	i64 sz    = type_size_of(e->type);
	i64 align = type_align_of(e->type);
	if (sz <= 0) sz = 1;
	if (align <= 0) align = 1;

	// Zero-filled storage in WRITABLE .bss. Even @(rodata) globals must be writable
	// here: their value is stored at runtime by the startup init loop (see __entry_point),
	// so read-only .rdata would fault. RO protection sacrificed until real static-init emission.
	i16 section_number = 4; // .bss (writable)
	coff_section_align(m->bss, (isize)align);
	u32 off = (u32)coff_section_len(m->bss);
	for (i64 b = 0; b < sz; b++) coff_section_write_u8(m->bss, 0);

	// ALL global vars must be EXTERNAL: a STATIC symbol is file-local and won't resolve
	// a UNDEF external ref from another .obj (the encoding_base64 ENC_TABLE/DEC_TABLE bug:
	// coff_sym_add_proc mis-read is_export from a Variable union as 0 → STATIC + FUNCTION).
	// Upgrade an existing UNDEF entry in place so its relocs point at the definition.
	u32 *existing = string_map_get(&m->coff.sym_map, name);
	if (existing != nullptr) {
		CoffSymEntry &se = m->coff.syms[*existing];
		se.value          = off;
		se.section_number = section_number;
		se.type           = COFF_SYM_TYPE_NULL;
		se.storage_class  = COFF_SYM_CLASS_EXTERNAL;
	} else {
		u32 sym_idx = (u32)m->coff.syms.count;
		CoffSymEntry sym_e = {};
		sym_e.name          = name;
		sym_e.value         = off;
		sym_e.section_number = section_number;
		sym_e.type          = COFF_SYM_TYPE_NULL;
		sym_e.storage_class = COFF_SYM_CLASS_EXTERNAL;
		array_add(&m->coff.syms, sym_e);
		string_map_set(&m->coff.sym_map, name, sym_idx);
	}
	x64_emit_cv_global(m, name, e);
}

// Does this file-scope global have a COMPILE-TIME-CONSTANT initializer? If so it is
// emitted as STATIC data (x64_emit_global_static) and does NOT need runtime init —
// mirrors LLVM's `tav.value.kind != ExactValue_Invalid` test (llvm_backend.cpp:3417).
gb_internal bool x64_global_has_const_init(DeclInfo *d, Entity *e) {
	if (d == nullptr || d->init_expr == nullptr) return false;
	if (e == nullptr || e->kind != Entity_Variable) return false;
	if (e->Variable.is_foreign) return false;
	if (is_type_any(e->type)) return false; // `any` globals need runtime typeid/data
	if (!x64_const_type_supported(e->type)) return false; // else x64_const_value would zero-fill
	TypeAndValue tav = type_and_value_of_expr(d->init_expr);
	return tav.mode != Addressing_Invalid && tav.value.kind != ExactValue_Invalid;
}

// Emit a compile-time-const global as STATIC data (mirrors LLVM's LLVMSetInitializer).
// @(rodata) → read-only .rdata; else writable .data. No startup-runtime store generated.
// `init_expr` is the constant initializer (file-scope: d->init_expr; proc-local @(static):
// the ValueDecl value directly, mirroring lb_build_static_variables).
gb_internal void x64_emit_global_static_value(x64Module *m, Entity *e, Ast *init_expr) {
	String name = x64_get_entity_name(e);
	{
		u32 *existing = string_map_get(&m->coff.sym_map, name);
		if (existing != nullptr && m->coff.syms[*existing].section_number != COFF_SECT_UNDEF) {
			return; // already defined
		}
	}
	Type *t  = x64_typed(e->type);
	i64   sz = type_size_of(t); if (sz <= 0) sz = 1;
	i64   al = type_align_of(t); if (al <= 0) al = 1;

	bool ro = e->Variable.is_rodata;                 // only @(rodata) becomes read-only
	CoffSection *sec       = ro ? m->rdata : m->data;
	i16          section_number = ro ? 2 : 3;         // .rdata=2, .data=3

	u32 off = x64_const_reserve(sec, al, sz);
	TypeAndValue tav = type_and_value_of_expr(init_expr);
	x64_const_value(m, sec, off, t, tav.value, tav.type);

	x64_const_define_symbol(m, name, section_number, off);
	x64_emit_cv_global(m, name, e);
}
gb_internal void x64_emit_global_static(x64Module *m, Entity *e, DeclInfo *d) {
	x64_emit_global_static_value(m, e, d->init_expr);
}

// Lazily create this module's `.tls$` section (the thread-local template the OS copies per
// thread). Name mirrors LLVM's COFF TLSDataSection so it sorts between the CRT's `_tls_start`
// (.tls) and `_tls_end` (.tls$ZZZ) markers bounding the TLS directory.
gb_internal CoffSection *x64_module_tls_section(x64Module *m) {
	if (m->tls == nullptr) {
		m->tls = coff_section_add(&m->coff, str_lit(".tls$"),
		    COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ | COFF_SCN_MEM_WRITE | COFF_SCN_ALIGN_16);
		m->tls_secnum = (i16)m->coff.sections.count; // 1-based index == section number

		// Force the linker to build the PE TLS directory by pulling in `_tls_used`
		// (IMAGE_TLS_DIRECTORY, from the CRT's tlssup); without it the OS never copies the
		// template per-thread nor fills `_tls_index`. LLVM emits this /INCLUDE automatically.
		// `.drectve` is link-only metadata (not in the image). Skip for -no-crt (no _tls_used).
		if (!build_context.no_crt) {
			CoffSection *dr = coff_section_add(&m->coff, str_lit(".drectve"),
			    0x00000200u | 0x00000800u); // IMAGE_SCN_LNK_INFO | IMAGE_SCN_LNK_REMOVE
			String d = str_lit(" /INCLUDE:_tls_used");
			coff_section_write(dr, d.text, d.len);
		}
	}
	return m->tls;
}

// Emit a `@(thread_local)` global into `.tls$` (storage is the per-thread template;
// references go through the TLS access sequence — see x64_emit_tls_addr). Const initializer
// becomes the template (copied to every thread); else zero-fill (a runtime initializer sets
// only the running thread, matching LLVM/Odin). EXTERNAL symbol (cross-module via SECREL).
gb_internal void x64_emit_global_tls(x64Module *m, Entity *e, DeclInfo *d) {
	if (e->Variable.is_foreign) {
		coff_sym_find_or_add_extern(&m->coff, x64_get_entity_name(e));
		return;
	}
	String name = x64_get_entity_name(e);
	{
		u32 *existing = string_map_get(&m->coff.sym_map, name);
		if (existing != nullptr && m->coff.syms[*existing].section_number != COFF_SECT_UNDEF) {
			return; // already defined
		}
	}
	Type *t  = x64_typed(e->type);
	i64   sz = type_size_of(t); if (sz <= 0) sz = 1;
	i64   al = type_align_of(t); if (al <= 0) al = 1;

	CoffSection *sec = x64_module_tls_section(m);
	u32 off;
	if (x64_global_has_const_init(d, e)) {
		off = x64_const_reserve(sec, al, sz);
		TypeAndValue tav = type_and_value_of_expr(d->init_expr);
		x64_const_value(m, sec, off, t, tav.value, tav.type);
	} else {
		coff_section_align(sec, (isize)al);
		off = (u32)coff_section_len(sec);
		for (i64 b = 0; b < sz; b++) coff_section_write_u8(sec, 0);
	}

	// EXTERNAL symbol in `.tls$` (upgrade an UNDEF entry from an earlier call-site reloc).
	u32 *existing = string_map_get(&m->coff.sym_map, name);
	if (existing != nullptr) {
		CoffSymEntry &se = m->coff.syms[*existing];
		se.value          = off;
		se.section_number = m->tls_secnum;
		se.type           = COFF_SYM_TYPE_NULL;
		se.storage_class  = COFF_SYM_CLASS_EXTERNAL;
	} else {
		u32 sym_idx = (u32)m->coff.syms.count;
		CoffSymEntry sym_e = {};
		sym_e.name          = name;
		sym_e.value         = off;
		sym_e.section_number = m->tls_secnum;
		sym_e.type          = COFF_SYM_TYPE_NULL;
		sym_e.storage_class = COFF_SYM_CLASS_EXTERNAL;
		array_add(&m->coff.syms, sym_e);
		string_map_set(&m->coff.sym_map, name, sym_idx);
	}
	x64_emit_cv_global(m, name, e, 0x1113u); // S_GTHREAD32
}

// Anonymous EXTERNAL global with zero-init .bss storage — the static backing a
// `&CompoundLit` in the startup runtime (e.g. `INT_ZERO := &Int{}`). Unique
// module-qualified name so the lea_sym reloc resolves to this same .obj.
gb_internal String x64_add_global_generated(x64Module *m, Type *type) {
	i64 sz = type_size_of(type); if (sz <= 0) sz = 1;
	i64 al = type_align_of(type); if (al <= 0) al = 1;
	coff_section_align(m->bss, (isize)al);
	u32 off = (u32)coff_section_len(m->bss);
	for (i64 b = 0; b < sz; b++) coff_section_write_u8(m->bss, 0);

	gbString gs = gb_string_make(m->alloc, "__$ag$");
	gs = gb_string_append_length(gs, m->pkg->name.text, m->pkg->name.len);
	if (m->file != nullptr) gs = gb_string_append_fmt(gs, "$%d", (int)m->file->id);
	gs = gb_string_append_fmt(gs, "$%u", m->gen_global_count++);
	String name = make_string(cast(u8 const *)gs, gb_string_length(gs));

	u32 sym_idx = (u32)m->coff.syms.count;
	CoffSymEntry se = {};
	se.name           = name;
	se.value          = off;
	se.section_number = 4; // .bss
	se.type           = COFF_SYM_TYPE_NULL;
	se.storage_class  = COFF_SYM_CLASS_EXTERNAL;
	array_add(&m->coff.syms, se);
	string_map_set(&m->coff.sym_map, name, sym_idx);
	return name;
}

// Emit a minimal empty procedure, for synthesized procs the LLVM backend generates
// but we stub out.
gb_internal void x64_emit_empty_stub(x64Module *m, String name) {
	X64Assembler a = {};
	x64_asm_init(&a, heap_allocator());

	x64_emit_push_r(&a, X64Reg_RBP);
	x64_emit_mov_rr(&a, X64OpSize_64, X64Reg_RBP, X64Reg_RSP);
	x64_emit_sub_ri(&a, X64OpSize_64, X64Reg_RSP, 0x20); // 32-byte shadow space
	x64_emit_mov_rr(&a, X64OpSize_64, X64Reg_RSP, X64Reg_RBP);
	x64_emit_pop_r(&a, X64Reg_RBP);
	x64_emit_ret(&a);

	u32 base_off = (u32)coff_section_len(m->text);
	coff_sym_add_proc(&m->coff, name, 1 /*text*/, base_off, true);
	coff_section_write(m->text, a.code.data, a.code.count);

	array_free(&a.code);
	array_free(&a.labels);
	array_free(&a.fixups);
	array_free(&a.relocs);
}

// Initialize one file-scope global with a non-const initializer (deduped via `inited`).
gb_internal void x64_startup_init_global(x64Procedure *p, PtrSet<Entity *> *inited, Entity *e, DeclInfo *d) {
	if (e == nullptr || e->kind != Entity_Variable) return;
	if (e->Variable.is_foreign) return;
	Scope *gs = e->scope;
	if (gs == nullptr || !(gs->flags & ScopeFlag_File)) return;
	if (d == nullptr || d->init_expr == nullptr) return; // zero-init: already zeroed
	if (x64_global_has_const_init(d, e)) return;         // emitted as static data
	if (ptr_set_update(inited, e)) return;
	Type *et = x64_typed(e->type);
	i64 gsz = type_size_of(et); if (gsz <= 0) gsz = 1;
	i64 gal = type_align_of(et); if (gal <= 0) gal = 1;
	i32 saved_local = p->local_size;
	x64Value gv = x64_build_expr(p, d->init_expr);
	i32 slv = x64_alloc_local(p, gsz, gal);
	x64_store_value(p, x64addr(x64_rbp_mem(slv), et), gv);
	x64_load_global_addr(p, e); // RAX = &global
	x64_copy_fixed(p, x64_mem(X64Reg_RAX, 0), x64_rbp_mem(slv), gsz);
	p->local_size = saved_local;
}

// Emit __$startup_runtime: a REAL proc running the initializers of LAZILY-defined
// (min_dep==0) globals that the eager __entry_point loop skipped (e.g. math/big's
// `INT_ZERO := &Int{}`); without it the global stays null and the @(init) proc using it
// faults. Mirrors lb_create_startup_runtime; the runtime entry calls it before
// intrinsics.__entry_point. `defined` = globals that received lazy storage.
gb_internal void x64_emit_startup_runtime(x64Module *m, x64Generator *gen, PtrSet<Entity *> *defined) {
	CheckerInfo *info = gen->info;

	Arena         *scratch = get_arena(ThreadArena_Temporary);
	ArenaTempGuard scratch_guard(scratch);
	gbAllocator    a = arena_allocator(scratch);

	x64Procedure *p = gb_alloc_item(a, x64Procedure);
	p->module    = m;
	p->alloc     = a;
	p->entity    = nullptr; // synthetic — proc_end driven by link_name, not entity
	p->type      = alloc_type_proc(nullptr, nullptr, 0, nullptr, 0, false, ProcCC_Odin);
	p->link_name = str_lit("__$startup_runtime");

	x64_asm_init(&p->asm_, a);
	x64_var_map_init(&p->var_offsets, a, 16);
	array_init(&p->deferred, a, 0, 4);
	array_init(&p->loops,    a, 0, 4);
	array_init(&p->lines,    a, 0, 8);
	array_init(&p->indirect_params, a, 0, 2);
	array_init(&p->inline_frames,   a, 0, 2);
	array_init(&p->inline_sites,    a, 0, 2);
	p->cur_inline_site = -1;
	p->fallthrough_lbl = -1;
	p->cur_line = -1;
	p->cur_file_id = -1;
	p->file_id  = 0;

	x64_proc_begin(p);

	// Initialize EVERY file-scope global that has storage (eager min_dep!=0, or lazy
	// in `defined`). Mirrors LLVM, which runs ALL initializers here over its created-
	// globals list. Doing it after the on-demand closure (every referenced global now
	// defined) avoids the eager/lazy asymmetry that left math/big's INT_ONE uninitialized.
	// p->is_startup makes &CompoundLit allocate a static global (so `&Int{}` escapes).
	p->is_startup = true;

	PtrSet<Entity *> inited = {};
	ptr_set_init(&inited, 64);
	defer (ptr_set_destroy(&inited));

	// Pass 1: dependency order — eager globals + any lazy global in variable_init_order.
	for (DeclInfo *d : info->variable_init_order) {
		Entity *e = d->entity.load(std::memory_order_relaxed);
		if (e == nullptr) continue;
		bool has_storage = (e->min_dep_count.load(std::memory_order_relaxed) != 0) ||
		                   ptr_set_exists(defined, e);
		if (!has_storage) continue;
		x64_startup_init_global(p, &inited, e, d);
	}
	// Pass 2: lazy globals not in variable_init_order (LLVM iterates created-globals,
	// not init order, so e.g. some math/big constants must still be initialized).
	FOR_PTR_SET(e, *defined) {
		x64_startup_init_global(p, &inited, e, decl_info_of_entity(e));
	}
	p->is_startup = false;

	x64_run_deferred(p);
	x64_emit_named_returns(p);
	x64_proc_emit_epilogue(p);
	x64_emit_ret(&p->asm_);
	x64_proc_end(p);
}

// `odin test` entry: the runtime's `main` is `when !ODIN_TEST` (absent), so generate one here
// (mirrors lb_create_main_procedure's test path). Sets runtime.args__ from argc/argv, runs the
// startup runtime, builds a []testing.Internal_Test{pkg,name,proc} from info->testing_procedures,
// calls testing.runner(slice), then runtime.exit(0/1). C entry: argc in ECX, argv in RDX (Win64).
gb_internal void x64_emit_test_main(x64Module *m, x64Generator *gen) {
	CheckerInfo *info = gen->info;
	AstPackage *rt = get_runtime_package(info);
	AstPackage *tp = try_get_core_package(info, str_lit("testing"));
	if (rt == nullptr || tp == nullptr) return;
	Entity *startup = scope_lookup_current(rt->scope, string_interner_insert(str_lit("_startup_runtime")));
	Entity *args_e  = scope_lookup_current(rt->scope, string_interner_insert(str_lit("args__")));
	Entity *exit_e  = scope_lookup_current(rt->scope, string_interner_insert(str_lit("exit")));
	Entity *runner  = scope_lookup_current(tp->scope, string_interner_insert(str_lit("runner")));
	Type   *it_type = find_type_in_pkg(info, str_lit("testing"), str_lit("Internal_Test"));
	if (runner == nullptr || it_type == nullptr || exit_e == nullptr) return;

	Arena         *scratch = get_arena(ThreadArena_Temporary);
	ArenaTempGuard scratch_guard(scratch);
	gbAllocator    a = arena_allocator(scratch);

	x64Procedure *p = gb_alloc_item(a, x64Procedure);
	p->module    = m;
	p->alloc     = a;
	p->entity    = nullptr;
	p->type      = alloc_type_proc(nullptr, nullptr, 0, nullptr, 0, false, ProcCC_CDecl); // C entry → generated context
	p->link_name = str_lit("main");
	x64_asm_init(&p->asm_, a);
	x64_var_map_init(&p->var_offsets, a, 16);
	array_init(&p->deferred, a, 0, 4);
	array_init(&p->loops,    a, 0, 4);
	array_init(&p->lines,    a, 0, 8);
	array_init(&p->indirect_params, a, 0, 2);
	array_init(&p->inline_frames,   a, 0, 2);
	array_init(&p->inline_sites,    a, 0, 2);
	p->cur_inline_site = -1;
	p->fallthrough_lbl = -1;
	p->cur_line = -1;
	p->cur_file_id = -1;
	p->file_id  = 0;

	x64_proc_begin(p);
	p->is_startup = true;

	// args__ = argv[:argc]  (argc=ECX, argv=RDX at entry; the prologue doesn't touch them).
	if (args_e != nullptr) {
		x64_emit_mov_rr(&p->asm_, X64OpSize_32, X64Reg_R8, X64Reg_RCX);   // R8 = argc (zero-extended; argc≥0)
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(args_e));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_mem(X64Reg_RAX, 0), X64Reg_RDX); // .data = argv
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_mem(X64Reg_RAX, 8), X64Reg_R8);  // .len  = argc
	}

	// startup runtime (global var initializers); _startup_runtime is foreign → link __$startup_runtime.
	if (startup != nullptr) x64_call_no_arg(p, startup);

	// Run the @(init) procs (e.g. os.init_std_files setting up stdin/stdout/stderr). Normally these
	// run inside intrinsics.__entry_point (x64_build_expr), which the test entry bypasses — so call
	// them here, mirroring LLVM's _startup_runtime (lb_create_startup_runtime runs init_procedures).
	for (Entity *ie : info->init_procedures) {
		if (ie != nullptr && ie->kind == Entity_Procedure) x64_call_no_arg(p, ie);
	}

	// Build the []testing.Internal_Test backing array on the stack.
	i64 itsz = type_size_of(it_type); if (itsz <= 0) itsz = 40;
	i64 N    = info->testing_procedures.count;
	i32 arr  = x64_alloc_local(p, gb_max(N*itsz, (i64)itsz), gb_max(type_align_of(it_type), (i64)8));
	for (i64 i = 0; i < N; i++) {
		Entity *tproc = info->testing_procedures[i];
		if (tproc == nullptr) continue;
		String pkg_name = (tproc->pkg != nullptr) ? tproc->pkg->name : str_lit("");
		String tname    = tproc->token.string;
		i32 base = arr + (i32)(i*itsz);
		// pkg string {data,len} @ +0
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_const_intern_string(m, pkg_name));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base + 0), X64Reg_RAX);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, pkg_name.len);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base + 8), X64Reg_RAX);
		// name string {data,len} @ +16
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_const_intern_string(m, tname));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base + 16), X64Reg_RAX);
		x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, tname.len);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base + 24), X64Reg_RAX);
		// proc pointer @ +32
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, x64_get_entity_name(tproc));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(base + 32), X64Reg_RAX);
	}

	// slice {&arr[0], N}
	i32 sl = x64_alloc_local(p, 16, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(arr));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sl + 0), X64Reg_RAX);
	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, N);
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(sl + 8), X64Reg_RAX);

	// runner(slice) — the slice (16B) is passed indirect (by pointer); context is the last Odin arg.
	i32 slp = x64_alloc_local(p, 8, 8);
	x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(sl));
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(slp), X64Reg_RAX);
	x64Value rargs[2];
	rargs[0] = x64v_mem(t_rawptr, x64_rbp_mem(slp));
	rargs[1] = x64_context_ptr_value(p);
	x64Value rres = x64_emit_call(p, x64_get_entity_name(runner), runner->type, rargs, 2); // bool → RAX

	// exit(success ? 0 : 1): code = (runner result == 0) ? 1 : 0.
	x64_value_to_reg(p, rres, X64Reg_RAX);
	x64_emit_test_rr(&p->asm_, X64OpSize_8, X64Reg_RAX, X64Reg_RAX);
	x64_emit_setcc_r(&p->asm_, X64Cc_E, X64Reg_RCX);                 // RCX = 1 if failed (result==0)
	x64_emit_movzx_rr(&p->asm_, X64OpSize_8, X64Reg_RCX, X64Reg_RCX);
	x64Value eargs[1];
	eargs[0] = x64v_reg(t_int, X64Reg_RCX);
	x64_emit_call(p, x64_get_entity_name(exit_e), exit_e->type, eargs, 1); // contextless (code:int) → noreturn

	x64_emit_mov_ri(&p->asm_, X64OpSize_64, X64Reg_RAX, 0); // unreachable; main returns i32 0
	x64_run_deferred(p);
	x64_proc_emit_epilogue(p);
	x64_emit_ret(&p->asm_);
	x64_proc_end(p);
}

// Mirror of lb_typeid_kind — type → Typeid_Kind value (= union variant index).
// Named types return Typeid_Invalid (0) → the Named variant (variants[0]).
gb_internal u64 x64_typeid_kind(Type *type) {
	type = default_type(type);
	Type *bt = base_type(type);
	u64 kind = Typeid_Invalid;
	switch (bt->kind) {
	case Type_Basic: {
		u32 flags = bt->Basic.flags;
		if (flags & BasicFlag_Boolean)  kind = Typeid_Boolean;
		if (flags & BasicFlag_Integer)  kind = Typeid_Integer;
		if (flags & BasicFlag_Unsigned) kind = Typeid_Integer;
		if (flags & BasicFlag_Float)    kind = Typeid_Float;
		if (flags & BasicFlag_Complex)  kind = Typeid_Complex;
		if (flags & BasicFlag_Quaternion) kind = Typeid_Quaternion; // was missing → quaternion type_info tagged Named → fmt read a garbage base → crash
		if (flags & BasicFlag_Pointer)  kind = Typeid_Pointer;
		if (flags & BasicFlag_String)   kind = Typeid_String;
		if (flags & BasicFlag_Rune)     kind = Typeid_Rune;
		if (bt->Basic.kind == Basic_typeid) kind = Typeid_Type_Id;
		break;
	}
	case Type_Pointer:                   kind = Typeid_Pointer;          break;
	case Type_MultiPointer:              kind = Typeid_Multi_Pointer;    break;
	case Type_Array:                     kind = Typeid_Array;            break;
	case Type_EnumeratedArray:           kind = Typeid_Enumerated_Array; break;
	case Type_Slice:                     kind = Typeid_Slice;            break;
	case Type_DynamicArray:              kind = Typeid_Dynamic_Array;    break;
	case Type_Map:                       kind = Typeid_Map;              break;
	case Type_Struct:                    kind = Typeid_Struct;           break;
	case Type_Enum:                      kind = Typeid_Enum;             break;
	case Type_Union:                     kind = Typeid_Union;            break;
	case Type_Tuple:                     kind = Typeid_Tuple;            break;
	case Type_Proc:                      kind = Typeid_Procedure;        break;
	case Type_BitSet:                    kind = Typeid_Bit_Set;          break;
	case Type_SimdVector:                kind = Typeid_Simd_Vector;      break;
	case Type_SoaPointer:                kind = Typeid_SoaPointer;       break;
	case Type_BitField:                  kind = Typeid_Bit_Field;        break;
	case Type_Matrix:                    kind = Typeid_Matrix;           break;
	case Type_FixedCapacityDynamicArray: kind = Typeid_Fixed_Capacity_Dynamic_Array; break;
	}
	return kind;
}

gb_internal String x64_ti_slot_sym(isize slot) {
	char b[48]; gb_snprintf(b, 47, "__$ti$%d", (int)slot);
	return copy_string(permanent_allocator(), make_string_c(b));
}
gb_internal String x64_ti_sym(char const *prefix, isize slot, char const *suffix) {
	char b[64]; gb_snprintf(b, 63, "__$%s$%d$%s", prefix, (int)slot, suffix);
	return copy_string(permanent_allocator(), make_string_c(b));
}

// [^]string sub-array: count × {ptr,len}; strings from fields->token.string or tags.
gb_internal String x64_ti_emit_strings(x64Module *rm, String name, isize count, Entity **fields, String *tags) {
	coff_section_align(rm->rdata, 8);
	u32 off = (u32)coff_section_len(rm->rdata);
	for (isize k = 0; k < count; k++) {
		u64 z = 0; i64 l = fields ? fields[k]->token.string.len : (tags ? tags[k].len : 0);
		coff_section_write(rm->rdata, &z, 8);
		coff_section_write(rm->rdata, &l, 8);
	}
	for (isize k = 0; k < count; k++) {
		String s = fields ? fields[k]->token.string : (tags ? tags[k] : String{});
		if (s.len == 0) continue;
		coff_reloc_add(rm->rdata, off + (u32)k*16, x64_const_intern_string(rm, s), COFF_REL_ADDR64);
	}
	coff_sym_add_proc(&rm->coff, name, 2, off, false);
	return name;
}

// [^]^Type_Info sub-array: count × ptr, reloc'd to each elem type's slot (nil if unemitted).
gb_internal String x64_ti_emit_pointers(x64Module *rm, CheckerInfo *info, bool *slot_emit, isize n,
                                        String name, isize count, Entity **fields, Type **variants) {
	coff_section_align(rm->rdata, 8);
	u32 off = (u32)coff_section_len(rm->rdata);
	for (isize k = 0; k < count; k++) { u64 z = 0; coff_section_write(rm->rdata, &z, 8); }
	for (isize k = 0; k < count; k++) {
		Type *et = fields ? fields[k]->type : variants[k];
		isize fs = type_info_index(info, et, false);
		if (fs >= 0 && fs < n && slot_emit[fs]) coff_reloc_add(rm->rdata, off + (u32)k*8, x64_ti_slot_sym(fs), COFF_REL_ADDR64);
	}
	coff_sym_add_proc(&rm->coff, name, 2, off, false);
	return name;
}

// Emit runtime.type_table so __type_info_of can do its id%n lookup.
gb_internal void x64_emit_type_table(x64Generator *gen) {
	CheckerInfo *info = gen->info;
	if (t_type_info == nullptr) return;

	isize n = (isize)info->type_info_types_hash_map.count;
	if (n == 0) return;

	// Runtime PACKAGE module (file == null) — stable home for the single type_table
	// global even when the runtime is split into per-file modules.
	x64Module *rm = nullptr;
	for (auto const &entry : gen->modules) {
		if (entry.value->pkg->kind == Package_Runtime && entry.value->file == nullptr) {
			rm = entry.value;
			break;
		}
	}
	if (rm == nullptr) return;

	// Type_Info struct layout from Odin's type system
	Type *ti_bt = base_type(t_type_info);
	GB_ASSERT(ti_bt->kind == Type_Struct && ti_bt->Struct.fields.count == 5);
	Type *variant_union = base_type(ti_bt->Struct.fields[4]->type);
	GB_ASSERT(variant_union->kind == Type_Union);

	i64 ti_size  = type_size_of(t_type_info);
	i64 ti_align = type_align_of(t_type_info);
	i64 off_sz   = type_offset_of(t_type_info, 0);
	i64 off_al   = type_offset_of(t_type_info, 1);
	i64 off_fl   = type_offset_of(t_type_info, 2);
	i64 off_id   = type_offset_of(t_type_info, 3);
	i64 off_var  = type_offset_of(t_type_info, 4);
	i64 vblock   = variant_union->Union.variant_block_size.load();
	i64 off_tag  = off_var + vblock;

	Arena         *scratch = get_arena(ThreadArena_Temporary);
	ArenaTempGuard scratch_guard(scratch);
	gbAllocator    a = arena_allocator(scratch);

	// Type_Info_Struct field offsets (relative to the variant payload at off_var):
	// 0 types, 1 names, 2 offsets, 3 usings, 4 tags, 5 field_count, 6 flags.
	i64 sf_types = 0, sf_names = 8, sf_offsets = 16, sf_usings = 24, sf_tags = 32, sf_fc = 40, sf_flags = 44;
	if (t_type_info_struct != nullptr) {
		Type *tis = base_type(t_type_info_struct);
		if (tis->kind == Type_Struct && tis->Struct.fields.count >= 7) {
			sf_types   = type_offset_of(t_type_info_struct, 0);
			sf_names   = type_offset_of(t_type_info_struct, 1);
			sf_offsets = type_offset_of(t_type_info_struct, 2);
			sf_usings  = type_offset_of(t_type_info_struct, 3);
			sf_tags    = type_offset_of(t_type_info_struct, 4);
			sf_fc      = type_offset_of(t_type_info_struct, 5);
			sf_flags   = type_offset_of(t_type_info_struct, 6);
		}
	}

	// Pass 1: mark which slots get a Type_Info and remember the type. Per-slot symbol
	// names are deterministic ("__$ti$<slot>") so a struct's types[] can reference
	// field-type slots emitted later in pass 2 (forward reloc-by-name resolves at link).
	Array<bool>  slot_emit; array_init(&slot_emit, a, (isize)n, (isize)n);
	Array<Type*> slot_type; array_init(&slot_type, a, (isize)n, (isize)n);
	for (isize i = 0; i < n; i++) { slot_emit[i] = false; slot_type[i] = nullptr; }

	for (isize i = 0; i < n; i++) {
		TypeInfoPair const &pair = info->type_info_types_hash_map[i];
		if (pair.hash == 0 || pair.type == nullptr || pair.type == t_invalid) continue;
		Type *t  = pair.type;
		Type *bt = base_type(t);
		bool emit = false;
		if      (bt->kind == Type_Basic)  emit = !(bt->Basic.flags & BasicFlag_Untyped);
		else if (bt->kind == Type_Struct) emit = bt->Struct.soa_kind == StructSoa_None; // incl. #raw_union
		else if (bt->kind == Type_BitField) emit = true;
		else if (bt->kind == Type_Pointer || bt->kind == Type_MultiPointer ||
		         bt->kind == Type_Slice   || bt->kind == Type_DynamicArray  ||
		         bt->kind == Type_Array   || bt->kind == Type_FixedCapacityDynamicArray) emit = true; // elem-based variants
		else if (bt->kind == Type_Proc)   emit = true;
		else if (bt->kind == Type_Enum)   emit = true;
		else if (bt->kind == Type_Union)  emit = true;
		else if (bt->kind == Type_Map)    emit = true;
		else if (bt->kind == Type_Matrix || bt->kind == Type_SimdVector ||
		         bt->kind == Type_BitSet || bt->kind == Type_EnumeratedArray) emit = true; // were unemitted → type_info nil/garbage → fmt(matrix)/flags(bit_set) crash
		if (!emit) continue;
		isize slot = type_info_index(info, pair, false);
		if (slot < 0 || slot >= n) continue;
		slot_emit[slot] = true;
		slot_type[slot] = t;
	}

	// Pass 2: emit each slot's Type_Info (and, for structs, the five parallel arrays).
	for (isize slot = 0; slot < n; slot++) {
		if (!slot_emit[slot]) continue;
		Type *t  = slot_type[slot];
		Type *bt = base_type(t);
		// Named types (distinct/aliased: time.Duration, MyDur :: distinct i64, named structs)
		// emit a Type_Info_Named{name, base, pkg, loc} pointing at the base type's slot — NOT
		// the base variant. Without this, fmt's named custom-formatters (Duration/Time/SCL)
		// never fire and reflect.Type_Info_Named matching fails. Mirrors lb_type_info Type_Named.
		bool is_named = (t->kind == Type_Named);

		// Struct variant: emit names/types/offsets/usings/tags arrays first, capturing
		// the symbol that names element 0 of each (mirrors lb_type_info Type_Struct).
		String a_types = {}, a_names = {}, a_offsets = {}, a_usings = {}, a_tags = {};
		i32  field_count = 0;
		u8   struct_flags = 0;
		Type *ci_elem = nullptr; // elem-based variants (pointer/slice/dyn-array/array/matrix/simd/bit_set/enum-array): elem @vd+0
		Type *ci_elem2 = nullptr; i64 ci_elem2_off = 0; // secondary ^Type_Info (bit_set.underlying / enum-array.index)
		String e_names = {}, e_values = {}; // enum names/values arrays
		i32   e_count = 0;
		Type *e_base  = nullptr;
		String u_variants = {}; // union variants [^]^Type_Info array
		i32   u_count = 0;
		String bf_names = {}, bf_types = {}, bf_bsizes = {}, bf_boffs = {}, bf_tags = {}; // bit_field arrays
		i32   bf_count = 0;
		Type *bf_backing = nullptr;
		if (is_named) {
			// no base sub-arrays for a Named variant
		} else if (bt->kind == Type_Struct) {
			type_set_offsets(bt);
			isize cnt = bt->Struct.fields.count;
			field_count = (i32)cnt;
			if (bt->Struct.is_packed)    struct_flags |= 1u<<0;
			if (bt->Struct.is_raw_union) struct_flags |= 1u<<1;
			if (bt->Struct.custom_align) struct_flags |= 1u<<3;

			if (cnt > 0) {
				a_names = x64_ti_emit_strings(rm, x64_ti_sym("tia", slot, "names"), cnt, bt->Struct.fields.data, nullptr);
				a_tags  = x64_ti_emit_strings(rm, x64_ti_sym("tia", slot, "tags"),  cnt, nullptr, bt->Struct.tags);
				a_types = x64_ti_emit_pointers(rm, info, slot_emit.data, n, x64_ti_sym("tia", slot, "types"), cnt, bt->Struct.fields.data, nullptr);

				coff_section_align(rm->rdata, 8); // offsets: [^]uintptr
				u32 offsets_off = (u32)coff_section_len(rm->rdata);
				for (isize k = 0; k < cnt; k++) { i64 fo=(bt->Struct.offsets!=nullptr)?bt->Struct.offsets[k]:0; coff_section_write(rm->rdata,&fo,8); }
				a_offsets = x64_ti_sym("tia", slot, "offsets"); coff_sym_add_proc(&rm->coff, a_offsets, 2, offsets_off, false);

				coff_section_align(rm->rdata, 1); // usings: [^]bool
				u32 usings_off = (u32)coff_section_len(rm->rdata);
				for (isize k = 0; k < cnt; k++) { u8 u=(bt->Struct.fields[k]->flags & EntityFlag_Using)?1:0; coff_section_write_u8(rm->rdata, u); }
				a_usings = x64_ti_sym("tia", slot, "usings"); coff_sym_add_proc(&rm->coff, a_usings, 2, usings_off, false);
			}
		} else if (bt->kind == Type_Enum) {
			e_base = bt->Enum.base_type;
			isize ecnt = bt->Enum.fields.count;
			e_count = (i32)ecnt;
			if (ecnt > 0) {
				e_names = x64_ti_emit_strings(rm, x64_ti_sym("tie", slot, "names"), ecnt, bt->Enum.fields.data, nullptr);
				i64 evsz = (t_type_info_enum_value != nullptr) ? type_size_of(t_type_info_enum_value) : 8;
				if (evsz <= 0) evsz = 8;
				coff_section_align(rm->rdata, 8); // values: each = the member's integer constant
				u32 voff = (u32)coff_section_len(rm->rdata);
				for (isize k = 0; k < ecnt; k++) {
					Entity *fe = bt->Enum.fields[k];
					i64 val = (fe->kind == Entity_Constant && fe->Constant.value.kind == ExactValue_Integer)
					        ? big_int_to_i64(&fe->Constant.value.value_integer) : 0;
					for (i64 b = 0; b < evsz; b++) coff_section_write_u8(rm->rdata, (b < 8) ? (u8)((u64)val >> (b*8)) : 0);
				}
				e_values = x64_ti_sym("tie", slot, "values");
				coff_sym_add_proc(&rm->coff, e_values, 2, voff, false);
			}
		} else if (bt->kind == Type_Union) {
			isize ucnt = bt->Union.variants.count;
			u_count = (i32)ucnt;
			if (ucnt > 0) {
				u_variants = x64_ti_emit_pointers(rm, info, slot_emit.data, n, x64_ti_sym("tiu", slot, "variants"), ucnt, nullptr, bt->Union.variants.data);
			}
		} else if (bt->kind == Type_BitField) {
			bf_backing = bt->BitField.backing_type;
			isize bcnt = bt->BitField.fields.count;
			bf_count = (i32)bcnt;
			if (bcnt > 0) {
				bf_names = x64_ti_emit_strings(rm, x64_ti_sym("tibf", slot, "names"), bcnt, bt->BitField.fields.data, nullptr);
				bf_types = x64_ti_emit_pointers(rm, info, slot_emit.data, n, x64_ti_sym("tibf", slot, "types"), bcnt, bt->BitField.fields.data, nullptr);
				coff_section_align(rm->rdata, 8); // bit_sizes: [^]uintptr
				u32 bsoff = (u32)coff_section_len(rm->rdata);
				for (isize k = 0; k < bcnt; k++) { i64 v = (k < bt->BitField.bit_sizes.count) ? (i64)bt->BitField.bit_sizes[k] : 0; coff_section_write(rm->rdata,&v,8); }
				bf_bsizes = x64_ti_sym("tibf", slot, "bsizes"); coff_sym_add_proc(&rm->coff, bf_bsizes, 2, bsoff, false);
				coff_section_align(rm->rdata, 8); // bit_offsets: [^]uintptr
				u32 booff = (u32)coff_section_len(rm->rdata);
				for (isize k = 0; k < bcnt; k++) { i64 v = (k < bt->BitField.bit_offsets.count) ? bt->BitField.bit_offsets[k] : 0; coff_section_write(rm->rdata,&v,8); }
				bf_boffs = x64_ti_sym("tibf", slot, "boffs"); coff_sym_add_proc(&rm->coff, bf_boffs, 2, booff, false);
				bf_tags = x64_ti_emit_strings(rm, x64_ti_sym("tibf", slot, "tags"), bcnt, nullptr, bt->BitField.tags);
			}
		}

		u8 *bytes = gb_alloc_array(a, u8, (isize)ti_size);
		gb_zero_size(bytes, (isize)ti_size);

		// Common fields
		i64 sz    = type_size_of(t);
		i64 al    = type_align_of(t);
		u32 flags = type_info_flags_of_type(t);
		u64 id    = type_hash_canonical_type(t);
		gb_memmove(bytes + off_sz,  &sz,    8);
		gb_memmove(bytes + off_al,  &al,    8);
		gb_memmove(bytes + off_fl,  &flags, 4);
		gb_memmove(bytes + off_id,  &id,    8);

		// Union tag (1-indexed variant index). Type_Info_Named is variants[0] → tag 1.
		u64 tag = is_named ? 1 : (x64_typeid_kind(t) + 1);
		gb_memmove(bytes + off_tag, &tag, 8);

		// Variant-specific data (at off_var)
		u8 *vd = bytes + (isize)off_var;
		if (is_named) {
			// Type_Info_Named{name: string, base: ^Type_Info, pkg: string, loc: ^SCL}.
			// Scalar lengths here; data ptrs + base via relocs below.
			i64 nm_off = 0, pkg_off = 24;
			if (t_type_info_named != nullptr) {
				Type *tin = base_type(t_type_info_named);
				if (tin->kind == Type_Struct && tin->Struct.fields.count >= 3) {
					nm_off  = type_offset_of(t_type_info_named, 0);
					pkg_off = type_offset_of(t_type_info_named, 2);
				}
			}
			String nm = t->Named.type_name->token.string;
			String pk = (t->Named.type_name->pkg != nullptr) ? t->Named.type_name->pkg->name : String{};
			i64 nl = nm.len, pl = pk.len;
			if (nl) gb_memmove(vd + nm_off  + 8, &nl, 8);
			if (pl) gb_memmove(vd + pkg_off + 8, &pl, 8);
		} else if (bt->kind == Type_Struct) {
			// Array pointers (vd+sf_*) are filled by ADDR64 relocs below; here write the
			// scalar fields (field_count, flags). Pointers stay zero in the buffer.
			gb_memmove(vd + sf_fc,    &field_count,  4);
			gb_memmove(vd + sf_flags, &struct_flags, 1);
		} else if (bt->kind == Type_Pointer || bt->kind == Type_MultiPointer ||
		           bt->kind == Type_Slice   || bt->kind == Type_DynamicArray  ||
		           bt->kind == Type_Array   || bt->kind == Type_FixedCapacityDynamicArray) {
			// Elem-based variants: elem: ^Type_Info @vd+0 (ADDR64 reloc below),
			// elem_size: int @vd+8 (slice/dyn-array/array/fixed-cap), count: int @vd+16 (array/fixed-cap).
			i64 esz = 0, cnt = 0;
			bool has_count = false;
			switch (bt->kind) {
			case Type_Pointer:      ci_elem = bt->Pointer.elem;      break;
			case Type_MultiPointer: ci_elem = bt->MultiPointer.elem; break;
			case Type_Slice:        ci_elem = bt->Slice.elem;        esz = ci_elem ? type_size_of(ci_elem) : 0; break;
			case Type_DynamicArray: ci_elem = bt->DynamicArray.elem; esz = ci_elem ? type_size_of(ci_elem) : 0; break;
			case Type_Array:        ci_elem = bt->Array.elem;        esz = ci_elem ? type_size_of(ci_elem) : 0; cnt = bt->Array.count; has_count = true; break;
			case Type_FixedCapacityDynamicArray: ci_elem = bt->FixedCapacityDynamicArray.elem; esz = ci_elem ? type_size_of(ci_elem) : 0; cnt = bt->FixedCapacityDynamicArray.capacity; has_count = true; break;
			default: break;
			}
			if (esz != 0)   gb_memmove(vd + 8,  &esz, 8);
			if (has_count)  gb_memmove(vd + 16, &cnt, 8);
		} else if (bt->kind == Type_Proc) {
			// Type_Info_Procedure: params/results (^Type_Info) left nil (Type_Info_Parameters
			// not emitted yet), variadic: bool, convention: u8 (= calling_convention).
			// Mirrors lb_type_info Type_Proc; enough for `field.type.variant.(Type_Info_Procedure)`.
			i64 voff = 16, coff = 17, csz = 1;
			if (t_type_info_procedure != nullptr) {
				Type *tip = base_type(t_type_info_procedure);
				if (tip->kind == Type_Struct && tip->Struct.fields.count >= 4) {
					voff = type_offset_of(t_type_info_procedure, 2);
					coff = type_offset_of(t_type_info_procedure, 3);
					csz  = type_size_of(tip->Struct.fields[3]->type); if (csz <= 0) csz = 1;
				}
			}
			u8 variadic = bt->Proc.variadic ? 1 : 0;
			u64 conv = (u64)bt->Proc.calling_convention;
			gb_memmove(vd + (isize)voff, &variadic, 1);
			gb_memmove(vd + (isize)coff, &conv, (isize)gb_min(csz, (i64)8));
		} else if (bt->kind == Type_Enum) {
			// names/values slice LENs here; base + data ptrs via relocs below.
			i64 en_off = 8, ev_off = 24;
			if (t_type_info_enum != nullptr) {
				Type *tie = base_type(t_type_info_enum);
				if (tie->kind == Type_Struct && tie->Struct.fields.count >= 3) {
					en_off = type_offset_of(t_type_info_enum, 1);
					ev_off = type_offset_of(t_type_info_enum, 2);
				}
			}
			i64 cnt64 = e_count;
			if (e_names.len)  gb_memmove(vd + en_off + 8, &cnt64, 8);
			if (e_values.len) gb_memmove(vd + ev_off + 8, &cnt64, 8);
		} else if (bt->kind == Type_Union) {
			// variants.len, tag_offset, custom_align/no_nil/shared_nil; ptr relocs below.
			i64 ov = 0, otoff = 16, ott = 24, oca = 40, onn = 41, osn = 42;
			if (t_type_info_union != nullptr) {
				Type *tiu = base_type(t_type_info_union);
				if (tiu->kind == Type_Struct && tiu->Struct.fields.count >= 7) {
					ov = type_offset_of(t_type_info_union, 0);
					otoff = type_offset_of(t_type_info_union, 1);
					ott = type_offset_of(t_type_info_union, 2); (void)ott;
					oca = type_offset_of(t_type_info_union, 4);
					onn = type_offset_of(t_type_info_union, 5);
					osn = type_offset_of(t_type_info_union, 6);
				}
			}
			i64 ucnt = u_count;
			if (u_variants.len) gb_memmove(vd + ov + 8, &ucnt, 8); // variants.len
			i64 tag_sz = union_tag_size(bt);
			if (tag_sz > 0) {
				i64 vbs = (i64)bt->Union.variant_block_size;
				i64 tag_off = (vbs + tag_sz - 1) & ~(tag_sz - 1);
				gb_memmove(vd + otoff, &tag_off, 8); // tag_offset: uintptr
			}
			u8 ca = bt->Union.custom_align != 0 ? 1 : 0;
			u8 nn = bt->Union.kind == UnionType_no_nil ? 1 : 0;
			u8 sn = bt->Union.kind == UnionType_shared_nil ? 1 : 0;
			vd[oca] = ca; vd[onn] = nn; vd[osn] = sn;
		} else if (bt->kind == Type_BitField) {
			// field_count; backing_type + array data ptrs via relocs below.
			i64 fcoff = 48;
			if (t_type_info_bit_field != nullptr) {
				Type *tibf = base_type(t_type_info_bit_field);
				if (tibf->kind == Type_Struct && tibf->Struct.fields.count >= 7) {
					fcoff = type_offset_of(t_type_info_bit_field, 6);
				}
			}
			i64 cnt64 = bf_count;
			gb_memmove(vd + fcoff, &cnt64, 8);
		} else if (bt->kind == Type_Map) {
			// key/value (^Type_Info) via relocs below; map_info (^Map_Info) left nil
			// (reflection's variant check doesn't use it).
		} else if (bt->kind == Type_Matrix) {
			// Type_Info_Matrix{elem@0, elem_size@8, elem_stride@16, row_count@24, column_count@32,
			// layout(u8)@40}. elem ptr via ci_elem reloc. Mirrors lb_type_info Type_Matrix.
			ci_elem = bt->Matrix.elem;
			i64 esz = ci_elem ? type_size_of(ci_elem) : 0;
			i64 stride = matrix_type_stride_in_elems(bt);
			i64 rc = bt->Matrix.row_count, cc = bt->Matrix.column_count;
			gb_memmove(vd + 8,  &esz,    8);
			gb_memmove(vd + 16, &stride, 8);
			gb_memmove(vd + 24, &rc,     8);
			gb_memmove(vd + 32, &cc,     8);
			vd[40] = bt->Matrix.is_row_major ? 1 : 0;
		} else if (bt->kind == Type_SimdVector) {
			// Type_Info_Simd_Vector{elem@0, elem_size@8, count@16}.
			ci_elem = bt->SimdVector.elem;
			i64 esz = ci_elem ? type_size_of(ci_elem) : 0;
			i64 cnt = bt->SimdVector.count;
			gb_memmove(vd + 8,  &esz, 8);
			gb_memmove(vd + 16, &cnt, 8);
		} else if (bt->kind == Type_BitSet) {
			// Type_Info_Bit_Set{elem@0, underlying@8, explicit_underlying(bool)@16, lower(i64)@24,
			// upper(i64)@32}. underlying reloc only when EXPLICIT (bit_set[E; U]); mirrors LLVM.
			ci_elem = bt->BitSet.elem;
			i64 lo = bt->BitSet.lower, up = bt->BitSet.upper;
			if (bt->BitSet.underlying != nullptr) { ci_elem2 = bt->BitSet.underlying; ci_elem2_off = 8; vd[16] = 1; }
			gb_memmove(vd + 24, &lo, 8);
			gb_memmove(vd + 32, &up, 8);
		} else if (bt->kind == Type_EnumeratedArray) {
			// Type_Info_Enumerated_Array{elem@0, index@8, elem_size@16, count@24, min_value@32,
			// max_value@40, is_sparse@48}. min/max (Type_Info_Enum_Value = i64) written directly.
			ci_elem = bt->EnumeratedArray.elem;
			ci_elem2 = bt->EnumeratedArray.index; ci_elem2_off = 8;
			i64 esz = ci_elem ? type_size_of(ci_elem) : 0;
			i64 cnt = bt->EnumeratedArray.count;
			gb_memmove(vd + 16, &esz, 8);
			gb_memmove(vd + 24, &cnt, 8);
			if (bt->EnumeratedArray.min_value) { i64 mn = exact_value_to_i64(*bt->EnumeratedArray.min_value); gb_memmove(vd + 32, &mn, 8); }
			if (bt->EnumeratedArray.max_value) { i64 mx = exact_value_to_i64(*bt->EnumeratedArray.max_value); gb_memmove(vd + 40, &mx, 8); }
			vd[48] = bt->EnumeratedArray.is_sparse ? 1 : 0;
		} else
		switch (bt->Basic.kind) {
		case Basic_i8:    case Basic_u8:
		case Basic_i16:   case Basic_u16:
		case Basic_i32:   case Basic_u32:
		case Basic_i64:   case Basic_u64:
		case Basic_i128:  case Basic_u128:
		case Basic_i16le: case Basic_u16le: case Basic_i32le: case Basic_u32le:
		case Basic_i64le: case Basic_u64le: case Basic_i128le: case Basic_u128le:
		case Basic_i16be: case Basic_u16be: case Basic_i32be: case Basic_u32be:
		case Basic_i64be: case Basic_u64be: case Basic_i128be: case Basic_u128be:
		case Basic_int: case Basic_uint: case Basic_uintptr: {
			// Type_Info_Integer: {signed: bool, endianness: u8}
			vd[0] = (bt->Basic.flags & BasicFlag_Unsigned) ? 0 : 1;
			u8 endian = 0;
			if (bt->Basic.flags & BasicFlag_EndianLittle) endian = 1;
			else if (bt->Basic.flags & BasicFlag_EndianBig) endian = 2;
			vd[1] = endian;
			break;
		}
		case Basic_f16: case Basic_f32: case Basic_f64:
		case Basic_f16le: case Basic_f32le: case Basic_f64le:
		case Basic_f16be: case Basic_f32be: case Basic_f64be: {
			// Type_Info_Float: {endianness: u8}
			u8 endian = 0;
			if (bt->Basic.flags & BasicFlag_EndianLittle) endian = 1;
			else if (bt->Basic.flags & BasicFlag_EndianBig) endian = 2;
			vd[0] = endian;
			break;
		}
		case Basic_string: {
			// Type_Info_String: {is_cstring: bool, encoding: u8}
			vd[0] = 0; vd[1] = 0; // is_cstring=false, UTF_8
			break;
		}
		case Basic_cstring: {
			vd[0] = 1; vd[1] = 0; // is_cstring=true, UTF_8
			break;
		}
		case Basic_string16: {
			vd[0] = 0; vd[1] = 1; // is_cstring=false, UTF_16
			break;
		}
		case Basic_cstring16: {
			vd[0] = 1; vd[1] = 1; // is_cstring=true, UTF_16
			break;
		}
		default:
			// Boolean, Rune, Any, Type_Id, Complex, Quaternion, rawptr: zero variant data
			break;
		}

		coff_section_align(rm->rdata, (isize)ti_align);
		u32 rdata_off = (u32)coff_section_len(rm->rdata);
		coff_section_write(rm->rdata, bytes, (isize)ti_size);
		gb_free(a, bytes);

		// Internal symbol so the pointer table reloc can reference it
		coff_sym_add_proc(&rm->coff, x64_ti_slot_sym(slot), 2 /*rdata*/, rdata_off, false);

		// Named variant: name.data + pkg.data string ptrs; base → base type's Type_Info slot.
		if (is_named) {
			i64 nm_off = 0, base_off = 16, pkg_off = 24;
			if (t_type_info_named != nullptr) {
				Type *tin = base_type(t_type_info_named);
				if (tin->kind == Type_Struct && tin->Struct.fields.count >= 3) {
					nm_off   = type_offset_of(t_type_info_named, 0);
					base_off = type_offset_of(t_type_info_named, 1);
					pkg_off  = type_offset_of(t_type_info_named, 2);
				}
			}
			String nm = t->Named.type_name->token.string;
			String pk = (t->Named.type_name->pkg != nullptr) ? t->Named.type_name->pkg->name : String{};
			if (nm.len) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)nm_off,  x64_const_intern_string(rm, nm), COFF_REL_ADDR64);
			if (pk.len) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)pkg_off, x64_const_intern_string(rm, pk), COFF_REL_ADDR64);
			Type *base = t->Named.base;
			if (base != nullptr) {
				isize bs = type_info_index(info, base, false);
				if (bs >= 0 && bs < n && slot_emit[bs]) {
					coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)base_off, x64_ti_slot_sym(bs), COFF_REL_ADDR64);
				}
			}
		}

		// Struct variant: patch the five array pointers (vd+sf_*) with ADDR64 relocs.
		if (bt->kind == Type_Struct) {
			u32 vbase = rdata_off + (u32)off_var;
			if (a_types.len)   coff_reloc_add(rm->rdata, vbase + (u32)sf_types,   a_types,   COFF_REL_ADDR64);
			if (a_names.len)   coff_reloc_add(rm->rdata, vbase + (u32)sf_names,   a_names,   COFF_REL_ADDR64);
			if (a_offsets.len) coff_reloc_add(rm->rdata, vbase + (u32)sf_offsets, a_offsets, COFF_REL_ADDR64);
			if (a_usings.len)  coff_reloc_add(rm->rdata, vbase + (u32)sf_usings,  a_usings,  COFF_REL_ADDR64);
			if (a_tags.len)    coff_reloc_add(rm->rdata, vbase + (u32)sf_tags,    a_tags,    COFF_REL_ADDR64);
		}

		// Elem-based variant: point `elem` (@off_var+0) at the element type's Type_Info.
		if (ci_elem != nullptr) {
			isize es = type_info_index(info, ci_elem, false);
			if (es >= 0 && es < n && slot_emit[es]) {
				coff_reloc_add(rm->rdata, rdata_off + (u32)off_var, x64_ti_slot_sym(es), COFF_REL_ADDR64);
			}
		}
		// Secondary ^Type_Info (bit_set.underlying @+8 when explicit / enum-array.index @+8).
		if (ci_elem2 != nullptr) {
			isize es = type_info_index(info, ci_elem2, false);
			if (es >= 0 && es < n && slot_emit[es]) {
				coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)ci_elem2_off, x64_ti_slot_sym(es), COFF_REL_ADDR64);
			}
		}

		// Enum variant: base (@off_var+0) → underlying int type; names/values data ptrs.
		if (bt->kind == Type_Enum) {
			i64 en_off = 8, ev_off = 24;
			if (t_type_info_enum != nullptr) {
				Type *tie = base_type(t_type_info_enum);
				if (tie->kind == Type_Struct && tie->Struct.fields.count >= 3) {
					en_off = type_offset_of(t_type_info_enum, 1);
					ev_off = type_offset_of(t_type_info_enum, 2);
				}
			}
			if (e_base != nullptr) {
				isize bs = type_info_index(info, e_base, false);
				if (bs >= 0 && bs < n && slot_emit[bs]) {
					coff_reloc_add(rm->rdata, rdata_off + (u32)off_var, x64_ti_slot_sym(bs), COFF_REL_ADDR64);
				}
			}
			if (e_names.len)  coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)en_off, e_names,  COFF_REL_ADDR64);
			if (e_values.len) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)ev_off, e_values, COFF_REL_ADDR64);
		}

		// Union variant: variants.data ptr + tag_type ptr.
		if (bt->kind == Type_Union) {
			i64 ov = 0, ott = 24;
			if (t_type_info_union != nullptr) {
				Type *tiu = base_type(t_type_info_union);
				if (tiu->kind == Type_Struct && tiu->Struct.fields.count >= 7) {
					ov  = type_offset_of(t_type_info_union, 0);
					ott = type_offset_of(t_type_info_union, 2);
				}
			}
			if (u_variants.len) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)ov, u_variants, COFF_REL_ADDR64);
			if (union_tag_size(bt) > 0) {
				Type *tt = union_tag_type(bt);
				if (tt != nullptr) {
					isize ts = type_info_index(info, tt, false);
					if (ts >= 0 && ts < n && slot_emit[ts]) {
						coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)ott, x64_ti_slot_sym(ts), COFF_REL_ADDR64);
					}
				}
			}
		}

		// Bit_Field variant: backing_type (@+0) + names/types/bit_sizes/bit_offsets/tags ptrs.
		if (bt->kind == Type_BitField) {
			i64 o_nm=8, o_ty=16, o_bs=24, o_bo=32, o_tg=40;
			if (t_type_info_bit_field != nullptr) {
				Type *tibf = base_type(t_type_info_bit_field);
				if (tibf->kind == Type_Struct && tibf->Struct.fields.count >= 7) {
					o_nm = type_offset_of(t_type_info_bit_field, 1);
					o_ty = type_offset_of(t_type_info_bit_field, 2);
					o_bs = type_offset_of(t_type_info_bit_field, 3);
					o_bo = type_offset_of(t_type_info_bit_field, 4);
					o_tg = type_offset_of(t_type_info_bit_field, 5);
				}
			}
			if (bf_backing != nullptr) {
				isize bs = type_info_index(info, bf_backing, false);
				if (bs >= 0 && bs < n && slot_emit[bs]) {
					coff_reloc_add(rm->rdata, rdata_off + (u32)off_var, x64_ti_slot_sym(bs), COFF_REL_ADDR64);
				}
			}
			if (bf_names.len)  coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)o_nm, bf_names,  COFF_REL_ADDR64);
			if (bf_types.len)  coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)o_ty, bf_types,  COFF_REL_ADDR64);
			if (bf_bsizes.len) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)o_bs, bf_bsizes, COFF_REL_ADDR64);
			if (bf_boffs.len)  coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)o_bo, bf_boffs,  COFF_REL_ADDR64);
			if (bf_tags.len)   coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)o_tg, bf_tags,   COFF_REL_ADDR64);
		}

		// Map variant: key (@0) + value (@8) + map_info (@16). map_info was left nil, but reflection
		// DOES use it — core:flags calls `type_info.map_info.key_hasher(...)` through the reflected
		// type_info → null-call segfault (map[cstring]cstring test hung). Point it at the same
		// {ks,vs,key_hasher,key_equal} Map_Info global the compiled map ops use.
		if (bt->kind == Type_Map) {
			i64 ok2 = 0, ov2 = 8, om2 = 16;
			if (t_type_info_map != nullptr) {
				Type *tim = base_type(t_type_info_map);
				if (tim->kind == Type_Struct && tim->Struct.fields.count >= 3) {
					ok2 = type_offset_of(t_type_info_map, 0);
					ov2 = type_offset_of(t_type_info_map, 1);
					om2 = type_offset_of(t_type_info_map, 2);
				}
			}
			if (bt->Map.key != nullptr) {
				isize ks = type_info_index(info, bt->Map.key, false);
				if (ks >= 0 && ks < n && slot_emit[ks]) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)ok2, x64_ti_slot_sym(ks), COFF_REL_ADDR64);
			}
			if (bt->Map.value != nullptr) {
				isize vs = type_info_index(info, bt->Map.value, false);
				if (vs >= 0 && vs < n && slot_emit[vs]) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)ov2, x64_ti_slot_sym(vs), COFF_REL_ADDR64);
			}
			// x64_gen_map_info_ptr is dedup'd: for a map already used in compiled code it returns the
			// cached symbol (no new work); for a reflection-only map it emits the global + enqueues the
			// synth hasher/equal — drained right after x64_emit_type_table (else unresolved external).
			String mi_sym = x64_gen_map_info_ptr(rm, bt);
			if (mi_sym.len > 0) coff_reloc_add(rm->rdata, rdata_off + (u32)off_var + (u32)om2, mi_sym, COFF_REL_ADDR64);
		}
	}

	// Pointer table in .rdata: n consecutive 8-byte slots, patched by ADDR64 relocs below.
	coff_section_align(rm->rdata, 8);
	u32 ptr_table_off = (u32)coff_section_len(rm->rdata);
	{
		u64 zero = 0;
		for (isize i = 0; i < n; i++) coff_section_write(rm->rdata, &zero, 8);
	}
	for (isize i = 0; i < n; i++) {
		if (!slot_emit[i]) continue;
		u32 slot_off = ptr_table_off + (u32)i * 8;
		coff_reloc_add(rm->rdata, slot_off, x64_ti_slot_sym(i), COFF_REL_ADDR64);
	}
	String ptr_sym = str_lit("__$type_table_ptrs");
	coff_sym_add_proc(&rm->coff, ptr_sym, 2 /*rdata*/, ptr_table_off, false);

	// runtime.type_table slice in .data: {ptr: ^ptr_table[0], len: n}
	coff_section_align(rm->data, 8);
	u32 tt_data_off = (u32)coff_section_len(rm->data);
	{
		u64 zero = 0;
		coff_section_write(rm->data, &zero, 8); // ptr — filled by ADDR64 reloc
		i64 len_val = (i64)n;
		coff_section_write(rm->data, &len_val, 8); // len
	}
	coff_reloc_add(rm->data, tt_data_off, ptr_sym, COFF_REL_ADDR64);

	// Define runtime.type_table in .data. Procs referencing it (e.g. __type_info_of)
	// have already added it as an UNDEF external; upgrade that entry to DEFINED, or add
	// a fresh definition if not yet referenced.
	for (Entity *e : info->entities) {
		if (e->kind  != Entity_Variable) continue;
		if (e->pkg   == nullptr)         continue;
		if (e->pkg->kind != Package_Runtime) continue;
		if (e->token.string != str_lit("type_table")) continue;

		String tt_name   = x64_get_entity_name(e);
		u32 existing_idx = coff_sym_find(&rm->coff, tt_name);
		if (existing_idx != (u32)-1) {
			rm->coff.syms[existing_idx].value          = tt_data_off;
			rm->coff.syms[existing_idx].section_number = 3; // .data
			rm->coff.syms[existing_idx].storage_class  = COFF_SYM_CLASS_EXTERNAL;
		} else {
			u32 sym_idx = (u32)rm->coff.syms.count;
			CoffSymEntry sym_e = {};
			sym_e.name           = tt_name;
			sym_e.value          = tt_data_off;
			sym_e.section_number = 3; // .data
			sym_e.type           = COFF_SYM_TYPE_NULL;
			sym_e.storage_class  = COFF_SYM_CLASS_EXTERNAL;
			array_add(&rm->coff.syms, sym_e);
			string_map_set(&rm->coff.sym_map, tt_name, sym_idx);
		}
		break;
	}
}

// Resolve an entity's module: its per-file module if one exists (runtime split /
// -module-per-file), else its package module. Mirrors lb_module_of_entity_internal.
gb_internal x64Module *x64_module_of_entity(x64Generator *gen, Entity *e) {
	if (e->file != nullptr) {
		x64Module **m = map_get(&gen->modules, cast(void *)e->file);
		if (m != nullptr) return *m;
	}
	if (e->pkg != nullptr) {
		x64Module **m = map_get(&gen->modules, cast(void *)e->pkg);
		if (m != nullptr) return *m;
	}
	return nullptr;
}

// Enqueue a referenced min_dep==0 proc into its OWNER module's queue (compiled there once),
// falling back to the enclosing module if it has no mapping. MPSC-safe across worker threads.
gb_internal void x64_enqueue_oncall(x64Procedure *p, Entity *e) {
	x64Module *owner = x64_module_of_entity(p->module->gen, e);
	mpsc_enqueue(&(owner ? owner : p->module)->proc_queue, e);
}

// Compile one module's root procedures. Thread-pool task — one module per worker, so
// every write to this module's COFF happens on a single thread.
gb_internal WORKER_TASK_PROC(x64_compile_module_worker) {
	x64Module *m = cast(x64Module *)data;
	for_array(i, m->compile_roots) {
		Entity   *e    = m->compile_roots[i];
		DeclInfo *decl = decl_info_of_entity(e);
		if (decl == nullptr || decl->proc_lit == nullptr) continue;
		x64_compile_procedure(m, e, decl->proc_lit->ProcLit.body);
	}
	return 0;
}

// Finalize CodeView and serialize+write one module's COFF to disk. Thread-pool task —
// each module is an independent file; byte buffer from the module's own arena (no CRT
// heap lock). Results in m->obj_path/obj_failed, collected sequentially after the barrier.
gb_internal WORKER_TASK_PROC(x64_write_module_worker) {
	x64Module *m = cast(x64Module *)data;
	m->obj_path   = {};
	m->obj_failed = false;

	// Write any module with real content. .rdata matters: a package referenced only for
	// its read-only globals (e.g. encoding_base64's ENC_TABLE/DEC_TABLE) has empty text/
	// data/bss but a non-empty .rdata — skipping it leaves those globals undefined.
	if (coff_section_len(m->text)  == 0 &&
	    coff_section_len(m->rdata) == 0 &&
	    coff_section_len(m->data)  == 0 &&
	    coff_section_len(m->bss)   == 0) {
		return 0; // empty module — nothing to emit
	}

	x64_module_finalize_debug(m);

	String   output_base = build_context.build_paths[BuildPath_Output].basename;
	gbString path_s = gb_string_make_length(temporary_allocator(), output_base.text, output_base.len);
	path_s = gb_string_appendc(path_s, "/");
	path_s = gb_string_append_length(path_s, m->pkg->name.text, m->pkg->name.len);
	if (m->file != nullptr) {
		path_s = gb_string_append_fmt(path_s, "_%d", (int)m->file->id); // unique per file
	}
	path_s = gb_string_appendc(path_s, "_x64.obj");
	String filepath = make_string(cast(u8 const *)path_s, gb_string_length(path_s));

	if (!coff_writer_emit(&m->coff, filepath)) {
		m->obj_failed = true;
	} else {
		m->obj_path = copy_string(permanent_allocator(), filepath);
	}
	// No coff_writer_free: all COFF data is in the module arena (reclaimed at exit). The
	// old free did string_map_destroy on the heap-backed sym_map — a CRT-heap free under
	// the global lock per module in the hot parallel write path; dropping it removes that
	// contention and keeps section data live for the post-write byte-breakdown diagnostic.
	return 0;
}

// Drain THIS module's proc queue, compiling each entry into THIS module. Single consumer per
// module (so all of m's COFF writes stay on one thread); compiling a proc may MPSC-enqueue more
// into any module's queue. Mirrors lb_generate_procedures_worker_proc.
gb_internal WORKER_TASK_PROC(x64_generate_pending_worker) {
	x64Module *m = cast(x64Module *)data;
	// Loop: compiling a proc may emit a map op → enqueue synth hashers; a synth hasher may
	// enqueue child hashers (synth_queue) and runtime hashers (proc_queue). Drain both to a
	// local fixpoint; the outer driver handles cross-module enqueues.
	for (;;) {
		bool did = false;
		for (Entity *e = nullptr; mpsc_dequeue(&m->proc_queue, &e); /**/) {
			DeclInfo *decl = decl_info_of_entity(e);
			if (decl == nullptr || decl->proc_lit == nullptr) continue;
			x64_compile_procedure(m, e, decl->proc_lit->ProcLit.body); // dedups internally if already emitted
			did = true;
		}
		if (x64_generate_synth_procs(m)) did = true;
		if (!did) break;
	}
	return 0;
}

// Drive the per-module proc queues to a fixpoint: a worker compiling a proc may enqueue into a
// module whose worker already finished, so re-dispatch until every queue is empty. Mirrors
// lb_generate_missing_procedures (single-queue variant; x64_compile_procedure dedups re-enqueues).
gb_internal void x64_generate_pending(x64Generator *gen, bool threaded) {
	isize retry = 0;
	for (;;) {
		if (threaded) {
			for (auto const &entry : gen->modules) thread_pool_add_task(x64_generate_pending_worker, entry.value);
			thread_pool_wait();
		} else {
			for (auto const &entry : gen->modules) x64_generate_pending_worker(entry.value);
		}
		bool any = false;
		for (auto const &entry : gen->modules) {
			if (entry.value->proc_queue.count.load(std::memory_order_relaxed) != 0 ||
			    entry.value->synth_queue.count.load(std::memory_order_relaxed) != 0) { any = true; break; }
		}
		if (!any) break;
		GB_ASSERT(retry++ <= gen->modules.count); // bounded: each round drains ≥1 queue to no-ops
	}
}

// Lazy global definition — LLVM's lb_find_value_from_entity. Every global referenced by compiled
// code (onref_globals) the eager min_dep pass skipped is DEFINED in its OWNER module, once.
gb_internal void x64_run_lazy_globals(x64Generator *gen, PtrSet<Entity *> *defined) {
	for (auto const &entry : gen->modules) {
		x64Module *m = entry.value;
		for_array(i, m->onref_globals) {
			Entity *e = m->onref_globals[i];
			if (ptr_set_update(defined, e)) continue;
			x64Module *owner = x64_module_of_entity(gen, e);
			if (owner == nullptr) continue;
			DeclInfo *d = decl_info_of_entity(e);
			if (e->Variable.thread_local_model.len != 0) x64_emit_global_tls(owner, e, d);
			else if (x64_global_has_const_init(d, e))     x64_emit_global_static(owner, e, d);
			else                                          x64_emit_global_variable(owner, e);
		}
		array_clear(&m->onref_globals);
	}
}

// Register a foreign library entity once (thread-safe via foreign_mutex).
gb_internal void x64_add_foreign_lib(x64Generator *gen, Entity *lib) {
	if (lib == nullptr) return;
	GB_ASSERT(lib->kind == Entity_LibraryName);
	mutex_lock(&gen->foreign_mutex);
	if (!ptr_set_update(&gen->foreign_libraries_set, lib)) {
		array_add(&gen->foreign_libraries, lib);
	}
	mutex_unlock(&gen->foreign_mutex);
}

gb_internal x64Generator *x64_generate_code(Checker *c) {
	CheckerInfo *info = &c->info;
	gbAllocator  a    = heap_allocator();

	x64Generator *gen = gb_alloc_item(a, x64Generator);
	gen->info  = info;
	gen->alloc = a;

	linker_data_init(gen, info, c->parser->init_fullpath);

	map_init(&gen->modules, (isize)info->packages.count * 2);

	// One x64Module per package, plus per-file modules for base:runtime (and
	// -module-per-file) so the large runtime parallelizes. Mirrors lb_init_generator.
	bool module_per_file = build_context.module_per_file && build_context.optimization_level <= 0;
	for (auto const &pkg_entry : info->packages) {
		AstPackage *pkg = pkg_entry.value;
		x64Module *pm = x64_module_create(gen, pkg, nullptr);
		map_set(&gen->modules, cast(void *)pkg, pm);

		if (pkg->kind == Package_Runtime || module_per_file) {
			for (AstFile *file : pkg->files) {
				x64Module *fm = x64_module_create(gen, pkg, file);
				map_set(&gen->modules, cast(void *)file, fm);
			}
		}
	}

	// Globals first, so symbols are available. Iterate variable_init_order, the SAME
	// source LLVM uses: info->entities misses some package-level globals (encoding_base64's
	// ENC_TABLE/DEC_TABLE) → unresolved externals. We emit every non-foreign global
	// unconditionally — unlike LLVM we have no on-demand creation to backstop a
	// wrongly-skipped but still-referenced global (an unused definition is harmless).
	for (DeclInfo *d : info->variable_init_order) {
		Entity *e = d->entity.load(std::memory_order_relaxed); // d->entity is std::atomic<Entity*>
		if (e == nullptr || e->kind != Entity_Variable) continue;
		Scope *s = e->scope;
		if (s == nullptr || !(s->flags & ScopeFlag_File)) continue;
		// Eager pass mirrors lb_generate_code: skip min_dep==0. Such globals (referenced
		// only by an on-demand-compiled proc) are defined LAZILY from onref_globals after
		// the proc closure (mirrors lb_find_value_from_entity).
		if (e->min_dep_count.load(std::memory_order_relaxed) == 0) continue;

		x64Module *m = x64_module_of_entity(gen, e);
		if (m == nullptr) continue;
		// @(thread_local) → `.tls$`; const init → STATIC .rdata/.data (startup skips it);
		// else zero .bss + runtime init (mirrors LLVM).
		if (e->Variable.thread_local_model.len != 0) x64_emit_global_tls(m, e, d);
		else if (x64_global_has_const_init(d, e))     x64_emit_global_static(m, e, d);
		else                                          x64_emit_global_variable(m, e);
	}

	// Roots are file-scope procs something depends on (min_dep_count != 0); nested procs
	// flow through their parent so dead code is never emitted. Mirrors LLVM. Sorted for
	// deterministic output.
	{
		Arena         *scratch = get_arena(ThreadArena_Temporary);
		ArenaTempGuard scratch_guard(scratch);
		Array<Entity *> roots;
		array_init(&roots, arena_allocator(scratch), 0, info->entities.count);
		for (Entity *e : info->entities) {
			if (e->kind != Entity_Procedure) continue;
			Scope *s = e->scope;
			if (s == nullptr || !(s->flags & ScopeFlag_File)) continue;
			if (e->Procedure.is_foreign) continue;
			GenProcsData *gpd = e->Procedure.gen_procs;
			if (gpd != nullptr) {
				for (Entity *ge : gpd->procs) {
					if (ge->min_dep_count.load(std::memory_order_relaxed) == 0) continue;
					array_add(&roots, ge);
				}
			} else {
				if (e->min_dep_count.load(std::memory_order_relaxed) == 0) continue;
				array_add(&roots, e);
			}
		}
		// x64 has no on-demand path, so add the entry point explicitly — it's reached only
		// via the intrinsic and may have min_dep_count == 0 (LLVM creates it lazily).
		if (info->entry_point != nullptr && info->entry_point->kind == Entity_Procedure) {
			array_add(&roots, info->entry_point);
		}
		// @(init) procs are likewise reached only from the startup sequence — force them in.
		for (Entity *ie : info->init_procedures) {
			if (ie != nullptr && ie->kind == Entity_Procedure) array_add(&roots, ie);
		}
		// `odin test`: the @(test) procs are referenced only by the synthetic main we generate
		// (x64_emit_test_main), so force them in (they may have min_dep_count == 0).
		if (build_context.command_kind == Command_test) {
			for (Entity *te : info->testing_procedures) {
				if (te != nullptr && te->kind == Entity_Procedure) array_add(&roots, te);
			}
		}
		// Distribute roots to their owning module (token-pos order → deterministic
		// per-module COFF output regardless of thread scheduling).
		array_sort(roots, x64_proc_entity_cmp);
		for_array(i, roots) {
			Entity *e = roots[i];
			x64Module *m = x64_module_of_entity(gen, e);
			if (m == nullptr) continue;
			array_add(&m->compile_roots, e);
		}

		// Compile each module's procs in parallel — one worker per module (mirrors
		// lb_generate_procedures). Distinct COFF objects + per-thread arenas → no sharing.
		bool threaded = global_thread_pool.threads.count > 1;
		for (auto const &entry : gen->modules) {
			x64Module *m = entry.value;
			if (m->compile_roots.count == 0) continue;
			if (threaded) thread_pool_add_task(x64_compile_module_worker, m);
			else          x64_compile_module_worker(m);
		}
		if (threaded) thread_pool_wait();

		// On-demand closure (parallel pass done). Workers MPSC-enqueued every referenced
		// min_dep==0 proc + deferred nested proc into per-module queues; drain to a fixpoint,
		// then define the lazily-referenced globals.
		{
			PtrSet<Entity *> defined = {};
			ptr_set_init(&defined, 64);

			x64_generate_pending(gen, threaded);
			x64_run_lazy_globals(gen, &defined);

			// __$startup_runtime: runs the lazy globals' initializers (skipped by the eager
			// loop). Emit into the runtime default module; called before __entry_point.
			for (auto const &entry : gen->modules) {
				x64Module *m = entry.value;
				if (m->pkg->kind == Package_Runtime && m->file == nullptr) {
					x64_emit_startup_runtime(m, gen, &defined);
					// `odin test`: generate the synthetic `main` (runtime's is `when !ODIN_TEST`).
					if (build_context.command_kind == Command_test) x64_emit_test_main(m, gen);
					break;
				}
			}

			// Startup initializers may reference more min_dep==0 procs/globals — drain them.
			x64_generate_pending(gen, threaded);
			x64_run_lazy_globals(gen, &defined);
		}

		// __$cleanup_runtime: still an empty stub (no cleanup work yet). The LLVM
		// backend generates it as a real proc; we just resolve the linker reference.
		for (auto const &entry : gen->modules) {
			x64Module *m = entry.value;
			if (m->pkg->kind == Package_Runtime && m->file == nullptr) {
				x64_emit_empty_stub(m, str_lit("__$cleanup_runtime"));
				break;
			}
		}
	}

	// Emit runtime.type_table — after all procs compiled (modules exist), before write.
	x64_emit_type_table(gen);
	// The type table's map variants generate Map_Info globals for reflection-only map types, which
	// enqueue their synth hasher/equal procs. Drain so those symbols get defined before the write.
	x64_generate_pending(gen, global_thread_pool.threads.count > 1);

	// Serialize + write one COFF object per module in parallel (independent files).
	{
		bool threaded = global_thread_pool.threads.count > 1;
		for (auto const &entry : gen->modules) {
			if (threaded) thread_pool_add_task(x64_write_module_worker, entry.value);
			else          x64_write_module_worker(entry.value);
		}
		if (threaded) thread_pool_wait();
	}

	bool success = true;
	for (auto const &entry : gen->modules) {
		x64Module *m = entry.value;
		if (m->obj_failed) {
			gb_printf_err("x64 backend: failed to write object for %.*s\n", LIT(m->pkg->name));
			success = false;
		} else if (m->obj_path.len > 0) {
			array_add(&gen->output_object_paths, m->obj_path);
		}
	}

	if (!success) {
		return nullptr;
	}

	// Collect foreign library entities for the linker. Entity_LibraryName entities live
	// in file scopes and are NOT in info->entities/all_procedures (foreign procs have no
	// body → never enqueued), so walk every file scope in every package.
	{
		for (auto const &pkg_entry : info->packages) {
			AstPackage *pkg = pkg_entry.value;
			for (AstFile *file : pkg->files) {
				Scope *s = file->scope;
				if (s == nullptr) continue;
				for (u32 i = 0; i < s->elements.cap; i++) {
					ScopeMapSlot *slot = &s->elements.slots[i];
					if (slot->hash == 0) continue;
					Entity *e = slot->value;
					// Match LLVM (lb_add_foreign_library_path asserts EntityFlag_Used): only
					// collect libraries whose foreign entities are actually USED. Passing stray
					// declared-but-unused libs lets one win symbol resolution over the correct one
					// (Kernel32.lib vs Synchronization.lib for WakeByAddressSingle).
					if (e != nullptr && e->kind == Entity_LibraryName && (e->flags & EntityFlag_Used)) {
						x64_add_foreign_lib(gen, e);
					}
				}
			}
		}
		// Also @require-attributed imports (may live in package scope, not file scope).
		for (Entity *lib : info->required_foreign_imports_through_force) {
			x64_add_foreign_lib(gen, lib);
		}
	}

	// Deterministic link order, EXACTLY like LLVM (lb_generate_code): sort by priority_index, then
	// package/file/source order. The unsorted scope-walk order let Kernel32.lib precede
	// Synchronization.lib, so WakeByAddressSingle resolved against the wrong DLL
	// (STATUS_ENTRYPOINT_NOT_FOUND) once the Windows SDK changed (VS uninstall).
	array_sort(gen->foreign_libraries, foreign_library_cmp);

	return gen;
}
