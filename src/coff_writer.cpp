#include "coff_writer.hpp"

// ===========================================================================
// Internal helpers
// ===========================================================================

// Append little-endian values to a byte array.
gb_internal void coff_buf_u8 (Array<u8> *buf, u8  v) { array_add(buf, v); }
gb_internal void coff_buf_u16(Array<u8> *buf, u16 v) {
	array_add(buf, cast(u8)( v       & 0xFFu));
	array_add(buf, cast(u8)((v >> 8) & 0xFFu));
}
gb_internal void coff_buf_u32(Array<u8> *buf, u32 v) {
	array_add(buf, cast(u8)( v        & 0xFFu));
	array_add(buf, cast(u8)((v >>  8) & 0xFFu));
	array_add(buf, cast(u8)((v >> 16) & 0xFFu));
	array_add(buf, cast(u8)((v >> 24) & 0xFFu));
}
gb_internal void coff_buf_u64(Array<u8> *buf, u64 v) {
	coff_buf_u32(buf, cast(u32)( v        & 0xFFFFFFFFull));
	coff_buf_u32(buf, cast(u32)((v >> 32) & 0xFFFFFFFFull));
}
gb_internal void coff_buf_bytes(Array<u8> *buf, void const *data, isize size) {
	array_add_elems(buf, cast(u8 const *)data, size);
}

// Patch a u32 at a given offset in the buffer
gb_internal void coff_buf_patch_u32(Array<u8> *buf, isize offset, u32 v) {
	GB_ASSERT(offset + 4 <= buf->count);
	u8 *p = buf->data + offset;
	p[0] = cast(u8)( v        & 0xFFu);
	p[1] = cast(u8)((v >>  8) & 0xFFu);
	p[2] = cast(u8)((v >> 16) & 0xFFu);
	p[3] = cast(u8)((v >> 24) & 0xFFu);
}

// Write a symbol name into a CoffSymbol.  Names <= 8 chars go directly;
// longer names are appended to the string table.
gb_internal void coff_sym_write_name(CoffSymbol *sym, String name, Array<u8> *str_table) {
	if (name.len <= 8) {
		gb_memset(sym->short_name, 0, 8);
		gb_memcopy(sym->short_name, name.text, name.len);
	} else {
		sym->long_name.zeroes = 0;
		// String-table offset includes the 4-byte size prefix at the front
		sym->long_name.str_offset = cast(u32)(str_table->count + 4);
		array_add_elems(str_table, name.text, name.len);
		array_add(str_table, cast(u8)0); // null terminator
	}
}

// Intern a string into the string table; return the offset (including the 4-byte prefix)
gb_internal u32 coff_str_intern(Array<u8> *str_table, String name) {
	u32 offset = cast(u32)(str_table->count + 4);
	array_add_elems(str_table, name.text, name.len);
	array_add(str_table, cast(u8)0);
	return offset;
}

// Derive the section name as stored in the header (at most 8 chars, null-padded)
gb_internal void coff_section_fill_name(char out[8], String name, Array<u8> *str_table) {
	if (name.len <= 8) {
		gb_memset(out, 0, 8);
		gb_memcopy(out, name.text, name.len);
	} else {
		// Long section name: "/offset" in the name field
		u32 offset = coff_str_intern(str_table, name);
		char tmp[16];
		gb_snprintf(tmp, gb_size_of(tmp), "/%u", cast(unsigned)offset);
		gb_memset(out, 0, 8);
		isize len = cast(isize)gb_strlen(tmp);
		if (len > 8) len = 8;
		gb_memcopy(out, tmp, len);
	}
}

// ===========================================================================
// Writer lifecycle
// ===========================================================================

gb_internal void coff_writer_init(CoffWriter *cw, gbAllocator alloc) {
	cw->alloc = alloc;
	array_init(&cw->sections,  alloc, 0, 8);
	array_init(&cw->syms,      alloc, 0, 64);
	array_init(&cw->str_table, alloc, 0, 256);
	string_map_init(&cw->sym_map, 64);
}

gb_internal void coff_writer_free(CoffWriter *cw) {
	for_array(i, cw->sections) {
		CoffSection *sec = cw->sections[i];
		array_free(&sec->data);
		array_free(&sec->relocs);
		gb_free(cw->alloc, sec);
	}
	array_free(&cw->sections);
	array_free(&cw->syms);
	array_free(&cw->str_table);
	string_map_destroy(&cw->sym_map);
}

// ===========================================================================
// Sections
// ===========================================================================

gb_internal CoffSection *coff_section_add(CoffWriter *cw, String name, u32 characteristics) {
	CoffSection *sec = cast(CoffSection *)gb_alloc(cw->alloc, gb_size_of(CoffSection));
	gb_memset(sec, 0, gb_size_of(CoffSection));
	gb_memset(sec->name, 0, 8);
	// Fill the 8-byte name field; long names via string table are handled in emit
	isize copy_len = gb_min(name.len, cast(isize)8);
	gb_memcopy(sec->name, name.text, copy_len);
	sec->characteristics = characteristics;
	array_init(&sec->data,   cw->alloc, 0, 256);
	array_init(&sec->relocs, cw->alloc, 0, 16);
	sec->sym_idx = cast(u32)-1;
	array_add(&cw->sections, sec);
	return sec;
}

gb_internal void coff_section_write(CoffSection *sec, void const *data, isize size) {
	array_add_elems(&sec->data, cast(u8 const *)data, size);
}
gb_internal void coff_section_write_u8(CoffSection *sec, u8 v) {
	array_add(&sec->data, v);
}
gb_internal void coff_section_write_u16(CoffSection *sec, u16 v) {
	array_add(&sec->data, cast(u8)( v       & 0xFFu));
	array_add(&sec->data, cast(u8)((v >> 8) & 0xFFu));
}
gb_internal void coff_section_write_u32(CoffSection *sec, u32 v) {
	array_add(&sec->data, cast(u8)( v        & 0xFFu));
	array_add(&sec->data, cast(u8)((v >>  8) & 0xFFu));
	array_add(&sec->data, cast(u8)((v >> 16) & 0xFFu));
	array_add(&sec->data, cast(u8)((v >> 24) & 0xFFu));
}
gb_internal void coff_section_write_u64(CoffSection *sec, u64 v) {
	coff_section_write_u32(sec, cast(u32)( v        & 0xFFFFFFFFull));
	coff_section_write_u32(sec, cast(u32)((v >> 32) & 0xFFFFFFFFull));
}
gb_internal void coff_section_align(CoffSection *sec, isize align, u8 fill) {
	isize cur = sec->data.count;
	isize rem = cur % align;
	if (rem == 0) return;
	isize pad = align - rem;
	for (isize i = 0; i < pad; i++) array_add(&sec->data, fill);
}
gb_internal isize coff_section_len(CoffSection *sec) {
	return sec->data.count;
}

// ===========================================================================
// Symbols
// ===========================================================================

gb_internal u32 coff_sym_add_section(CoffWriter *cw, i16 section_number, CoffSection *sec) {
	u32 idx = cast(u32)cw->syms.count;

	CoffSymEntry e = {};
	// A section symbol's name is the section name itself.
	char tmp[9] = {};
	gb_memcopy(tmp, sec->name, 8);
	e.name = make_string(cast(u8 const *)gb_alloc_copy(cw->alloc, tmp, 9), 8);
	while (e.name.len > 0 && e.name.text[e.name.len-1] == 0) e.name.len--;
	e.value          = 0;
	e.section_number = section_number;
	e.type           = COFF_SYM_TYPE_NULL;
	e.storage_class  = COFF_SYM_CLASS_STATIC;
	e.has_aux_section   = true;
	e.aux_section_length = cast(u32)sec->data.count; // updated during emit
	e.aux_num_relocs     = cast(u16)sec->relocs.count;
	array_add(&cw->syms, e);

	if (sec) sec->sym_idx = idx;
	return idx;
}

gb_internal u32 coff_sym_add_proc(CoffWriter *cw, String name, i16 section_number, u32 value, bool is_export) {
	// If the symbol was already added as an UNDEF external (from a call-site
	// relocation arriving before the definition), update it in place so that
	// all existing relocations pointing at that index get the definition.
	u32 *existing_idx = string_map_get(&cw->sym_map, name);
	if (existing_idx != nullptr) {
		CoffSymEntry &existing = cw->syms[*existing_idx];
		existing.value          = value;
		existing.section_number = section_number;
		existing.type           = COFF_SYM_TYPE_FUNCTION;
		existing.storage_class  = is_export ? cast(u8)COFF_SYM_CLASS_EXTERNAL : cast(u8)COFF_SYM_CLASS_STATIC;
		return *existing_idx;
	}

	u32 idx = cast(u32)cw->syms.count;
	CoffSymEntry e = {};
	e.name           = name;
	e.value          = value;
	e.section_number = section_number;
	e.type           = COFF_SYM_TYPE_FUNCTION;
	e.storage_class  = is_export ? cast(u8)COFF_SYM_CLASS_EXTERNAL : cast(u8)COFF_SYM_CLASS_STATIC;
	array_add(&cw->syms, e);
	string_map_set(&cw->sym_map, name, idx);
	return idx;
}

gb_internal u32 coff_sym_find_or_add_extern(CoffWriter *cw, String name) {
	u32 *existing = string_map_get(&cw->sym_map, name);
	if (existing) return *existing;

	u32 idx = cast(u32)cw->syms.count;
	CoffSymEntry e = {};
	e.name           = name;
	e.value          = 0;
	e.section_number = COFF_SECT_UNDEF;
	e.type           = COFF_SYM_TYPE_NULL;
	e.storage_class  = COFF_SYM_CLASS_EXTERNAL;
	array_add(&cw->syms, e);
	string_map_set(&cw->sym_map, name, idx);
	return idx;
}

gb_internal u32 coff_sym_find(CoffWriter *cw, String name) {
	u32 *p = string_map_get(&cw->sym_map, name);
	return p ? *p : cast(u32)-1;
}

// ===========================================================================
// Relocations
// ===========================================================================

gb_internal void coff_reloc_add(CoffSection *sec, u32 section_offset, String sym_name, u16 type) {
	CoffSectionReloc r = {};
	r.section_offset = section_offset;
	r.sym_idx        = cast(u32)-1; // resolved during emit
	r.type           = type;
	r.sym_name       = sym_name;
	array_add(&sec->relocs, r);
}

gb_internal void coff_reloc_add_idx(CoffSection *sec, u32 section_offset, u32 sym_idx, u16 type) {
	CoffSectionReloc r = {};
	r.section_offset = section_offset;
	r.sym_idx        = sym_idx;
	r.type           = type;
	array_add(&sec->relocs, r);
}

gb_internal void coff_apply_x64_relocs(CoffSection *sec, X64Assembler *asm_, CoffWriter *cw, u32 base_offset) {
	for_array(i, asm_->relocs) {
		X64RelocEntry const &xr = asm_->relocs[i];
		u32 offset = base_offset + cast(u32)xr.code_offset;
		u32 sym_idx = coff_sym_find_or_add_extern(cw, xr.sym_name);
		coff_reloc_add_idx(sec, offset, sym_idx, cast(u16)xr.type);
	}
}

// ===========================================================================
// Serialization
// ===========================================================================

gb_internal Array<u8> coff_writer_to_bytes(CoffWriter *cw) {
	gbAllocator alloc = cw->alloc;

	// Pass 1: resolve pending symbol indices in relocations.
	for_array(si, cw->sections) {
		CoffSection *sec = cw->sections[si];
		for_array(ri, sec->relocs) {
			CoffSectionReloc *r = &sec->relocs[ri];
			if (r->sym_idx == cast(u32)-1) {
				r->sym_idx = coff_sym_find_or_add_extern(cw, r->sym_name);
			}
		}
	}

	// Pass 2: fill aux_section_length / aux_num_relocs now that final sizes are known.
	for_array(si, cw->sections) {
		CoffSection *sec = cw->sections[si];
		if (sec->sym_idx != cast(u32)-1) {
			CoffSymEntry *se = &cw->syms[sec->sym_idx];
			if (se->has_aux_section) {
				se->aux_section_length = cast(u32)sec->data.count;
				se->aux_num_relocs     = cast(u16)sec->relocs.count;
			}
		}
	}

	// Pass 3: compute file-layout offsets.
	u32 num_sections = cast(u32)cw->sections.count;
	u32 num_symbols  = cast(u32)cw->syms.count;
	// Each symbol with aux takes 2 x 18-byte slots
	for_array(i, cw->syms) {
		if (cw->syms[i].has_aux_section) num_symbols++;
	}

	u32 offset_sections = cast(u32)sizeof(CoffFileHeader);
	u32 offset_data     = offset_sections + num_sections * cast(u32)sizeof(CoffSectionHeader);

	// Accumulate section data offsets
	Array<u32> sec_data_offsets;
	array_init(&sec_data_offsets, alloc, num_sections, num_sections);
	Array<u32> sec_reloc_offsets;
	array_init(&sec_reloc_offsets, alloc, num_sections, num_sections);

	u32 cur = offset_data;
	for (u32 si = 0; si < cast(u32)cw->sections.count; si++) {
		CoffSection *sec = cw->sections[si];
		sec_data_offsets[si]  = (sec->data.count > 0) ? cur : 0u;
		cur += cast(u32)sec->data.count;
	}
	for (u32 si = 0; si < cast(u32)cw->sections.count; si++) {
		CoffSection *sec = cw->sections[si];
		sec_reloc_offsets[si] = (sec->relocs.count > 0) ? cur : 0u;
		cur += cast(u32)sec->relocs.count * cast(u32)sizeof(CoffReloc);
	}

	u32 offset_symtable = cur;
	u32 offset_strtable = offset_symtable + num_symbols * 18u;

	// Pass 4: write output buffer.
	Array<u8> out;
	array_init(&out, alloc, 0, cast(isize)(offset_strtable + cw->str_table.count + 8));

	// File header
	{
		CoffFileHeader hdr = {};
		hdr.machine         = COFF_MACHINE_AMD64;
		hdr.num_sections    = cast(u16)cw->sections.count;
		hdr.timestamp       = 0; // deterministic
		hdr.ptr_sym_table   = offset_symtable;
		hdr.num_symbols     = num_symbols;
		hdr.opt_header_size = 0;
		hdr.characteristics = 0;
		coff_buf_bytes(&out, &hdr, sizeof(hdr));
	}

	// Section headers
	for (u32 si = 0; si < cast(u32)cw->sections.count; si++) {
		CoffSection *sec = cw->sections[si];
		CoffSectionHeader sh = {};
		gb_memcopy(sh.name, sec->name, 8);
		sh.virtual_size    = 0;
		sh.virtual_address = 0;
		sh.raw_data_size   = cast(u32)sec->data.count;
		sh.ptr_raw_data    = sec_data_offsets[si];
		sh.ptr_relocs      = sec_reloc_offsets[si];
		sh.ptr_line_numbers = 0;
		sh.num_relocs      = cast(u16)gb_min(sec->relocs.count, cast(isize)0xFFFF);
		sh.num_line_numbers = 0;
		sh.characteristics  = sec->characteristics;
		coff_buf_bytes(&out, &sh, sizeof(sh));
	}

	// Section raw data
	for_array(si, cw->sections) {
		CoffSection *sec = cw->sections[si];
		if (sec->data.count > 0) {
			coff_buf_bytes(&out, sec->data.data, sec->data.count);
		}
	}

	// Map each cw->syms index to its real file symbol-table index. Symbols with
	// an aux record occupy two slots (symbol + aux), so every symbol after one is
	// shifted. Relocations must reference the FILE index, not the cw->syms index.
	Array<u32> sym_file_index;
	array_init(&sym_file_index, alloc, cw->syms.count, cw->syms.count);
	{
		u32 fi = 0;
		for_array(i, cw->syms) {
			sym_file_index[i] = fi;
			fi += cw->syms[i].has_aux_section ? 2u : 1u;
		}
	}

	// Section relocations
	for_array(si, cw->sections) {
		CoffSection *sec = cw->sections[si];
		for_array(ri, sec->relocs) {
			CoffSectionReloc const &r = sec->relocs[ri];
			CoffReloc cr = {};
			cr.virtual_address = r.section_offset;
			cr.sym_table_idx   = (r.sym_idx < cast(u32)sym_file_index.count)
			                     ? sym_file_index[r.sym_idx] : r.sym_idx;
			cr.type            = r.type;
			coff_buf_bytes(&out, &cr, sizeof(cr));
		}
	}

	// Symbol table
	for_array(i, cw->syms) {
		CoffSymEntry const &se = cw->syms[i];
		CoffSymbol sym = {};
		coff_sym_write_name(&sym, se.name, &cw->str_table);
		sym.value          = se.value;
		sym.section_number = se.section_number;
		sym.type           = se.type;
		sym.storage_class  = se.storage_class;
		sym.num_aux        = se.has_aux_section ? 1u : 0u;
		coff_buf_bytes(&out, &sym, sizeof(sym));

		if (se.has_aux_section) {
			CoffAuxSection aux = {};
			aux.section_length   = se.aux_section_length;
			aux.num_relocs       = se.aux_num_relocs;
			aux.num_line_numbers = 0;
			coff_buf_bytes(&out, &aux, sizeof(aux));
		}
	}

	// String table: 4-byte total-size prefix followed by the raw strings
	{
		u32 str_table_size = cast(u32)(cw->str_table.count) + 4u;
		coff_buf_u32(&out, str_table_size);
		if (cw->str_table.count > 0) {
			coff_buf_bytes(&out, cw->str_table.data, cw->str_table.count);
		}
	}

	array_free(&sec_data_offsets);
	array_free(&sec_reloc_offsets);
	array_free(&sym_file_index);
	return out;
}

gb_internal bool coff_writer_emit(CoffWriter *cw, String filepath) {
	Array<u8> bytes = coff_writer_to_bytes(cw);
	defer (array_free(&bytes));

	gbFile f = {};
	gbFileError err = gb_file_create(&f, cast(char const *)filepath.text);
	if (err != gbFileError_None) return false;
	b32 ok = gb_file_write(&f, bytes.data, bytes.count);
	gb_file_close(&f);
	return cast(bool)ok;
}
