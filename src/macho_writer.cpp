// Mach-O x86-64 relocatable-object writer for the Odin fast x64 backend.
//
// Consumes the SAME in-memory model the COFF writer uses (CoffWriter: sections with
// raw data + named relocations, and a symbol table) and serializes an MH_OBJECT .o
// that Apple's ld64 links directly — used for darwin_amd64 (runs under Rosetta 2).
//
// Mapping:
//   .text  → __TEXT,__text        .rdata → __DATA,__const (may contain ADDR64 ptrs,
//   .data  → __DATA,__data          so it cannot live in __TEXT of a PIE image)
//   .bss   → __DATA,__bss (S_ZEROFILL; COFF model stores literal zeros, Mach-O
//                          stores none — only the vm size)
//   .pdata/.xdata/.debug$*/.tls$/.drectve → skipped (Win64/PE-only concepts)
//
// Relocations (the live sections only ever carry two COFF kinds):
//   COFF REL32  → X86_64_RELOC_BRANCH (byte before the disp32 is E8/E9: CALL/JMP)
//                 or X86_64_RELOC_SIGNED (RIP-relative LEA/MOV ModRM), pcrel, len 4.
//                 Both formats store the addend in the field with target = S + A
//                 relative to the END of the fixup — bit-identical, no rewrite.
//   COFF ADDR64 → X86_64_RELOC_UNSIGNED, absolute, len 8. Addend in the field.
// All relocs are emitted r_extern=1 against a symbol-table entry (works for local
// symbols too, and sidesteps section-index + absolute-addend bookkeeping).
//
// Symbols get the darwin C underscore prefix ("main" → "_main", "write" → "_write").
// MH_SUBSECTIONS_VIA_SYMBOLS is set (required for ld64's DEBUG MAP: without it the
// linker emits no N_OSO/N_FUN stabs and lldb never finds the DWARF in the .o files).
// Safe because every reloc is extern against a NAMED symbol and embedded addends
// stay within that symbol's own atom (field offsets into its own global/proc).

#define MACHO_MH_MAGIC_64        0xFEEDFACFu
#define MACHO_CPU_TYPE_X86_64    0x01000007u
#define MACHO_CPU_SUBTYPE_X86_64 0x00000003u
#define MACHO_MH_OBJECT          0x1u

#define MACHO_LC_SEGMENT_64      0x19u
#define MACHO_LC_SYMTAB          0x02u
#define MACHO_LC_DYSYMTAB        0x0Bu
#define MACHO_LC_BUILD_VERSION   0x32u

#define MACHO_PLATFORM_MACOS     1u
#define MACHO_MH_SUBSECTIONS_VIA_SYMBOLS 0x2000u

#define MACHO_S_REGULAR                 0x0u
#define MACHO_S_ZEROFILL                0x1u
#define MACHO_S_THREAD_LOCAL_REGULAR    0x11u
#define MACHO_S_THREAD_LOCAL_ZEROFILL   0x12u
#define MACHO_S_THREAD_LOCAL_VARIABLES  0x13u
#define MACHO_S_ATTR_PURE_INSTRUCTIONS  0x80000000u
#define MACHO_S_ATTR_SOME_INSTRUCTIONS  0x00000400u
#define MACHO_S_ATTR_DEBUG              0x02000000u

#define MACHO_N_UNDF 0x00u
#define MACHO_N_SECT 0x0Eu
#define MACHO_N_EXT  0x01u

#define MACHO_X86_64_RELOC_UNSIGNED 0u
#define MACHO_X86_64_RELOC_SIGNED   1u
#define MACHO_X86_64_RELOC_BRANCH   2u
#define MACHO_X86_64_RELOC_GOT_LOAD 3u
#define MACHO_X86_64_RELOC_TLV      9u

struct MachoOutSection {
	CoffSection *src;
	char const  *sectname;
	char const  *segname;
	u32          flags;
	u32          align_log2;
	bool         zerofill;
	u64          vmaddr;
	u64          size;
	u32          fileoff;   // 0 for zerofill
	u32          reloff;
	u32          nreloc;
};

struct MachoSym {
	String name;      // already underscore-prefixed
	u8     n_type;
	u8     n_sect;    // 1-based mach-o ordinal, 0 for undef
	u64    n_value;
};

static u32 macho_align_log2_from_coff(u32 characteristics, u32 fallback) {
	u32 a = (characteristics & COFF_SCN_ALIGN_MASK) >> 20;
	return (a == 0) ? fallback : (a - 1);
}

static String macho_prefix_underscore(gbAllocator a, String name) {
	u8 *buf = gb_alloc_array(a, u8, name.len + 1);
	buf[0] = '_';
	gb_memcopy(buf + 1, name.text, name.len);
	return make_string(buf, name.len + 1);
}

gb_internal bool macho_writer_emit(CoffWriter *cw, String filepath) {
	gbAllocator a = cw->alloc; // module arena — thread-owned, freed at exit

	// ── Collect mapped sections in a FIXED rank order: content sections first,
	// zerofill (__bss, __thread_bss) LAST in vm order (ld64 requirement). Lazily
	// created sections (.tdata/.tbss/.tlv) appear after .bss in cw->sections, so
	// rank-slotting (not source order) is required.
	// __DWARF debug sections rank after the __DATA zerofills: ld64's zerofill-last
	// rule is per SEGMENT name, and clang's own objects use this exact layout.
	enum { MO_TEXT, MO_CONST, MO_DATA, MO_TDATA, MO_TLV, MO_BSS, MO_TBSS,
	       MO_DWABB, MO_DWINF, MO_DWLIN, MO_COUNT };
	MachoOutSection slot[MO_COUNT] = {};
	for_array(i, cw->sections) {
		CoffSection *sec = cw->sections[i];
		if (sec->data.count == 0) continue;
		String name = make_string(cast(u8 const *)sec->name, gb_strnlen(sec->name, 8));
		MachoOutSection *o = nullptr;
		if (name == ".text") {
			o = &slot[MO_TEXT];
			o->sectname = "__text"; o->segname = "__TEXT";
			o->flags = MACHO_S_ATTR_PURE_INSTRUCTIONS | MACHO_S_ATTR_SOME_INSTRUCTIONS;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 4);
		} else if (name == ".rdata") {
			o = &slot[MO_CONST];
			o->sectname = "__const"; o->segname = "__DATA";
			o->flags = MACHO_S_REGULAR;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 3);
		} else if (name == ".data") {
			o = &slot[MO_DATA];
			o->sectname = "__data"; o->segname = "__DATA";
			o->flags = MACHO_S_REGULAR;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 3);
		} else if (name == ".tdata") {
			o = &slot[MO_TDATA];
			o->sectname = "__thread_data"; o->segname = "__DATA";
			o->flags = MACHO_S_THREAD_LOCAL_REGULAR;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 4);
		} else if (name == ".tlv") {
			o = &slot[MO_TLV];
			o->sectname = "__thread_vars"; o->segname = "__DATA";
			o->flags = MACHO_S_THREAD_LOCAL_VARIABLES;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 3);
		} else if (name == ".bss") {
			o = &slot[MO_BSS];
			o->sectname = "__bss"; o->segname = "__DATA";
			o->flags = MACHO_S_ZEROFILL;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 3);
			o->zerofill = true;
		} else if (name == ".tbss") {
			o = &slot[MO_TBSS];
			o->sectname = "__thread_bss"; o->segname = "__DATA";
			o->flags = MACHO_S_THREAD_LOCAL_ZEROFILL;
			o->align_log2 = macho_align_log2_from_coff(sec->characteristics, 4);
			o->zerofill = true;
		} else if (name == ".dwabb") {
			o = &slot[MO_DWABB];
			o->sectname = "__debug_abbrev"; o->segname = "__DWARF";
			o->flags = MACHO_S_REGULAR | MACHO_S_ATTR_DEBUG;
			o->align_log2 = 0;
		} else if (name == ".dwinf") {
			o = &slot[MO_DWINF];
			o->sectname = "__debug_info"; o->segname = "__DWARF";
			o->flags = MACHO_S_REGULAR | MACHO_S_ATTR_DEBUG;
			o->align_log2 = 0;
		} else if (name == ".dwlin") {
			o = &slot[MO_DWLIN];
			o->sectname = "__debug_line"; o->segname = "__DWARF";
			o->flags = MACHO_S_REGULAR | MACHO_S_ATTR_DEBUG;
			o->align_log2 = 0;
		} else {
			continue; // .pdata/.xdata/.debug$*/.tls$/.drectve: PE-only, dropped
		}
		o->src  = sec;
		o->size = (u64)sec->data.count;
	}
	MachoOutSection out[MO_COUNT] = {};
	int nsects = 0;
	for (int r = 0; r < MO_COUNT; r++) {
		if (slot[r].src != nullptr) out[nsects++] = slot[r];
	}
	// zerofill-last check is per SEGMENT name (a __DWARF section after __bss is fine).
	for (int i = 0; i + 1 < nsects; i++) {
		bool same_seg = gb_strcmp(out[i].segname, out[i+1].segname) == 0;
		GB_ASSERT_MSG(!(same_seg && out[i].zerofill && !out[i+1].zerofill),
		              "macho: zerofill sections must be last within their segment");
	}

	// COFF section index (1-based) → mach-o ordinal (1-based), 0 = unmapped.
	i32 coff_to_ord[24] = {};
	for (int o = 0; o < nsects; o++) {
		for_array(i, cw->sections) {
			if (cw->sections[i] == out[o].src) {
				GB_ASSERT(i + 1 < (isize)gb_count_of(coff_to_ord));
				coff_to_ord[i + 1] = (i32)(o + 1);
			}
		}
	}

	// ── Symbols: defined (local | external) from cw->syms, undef from reloc names ──
	Array<MachoSym> locals, exts, undefs;
	array_init(&locals, a, 0, 16);
	array_init(&exts,   a, 0, 64);
	array_init(&undefs, a, 0, 32);
	StringMap<i32> group_and_pos = {}; // unprefixed name → (group<<24)|pos, resolved to final idx later
	string_map_init(&group_and_pos, 128);

	// vm layout first (symbol values need section addresses).
	{
		u64 vm = 0;
		for (int o = 0; o < nsects; o++) {
			u64 al = (u64)1 << out[o].align_log2;
			vm = (vm + al - 1) & ~(al - 1);
			out[o].vmaddr = vm;
			vm += out[o].size;
		}
	}

	for_array(i, cw->syms) {
		CoffSymEntry *s = &cw->syms[i];
		if (s->has_aux_section) continue;                 // COFF section symbol
		if (s->section_number == COFF_SECT_UNDEF) continue; // externs re-derived from relocs
		if (s->section_number < 0) continue;              // absolute/debug
		i32 ord = (s->section_number < (i16)gb_count_of(coff_to_ord)) ? coff_to_ord[s->section_number] : 0;
		if (ord == 0) continue;                          // symbol in a dropped section
		if (string_map_get(&group_and_pos, s->name) != nullptr) continue;

		MachoSym ms = {};
		ms.name    = macho_prefix_underscore(a, s->name);
		ms.n_sect  = (u8)ord;
		ms.n_value = out[ord - 1].vmaddr + s->value;
		bool is_ext = s->storage_class == COFF_SYM_CLASS_EXTERNAL;
		ms.n_type  = (u8)(MACHO_N_SECT | (is_ext ? MACHO_N_EXT : 0));
		if (is_ext) {
			string_map_set(&group_and_pos, s->name, (i32)((1 << 24) | exts.count));
			array_add(&exts, ms);
		} else {
			string_map_set(&group_and_pos, s->name, (i32)((0 << 24) | locals.count));
			array_add(&locals, ms);
		}
	}

	// Undefined externals: every reloc target not defined above.
	for (int o = 0; o < nsects; o++) {
		CoffSection *sec = out[o].src;
		for_array(r, sec->relocs) {
			CoffSectionReloc *rel = &sec->relocs[r];
			String name = rel->sym_name;
			if (rel->sym_idx != cast(u32)-1) name = cw->syms[rel->sym_idx].name;
			GB_ASSERT_MSG(name.len > 0, "macho: reloc with no symbol name");
			if (string_map_get(&group_and_pos, name) != nullptr) continue;
			MachoSym ms = {};
			ms.name   = macho_prefix_underscore(a, name);
			ms.n_type = (u8)(MACHO_N_UNDF | MACHO_N_EXT);
			string_map_set(&group_and_pos, name, (i32)((2 << 24) | undefs.count));
			array_add(&undefs, ms);
		}
	}

	isize nlocal = locals.count, next = exts.count, nundef = undefs.count;
	isize nsyms  = nlocal + next + nundef;

	// ── File layout ──
	u32 const seg_cmd_size   = 72 + 80 * (u32)nsects;
	u32 const build_ver_size = 24;
	u32 const symtab_size    = 24;
	u32 const dysymtab_size  = 80;
	u32 sizeofcmds = seg_cmd_size + build_ver_size + symtab_size + dysymtab_size;
	u32 data_start = 32 + sizeofcmds;

	u32 off = data_start;
	for (int o = 0; o < nsects; o++) {
		if (out[o].zerofill) { out[o].fileoff = 0; continue; }
		u32 al = 1u << out[o].align_log2;
		off = (off + al - 1) & ~(al - 1);
		out[o].fileoff = off;
		off += (u32)out[o].size;
	}
	u32 relocs_start = (off + 7) & ~7u;
	u32 roff = relocs_start;
	for (int o = 0; o < nsects; o++) {
		out[o].nreloc = (u32)out[o].src->relocs.count;
		if (out[o].zerofill) GB_ASSERT(out[o].nreloc == 0);
		out[o].reloff = out[o].nreloc ? roff : 0;
		roff += out[o].nreloc * 8;
	}
	u32 symoff = roff;
	u32 stroff = symoff + (u32)nsyms * 16;

	// String table: index 0 reserved (empty).
	Array<u8> strtab;
	array_init(&strtab, a, 0, 1024);
	array_add(&strtab, (u8)0);
	auto intern_str = [&](String s) -> u32 {
		u32 pos = (u32)strtab.count;
		array_add_elems(&strtab, s.text, s.len);
		array_add(&strtab, (u8)0);
		return pos;
	};

	// ── Serialize ──
	Array<u8> buf;
	array_init(&buf, a, 0, (isize)(stroff + 4096));

	// mach_header_64
	coff_buf_u32(&buf, MACHO_MH_MAGIC_64);
	coff_buf_u32(&buf, MACHO_CPU_TYPE_X86_64);
	coff_buf_u32(&buf, MACHO_CPU_SUBTYPE_X86_64);
	coff_buf_u32(&buf, MACHO_MH_OBJECT);
	coff_buf_u32(&buf, 4);            // ncmds
	coff_buf_u32(&buf, sizeofcmds);
	coff_buf_u32(&buf, MACHO_MH_SUBSECTIONS_VIA_SYMBOLS); // flags
	coff_buf_u32(&buf, 0);            // reserved

	// LC_SEGMENT_64
	{
		u64 vmsize = nsects ? (out[nsects-1].vmaddr + out[nsects-1].size) : 0;
		// The segment's file extent starts at the FIRST content section (not data_start —
		// the alignment pad before it isn't part of the segment, and counting it made
		// filesize exceed vmsize, which ld64 rejects).
		u32 seg_fileoff = data_start;
		for (int o = 0; o < nsects; o++) {
			if (!out[o].zerofill) { seg_fileoff = out[o].fileoff; break; }
		}
		u32 filesize = (off > seg_fileoff) ? (off - seg_fileoff) : 0;
		coff_buf_u32(&buf, MACHO_LC_SEGMENT_64);
		coff_buf_u32(&buf, seg_cmd_size);
		char segname[16] = {};                    // one anonymous segment (MH_OBJECT convention)
		coff_buf_bytes(&buf, segname, 16);
		coff_buf_u64(&buf, 0);                    // vmaddr
		coff_buf_u64(&buf, vmsize);
		coff_buf_u64(&buf, seg_fileoff);          // fileoff
		coff_buf_u64(&buf, filesize);
		coff_buf_u32(&buf, 7);                    // maxprot rwx
		coff_buf_u32(&buf, 7);                    // initprot rwx
		coff_buf_u32(&buf, (u32)nsects);
		coff_buf_u32(&buf, 0);                    // flags

		for (int o = 0; o < nsects; o++) {
			char sectname[16] = {}, segname2[16] = {};
			gb_memcopy(sectname, out[o].sectname, gb_strlen(out[o].sectname));
			gb_memcopy(segname2, out[o].segname,  gb_strlen(out[o].segname));
			coff_buf_bytes(&buf, sectname, 16);
			coff_buf_bytes(&buf, segname2, 16);
			coff_buf_u64(&buf, out[o].vmaddr);
			coff_buf_u64(&buf, out[o].size);
			coff_buf_u32(&buf, out[o].fileoff);
			coff_buf_u32(&buf, out[o].align_log2);
			coff_buf_u32(&buf, out[o].reloff);
			coff_buf_u32(&buf, out[o].nreloc);
			coff_buf_u32(&buf, out[o].flags);
			coff_buf_u32(&buf, 0); // reserved1
			coff_buf_u32(&buf, 0); // reserved2
			coff_buf_u32(&buf, 0); // reserved3
		}
	}

	// LC_BUILD_VERSION (platform macos, minos 11.0; silences ld64 platform warnings)
	coff_buf_u32(&buf, MACHO_LC_BUILD_VERSION);
	coff_buf_u32(&buf, build_ver_size);
	coff_buf_u32(&buf, MACHO_PLATFORM_MACOS);
	coff_buf_u32(&buf, 0x000B0000);  // minos 11.0.0
	coff_buf_u32(&buf, 0);           // sdk (unknown)
	coff_buf_u32(&buf, 0);           // ntools

	// LC_SYMTAB
	coff_buf_u32(&buf, MACHO_LC_SYMTAB);
	coff_buf_u32(&buf, symtab_size);
	coff_buf_u32(&buf, symoff);
	coff_buf_u32(&buf, (u32)nsyms);
	coff_buf_u32(&buf, stroff);
	isize strsize_patch = buf.count;
	coff_buf_u32(&buf, 0);           // strsize — patched after interning

	// LC_DYSYMTAB (only the local/extdef/undef ranges)
	coff_buf_u32(&buf, MACHO_LC_DYSYMTAB);
	coff_buf_u32(&buf, dysymtab_size);
	coff_buf_u32(&buf, 0);              coff_buf_u32(&buf, (u32)nlocal);  // ilocalsym/nlocalsym
	coff_buf_u32(&buf, (u32)nlocal);    coff_buf_u32(&buf, (u32)next);    // iextdefsym/nextdefsym
	coff_buf_u32(&buf, (u32)(nlocal + next)); coff_buf_u32(&buf, (u32)nundef); // iundefsym/nundefsym
	for (int k = 0; k < 12; k++) coff_buf_u32(&buf, 0); // toc/modtab/extref/indirect/extrel/locrel

	// Resolve relocations FIRST — __DWARF relocs patch their section data (below),
	// so this must run before the data is copied into the output buffer.
	Array<u32> rel_words;
	array_init(&rel_words, a, 0, 64);
	for (int o = 0; o < nsects; o++) {
		CoffSection *sec = out[o].src;
		bool is_dwarf = gb_strcmp(out[o].segname, "__DWARF") == 0;
		for_array(r, sec->relocs) {
			CoffSectionReloc *rel = &sec->relocs[r];
			String name = rel->sym_name;
			if (rel->sym_idx != cast(u32)-1) name = cw->syms[rel->sym_idx].name;
			i32 *gp = string_map_get(&group_and_pos, name);
			GB_ASSERT_MSG(gp != nullptr, "macho: unresolved reloc symbol %.*s", LIT(name));
			i32 group = *gp >> 24, pos = *gp & 0xFFFFFF;
			MachoSym *ms = (group == 0) ? &locals[pos] : (group == 1) ? &exts[pos] : &undefs[pos];
			u32 symnum = (u32)(group == 0 ? pos : group == 1 ? nlocal + pos : nlocal + next + pos);

			if (is_dwarf) {
				// DWARF consumers (lldb via the debug map) do NOT bind extern relocs in
				// debug sections. Use clang's convention: a NON-extern reloc against the
				// target's SECTION ordinal, with the target's object-VM address folded
				// into the section data.
				GB_ASSERT_MSG(rel->type == COFF_REL_ADDR64 && ms->n_sect != 0,
				              "macho: __DWARF reloc must be ADDR64 to a defined symbol (%.*s)", LIT(name));
				u32 at = rel->section_offset;
				u64 cur = 0;
				for (int b = 7; b >= 0; b--) cur = (cur << 8) | sec->data[at + b];
				u64 v = cur + ms->n_value;
				for (int b = 0; b < 8; b++) { sec->data[at + b] = (u8)(v & 0xFF); v >>= 8; }
				array_add(&rel_words, at);
				array_add(&rel_words, (u32)((ms->n_sect & 0x00FFFFFFu) | (3u << 25) | (MACHO_X86_64_RELOC_UNSIGNED << 28)));
				continue;
			}

			u32 pcrel, length, type;
			switch (rel->type) {
			case COFF_REL_REL32: {
				u8 prev = (rel->section_offset > 0) ? sec->data[rel->section_offset - 1] : 0;
				bool is_branch = (prev == 0xE8 || prev == 0xE9); // CALL/JMP rel32
				pcrel = 1; length = 2;
				type = is_branch ? MACHO_X86_64_RELOC_BRANCH : MACHO_X86_64_RELOC_SIGNED;
			} break;
			case COFF_REL_ADDR64:
				pcrel = 0; length = 3;
				type = MACHO_X86_64_RELOC_UNSIGNED;
				break;
			case X64Reloc_TLV: // RIP-relative ref to a __thread_vars descriptor
				pcrel = 1; length = 2;
				type = MACHO_X86_64_RELOC_TLV;
				break;
			case X64Reloc_GOTLD: // RIP-relative GOT load of a dylib-external symbol
				pcrel = 1; length = 2;
				type = MACHO_X86_64_RELOC_GOT_LOAD;
				break;
			default:
				GB_PANIC("macho: unsupported reloc type %u -> %.*s (win64-only reloc in a live section?)",
				         rel->type, LIT(name));
				return false;
			}
			array_add(&rel_words, rel->section_offset); // r_address (section-relative)
			array_add(&rel_words, (symnum & 0x00FFFFFFu) | (pcrel << 24) | (length << 25) | (1u << 27) | (type << 28));
		}
	}

	// Section data
	for (int o = 0; o < nsects; o++) {
		if (out[o].zerofill) continue;
		while ((u32)buf.count < out[o].fileoff) array_add(&buf, (u8)0);
		coff_buf_bytes(&buf, out[o].src->data.data, out[o].src->data.count);
	}
	while ((u32)buf.count < relocs_start) array_add(&buf, (u8)0);

	// Relocations (resolved above, in the same section order the headers point at)
	for_array(w, rel_words) coff_buf_u32(&buf, rel_words[w]);

	// Symbol table (locals, extdefs, undefs — matching LC_DYSYMTAB ranges)
	GB_ASSERT((u32)buf.count == symoff);
	Array<MachoSym> *groups[3] = { &locals, &exts, &undefs };
	for (int g = 0; g < 3; g++) {
		for_array(i, *groups[g]) {
			MachoSym *ms = &(*groups[g])[i];
			coff_buf_u32(&buf, intern_str(ms->name)); // n_strx
			coff_buf_u8 (&buf, ms->n_type);
			coff_buf_u8 (&buf, ms->n_sect);
			coff_buf_u16(&buf, 0);                    // n_desc
			coff_buf_u64(&buf, ms->n_value);
		}
	}

	// String table
	GB_ASSERT((u32)buf.count == stroff);
	coff_buf_bytes(&buf, strtab.data, strtab.count);
	coff_buf_patch_u32(&buf, strsize_patch, (u32)strtab.count);

	gbFile f = {};
	gbFileError err = gb_file_create(&f, cast(char const *)filepath.text);
	if (err != gbFileError_None) return false;
	b32 ok = gb_file_write(&f, buf.data, buf.count);
	gb_file_close(&f);
	return cast(bool)ok;
}
