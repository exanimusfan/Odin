// Windows COFF object-file writer for the Odin fast debug backend.
//
// Produces a minimal AMD64 COFF .obj that the existing LLD linker stage can
// consume directly, just like the LLVM backend's output.
//
// Usage pattern:
//   CoffWriter cw;
//   coff_writer_init(&cw, allocator);
//
//   CoffSection *text = coff_section_add(&cw, str_lit(".text"),
//                           COFF_SCN_TEXT | COFF_SCN_ALIGN_16);
//   u32 main_sym = coff_sym_add_proc(&cw, str_lit("main"), 1/*section*/, 0/*offset*/);
//   coff_apply_x64_relocs(text, &asm, &cw);   // pull relocs from the assembler
//   coff_writer_emit(&cw, filepath);
//   coff_writer_free(&cw);

// ---------------------------------------------------------------------------
// COFF machine / file constants
// ---------------------------------------------------------------------------

#define COFF_MACHINE_AMD64      0x8664u

// Section characteristics
#define COFF_SCN_CNT_CODE       0x00000020u
#define COFF_SCN_CNT_IDATA      0x00000040u  // initialized data
#define COFF_SCN_CNT_UDATA      0x00000080u  // uninitialized data
#define COFF_SCN_LNK_COMDAT     0x00001000u
#define COFF_SCN_ALIGN_1        0x00100000u
#define COFF_SCN_ALIGN_2        0x00200000u
#define COFF_SCN_ALIGN_4        0x00300000u
#define COFF_SCN_ALIGN_8        0x00400000u
#define COFF_SCN_ALIGN_16       0x00500000u
#define COFF_SCN_ALIGN_32       0x00600000u
#define COFF_SCN_ALIGN_64       0x00700000u
#define COFF_SCN_ALIGN_MASK     0x00F00000u
#define COFF_SCN_MEM_DISCARDABLE 0x02000000u
#define COFF_SCN_MEM_EXECUTE    0x20000000u
#define COFF_SCN_MEM_READ       0x40000000u
#define COFF_SCN_MEM_WRITE      0x80000000u

// Combined flags for common section types
#define COFF_SCN_TEXT   (COFF_SCN_CNT_CODE  | COFF_SCN_MEM_EXECUTE | COFF_SCN_MEM_READ)
#define COFF_SCN_RDATA  (COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ)
#define COFF_SCN_DATA   (COFF_SCN_CNT_IDATA | COFF_SCN_MEM_READ | COFF_SCN_MEM_WRITE)
#define COFF_SCN_BSS    (COFF_SCN_CNT_UDATA | COFF_SCN_MEM_READ | COFF_SCN_MEM_WRITE)

// Symbol storage classes
#define COFF_SYM_CLASS_EXTERNAL   2u
#define COFF_SYM_CLASS_STATIC     3u
#define COFF_SYM_CLASS_FUNCTION   101u
#define COFF_SYM_CLASS_FILE       103u

// Symbol types
#define COFF_SYM_TYPE_NULL        0x0000u
#define COFF_SYM_TYPE_FUNCTION    0x0020u

// Special section numbers (signed 16-bit)
#define COFF_SECT_UNDEF   ((i16)0)   // external / undefined
#define COFF_SECT_ABS     ((i16)-1)  // absolute constant
#define COFF_SECT_DEBUG   ((i16)-2)

// Relocation types (AMD64 COFF, match IMAGE_REL_AMD64_*)
#define COFF_REL_ADDR64   0x0001u  // 64-bit absolute VA
#define COFF_REL_ADDR32   0x0002u  // 32-bit absolute VA
#define COFF_REL_ADDR32NB 0x0003u  // 32-bit relative (no image base)
#define COFF_REL_REL32    0x0004u  // 32-bit PC-relative
#define COFF_REL_REL32_1  0x0005u
#define COFF_REL_REL32_2  0x0006u
#define COFF_REL_REL32_3  0x0007u
#define COFF_REL_REL32_4  0x0008u
#define COFF_REL_REL32_5  0x0009u
#define COFF_REL_SECTION  0x000Au  // 16-bit section index (CodeView seg field)
#define COFF_REL_SECREL   0x000Bu  // 32-bit section-relative

// ---------------------------------------------------------------------------
// Binary layout structs (packed — written directly into the output buffer)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)

struct CoffFileHeader {
	u16 machine;           // COFF_MACHINE_AMD64
	u16 num_sections;
	u32 timestamp;
	u32 ptr_sym_table;
	u32 num_symbols;
	u16 opt_header_size;   // 0 for object files
	u16 characteristics;   // 0 for object files
};

struct CoffSectionHeader {
	char name[8];          // null-padded; leading "/" + decimal = string-table offset
	u32 virtual_size;      // 0 in object files
	u32 virtual_address;   // 0 in object files
	u32 raw_data_size;
	u32 ptr_raw_data;
	u32 ptr_relocs;
	u32 ptr_line_numbers;  // 0
	u16 num_relocs;
	u16 num_line_numbers;  // 0
	u32 characteristics;
};

struct CoffReloc {
	u32 virtual_address;   // byte offset within section
	u32 sym_table_idx;
	u16 type;
};

struct CoffSymbol {
	union {
		char  short_name[8];
		struct { u32 zeroes; u32 str_offset; } long_name;
	};
	u32 value;
	i16 section_number;
	u16 type;
	u8  storage_class;
	u8  num_aux;
};

// Auxiliary record for a section symbol
struct CoffAuxSection {
	u32 section_length;
	u16 num_relocs;
	u16 num_line_numbers;
	u32 checksum;
	u16 section_number;    // COMDAT: low 16 bits of assoc. section number
	u8  selection;         // COMDAT selection type
	u8  unused[3];
};

#pragma pack(pop)

GB_STATIC_ASSERT(sizeof(CoffFileHeader)    == 20);
GB_STATIC_ASSERT(sizeof(CoffSectionHeader) == 40);
GB_STATIC_ASSERT(sizeof(CoffReloc)         == 10);
GB_STATIC_ASSERT(sizeof(CoffSymbol)        == 18);
GB_STATIC_ASSERT(sizeof(CoffAuxSection)    == 18);

// ---------------------------------------------------------------------------
// High-level section representation
// ---------------------------------------------------------------------------

struct CoffSectionReloc {
	u32    section_offset; // byte offset within the section
	u32    sym_idx;        // index into CoffWriter::syms  (resolved during emit)
	u16    type;
	String sym_name;       // used to resolve sym_idx if sym_idx == (u32)-1
};

struct CoffSection {
	char  name[8];         // as stored in the section header
	u32   characteristics;
	Array<u8>              data;
	Array<CoffSectionReloc> relocs;
	u32   sym_idx;         // index of this section's own symbol in CoffWriter::syms
};

// ---------------------------------------------------------------------------
// High-level symbol representation
// ---------------------------------------------------------------------------

struct CoffSymEntry {
	String name;
	u32    value;           // offset within section (or 0 for externals)
	i16    section_number;  // 1-based; COFF_SECT_UNDEF for external refs
	u16    type;
	u8     storage_class;
	bool   has_aux_section;
	// Auxiliary section data (only valid when has_aux_section)
	u32    aux_section_length;
	u16    aux_num_relocs;
};

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

struct CoffWriter {
	gbAllocator            alloc;
	Array<CoffSection *>   sections;
	Array<CoffSymEntry>    syms;
	Array<u8>              str_table; // raw bytes AFTER the 4-byte size field
	StringMap<u32>         sym_map;   // name → index in syms
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

gb_internal void coff_writer_init(CoffWriter *cw, gbAllocator alloc);
gb_internal void coff_writer_free(CoffWriter *cw);

// Sections
gb_internal CoffSection *coff_section_add(CoffWriter *cw, String name, u32 characteristics);
gb_internal void         coff_section_write(CoffSection *sec, void const *data, isize size);
gb_internal void         coff_section_write_u8 (CoffSection *sec, u8  v);
gb_internal void         coff_section_write_u16(CoffSection *sec, u16 v);
gb_internal void         coff_section_write_u32(CoffSection *sec, u32 v);
gb_internal void         coff_section_write_u64(CoffSection *sec, u64 v);
gb_internal void         coff_section_align(CoffSection *sec, isize align, u8 fill = 0);
gb_internal isize        coff_section_len(CoffSection *sec);

// Symbols — all return the 0-based symbol-table index.
gb_internal u32 coff_sym_add_section(CoffWriter *cw, i16 section_number, CoffSection *sec);
gb_internal u32 coff_sym_add_proc   (CoffWriter *cw, String name, i16 section_number, u32 value, bool is_export);
gb_internal u32 coff_sym_find_or_add_extern(CoffWriter *cw, String name); // external reference
gb_internal u32 coff_sym_find(CoffWriter *cw, String name); // (u32)-1 if not found

// Symbol resolved by name at emit time.
gb_internal void coff_reloc_add(CoffSection *sec, u32 section_offset, String sym_name, u16 type);
// Symbol index already known.
gb_internal void coff_reloc_add_idx(CoffSection *sec, u32 section_offset, u32 sym_idx, u16 type);

// Transfer all X64RelocEntry items into a section. base_offset = byte in the section
// where the assembler's code starts.
gb_internal void coff_apply_x64_relocs(CoffSection *sec, X64Assembler *asm_, CoffWriter *cw, u32 base_offset);

gb_internal Array<u8> coff_writer_to_bytes(CoffWriter *cw); // caller must free
gb_internal bool coff_writer_emit(CoffWriter *cw, String filepath);
