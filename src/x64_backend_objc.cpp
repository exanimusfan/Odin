// x64_backend_objc.cpp — Objective-C interop for the darwin x64 backend.
//
// Mirrors llvm_backend's objc handling. A message send compiles to a call to the
// right objc_msgSend variant with (self, _cmd, args...); `self` is the receiver (or a
// looked-up class for class methods) and `_cmd` is a SEL. Selectors and classes are
// resolved ONCE at startup: each unique name gets a global (`__$objc_SEL::name` /
// `__$objc_CLASS::name`) that x64_objc_emit_registration fills via sel_registerName /
// objc_lookUpClass; access sites just load the global. All the objc_* runtime symbols
// come from libobjc (pulled in by the Foundation framework the program imports).
//
// Implemented: intrinsics.objc_send, objc_find_selector, objc_find_class, and the
// automatic `obj->method()` send (is_objc_impl_or_import). NOT yet: objc_super (super
// sends), objc_block, objc_ivar_get, class-implementation registration.

// ── selector / class name collection (thread-safe; one global per unique name) ──
gb_internal void x64_objc_note_selector(x64Generator *gen, String name) {
	mutex_lock(&gen->foreign_mutex);
	string_set_update(&gen->objc_selectors, name);
	mutex_unlock(&gen->foreign_mutex);
}
gb_internal void x64_objc_note_class(x64Generator *gen, String name) {
	mutex_lock(&gen->foreign_mutex);
	string_set_update(&gen->objc_classes, name);
	mutex_unlock(&gen->foreign_mutex);
}

// `__$objc_SEL::<name>` / `__$objc_CLASS::<name>` — same mangling LLVM uses.
static String x64_objc_mangle(x64Module *m, char const *prefix, String name) {
	gbString s = gb_string_make(m->alloc, prefix);
	s = gb_string_append_length(s, name.text, name.len);
	return make_string(cast(u8 const *)s, gb_string_length(s));
}

// Load a SEL/Class global (defined in the runtime module, filled at startup) into a
// fresh RBP local and return it as a stable rawptr memory value.
static x64Value x64_objc_load_global(x64Procedure *p, String mangled) {
	i32 off = x64_alloc_local(p, 8, 8);
	x64_emit_lea_sym(&p->asm_, X64Reg_RAX, mangled); // &global
	x64_emit_mov_rm(&p->asm_, X64OpSize_64, X64Reg_RAX, x64_mem(X64Reg_RAX, 0)); // *global
	x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(off), X64Reg_RAX);
	return x64v_mem(t_rawptr, x64_rbp_mem(off));
}

gb_internal x64Value x64_objc_selector(x64Procedure *p, String name) {
	x64_objc_note_selector(p->module->gen, name);
	return x64_objc_load_global(p, x64_objc_mangle(p->module, "__$objc_SEL::", name));
}
gb_internal x64Value x64_objc_class(x64Procedure *p, String name) {
	x64_objc_note_class(p->module->gen, name);
	return x64_objc_load_global(p, x64_objc_mangle(p->module, "__$objc_CLASS::", name));
}

// The message receiver for an explicit objc_send: a class type expr resolves to the
// class object; anything else is an ordinary value (the object pointer).
static x64Value x64_objc_id_value(x64Procedure *p, Ast *arg) {
	if (arg->tav.mode == Addressing_Type && arg->tav.type != nullptr) {
		Type *t = arg->tav.type;
		if (t->kind == Type_Named && t->Named.type_name != nullptr &&
		    t->Named.type_name->TypeName.objc_class_name.len != 0) {
			return x64_objc_class(p, t->Named.type_name->TypeName.objc_class_name);
		}
	}
	return x64_stabilize_value(p, x64_build_expr(p, arg));
}

// The objc_msgSend variant name for a message's return-ABI class.
static String x64_objc_msgsend_name(ObjcMsgKind kind) {
	switch (kind) {
	case ObjcMsg_normal: return str_lit("objc_msgSend");
	case ObjcMsg_fpret:  return str_lit("objc_msgSend_fpret");
	case ObjcMsg_fp2ret: return str_lit("objc_msgSend_fp2ret");
	case ObjcMsg_stret:  return str_lit("objc_msgSend_stret");
	}
	GB_PANIC("x64 objc: unhandled ObjcMsgKind %u", kind);
	return str_lit("objc_msgSend");
}

// Emit the call: objc_msgSend[variant](self, _cmd, explicit...) typed as data.proc_type.
// A struct returned by pointer (stret) gets a caller sret local passed as the hidden
// first arg (SysV: RDI), the same slot objc_msgSend_stret expects.
static x64Value x64_emit_objc_msgsend(x64Procedure *p, ObjcMsgData data,
                                      x64Value self, x64Value cmd,
                                      x64Value *explicit_args, int explicit_count) {
	String name = x64_objc_msgsend_name(data.kind);

	self.type = t_rawptr;
	cmd.type  = t_rawptr;

	bool sret = x64_returns_by_pointer(data.proc_type);
	int  n    = explicit_count + 2 + (sret ? 1 : 0);
	x64Value *args = gb_alloc_array(temporary_allocator(), x64Value, n);
	int k = 0;

	i32 ret_off = 0;
	Type *ret_type = x64_last_result_type(data.proc_type);
	if (sret) {
		ret_off = x64_alloc_local(p, type_size_of(ret_type), type_align_of(ret_type));
		i32 ptr = x64_alloc_local(p, 8, 8);
		x64_emit_lea(&p->asm_, X64Reg_RAX, x64_rbp_mem(ret_off));
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
		args[k++] = x64v_mem(t_rawptr, x64_rbp_mem(ptr));
	}
	args[k++] = self;
	args[k++] = cmd;
	for (int i = 0; i < explicit_count; i++) args[k++] = explicit_args[i];

	x64Value rv = x64_emit_call(p, name, data.proc_type, args, k);
	if (sret) return x64v_mem(ret_type, x64_rbp_mem(ret_off));
	return rv;
}

// intrinsics.objc_send(RetType, receiver, "selector", args...)
gb_internal x64Value x64_build_objc_send(x64Procedure *p, Ast *expr) {
	ast_node(ce, CallExpr, expr);
	CheckerInfo *info = p->module->gen->info;
	ObjcMsgData *pdata = map_get(&info->objc_msgSend_types, expr);
	GB_ASSERT_MSG(pdata != nullptr && pdata->proc_type != nullptr, "x64 objc: missing objc_msgSend_types");
	GB_ASSERT(ce->args.count >= 3);

	x64Value self = x64_objc_id_value(p, ce->args[1]);
	Ast *sel_expr = ce->args[2];
	GB_ASSERT_MSG(sel_expr->tav.value.kind == ExactValue_String, "x64 objc: selector must be a string literal");
	x64Value cmd = x64_objc_selector(p, sel_expr->tav.value.value_string);

	int ecount = (int)ce->args.count - 3;
	x64Value *eargs = (ecount > 0) ? gb_alloc_array(temporary_allocator(), x64Value, ecount) : nullptr;
	for (int i = 0; i < ecount; i++) {
		eargs[i] = x64_stabilize_value(p, x64_build_expr(p, ce->args[3 + i]));
	}
	return x64_emit_objc_msgsend(p, *pdata, self, cmd, eargs, ecount);
}

// Automatic `obj->method(args)` / `Class.method(args)` send. `method` is the objc
// method entity (is_objc_impl_or_import); its selector name and class are on the entity.
gb_internal x64Value x64_build_objc_auto_send(x64Procedure *p, Ast *expr, Entity *method) {
	ast_node(ce, CallExpr, expr);
	CheckerInfo *info = p->module->gen->info;
	ObjcMsgData *pdata = map_get(&info->objc_msgSend_types, expr);
	GB_ASSERT_MSG(pdata != nullptr && pdata->proc_type != nullptr, "x64 objc: missing objc_msgSend_types (auto)");
	GB_ASSERT(method->kind == Entity_Procedure && method->Procedure.objc_selector_name.len != 0);

	GB_ASSERT_MSG(unparen_expr(ce->args.count > 0 ? ce->args[0] : expr)->tav.objc_super_target == nullptr,
	              "x64 objc: super sends not implemented");

	x64Value self;
	int arg_offset;
	if (method->Procedure.is_objc_class_method) {
		Entity *cls = method->Procedure.objc_class;
		if (ce->proc->kind == Ast_SelectorExpr) {
			// `Foo.method()`: the class is the selector's lhs type (may be a subclass).
			ast_node(se, SelectorExpr, ce->proc);
			if (se->expr->tav.mode == Addressing_Type && se->expr->tav.type != nullptr &&
			    se->expr->tav.type->kind == Type_Named) {
				Entity *e = se->expr->tav.type->Named.type_name;
				if (e != nullptr && e->kind == Entity_TypeName) {
					if (e->TypeName.is_type_alias && e->type != nullptr && e->type->kind == Type_Named) {
						e = e->type->Named.type_name;
					}
					cls = e;
				}
			}
		}
		GB_ASSERT_MSG(cls != nullptr && cls->TypeName.objc_class_name.len != 0, "x64 objc: class method without class");
		self = x64_objc_class(p, cls->TypeName.objc_class_name);
		arg_offset = 0;
	} else {
		GB_ASSERT(ce->args.count > 0);
		self = x64_stabilize_value(p, x64_build_expr(p, ce->args[0]));
		arg_offset = 1;
	}

	x64Value cmd = x64_objc_selector(p, method->Procedure.objc_selector_name);

	int ecount = (int)ce->args.count - arg_offset;
	if (ecount < 0) ecount = 0;
	x64Value *eargs = (ecount > 0) ? gb_alloc_array(temporary_allocator(), x64Value, ecount) : nullptr;
	for (int i = 0; i < ecount; i++) {
		eargs[i] = x64_stabilize_value(p, x64_build_expr(p, ce->args[arg_offset + i]));
	}
	return x64_emit_objc_msgsend(p, *pdata, self, cmd, eargs, ecount);
}

// ── intrinsics.objc_block ───────────────────────────────────────────────────
// Build an Apple ABI block object wrapping an Odin proc (#See clang Block-ABI-Apple).
// Layout: { isa, i32 flags, i32 reserved, invoke, ^descriptor, [context (Odin cc)] }.
// A no-capture proc "c"/"contextless" → GLOBAL block (static, _NSConcreteGlobalBlock);
// a no-capture proc "odin" → STACK block (_NSConcreteStackBlock) with the current
// context copied in, so the callback runs with a valid context. NOT implemented:
// captured variables (GB_PANIC), by-pointer (sret) returns, or stack-passed forwards.
static const X64Reg X64_SYSV_GP[6] = {
	X64Reg_RDI, X64Reg_RSI, X64Reg_RDX, X64Reg_RCX, X64Reg_R8, X64Reg_R9,
};

// Emit the invoker: (block, [forwards...]) → user_proc(forwards..., [ctx]). The block
// occupies GP[0]; INTEGER forwards shift down one GP slot (XMM forwards stay), and for
// Odin cc a pointer to the block's context field is appended as the trailing arg.
static String x64_objc_emit_block_invoker(x64Module *m, String user_name, Type *uproc,
                                          bool odin_cc, i32 ctx_off) {
	X64Assembler a = {};
	x64_asm_init(&a, heap_allocator());

	// Count integer (GP) forward params (floats travel in XMM, unshifted).
	int n_int = 0;
	if (uproc->Proc.params != nullptr) {
		for_array(i, uproc->Proc.params->Tuple.variables) {
			Entity *pe = uproc->Proc.params->Tuple.variables[i];
			if (pe->kind != Entity_Variable) continue;
			if (!x64_arg_is_float(pe->type)) n_int++;
		}
	}
	GB_ASSERT_MSG(n_int + (odin_cc ? 1 : 0) <= 6, "x64 objc_block: too many integer forward args");

	// Compute &block.context (Odin cc) before RDI (the block ptr) is overwritten.
	if (odin_cc) x64_emit_lea(&a, X64Reg_R11, x64_mem(X64Reg_RDI, ctx_off));
	// Shift GP[i+1] → GP[i] for each integer forward (ascending is clobber-safe).
	for (int i = 0; i < n_int; i++) x64_emit_mov_rr(&a, X64OpSize_64, X64_SYSV_GP[i], X64_SYSV_GP[i+1]);
	// Odin cc: context pointer becomes the trailing GP argument.
	if (odin_cc) x64_emit_mov_rr(&a, X64OpSize_64, X64_SYSV_GP[n_int], X64Reg_R11);
	x64_emit_jmp_sym(&a, user_name); // tail-call the user proc

	u8 nbuf[32]; gb_snprintf((char*)nbuf, sizeof(nbuf), "__$objc_blk_inv_%u", m->objc_block_id);
	String name = x64_objc_mangle(m, "", make_string_c((char const*)nbuf));

	u32 base_off = (u32)coff_section_len(m->text);
	coff_sym_add_proc(&m->coff, name, 1 /*text*/, base_off, false /*static/local*/);
	coff_section_write(m->text, a.code.data, a.code.count);
	coff_apply_x64_relocs(m->text, &a, &m->coff, base_off); // the jmp REL32

	array_free(&a.code); array_free(&a.labels); array_free(&a.fixups); array_free(&a.relocs);
	return name;
}

gb_internal x64Value x64_build_objc_block(x64Procedure *p, Ast *expr) {
	ast_node(ce, CallExpr, expr);
	GB_ASSERT(ce->args.count > 0);
	GB_ASSERT_MSG(ce->args.count == 1, "x64 objc_block: captured variables not implemented");
	x64Module *m = p->module;

	Ast *proc_arg = ce->args[0];
	Type *uproc = base_type(proc_arg->tav.type);
	GB_ASSERT(uproc != nullptr && uproc->kind == Type_Proc);
	Entity *ue = entity_of_node(unparen_expr(proc_arg));
	GB_ASSERT_MSG(ue != nullptr && ue->kind == Entity_Procedure, "x64 objc_block: arg must be a named proc");
	String user_name = x64_get_entity_name(ue);
	GB_ASSERT_MSG(!x64_returns_by_pointer(uproc), "x64 objc_block: sret-returning block not implemented");

	bool odin_cc = is_calling_convention_odin(uproc->Proc.calling_convention);
	i64  ctx_off = 32;               // isa(8)+flags(4)+reserved(4)+invoke(8)+desc(8)
	i64  blk_size = odin_cc ? (32 + type_size_of(t_context)) : 32;

	u32 bid = m->objc_block_id;
	String invoker = x64_objc_emit_block_invoker(m, user_name, uproc, odin_cc, (i32)ctx_off);

	// Descriptor: { reserved: u64 = 0, size: u64 = blk_size } in .rdata (no copy/dispose).
	coff_section_align(m->rdata, 8);
	u32 desc_off = (u32)coff_section_len(m->rdata);
	coff_section_write_u64(m->rdata, 0);
	coff_section_write_u64(m->rdata, (u64)blk_size);
	u8 dbuf[32]; gb_snprintf((char*)dbuf, sizeof(dbuf), "__$objc_blk_desc_%u", bid);
	String desc = x64_objc_mangle(m, "", make_string_c((char const*)dbuf));
	x64_define_data_sym(m, desc, 2 /*.rdata*/, desc_off, COFF_SYM_CLASS_STATIC);

	m->objc_block_id++;

	int const BLOCK_IS_GLOBAL = 1 << 28;

	if (!odin_cc) {
		// GLOBAL block: a static literal, isa=_NSConcreteGlobalBlock (dyld-bound).
		coff_section_align(m->data, 8);
		u32 boff = (u32)coff_section_len(m->data);
		coff_reloc_add(m->data, boff + 0, str_lit("_NSConcreteGlobalBlock"), COFF_REL_ADDR64);
		coff_section_write_u64(m->data, 0);                       // isa (reloc)
		coff_section_write_u32(m->data, (u32)BLOCK_IS_GLOBAL);    // flags
		coff_section_write_u32(m->data, 0);                       // reserved
		coff_reloc_add(m->data, boff + 16, invoker, COFF_REL_ADDR64);
		coff_section_write_u64(m->data, 0);                       // invoke (reloc)
		coff_reloc_add(m->data, boff + 24, desc, COFF_REL_ADDR64);
		coff_section_write_u64(m->data, 0);                       // descriptor (reloc)
		u8 gbuf[32]; gb_snprintf((char*)gbuf, sizeof(gbuf), "__$objc_blk_%u", bid);
		String bsym = x64_objc_mangle(m, "", make_string_c((char const*)gbuf));
		x64_define_data_sym(m, bsym, 3 /*.data*/, boff, COFF_SYM_CLASS_STATIC);
		i32 ptr = x64_alloc_local(p, 8, 8);
		x64_emit_lea_sym(&p->asm_, X64Reg_RAX, bsym);
		x64_emit_mov_mr(&p->asm_, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
		return x64v_mem(t_rawptr, x64_rbp_mem(ptr));
	}

	// STACK block (Odin cc): alloca + init at runtime, with the current context copied in.
	X64Assembler *a = &p->asm_;
	i32 blk = x64_alloc_local(p, blk_size, 16);
	x64_emit_got_load_sym(a, X64Reg_RAX, str_lit("_NSConcreteStackBlock")); // isa (dylib data)
	x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(blk + 0), X64Reg_RAX);
	x64_emit_mov_mi(a, X64OpSize_32, x64_rbp_mem(blk + 8),  0); // flags (stack block)
	x64_emit_mov_mi(a, X64OpSize_32, x64_rbp_mem(blk + 12), 0); // reserved
	x64_emit_lea_sym(a, X64Reg_RAX, invoker);
	x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(blk + 16), X64Reg_RAX);
	x64_emit_lea_sym(a, X64Reg_RAX, desc);
	x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(blk + 24), X64Reg_RAX);
	// context field = copy of the current context
	X64Mem cur = x64_current_context_body(p);           // RAX-based or RBP-based
	x64_emit_lea(a, X64Reg_RCX, cur);                    // &current context
	x64_copy_fixed(p, x64_rbp_mem(blk + (i32)ctx_off), x64_mem(X64Reg_RCX, 0), type_size_of(t_context));
	i32 ptr = x64_alloc_local(p, 8, 8);
	x64_emit_lea(a, X64Reg_RAX, x64_rbp_mem(blk));
	x64_emit_mov_mr(a, X64OpSize_64, x64_rbp_mem(ptr), X64Reg_RAX);
	return x64v_mem(t_rawptr, x64_rbp_mem(ptr));
}

// ── startup: define + fill the SEL / class globals (runtime module only) ────────
// For each unique name: an 8-byte external .bss global set once to
// sel_registerName("name") / objc_lookUpClass("name"). Emitted into the startup
// runtime proc, so it runs before any @(init) proc or main.
gb_internal void x64_objc_emit_registration(x64Procedure *p, x64Generator *gen) {
	x64Module *m = p->module;

	X64Assembler *a = &p->asm_;
	auto emit_one = [&](String name, char const *prefix, char const *runtime_fn) {
		// define the global (8-byte zero .bss, external, mangled name)
		String sym = x64_objc_mangle(m, prefix, name);
		coff_section_align(m->bss, 8);
		u32 off = (u32)coff_section_len(m->bss);
		for (int b = 0; b < 8; b++) coff_section_write_u8(m->bss, 0);
		x64_define_data_sym(m, sym, 4 /*.bss*/, off, COFF_SYM_CLASS_EXTERNAL);

		// g = runtime_fn(cstr(name));  store g into the global. Hand-emitted SysV call
		// (objc is darwin-only): cstring in RDI, result in RAX. Foreign libobjc symbol.
		String cstr = x64_const_intern_string(m, name);
		x64_emit_lea_sym(a, X64Reg_RDI, cstr);            // RDI = &"name"
		x64_emit_call_sym(a, make_string_c(runtime_fn));  // sel_registerName / objc_lookUpClass
		x64_emit_lea_sym(a, X64Reg_RCX, sym);             // &global
		x64_emit_mov_mr(a, X64OpSize_64, x64_mem(X64Reg_RCX, 0), X64Reg_RAX); // *global = result
	};

	for_array(i, gen->objc_classes.entries) {
		if (gen->objc_classes.entries[i].hash == 0 && gen->objc_classes.entries[i].value.len == 0) continue;
		emit_one(gen->objc_classes.entries[i].value, "__$objc_CLASS::", "objc_lookUpClass");
	}
	for_array(i, gen->objc_selectors.entries) {
		if (gen->objc_selectors.entries[i].hash == 0 && gen->objc_selectors.entries[i].value.len == 0) continue;
		emit_one(gen->objc_selectors.entries[i].value, "__$objc_SEL::", "sel_registerName");
	}
}
