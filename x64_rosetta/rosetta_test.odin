package rosetta_test

@(export, link_name="ros_add")
ros_add :: proc "c" (a: i64, b: i64) -> i64 {
	return a + b
}

@(export, link_name="ros_sum")
ros_sum :: proc "c" (n: i64) -> i64 {
	s: i64 = 0
	for i: i64 = 1; i <= n; i += 1 {
		s += i
	}
	return s
}

@(export, link_name="ros_fib")
ros_fib :: proc "c" (n: i64) -> i64 {
	if n < 2 {
		return n
	}
	return ros_fib(n - 1) + ros_fib(n - 2)
}

// Mixed float/int args: SysV uses SEPARATE XMM and GP counters (a→XMM0, b→RDI,
// c→XMM1, d→RSI), unlike Win64's unified slots.
@(export, link_name="ros_mix")
ros_mix :: proc "c" (a: f64, b: i64, c: f64, d: i64) -> f64 {
	return a*f64(b) + c*f64(d)
}

// 8 integer args: SysV passes 6 in registers, g and h on the stack (callee side).
@(export, link_name="ros_many")
ros_many :: proc "c" (a, b, c, d, e, f, g, h: i64) -> i64 {
	return a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h
}

// Internal call with stack args (caller side of the same layout).
@(export, link_name="ros_call_many")
ros_call_many :: proc "c" (n: i64) -> i64 {
	return ros_many(n, n+1, n+2, n+3, n+4, n+5, n+6, n+7)
}

// ── Aggregate classification tests (SysV eightbyte pairs / MEMORY / sret) ────

Pair_II  :: struct { a, b: i64 }      // 16B INTEGER×2: RDI,RSI in / RAX:RDX out
Pair_DD  :: struct { x, y: f64 }      // 16B SSE×2: XMM0,XMM1 in / XMM0:XMM1 out
Mixed    :: struct { i: i64, d: f64 } // 16B INT+SSE: RDI+XMM0 in / RAX+XMM0 out
Big      :: struct { a, b, c: i64 }   // 24B MEMORY: by-value stack copy / sret return
Small_FF :: struct { a, b: f32 }      // 8B single SSE eightbyte: XMM0 in/out

@(export, link_name="ros_pair_sum")
ros_pair_sum :: proc "c" (p: Pair_II) -> i64 {
	return p.a + p.b
}

@(export, link_name="ros_make_pair")
ros_make_pair :: proc "c" (a: i64, b: i64) -> Pair_II {
	return {a, b}
}

@(export, link_name="ros_dd_dot")
ros_dd_dot :: proc "c" (v: Pair_DD, w: Pair_DD) -> f64 {
	return v.x*w.x + v.y*w.y
}

@(export, link_name="ros_make_dd")
ros_make_dd :: proc "c" (x: f64, y: f64) -> Pair_DD {
	return {x, y}
}

@(export, link_name="ros_mixed")
ros_mixed :: proc "c" (m: Mixed) -> Mixed {
	return {m.i * 2, m.d * 3}
}

@(export, link_name="ros_big_sum")
ros_big_sum :: proc "c" (b: Big) -> i64 {
	return b.a + b.b + b.c
}

@(export, link_name="ros_make_big")
ros_make_big :: proc "c" (a: i64) -> Big {
	return {a, a + 1, a + 2}
}

@(export, link_name="ros_ff")
ros_ff :: proc "c" (s: Small_FF) -> f32 {
	return s.a + s.b
}

@(export, link_name="ros_str_bytes")
ros_str_bytes :: proc "c" (s: string) -> i64 {
	t: i64 = 0
	for i := 0; i < len(s); i += 1 {
		t += i64(s[i])
	}
	return t
}

// Odin→Odin aggregate calls: exercises the caller AND callee sides of pair,
// MEMORY, and sret handling inside generated code.
@(export, link_name="ros_roundtrip")
ros_roundtrip :: proc "c" (a: i64, b: i64) -> i64 {
	p := ros_make_pair(a, b)
	q := ros_mixed(Mixed{p.a, f64(p.b)})
	g := ros_make_big(p.a)
	return ros_pair_sum(p) + i64(q.d) + q.i + ros_big_sum(g)
}
