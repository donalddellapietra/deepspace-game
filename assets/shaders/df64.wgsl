// Double-float (df64) emulation. Mirror of `src/world/df64.rs`. Each
// value is a (hi, lo) pair where `hi` is the f32 nearest the true value
// and `lo` is the f32 residual `value - hi`. Gives ~46-bit mantissa
// (vs 23 for f32) at 2-3× compute. Used by sphere_descend_anchor to
// preserve precision past the f32 floor at depth ~10-12.
//
// Apple Silicon's MSL compiler aggressively fuses adjacent muls/adds
// into FMA. Pure-add error-free transforms (two_sum, quick_two_sum)
// are FMA-immune. The product transform uses Veltkamp/Dekker splits
// so the residual is computed without relying on FMA contraction.
//
// Each helper is a standalone function with explicit `let` bindings
// to keep the data-flow visible to the compiler — no inline arithmetic
// chains the optimizer can collapse.

struct DF { hi: f32, lo: f32, };
struct DFv3 { x: DF, y: DF, z: DF, };

fn df_make(hi: f32, lo: f32) -> DF { return DF(hi, lo); }
fn df_from_f32(a: f32) -> DF { return DF(a, 0.0); }
fn df_to_f32(a: DF) -> f32 { return a.hi + a.lo; }
fn df_neg(a: DF) -> DF { return DF(-a.hi, -a.lo); }
fn df_abs(a: DF) -> DF {
    if a.hi < 0.0 { return df_neg(a); }
    return a;
}

// f32 nearest 1/3, with f64 residual cast to f32 (~9.93e-9). The exact
// hi+lo reconstructs 1/3 to ~6e-16 absolute. Multiplying a DF by
// INV3 (rather than literal 1/3) is the precision-stable divide-by-3
// the descent's pop step needs.
fn df_inv3() -> DF { return DF(0.33333334, -9.934108e-9); }

// Knuth two-sum. Exact: a + b = s + e where s = round(a+b),
// |e| <= ulp(s)/2. Independent of input order.
fn two_sum(a: f32, b: f32) -> DF {
    let s  = a + b;
    let bb = s - a;
    let aa = s - bb;
    let er = (a - aa) + (b - bb);
    return DF(s, er);
}

// Quick two-sum. Requires |a| >= |b| (saves 3 ops).
fn quick_two_sum(a: f32, b: f32) -> DF {
    let s = a + b;
    let e = b - (s - a);
    return DF(s, e);
}

// Veltkamp split: 4097 = 2^12 + 1. Splits a f32 into (hi, lo) of 12
// bits each, so a*b can be reconstructed exactly via Dekker.
const DF_SPLITTER: f32 = 4097.0;

fn df_split(a: f32) -> vec2<f32> {
    let t = a * DF_SPLITTER;
    let hi = t - (t - a);
    let lo = a - hi;
    return vec2<f32>(hi, lo);
}

fn two_prod(a: f32, b: f32) -> DF {
    let p   = a * b;
    let sa  = df_split(a);
    let sb  = df_split(b);
    let e1  = sa.x * sb.x - p;
    let e2  = e1 + sa.x * sb.y;
    let e3  = e2 + sa.y * sb.x;
    let er  = e3 + sa.y * sb.y;
    return DF(p, er);
}

fn df_add(a: DF, b: DF) -> DF {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    let mid_lo = s.lo + t.hi;
    let mid = quick_two_sum(s.hi, mid_lo);
    let final_lo = mid.lo + t.lo;
    return quick_two_sum(mid.hi, final_lo);
}

fn df_sub(a: DF, b: DF) -> DF { return df_add(a, df_neg(b)); }

fn df_mul(a: DF, b: DF) -> DF {
    let p = two_prod(a.hi, b.hi);
    let cross = a.hi * b.lo + a.lo * b.hi;
    let sum_lo = p.lo + cross;
    return quick_two_sum(p.hi, sum_lo);
}

fn df_mul_f32(a: DF, k: f32) -> DF {
    let p = two_prod(a.hi, k);
    let sum_lo = p.lo + a.lo * k;
    return quick_two_sum(p.hi, sum_lo);
}

// 1 / a — Newton refinement of the f32 reciprocal seed (one iter).
fn df_inv(a: DF) -> DF {
    let r0 = 1.0 / a.hi;
    let ar0 = df_mul_f32(a, r0);
    let two_minus = df_sub(df_from_f32(2.0), ar0);
    return df_mul_f32(two_minus, r0);
}

fn df_lt(a: DF, b: DF) -> bool {
    if a.hi < b.hi { return true; }
    if a.hi > b.hi { return false; }
    return a.lo < b.lo;
}

fn df_le(a: DF, b: DF) -> bool {
    if a.hi < b.hi { return true; }
    if a.hi > b.hi { return false; }
    return a.lo <= b.lo;
}

// ─── Vec3 helpers ──────────────────────────────────────────────────

fn dfv3_make(hi: vec3<f32>, lo: vec3<f32>) -> DFv3 {
    return DFv3(DF(hi.x, lo.x), DF(hi.y, lo.y), DF(hi.z, lo.z));
}

fn dfv3_from_f32(v: vec3<f32>) -> DFv3 {
    return DFv3(DF(v.x, 0.0), DF(v.y, 0.0), DF(v.z, 0.0));
}

fn dfv3_to_f32(v: DFv3) -> vec3<f32> {
    return vec3<f32>(df_to_f32(v.x), df_to_f32(v.y), df_to_f32(v.z));
}

fn dfv3_hi(v: DFv3) -> vec3<f32> { return vec3<f32>(v.x.hi, v.y.hi, v.z.hi); }

fn dfv3_neg(v: DFv3) -> DFv3 {
    return DFv3(df_neg(v.x), df_neg(v.y), df_neg(v.z));
}

fn dfv3_abs(v: DFv3) -> DFv3 {
    return DFv3(df_abs(v.x), df_abs(v.y), df_abs(v.z));
}

fn dfv3_add(a: DFv3, b: DFv3) -> DFv3 {
    return DFv3(df_add(a.x, b.x), df_add(a.y, b.y), df_add(a.z, b.z));
}

fn dfv3_sub(a: DFv3, b: DFv3) -> DFv3 {
    return DFv3(df_sub(a.x, b.x), df_sub(a.y, b.y), df_sub(a.z, b.z));
}

fn dfv3_mul(a: DFv3, b: DFv3) -> DFv3 {
    return DFv3(df_mul(a.x, b.x), df_mul(a.y, b.y), df_mul(a.z, b.z));
}

fn dfv3_scale(a: DFv3, k: DF) -> DFv3 {
    return DFv3(df_mul(a.x, k), df_mul(a.y, k), df_mul(a.z, k));
}

fn dfv3_mul_f32(a: DFv3, k: f32) -> DFv3 {
    return DFv3(df_mul_f32(a.x, k), df_mul_f32(a.y, k), df_mul_f32(a.z, k));
}

// `a * 3` — exact in f32 for inputs whose mantissa has 2 free bits;
// lo*3 may need renormalization via quick_two_sum.
fn df_times3(a: DF) -> DF {
    let h = a.hi * 3.0;
    let l = a.lo * 3.0;
    return quick_two_sum(h, l);
}

fn dfv3_times3(a: DFv3) -> DFv3 {
    return DFv3(df_times3(a.x), df_times3(a.y), df_times3(a.z));
}

// `a / 3` via INV3 multiplication. Raw `1/3` in f32 has ~3e-8 relative
// error that compounds to f32 floor after ~6 push-pop pairs.
fn df_div3(a: DF) -> DF { return df_mul(a, df_inv3()); }

fn dfv3_div3(a: DFv3) -> DFv3 {
    return DFv3(df_div3(a.x), df_div3(a.y), df_div3(a.z));
}

// Branchless argmin axis (slab convention: x > y > z tie-break)
// returning a 0/1 mask vec3 for use in DDA advance.
fn df_min_axis_mask(sd: DFv3) -> vec3<f32> {
    let xy_lt = df_lt(sd.x, sd.y);
    let xz_lt = df_lt(sd.x, sd.z);
    let yx_le = df_le(sd.y, sd.x);
    let yz_lt = df_lt(sd.y, sd.z);
    let pick_x = xy_lt && xz_lt;
    let pick_y = yx_le && yz_lt;
    let pick_z = !pick_x && !pick_y;
    return vec3<f32>(
        select(0.0, 1.0, pick_x),
        select(0.0, 1.0, pick_y),
        select(0.0, 1.0, pick_z),
    );
}

// Pick a single component of a DFv3 by integer index (0/1/2).
fn dfv3_get(v: DFv3, i: i32) -> DF {
    if i == 0 { return v.x; }
    if i == 1 { return v.y; }
    return v.z;
}

fn dfv3_set(v: DFv3, i: i32, c: DF) -> DFv3 {
    var r = v;
    if i == 0 { r.x = c; }
    else if i == 1 { r.y = c; }
    else { r.z = c; }
    return r;
}

// Dot of mask*sd (single non-zero component selects one DF).
fn dfv3_dot_mask(sd: DFv3, m: vec3<f32>) -> DF {
    var r = DF(0.0, 0.0);
    if m.x > 0.5 { r = sd.x; }
    else if m.y > 0.5 { r = sd.y; }
    else { r = sd.z; }
    return r;
}
