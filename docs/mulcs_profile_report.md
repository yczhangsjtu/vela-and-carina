# HyperPlonk + Mulcs PCS Profiling Report

## 1. Experiment Setup

| Parameter | Value |
|-----------|-------|
| CPU | Apple M3 Pro (12 threads) |
| Curve | BLS12-381 |
| Gate | vanilla Plonk |
| Build | `cargo test --release` |
| Profiling | `MULCS_PROFILE=1`, `NV_RANGE=8,10,12,14`, `BACKEND=both` |
| Repeat | 1 (single run) |
| SRS | `gen_srs_for_testing` (dummy trusted setup) |

## 2. Top-Level Comparison

| Backend | nv | N | SRS gen (ms) | Preprocess (ms) | Prove (ms) | Verify (ms) |
|---------|----|---|-------------|-----------------|-----------|-------------|
| mKZG | 8 | 256 | 6.8 | 4.7 | 28.4 | 5.2 |
| **Mulcs** | 8 | 256 | **85.6** | 5.3 | **85.5** | **17.2** |
| mKZG | 10 | 1024 | 10.5 | 14.2 | 53.3 | 5.6 |
| **Mulcs** | 10 | 1024 | **339.5** | 14.4 | **194.9** | **17.5** |
| mKZG | 12 | 4096 | 24.0 | 42.4 | 83.3 | 6.1 |
| **Mulcs** | 12 | 4096 | **1333.4** | 45.4 | **537.2** | **17.4** |
| mKZG | 14 | 16384 | 61.4 | 141.1 | 208.5 | 6.3 |
| **Mulcs** | 14 | 16384 | **5378.1** | 149.1 | **1589.7** | **17.5** |

### Ratios (Mulcs / mKZG)

| nv | SRS ratio | Prove ratio | Verify ratio |
|----|-----------|-------------|--------------|
| 8 | 12.6x | 3.0x | 3.3x |
| 10 | 32.3x | 3.7x | 3.1x |
| 12 | 55.6x | 6.4x | 2.9x |
| 14 | 87.6x | 7.6x | 2.8x |

**Key observation**: Mulcs verify is nearly constant (~17.5ms) across nv, while mKZG grows from 5.2ms to 6.3ms. Mulcs prove ratio worsens with nv (from 3x to 7.6x). Mulcs SRS gen is catastrophically slower (12.6x–87.6x).

## 3. Mulcs SRS Generation Breakdown

For nv=12 (N=4096, max_degree=8192):

| Phase | Time (ms) | % of SRS |
|-------|-----------|----------|
| `srs_gen_g1_powers` (8193 G1 scalar mults) | 1329.0 | **99.7%** |
| `srs_gen_g2` (3 G2 mults) | 1.4 | 0.1% |
| `srs_gen_x_pows` (8193 field mults) | 0.2 | 0.01% |
| `srs_gen_sample` (random x,g1,g2) | 0.9 | 0.07% |
| `srs_gen_gamma` | 0.0002 | 0% |

**Why Mulcs SRS is slow**: Mulcs generates 2N G1 powers via sequential `g1 * xi` scalar multiplications (8193 for nv=12, 131073 for nv=16). Each is a G1 scalar mult (~0.16ms per mult at nv=12). mKZG uses FixedBaseMSM with precomputed window tables — a single multi-scalar multiplication over 2^NV scalars leverages batch optimization, making it O(NV·2^NV) vs Mulcs's naive O(2^NV) individual scalar mults.

The mKZG SRS generator uses `FixedBase::msm()` which precomputes a window table for multiple scalars, dramatically reducing per-scalar cost. Mulcs currently uses naive `g1 * xi` for each element.

## 4. Mulcs Prover Breakdown

For nv=12, 22 polynomials in the batch:

| Phase | Time (ms) | % of multi_open |
|-------|-----------|-----------------|
| `multi_open_per_poly` (h + h̄ + commit per poly) | 375.5 | **85.8%** |
| `multi_open_quotient_construction` (poly_div per poly) | 34.2 | 7.8% |
| `multi_open_eval_zgz` (evaluate f/h̄ at z, γz) | 13.5 | 3.1% |
| `multi_open_commit_q` (final KZG MSM) | 14.1 | 3.2% |
| `multi_open_append_pts_evals` | 0.05 | 0.01% |
| Fiat-Shamir (z + inner/outer) | 0.01 | 0% |
| **Total multi_open** | **437.5** | 100% |

### Within `per_poly` (avg per polynomial, nv=12):

| Sub-phase | Avg ms |
|-----------|--------|
| `util_mul_structured_eq` (f_v * P_eq) | ~3.0ms |
| + `util_compute_h` total (includes mul_structured_eq) | ~3.2ms |
| `util_hbar_denoms` (build + batch_inversion) | ~0.7ms |
| `util_hbar_scale` (scale coeffs) | ~0.2ms |
| `util_hbar_gamma_pows` (compute γ powers) | ~0.05ms |
| MSM commit of h̄ (KZG MSM over 2N coeffs) | ~3–5ms |

**Why prove gets slower with nv**:
- `mul_structured_eq` does O(N·μ) field operations and N allocations per factor. At nv=12, N=4096, each `mul_structured_eq` traverses ~N * μ ~ 49K field mults. At nv=16 (N=65536), this becomes ~1M field mults.
- h̄ has ~2N coefficients (8191 at nv=12), requiring a large KZG MSM to commit.
- Quotient construction also grows with N (poly_div over degree-2N polynomials).
- Overall compute_h + compute_h_bar + commit_hbar per poly dominates.

## 5. Mulcs Verifier Breakdown

For nv=12 (constant across all nv in our data):

| Phase | Time (ms) | % of verify |
|-------|-----------|-------------|
| `batch_verify_aggregate_cm` (build interpolation + group ops) | **14.7** | **87.1%** |
| `batch_verify_pairing` (1 pairing check) | 2.0 | 11.8% |
| `batch_verify_claymore` (per-poly Claymore identity) | 0.06 | 0.4% |
| `batch_verify_transcript` (absorb data) | 0.04 | 0.2% |
| `batch_verify_fs` (get challenges) | 0.03 | 0.2% |
| **Total batch_verify** | **16.9** | 100% |

**Why verify is ~17ms and constant**:
- The dominant cost is `batch_verify_aggregate_cm` (14.7ms), which for each of the 22 polynomials rebuilds the Lagrange interpolation coefficients via `build_multi_point_polys` and does group operations (`cm * scalar + ...`).
- `build_multi_point_polys` runs O(k²) for k=2 points, but it runs 22 times for 22 polynomials. Each call does 2 inversions + Lagrange interpolation.
- The constant nature across nv suggests the group operations (G1 scalar mults) dominate. Each poly requires 2 G1 scalar mults (for cm_R construction) + 1 group addition per poly.
- The pairing check is O(1) regardless of nv — the same 1 pairing for the batched proof.
- Claymore identity is cheap (~3μ field mults per poly, <0.1ms total).

**Why 1 pairing but still slow**: The per-polynomial Lagrange interpolation and group operations dominate. For 22 polys, that's ~44 G1 scalars + 22 Lagrange reconstructions. Even though the final pairing is O(1), the reconstruction cost scales with number of openings.

## 6. What's HyperPlonk PIOP vs PCS Backend?

| Phase | PIOP (mKZG) | PCS Backend (Mulcs) |
|-------|-------------|---------------------|
| SRS gen | 24ms | 1333ms → **Mulcs responsible** |
| Preprocess | 42ms | 45ms → **Equal (PIOP dominated)** |
| Prove | 83ms | 537ms → **Mulcs 6.4x slower** |
| Verify | 6.1ms | 17.4ms → **Mulcs 2.9x slower** |

The PIOP overhead (preprocess specifically) is nearly identical between backends (~42ms), confirming it's from HyperPlonk's commitment/sumcheck independent of PCS. The prove overhead difference (454ms) is entirely Mulcs's compute_h/h_bar + KZG MSM per poly. The verify difference (11.3ms) comes from the per-poly Lagrange interpolation + group operations in Mulcs verifier, vs mKZG's batched sumcheck-based verify.

## 7. Bottleneck Ranking (nv=12, prove)

| Rank | Phase | Time (ms) | % of prove | Why Expensive |
|------|-------|-----------|------------|---------------|
| 1 | `multi_open_per_poly` | 375.5 | 69.9% | compute_h + compute_h_bar + KZG commit per poly; most time is in `mul_structured_eq` (~3ms/poly) and MSM (~4ms/poly) |
| 2 | `multi_open_quotient_construction` | 34.2 | 6.4% | poly_div per poly for 2N-degree polynomials |
| 3 | `multi_open_commit_q` | 14.1 | 2.6% | final KZG MSM over ~2N scalars |
| 4 | `multi_open_eval_zgz` | 13.5 | 2.5% | Horner evaluation of 2N-degree polynomials |
| 5 | SRS gen (g1_powers) | 1329.0 | — | naive sequential G1 scalar mults × 2N |
| **—— HyperPlonk PIOP ——** | | ~80ms | ~15% | zero-check + perm-check + commit witnesses |

## 8. Verifier Bottleneck (nv=12)

| Rank | Phase | Time (ms) | % of verify |
|------|-------|-----------|-------------|
| 1 | `batch_verify_aggregate_cm` | 14.7 | 87.1% |
| 2 | `batch_verify_pairing` | 2.0 | 11.8% |
| 3 | `batch_verify_claymore` | 0.06 | 0.4% |

The per-poly `build_multi_point_polys` + group operations dominate. Even with 1 pairing, the verifier does O(num_polys) Lagrange interpolation + G1 scalar mults.

## 9. Interpretation vs mKZG

### Why mKZG is faster for prove:
- mKZG uses the multilinear basis directly — opening is NV MSMs each over ~2^{NV-i} G1 elements, totaling ~2·2^NV scalar mults. No quotient construction.
- Mulcs uses univariate representation — needs compute_h (structured polynomial multiplication), compute_h_bar (batch inversion), and KZG MSM for each opening. This is O(N·μ) field operations plus KZG commit.

### Why mKZG is faster for verify:
- mKZG batch_verify uses a sumcheck-based approach: verify sumcheck, then OPEN a single g' polynomial at a single point. Sumcheck is O(N) field ops (cheap), then 1 pairing.
- Mulcs verifier must reconstruct per-poly commitment differences (cm_f + r·cm_h̄ - cm_R), which requires Lagrange interpolation + G1 group ops per opening. While the pairing is also O(1), the reconstruction is O(num_polys · G1_ops).

### Why Mulcs has O(1) pairing but slower verifier:
The pairing IS O(1), but the preprocessing (Lagrange interpolation, remainder polynomial construction, group operations) is O(num_polys · G1_scalar_mult). For HyperPlonk, num_polys can be large (22 at nv=12, ~30+ at nv=16).

## 10. Optimization Suggestions (by priority)

1. **SRS gen — use FixedBaseMSM or windowed MSM** (Priority: HIGH)
   - Current: `g1 * xi` for each of 2N elements sequentially.
   - Fix: Use `FixedBase::msm()` like mKZG does, or at least use rayon parallel iteration.
   - Impact: SRS gen time reduced from seconds to tens of ms. Would make SRS gen faster than mKZG for moderate nv.

2. **compute_h / mul_structured_eq — reduce allocation** (Priority: HIGH)
   - Current: Each factor (μ rounds) allocates a new Vec of growing size.
   - Fix: Pre-allocate the final size once (`f_v.coeffs.len() + 2^mu - 1`) and update in-place.
   - Impact: ~30% reduction in per-poly compute time. Combined with rayon parallelism already in place.

3. **Verifier — eliminate per-poly Lagrange interpolation** (Priority: HIGH)
   - Current: Each opening rebuilds `build_multi_point_polys` from scratch (2 inversions per poly).
   - Fix: Since all openings share the same z/γz, precompute the Z-polynomial and the Lagrange basis once.
   - Impact: ~70% reduction in verifier time (the 14.7ms aggregate_cm phase).

4. **compute_h_bar — reuse gamma_pows across multiple h̄ computations** (Priority: MEDIUM)
   - Current: Gamma powers recomputed per polynomial (O(N) field mults each).
   - Fix: Precompute gamma_pows once per batch and reuse across all h̄ computations.
   - Impact: ~5ms reduction per batch at nv=12.

5. **KZG quotient construction — use Horner's method for poly_div** (Priority: LOW)
   - Current: Naive polynomial long division.
   - Fix: Use schoolbook division optimized for the specific (X-z)(X-γz) divisor.
   - Impact: Modest improvement (divisor is only degree 2).

6. **Verifier batch Claymore identity — precompute z_n1, gamma_n1, z_inv once** (Priority: LOW)
   - Current: Recomputes for each poly.
   - Fix: Precompute per batch.
   - Impact: Already cheap (0.06ms), minor improvement.

## 11. Caveats

- SRS is `gen_srs_for_testing` (dummy trusted setup) — not a production trusted setup.
- Single-run benchmarks have noise; averaging over repeats would improve accuracy.
- The mKZG internal phases are NOT instrumented; only top-level SRS/preprocess/prove/verify times are available. mKZG internal breakdown is speculative based on code analysis.
- Proof size is `unavailable` — `HyperPlonkProof` derives `PartialEq` but not `CanonicalSerialize`. Future work.
- The Mulcs batch opening implementation is the current engineering prototype, not the optimal version.
- `delta` uses fixed `F::one()` instead of random — acceptable for profiling, not production security.
- Fixed `z = F::from(2u64)` in standalone `open` will make single-open profiling inaccurate for verifier time.

## 12. Proof Size Status

`HyperPlonkProof` does NOT implement `CanonicalSerialize`. The PCS batch proof (`MulcsBatchProof`) also lacks serialization. Approximate proxy:
- G1 elements: 1 commitment per witness (3) + 1 pi + `cm_hbars` (22) = ~26 G1 points
- Field elements: f_i_eval_at_point_i (22) + mulcs_evals (22×4) + sumcheck proofs + perm proof
- Rough proxy: ~26×48 bytes (G1 compressed) + ~200×32 bytes (Fr) ≈ 7.6 KB (excluding sumcheck/perm proofs)

mKZG proof uses similar count of G1 elements + sumcheck data.
