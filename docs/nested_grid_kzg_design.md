# Nested Reciprocal Grid-KZG (NRG-KZG) — implementation design

This document describes the ScholarDesk implementation of the candidate
multilinear polynomial commitment scheme analysed in
`research/pcs-field-map/proof-notes/nested-reciprocal-grid-kzg-mlpcs.md` and
`...-soundness.md`. It is the engineering companion to those notes; the algebra
and the (partial) soundness analysis live there.

Module: `subroutines/src/pcs/nested_grid_kzg/` (`mod.rs`, `srs.rs`, `tests.rs`).
Public types: `NestedGridKzgPCS`, `NestedGridKzgProof`,
`NestedGridKzgUniversalParams`, `NestedGridKzgProverParam`,
`NestedGridKzgVerifierParam`.

## 1. Security caveats (read first)

- This implementation is **not hiding and not zero knowledge**.
- `gen_srs_for_testing` samples the trapdoors `tau, sigma` locally and is **for
  testing only**; it is not a production trusted setup.
- What is established (in the notes) is **statistical soundness in the
  online-extraction / ideal-polynomial model**.
- An **AGM instantiation still requires an adaptive two-trapdoor ideal-check
  lemma**.
- A **standard-model proof still requires the OnlineHomGridExt / ARSDH(2)**
  extractor.
- The Fiat–Shamir transcript here fully binds the statement, but **ROM
  knowledge soundness of the compiled protocol is not formally proved**.

No claim of "formally proven security" or "production ready" is made.

## 2. Representation

For `mu` variables (`mu >= 4`) use a non-padded rectangular split:

```
m_left  = ceil(mu/2)   M_L = 2^m_left
m_right = floor(mu/2)  M_R = 2^m_right    N = M_L * M_R = 2^mu
```

`mu >= 4` guarantees `M_L, M_R >= 4`. Smaller `mu` returns a `PCSError`
(never panics).

The `DenseMultilinearExtension` evaluation index puts low-order variables
first. With `u = point`, `u_L = point[..m_left]`, `u_R = point[m_left..]`, the
canonical-index-to-matrix map is fixed:

```
F[i,j] = evaluations[i + M_L * j],   0 <= i < M_L, 0 <= j < M_R
```

so variables `0..m_left` select `i` (bound to `u_L`) and variables
`m_left..mu` select `j` (bound to `u_R`). The bivariate twin is
`f(X,Y) = sum_{i,j} F[i,j] X^i Y^j`, committed **directly on the evaluation
table** — no FFT, interpolation, or monomial-basis conversion.

Tensor polynomials `psi_L(X)`, `psi_R(Y)` have coefficient vectors
`build_eq_x_r_vec(u_L)` and `build_eq_x_r_vec(u_R)`; the claimed value is
`y = psi_L^T F psi_R`.

## 3. SRS

Two independent nonzero trapdoors `tau, sigma` (test SRS resamples zero and
avoids `tau == sigma`). The G1 key is exactly the `N`-element grid
`[tau^i sigma^j]_1` (`0 <= i < M_L`, `0 <= j < M_R`) — **never `2N`**. The G2
material is exactly the five elements `[1]_2, [tau]_2, [tau^2]_2, [sigma]_2,
[sigma^2]_2` — full G2 powers are never generated.

`NestedGridKzgVerifierParam` holds `g1_one, g1_tau, g1_sigma, g1_tau_sigma`
(the four fixed G1 bases, dimension-independent), the five G2 powers, and
`max_num_vars, m_left, m_right`.

All shifts use `checked_shl`; all dimension products use `checked_mul`;
malicious `mu`, wrong point length, or out-of-range dimensions return
`PCSError`.

### 3.1 Dominant-QX-first G1 layout and `base_index`

The G1 key is stored so the dominant `Pi_X` commitment is a single **contiguous
prefix** MSM.

- Segment 1 (`qx_len = (M_L-2)*M_R` elements): `for j in 0..M_R { for i in
  0..M_L-2 { [tau^i sigma^j] } }`.
- Segment 2 (`2*M_R` elements): `for j in 0..M_R { [tau^{M_L-2} sigma^j],
  [tau^{M_L-1} sigma^j] }`.

```
base_index(i, j) =
    j*(M_L-2) + i                          if i <  M_L-2
    (M_L-2)*M_R + 2*j + (i-(M_L-2))         if i >= M_L-2
```

`base_index` is a bijection of the full grid onto `[0, N)`. Consequences:

1. `Pi_X` MSM uses `g1_powers[..qx_len]`, length exactly `(M_L-2)*M_R`.
2. The full commitment reorders canonical `F[i + M_L*j]` into this layout and
   runs one `N`-MSM over the prefix `g1_powers[..N]`.
3. `S0`, `S1`, `Pi_Y` collect only `O(M_L + M_R)` bases; the key is never
   duplicated.
4. Trimming to the same size as the universal SRS **shares the `Arc`** (no
   `N`-copy). Trimming smaller rebuilds the target layout via the universal
   `base_index`.
5. SRS generation reuses one FixedBase window table and materialises scalars /
   projective points **one chunk at a time** (`SRS_GEN_CHUNK = 1<<16`), so
   setup never simultaneously holds `N` scalars, `N` projective points, and
   several `N`-affine buffers.

Unit tests cover the bijection, the prefix monomials, the full-commit vs
reference MSM, the four verifier bases, and smaller-trim consistency.

## 4. Proof

`NestedGridKzgProof` contains exactly `4 G1 + 8 F` plus non-cryptographic `mu`
(as `u32`):

- G1: `cm_s0 = [S0(tau)]`, `cm_s1 = [S1(sigma)]`, `pi_x = [Q_X + eta W0]`,
  `pi_y = [Q_Y + eta^2 W1]`.
- F: `a_plus = g(r)`, `a_minus = g(r^{-1})`, `t0_plus = S0(r)`,
  `v_pp = f(r,s)`, `v_pn = f(r,s^{-1})`, `v_np = f(r^{-1},s)`,
  `v_nn = f(r^{-1},s^{-1})`, `t1_plus = S1(s)`.

No `r, s, lambda, eta`, no `t0_minus`/`t1_minus`, no interpolants, no quotient
coefficients, no extra KZG witnesses are ever transmitted.

On BLS12-381 the cryptographic payload is `4*48 + 8*32 = 448 bytes`. The
canonical serialized size is `452 bytes` (payload + the 4-byte `mu`), reported
separately by benchmarks.

## 5. Transcript

`new_transcript` binds: domain separator `nested-grid-kzg-v1`, a protocol
version tag, `mu, m_left, m_right`, the original commitment `C_f`, the opening
`point`, and the claimed `value`. Strict order:

1. statement absorbed (above);
2. absorb `cm_s0`; derive `r` with `r != 0`, `r^2 != 1`;
3. absorb `a_plus, a_minus, t0_plus`; derive nonzero `lambda`;
4. absorb `cm_s1`; derive `s` with `s != 0`, `s^2 != 1`;
5. absorb `v_pp, v_pn, v_np, v_nn, t1_plus`; derive nonzero `eta`;
6. compute and send `pi_x, pi_y` (no further challenges).

Challenge rejection uses shared `draw_nonzero` / `draw_reciprocal` helpers
(prover and verifier identical). Each rejected draw re-derives the next
transcript challenge (counter-based resampling) up to `MAX_CHALLENGE_RETRY = 64`
attempts; exhaustion returns `PCSError`. There is no fixed fallback and no
silent coercion of a bad challenge.

## 6. Prover (algorithms, complexity, real MSM lengths)

Let `N = M_L*M_R`. Phases and MSM/element counts (as reported by the profiler
`count` field):

| Phase | Work | Count |
|---|---|---|
| `nrg_open_build_psi` | two `build_eq_x_r_vec` | `M_L+M_R` |
| `nrg_open_compute_g` | matrix-vector `g[i]=sum_j F[i,j]psi_R[j]`, O(N) | `N` |
| `nrg_open_compute_s0` | structured reciprocal witness, O(M_L log M_L) | `M_L` |
| `nrg_open_commit_s0` | collected MSM, bases `[tau^i]` | `M_L-1` |
| `nrg_open_eval_outer` | `g(r), g(r^{-1}), S0(r)` | `3` |
| `nrg_open_compute_restrictions` | per-column Horner at `r, r^{-1}`, O(2N) | `2N` |
| `nrg_open_compute_s1` | structured reciprocal witness, O(M_R log M_R) | `M_R` |
| `nrg_open_commit_s1` | collected MSM, bases `[sigma^j]` | `M_R-1` |
| `nrg_open_eval_grid` | four grid values + `S1(s)` | `5` |
| `nrg_open_interpolate` | tensor `I`, `L0`, `L1` (constant size) | `1` |
| `nrg_open_divide_grid` | quadratic synthetic division in X per column, then Y | `N` |
| `nrg_open_divide_witnesses` | two quadratic synthetic divisions | `2` |
| `nrg_open_commit_pi_x` | **contiguous prefix** MSM | `(M_L-2)*M_R` |
| `nrg_open_commit_pi_y` | small collected MSM | `2*(M_R-2)` |

`S0`/`S1` use `reciprocal_witness`, a shift-add reciprocal-Laurent helper over
two preallocated ping-pong buffers (no dense O(M^2) convolution); a
coefficient-level test compares it against a dense reference.

The bivariate quotient uses monic quadratic synthetic division
(`div_by_monic_quadratic`), not a general polynomial crate. All interpolation
inverses are fallible (duplicate points return `PCSError`). Nonzero grid or
witness remainders return `InvalidProver` (no `assert`/`panic`).

Theoretical proof-generation MSM scalar total (excluding the original `C_f`):

```
(M_L-1) + (M_R-1) + (M_L-2)*M_R + 2*(M_R-2) = N + M_L + M_R - 6
```

which is `N + 2*sqrt(N) - 6` in the balanced case. At `nv=20`
(`M_L=M_R=1024`) the profiler confirms the four lengths `1023, 1023, 1046528,
2044` summing to `1050618 = N + 2048 - 6`.

Only `Pi_X` is a size-`N` MSM; the other three are `O(sqrt N)`.

## 7. Verifier

Verify is `O(mu)` field work plus constant group work and never allocates
`O(N)`. Integrity checks (before any shift/allocation): `mu` fits in `usize`,
`4 <= mu < usize::BITS`, `mu <= vk.max_num_vars`, checked domain sizes,
`point.len() == mu`.

Code locations in `subroutines/src/pcs/nested_grid_kzg/mod.rs`,
`nested_grid_verify`:

- **7-base G1 MSM** (`nrg_verify_g1_msm`): bases `[C_f, cm_s0, cm_s1, [1],
  [tau], [sigma], [tau*sigma]]`, scalars `[1, eta, eta^2, -j00, -j10, -j01,
  -j11]` giving `C_E = C_f + eta*cm_s0 + eta^2*cm_s1 - C_J`.
- **2 dynamic G2 scalar multiplications** (`nrg_verify_g2_divisors`):
  `[Z_A(tau)]_2 = [tau^2] - (r+r^{-1})[tau] + [1]` and
  `[Z_B(sigma)]_2 = [sigma^2] - (s+s^{-1})[sigma] + [1]`.
- **single three-term multi-pairing** (`nrg_verify_pairing`):
  `e(C_E,[1]) * e(-pi_x,[Z_A(tau)]) * e(-pi_y,[Z_B(sigma)]) == 1`.

`psi_L(r), psi_L(r^{-1}), psi_R(s), psi_R(s^{-1})` use the product form (no full
`eq` vector). `t0_minus`/`t1_minus` are recomputed by the verifier from the
transmitted values (they are never trusted from the proof).

## 8. Trait / HyperPlonk integration

`NestedGridKzgPCS` implements `PolynomialCommitmentScheme<E>` with
`Polynomial = Arc<DenseMultilinearExtension>`, `Point = Vec<F>`,
`Proof = NestedGridKzgProof`, `BatchProof = BatchProof<E, Self>`. Batch opening
reuses the generic sum-check `multi_open_internal` / `batch_verify_internal`.

Because the trait `open` does not receive the existing commitment (which the
transcript must bind), `open` recomputes `C_f` (`nrg_open_statement_recommit`,
an `N`-MSM) and then calls the inherent
`NestedGridKzgPCS::open_with_commitment`. Benchmarks report:

- `commit` — the standalone commitment time;
- `core_open` — `open_with_commitment` using an existing commitment;
- `trait_open_total` — the trait `open`, i.e. `core_open` plus the
  recommitment.

The recommitment is a trait-API artifact and is **not** counted in the four
theoretical proof MSMs (`N + M_L + M_R - 6`).

## 9. Tests, profiling, benchmarks

Correctness/negative/panic-safety/batch tests are in
`subroutines/src/pcs/nested_grid_kzg/{srs.rs,tests.rs}`; HyperPlonk end-to-end
and 7-backend cross-checks in `hyperplonk/tests/nested_grid_kzg_backend.rs`.
Default `cargo test` runs no heavy benchmark.

Profiling (`PCS_PROFILE=1`) emits the unified 9-column CSV with backend
`NestedGridKZG` and real element/MSM counts. When disabled, timers take the
zero-overhead path (no `Instant::now()`).

Benchmarks: `subroutines/benches/nested_grid_kzg_bench.rs`
(`NRG_BENCH_NV_RANGE`, `NRG_VERIFY_REPETITIONS`), plus NestedGridKZG entries in
`pcs_bench.rs`, `pcs_single_verify_bench.rs`
(`PCS_VERIFY_NV_RANGE`/`PCS_VERIFY_BACKEND`), and
`hyperplonk/tests/pcs_compare_bench.rs` (`BACKEND=nrg`). Heavy phases run once;
only verify repeats.
