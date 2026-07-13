# CHOPIN multilinear PCS — engineering design

This document specifies the engineering-optimized implementation of the CHOPIN
multilinear polynomial commitment scheme (Belohorec, Hubáček, Kalsta, Mašková,
*Chopin: Optimal Pairing-Based Multilinear Polynomial Commitments from Bivariate
KZG*), wired into this repository's `PolynomialCommitmentScheme` trait, HyperPlonk,
profiling and benchmarks.

The canonical `ChopinPCS` implements **Figure 5 (optimized CHOPIN from bivariate
KZG) + Figure 6 (BDFG20 multi-polynomial / multi-point batch opening)**. This is
the variant whose costs are reported in the paper's Table 1
(`(5+2)G + 7F`, `1N + (4+2)√N` prover MSMs, `3+2` pairings).

This document is kept in sync with the final code. All function names below refer to
`subroutines/src/pcs/chopin/{mod.rs,srs.rs,tests.rs}` and
`subroutines/src/pcs/bdfg.rs`.

---

## 1. Paper Figure → local function map

| Paper object | Local function / type |
|---|---|
| Fig. 1 interactive proof (restriction, fold, checks) | conceptual blueprint; realized by `chopin_core_open` / `chopin_core_verify` |
| Fig. 2 bivariate KZG `Com` | `ChopinProverParam::msm_full_reordered` (`C_F = [f(τ,σ)]_1`) |
| Fig. 2 bivariate KZG `Open` (`q1`, `q2`, `π1`, `π2`) | `divide_x_at_alpha` (→ `q1`, `f_alpha`), `divide_y_at_beta` (→ `q2`), `msm_q1_prefix` (→ `pi_biv_x`), `msm_sigma_slice` (→ `pi_biv_y`) |
| Fig. 2 bivariate KZG `Ver` (3-term pairing) | `chopin_core_verify` bivariate check (one `E::multi_pairing` over 3 terms) |
| Fig. 4 batched Lagrange IPA (`S`, verify identity) | `symmetric_lagrange_witness` (S0, S1), IPA identity in `chopin_core_verify` |
| Fig. 5 optimized CHOPIN (whole opening) | `chopin_core_open` / `chopin_core_verify` |
| Fig. 6 BDFG20 batch opening (`W`, `W'`, verifier `Cs`) | `crate::pcs::bdfg::{bdfg_first_round, bdfg_second_round, bdfg_verifier_combination}` |
| Fig. 6 union set / `ℓ_t` / `Z_{T\St}` / `Z_T` | `bdfg::{union_points, lagrange_interpolate, vanishing_poly}` |
| §2 / §6 complexity, proof size | this document §7–§9 and the `chopin_bench` outputs |
| p.6 odd-`nv` rectangular extension | `split_exponents` (m_left=ceil, m_right=floor), asymmetric `M_L != M_R` throughout |

Figure 7 (the *modified* standard-model batch proof, `W` + two same-point KZG
openings, 3 G1 / 4 pairing terms) is **not** what `ChopinPCS` implements. See §10.

---

## 2. DenseMultilinearExtension → bivariate coefficient matrix

`DenseMultilinearExtension` stores the evaluation vector `F[0..N]` in
little-endian variable order (`F[idx]`, bit `k` of `idx` = variable `k`).

For `mu` variables we split the variables into a left block (low variables) and a
right block (high variables):

```
m_left  = ceil(mu / 2)        M_L = 2^m_left
m_right = floor(mu / 2)        M_R = 2^m_right       N = M_L * M_R = 2^mu
i in [0, M_L)   <-> point[0 .. m_left]   (low variables)
j in [0, M_R)   <-> point[m_left .. mu]  (high variables)
index = i + j * M_L
```

The **bivariate twin** is
```
f(X, Y) = sum_{j=0}^{M_R-1} sum_{i=0}^{M_L-1} F[i + j*M_L] X^i Y^j,
degX(f) < M_L,  degY(f) < M_R,   C_F = [f(τ, σ)]_1.
```

`Psi_L[i] = eq(i, z_L)` (length `M_L`), `Psi_R[j] = eq(j, z_R)` (length `M_R`),
with `z_L = point[0..m_left]`, `z_R = point[m_left..mu]`. Then
`eta = F(z) = Psi_L^T F Psi_R`.

---

## 3. even / odd nv dimensions

| mu | m_left | m_right | M_L | M_R | N | q1 len (N−M_R) | q2 len (M_R−1) |
|----|--------|---------|-----|-----|---|----------------|----------------|
| 2  | 1 | 1 | 2 | 2 | 4 | 2 | 1 |
| 3  | 2 | 1 | 4 | 2 | 8 | 6 | 1 |
| 4  | 2 | 2 | 4 | 4 | 16 | 12 | 3 |
| 5  | 3 | 2 | 8 | 4 | 32 | 24 | 3 |
| 6  | 3 | 3 | 8 | 8 | 64 | 56 | 7 |

Odd `nv` uses the **rectangular** `M_L × M_R` matrix (`M_L = 2 M_R`). The number of
matrix entries is exactly `M_L·M_R = 2^mu = N`; **no padding to `2N`**. The dominant
`q1` commitment stays at `N − M_R` scalars. Requires `mu >= 2`.

---

## 4. SRS structure (exact)

Logical SRS:

* **G1**: `[τ^i σ^j]_1` for `0 <= i < M_L`, `0 <= j < M_R` — exactly `N` affine points.
* **G2**: `[1]_2`, `[τ]_2`, `[σ]_2` — exactly 3 affine points.
* **Verifier also needs** `[1]_1` (= `[τ^0 σ^0]_1`).

Explicitly **absent**: `[τ^2]_2`, `[σ^2]_2`, full G2 powers, any `2N` G1 storage, and
any separate univariate KZG SRS.

`gen_srs_for_testing` (TESTING ONLY trusted setup):
1. Sample non-zero `τ`, `σ` with `τ != σ`.
2. Build `M_L` `τ`-powers and `M_R` `σ`-powers (never the full grid of scalars).
3. `FixedBase` window table + chunked MSM (`SRS_GEN_CHUNK = 1<<16`); at most one chunk
   of scalars/projectives is alive at once. Only the final `N` G1 affine grid is kept.
4. Emit the 3 G2 affine elements.

Trimming from `max_nv` to `nv`: if `nv == max_nv` the `Arc<Vec<G1Affine>>` is shared
(no copy). Otherwise the smaller grid is rebuilt by reading the maximal grid through
its own `base_index`/`inverse_base_index` and re-placing each `(i,j)` at the smaller
layout's `base_index` (correct `τ^i σ^j` extraction and reorder, never a raw prefix).

---

## 5. q1-prefix G1 storage layout

`q1(X,Y) = (f(X,Y) − f(α,Y)) / (X − α)` has coefficients `q1_j[i]`,
`0 <= i < M_L−1`, `0 <= j < M_R`; length `(M_L−1)·M_R = N − M_R`.

To make the dominant commitment a single **contiguous prefix MSM**, the G1 grid is
stored in **q1-prefix (j-major) layout**:

```
base_index(i, j):
  if i <  M_L-1:  j*(M_L-1) + i          # prefix region, size (M_L-1)*M_R
  if i == M_L-1:  (M_L-1)*M_R + j        # tail region, size M_R  (highest X power)
inverse_base_index(p):
  if p <  (M_L-1)*M_R:  ( p % (M_L-1),  p / (M_L-1) )
  else:                 ( M_L-1,        p - (M_L-1)*M_R )
```

This is a bijection of the full `M_L × M_R` grid onto `[0, N)`.

Dedicated MSM helpers (on `ChopinProverParam`):

* `msm_full_reordered(evals)` — reorder `F[i+jM_L]` into `scalars[base_index(i,j)]`,
  one `N`-MSM over the whole grid → `C_F`.
* `msm_q1_prefix(q1_coeffs)` — `q1` is produced directly in `j*(M_L-1)+i` order, so the
  commitment is a single contiguous MSM over `g1[0 .. (M_L-1)*M_R]`, real length `N − M_R`.
* `msm_tau_slice(coeffs)` — commit a univariate poly (degree `< M_L`) on the
  `σ^0` slice. Positions `0..M_L-2` are the contiguous prefix `[τ^i]_1`; the single
  top coefficient (index `M_L-1`, if present) sits at position `(M_L-1)*M_R`, so it is
  a contiguous prefix MSM **plus one scalar multiplication**. Used for `C0, C1, CS`.
* `msm_sigma_slice(coeffs)` — commit `q2(Y)` (degree `< M_R−1`) on the `τ^0` slice
  `[σ^j]_1` (positions `j*(M_L-1)`, strided). Bases are collected (size `M_R−1`) and the
  reported profile `count` is the real scalar count.

The G1 SRS is stored once (shared `Arc<Vec<G1Affine>>`); no view is duplicated.

This layout differs from `nested_grid_kzg` (which uses a `(M_L-2)*M_R` quotient prefix
and keeps two dominant powers per column); CHOPIN's `q1` length is `(M_L-1)*M_R`.

---

## 6. Proof struct and byte size

```rust
pub struct ChopinProof<E: Pairing> {
    comm_f_zr:    E::G1Affine,   // C0 = [f_zR(τ)]_1 (τ-slice)
    comm_f_alpha: E::G1Affine,   // C1 = [f_alpha(τ)]_1 (τ-slice)
    comm_s:       E::G1Affine,   // CS = [S(τ)]_1 (τ-slice)
    pi_biv_x:     E::G1Affine,   // π1 = [q1(τ,σ)]_1
    pi_biv_y:     E::G1Affine,   // π2 = [q2(σ)]_1
    batch_w:      E::G1Affine,   // BDFG20 W = [m/Z_T (τ)]_1
    batch_w_prime:E::G1Affine,   // BDFG20 W' = [L/(X-z) (τ)]_1
    a:  E::ScalarField,          // f_zR(α)
    a1: E::ScalarField,          // f_zR(β)
    a2: E::ScalarField,          // f_zR(β^{-1})
    b1: E::ScalarField,          // f_alpha(β) = f(α,β)
    b2: E::ScalarField,          // f_alpha(β^{-1})
    s1: E::ScalarField,          // S(β)
    s2: E::ScalarField,          // S(β^{-1})
    mu: u32,                     // metadata, bound into transcript
}
```

7 G1 + 7 scalars + `mu` metadata. NOT stored: `alpha, beta, gamma, beta_inv, rho, z`,
interpolation coefficients, `q1/q2`, extra remainders, or duplicate commitments.

Cryptographic payload on BLS12-381 (compressed): `7·48 + 7·32 = 560 bytes`.
Canonical compressed serialization including `mu: u32` = `564 bytes` (a test asserts
both `560` and `564`, and that the size is constant for `nv = 8..20`).

---

## 7. Transcript order (`chopin-mlpcs-v1`)

Statement binding (before the first challenge):
`protocol version, mu, m_left, m_right, M_L, M_R, C_F, point, eta`.

```
statement
comm_f_zr            -> derive alpha        (alpha unconstrained beyond field)
comm_f_alpha, a
                     -> derive gamma        (gamma != 0)
comm_s
                     -> derive beta         (beta != 0, beta^2 != 1, beta != alpha, beta^{-1} != alpha)
a1, a2, b1, b2, s1, s2, pi_biv_x, pi_biv_y
                     -> derive rho          (rho != 0)
batch_w
                     -> derive z            (z not in {alpha, beta, beta^{-1}})
batch_w_prime
```

Constrained challenges use deterministic rejection resampling (`get_and_append_challenge`
in a bounded loop) shared verbatim by prover and verifier, so both derive identical
values. Every proof G1/scalar field is either absorbed into the transcript or consumed
by the final equations; no field is ignored by the verifier.

---

## 8. Core prover MSM count (exact)

`chopin_core_open` produces exactly these MSMs (real scalar lengths):

| MSM | phase | real scalar length | note |
|---|---|---|---|
| `pi_biv_x = [q1(τ,σ)]_1` | `chopin_open_commit_q1` | `N − M_R` | the single dominant `N`-scale MSM |
| `C0 = [f_zR(τ)]_1` | `chopin_open_commit_f_zr` | `M_L` | τ-slice |
| `C1 = [f_alpha(τ)]_1` | `chopin_open_commit_f_alpha` | `M_R` | τ-slice |
| `CS = [S(τ)]_1` | `chopin_open_commit_s` | `M_L − 1` | τ-slice |
| `pi_biv_y = [q2(σ)]_1` | `chopin_open_commit_q2` | `M_R − 1` | σ-slice |
| `batch_w = [W(τ)]_1` | `chopin_open_bdfg_commit_w` | `<= M_L − 1` | τ-slice |
| `batch_w_prime = [W'(τ)]_1` | `chopin_open_bdfg_commit_w_prime` | `<= M_L − 1` | τ-slice |

For even `nv` (`M_L = M_R = M = √N`): dominant `q1` = `N − M`; `C0,C1 ≈ M`; `CS ≈ M`;
`pi_y = M − 1`; `W, W' ≈ M`. Core opening total ≈ `1·N + (4+2)·√N`, exactly matching
Table 1 (`1N + (4+2)√N`). There is exactly **one** `N`-scale MSM — Mercury needs two
(its `q` and `quot_f`). CHOPIN introduces **no** `N`-scale polynomial-division quotient.

The seven univariate evaluation claims (`a, a1, a2, b1, b2, s1, s2`) are resolved by a
**single** Figure-6 BDFG20 multi-polynomial/multi-point batch opening (two extra G1s,
`W` and `W'`), not per-claim KZG proofs.

---

## 9. Verifier work

Field work: `O(mu)` — `psi_L(β), psi_L(β^{-1}), psi_R(β), psi_R(β^{-1})` via the
product form, plus the IPA identity and BDFG scalar reconstruction.

G1 scalar mults / MSM:
* `chopin_verify_bdfg_msm`: one MSM with 6 bases `[C0, C1, CS, [1]_1, W, W']`.

G2 scalar mults (dynamic, `chopin_verify_g2_scalars`, 2 total):
* `[τ − α]_2 = [τ]_2 − α[1]_2`
* `[σ − β]_2 = [σ]_2 − β[1]_2`

Pairings (`5` terms, `2` product checks, **never merged**):
* bivariate KZG: one `E::multi_pairing` over **3 terms**
  `e(C_F − b1[1]_1, [1]_2)·e(−pi_biv_x, [τ−α]_2)·e(−pi_biv_y, [σ−β]_2) = 1_GT`.
* BDFG20: one `E::multi_pairing` over **2 terms**
  `e(C_s + z·W', [1]_2)·e(−W', [τ]_2) = 1_GT`, with
  `C_s = C0 + ρ(z−α)C1 + ρ^2(z−α)CS − ℓ(z)[1]_1 − Z_T(z)·W`.

The two pairing equations are checked independently; they are **not** collapsed into one
product check with an unproven random combiner.

The IPA identity checked by the verifier (note the parenthesisation — `γ` multiplies
**both** `b1` and `b2`):
```
a1·psi_L(β^{-1}) + a2·psi_L(β) + γ·( b1·psi_R(β^{-1}) + b2·psi_R(β) )
    = 2(eta + γ·a) + β·s1 + β^{-1}·s2.
```

---

## 10. Figure 6 vs Figure 7 — security & performance calibration

The paper contains two batch-opening protocols:

* **Figure 6** (BDFG20 original ePrint 2020/081 §4): sends `W`, `W'` (2 G1),
  and the verifier checks one two-term pairing product. This is the batch proof
  counted by Table 1's `(5+2)G + 7F`, `3+2 pairings`, and is the batch proof
  implemented by `ChopinPCS` together with Figure 5. The original BDFG20
  knowledge-soundness analysis is in the Algebraic Group Model (AGM).

* **Figure 7** (modified, for standard-model extraction): explicitly sends `W`
  plus two same-point KZG opening proofs (`π^(s)`, `π^(q)`), hence has a
  different proof size and pairing count. Section 7 proves standard-model
  knowledge soundness for this modified protocol when batching a constant number
  of polynomials; it does not establish that result for Figure 6 itself.

Calibration recorded here:
* `ChopinPCS` proof-size / performance numbers correspond to **Figure 5 +
  Figure 6**.
* We do **not** claim the code mechanically proves knowledge soundness in any
  model.  The code is tested for correctness via polynomial identities and
  random-instance verification, which is an engineering confidence measure,
  not a formal security proof.
* The paper's Figure 7 standard-model result is a **separate protocol
  variant** and is **not** implemented in `ChopinPCS`. It therefore does not
  establish end-to-end standard-model knowledge soundness for this Figure 5 +
  Figure 6 backend.
* If a Figure-7 variant is implemented in the future, it should be a separate
  type (e.g. `ChopinStandardBatchPCS`) reporting its own proof size and
  pairing count; it is not a substitute for the canonical `ChopinPCS`.
* The non-interactive Fiat-Shamir variant used here relies on the Random-Oracle
  Model (ROM) heuristic. The interactive protocol's statistical soundness bound
  is not, by itself, a ROM knowledge-soundness theorem.

### Odd-nv rectangular soundness

The paper's Lemma 1 (soundness of the interactive proof, §4.1) is stated and
proved for the **even** `n = 2m` case.  The rectangular generalisation `mu =
2m+1` (p.6, "Extensions to an odd number of variables") adapts the dimensions
(`M_L = 2M_R`) and the soundness bound adjusts to `(M_L + M_R - 2)/|F|`.
**The paper text describes this adjustment; the rectangular formula is the
paper's own.** The code handles both even and odd `nv` through the same
dimensional-split code path; correctness is tested for both even and odd `nv`
(3,5,7,4,6,8).

---

## 11. Security vs engineering calibration

* Statistical soundness of the interactive reduction is `2(M - 1)/|F|` in the
  even case `M_L = M_R = M`, and `(M_L + M_R - 2)/|F|` in the odd rectangular
  case. The latter is the formula stated by the paper's odd-variable extension.
* The paper proves standard-model knowledge soundness for its modified Figure 7
  batch protocol under its stated assumptions. This repository implements
  Figure 5 + Figure 6 instead, so it must not claim that Figure-7 theorem for
  the implemented backend. The implementation also does not mechanically
  verify any security proof.
* The implementation provides **no hiding / zero knowledge**; `gen_srs_for_testing`
  samples the trapdoors locally and is TESTING ONLY.
* All engineering claims (proof size, MSM counts, pairing counts, transcript, profiling,
  benchmarks, correctness/negative tests) are realized and tested in this repository.

---

## 12. Shared BDFG20 module (`subroutines/src/pcs/bdfg.rs`)

Both Mercury and CHOPIN use the same BDFG20 (ePrint 2020/081, §4) multi-polynomial /
multi-point batch algebra. The shared module holds only pure polynomial algebra:

* `BdfgClaim<'a, F> { poly, points, values }`.
* `union_points` (union set + pairwise-distinct validation).
* `lagrange_interpolate`, `vanishing_poly` (and `mul_by_linear`, `divide_by_linear`,
  `add_scaled`, `poly_sub`, `poly_eval`, `subtract_const` primitives).
* `bdfg_first_round` → interpolants `ℓ_t`, `m(X)`, `W = m/Z_T` (exact division, zero
  remainder checked).
* `bdfg_second_round` → `L(X)`, `W' = L/(X−z)` (exact division, zero remainder checked).
* `bdfg_verifier_combination` → per-commitment scalars `ρ^t Z_{T\St}(z)`, the constant
  `ℓ`-scalar `sum_t ρ^t Z_{T\St}(z) ℓ_t(z)`, and `Z_T(z)`.

Commitments, SRS slice choice and transcript labels are supplied by the caller
(Mercury / CHOPIN), so the shared module never depends on a concrete `ProverParam`.

Mercury's claim ordering (`g, h, S, D` at `{ζ,ζ^{-1}}` / `{ζ,ζ^{-1},α}` / `{ζ,ζ^{-1}}` /
`{ζ}`, batching challenge `β`) and CHOPIN's (`f_zR, f_alpha, S` at `{α,β,β^{-1}}` /
`{β,β^{-1}}` / `{β,β^{-1}}`, batching challenge `ρ`) are both expressed as `BdfgClaim`
lists. Mercury's proof bytes, transcript and claim ordering are unchanged by the
migration; its existing tests continue to pass.

### Figure 6 vs Figure 7 batching difference

`bdfg.rs` implements Figure 6 (two witnesses `W`, `W'`, two verifier pairing terms). It
does **not** implement Figure 7 (which would split the witness identity into two
same-point KZG openings and check two pairing equations with an extra group element).

---

## 13. `core_open` vs `trait_open`

`PolynomialCommitmentScheme::open` receives no commitment, so it must **re-commit** `C_F`
(an extra `N`-MSM) to bind it into the transcript. `ChopinPCS::open_with_commitment`
takes a caller-held `C_F` and skips that recommit; benchmarks label it `core_open`.

* `core_open` (`chopin_core_open`) reflects the paper's `1N + (4+2)√N` MSM count.
* `trait_open` = `core_open` + one `N`-MSM `C_F` recommit. Benchmarks report both and
  make clear that only `core_open` is valid for the paper's `1N + (4+2)√N` claim.
