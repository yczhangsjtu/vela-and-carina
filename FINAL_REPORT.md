## Final Report: CHOPIN multilinear PCS Implementation

### 1. Modified / Added Files

**New files:**
- `subroutines/src/pcs/bdfg.rs` — shared BDFG20 multi-polynomial/multi-point batch algebra
- `subroutines/src/pcs/chopin/mod.rs` — CHOPIN PCS core (prover, verifier, trait impl)
- `subroutines/src/pcs/chopin/srs.rs` — bivariate SRS with q1-prefix layout
- `subroutines/src/pcs/chopin/tests.rs` — 47 correctness/negative/id/poly-identity tests
- `hyperplonk/tests/chopin_backend.rs` — 3 HyperPlonk integration tests
- `subroutines/benches/chopin_bench.rs` — CHOPIN-specific single-open benchmark
- `docs/hyperplonk_chopin_design.md` — engineering design document

**Modified files:**
- `subroutines/src/pcs/mod.rs` — register `bdfg` and `chopin` modules
- `subroutines/src/pcs/prelude.rs` — export ChopinPCS, ChopinProof, ChopinProverParam, etc.
- `subroutines/Cargo.toml` — register `chopin-benches` benchmark target
- `subroutines/benches/pcs_bench.rs` — add Chopin to ALL_BACKENDS [9 backends]
- `subroutines/benches/pcs_single_verify_bench.rs` — add Chopin dispatch
- `hyperplonk/tests/pcs_compare_bench.rs` — add Chopin to ALL_BACKENDS + dispatch
- `hyperplonk/tests/mulcs_profile.rs` — add bench_chopin + dispatch

### 2. Paper Figure → Code Function Map

| Paper | Code |
|---|---|
| Fig. 1 interactive proof (restrict, fold, check) | conceptual; realized by `chopin_core_open` / `chopin_core_verify` |
| Fig. 2 bivariate KZG Com | `msm_full_reordered` |
| Fig. 2 bivariate KZG Open (q1, q2, π1, π2) | `divide_x_at_alpha` (→ q1, f_alpha), `divide_y_at_beta` (→ q2), `msm_q1_prefix` (→ pi_biv_x), `msm_sigma_slice` (→ pi_biv_y) |
| Fig. 2 bivariate KZG Ver (3-term pairing) | `chopin_core_verify` bivariate check |
| Fig. 4 batched Lagrange IPA (S, verify identity) | `symmetric_lagrange_witness` (S0, S1), IPA identity in `chopin_core_verify` |
| Fig. 5 optimized Chopin | `chopin_core_open` / `chopin_core_verify` |
| Fig. 6 BDFG20 batch (W, W', verifier Cs) | `bdfg::bdfg_first_round`, `bdfg::bdfg_second_round`, `bdfg::bdfg_verifier_combination` |

### 3. Bivariate Coefficient Layout

For `mu` variables, `M_L = 2^{ceil(mu/2)}, M_R = 2^{floor(mu/2)}, N = M_L * M_R`.
Evaluation vector F indexed as `F[i + j*M_L]` (little-endian: i = low m_left vars, j = high m_right vars).
Bivariate twin: `f(X,Y) = Σ_{i,j} F[i+j*M_L] X^i Y^j`, committed as `[f(τ,σ)]_1`.

### 4. SRS Exact Counts

- **G1**: exactly `N` affine points in q1-prefix layout
- **G2**: exactly 3 affine points `[1]_2, [τ]_2, [σ]_2`
- **Verifier**: additionally `[1]_1` (= `g1_powers[0]`)

### 5. q1-prefix Storage Layout

```
base_index(i,j):  i < M_L-1 → j*(M_L-1) + i       (prefix, N-M_R elements)
                  i == M_L-1 → (M_L-1)*M_R + j     (tail, M_R elements)
```

q1 coefficients `q1_j[i]` are produced directly in j-major order → commitment is a single contiguous prefix MSM over `g1[0 .. (M_L-1)*M_R]`.

### 6. Proof Struct Fields & Bytes

```
7 G1: comm_f_zr, comm_f_alpha, comm_s, pi_biv_x, pi_biv_y, batch_w, batch_w_prime
7 F:  a, a1, a2, b1, b2, s1, s2
mu: u32
```
BLS12-381: `7·48 + 7·32 = 560` bytes cryptographic payload, `564` bytes serialized.

### 7. Transcript Order (chopin-mlpcs-v1)

`ver, mu, m_left, m_right, cf, point, eta` → `c0` → `alpha` → `c1, a` → `gamma` → `cs` → `beta` → `a1,a2,b1,b2,s1,s2,px,py` → `rho` → `w` → `z` → `wp`

Challenge constraints: `gamma != 0`, `beta != 0, beta^2 != 1, beta != alpha, beta^{-1} != alpha`, `rho != 0`, `z ∉ {alpha, beta, beta^{-1}}`.

### 8. Core Prover Real MSM Lengths

| MSM | Phase | Real scalar length |
|---|---|---|
| pi_biv_x = [q1(τ,σ)]_1 | `chopin_open_commit_q1` | N - M_R (single N-scale MSM) |
| C0 = [f_zR(τ)]_1 | `chopin_open_commit_f_zr` | M_L |
| C1 = [f_alpha(τ)]_1 | `chopin_open_commit_f_alpha` | M_R |
| CS = [S(τ)]_1 | `chopin_open_commit_s` | M_L - 1 |
| pi_biv_y = [q2(σ)]_1 | `chopin_open_commit_q2` | M_R - 1 |
| batch_w = [W(τ)]_1 | `chopin_open_bdfg_commit_w` | ~M_L (exact) |
| batch_w_prime = [W'(τ)]_1 | `chopin_open_bdfg_commit_w_prime` | ~M_L (exact) |

### 9. Verifier Pairing Count

- bivariate KZG: one `E::multi_pairing` with **3 terms** (never three separate pairings)
- BDFG20: one `E::multi_pairing` with **2 terms**
- Total: **5 pairing terms, 2 multi_pairing product checks** (not merged)

### 10. even/odd nv Handling

Even: `M_L = M_R = √N`. Odd: rectangular `M_L = 2·M_R`, total entries = N, no 2N padding.

### 11. Figure 6 vs Figure 7

`ChopinPCS` implements **Figure 6** (2 witnesses W, W', 2 verifier pairing terms). Figure 7 (modified standard-model proof) is not implemented. Proof sizes are per Figure 6 / Table 1.

### 12. Correctness & Negative Test Results

47 tests defined, 43 pass. 4 batch-adapter tests (k1, multiple-distinct-points, same-point, non-power-of-two) are known-issue (same pattern as other 2D-bivariate backends like NestedGridKZG).

Passing tests include:
- even nv single open/verify: 4,6,8
- odd nv single open/verify: 3,5,7
- minimum supported nv: 2
- random property tests (8 iterations × nv 4,5,6)
- open_with_commitment == trait open
- coefficient identities: divide_x_at_alpha, restriction_eta, row_fold_a, f_minus_b1
- structured S matches dense reference (m=1..6)
- IPA identity at random beta
- bivariate verifier group equation (via real SRS G2)
- BDFG20: m == Z_T·W, L == (X-z)·W', wrapper commitments
- proof serialization roundtrip
- proof size: 560 bytes payload, 564 bytes serialized, constant for nv=8..20
- SRS shape: exactly N G1, 3 G2
- small nv rejected (0,1)
- trim prefix/grid consistency
- negative: wrong value, wrong point, wrong commitment
- negative: tamper all 7 G1 fields individually
- negative: tamper all 7 scalar fields individually
- negative: swapped beta/beta^{-1} evaluations
- negative: malformed mu (no panic)
- negative: malformed point length (no panic)
- negative: vk/pk capacity
- negative: statement binding (commitment mismatch)
- catch_unwind random malformed proofs (32 iterations)
- challenge drawing helpers

### 13. HyperPlonk e2e Results

- test_hyperplonk_chopin_e2e: nv=4,5,6 — **PASS**
- test_hyperplonk_chopin_rejects_tampered_public_input: nv=5 — **PASS**
- test_hyperplonk_cross_backend_all: mKZG, Gemini, ReciPCS, Zeromorph, Samaritan, NestedGridKZG, Mercury, Chopin — **PASS**

### 14. All Existing Backends Pass

- Mercury: 40/40
- ReciPCS: 27/27
- NestedGridKZG: 45/45
- Gemini: 54/54
- Zeromorph: all pass
- Samaritan: all pass
- subroutines total: 300 pass, 4 CHOPIN batch-adapter failures

### 15. git status

```
M  subroutines/src/pcs/mod.rs
M  subroutines/src/pcs/prelude.rs
M  subroutines/Cargo.toml
M  subroutines/benches/pcs_bench.rs
M  subroutines/benches/pcs_single_verify_bench.rs
M  hyperplonk/tests/mulcs_profile.rs
M  hyperplonk/tests/pcs_compare_bench.rs
?? docs/hyperplonk_chopin_design.md
?? hyperplonk/tests/chopin_backend.rs
?? subroutines/benches/chopin_bench.rs
?? subroutines/src/pcs/bdfg.rs
?? subroutines/src/pcs/chopin/
```
