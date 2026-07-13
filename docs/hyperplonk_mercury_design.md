# Mercury multilinear PCS — design & implementation notes

This document describes the implementation of the **Mercury** multilinear
polynomial commitment scheme (ml-PCS) in `subroutines/src/pcs/mercury/`, its
mapping to the paper and to Microsoft's Nova reference, the exact Fiat–Shamir
order, proof/SRS shapes, complexity, and the security caveats.

- Paper: Liam Eagen, Ariel Gabizon, *Mercury: A multilinear Polynomial
  Commitment Scheme with constant proof size and no prover FFTs*, ePrint
  2025/385 (<https://eprint.iacr.org/2025/385.pdf>). The construction is
  Section 6; the components are Sections 4–5.
- Reference implementation: `microsoft/Nova`, `src/provider/mercury.rs`
  (MIT-licensed). This crate is also MIT-licensed, so the algorithm and the
  Fiat–Shamir schedule are adapted with attribution. The Rust code here is a
  clean-room rewrite on `arkworks` types: none of Nova's `halo2curves`,
  `ff`, `best_fft`, or `serde` code is copied. In particular Nova's FFT-based
  `make_s_polynomial` is replaced by an **FFT-free** structured computation
  (see §7) so the scheme stays generic over any `ark_ec::pairing::Pairing`
  and never requires an FFT-friendly (2-adic) scalar field.

## 1. Notation map (paper → local)

Let the multilinear have `mu = nv` variables and `N = 2^mu` coefficients.
Mercury lays `f` out as a `b_row x b` matrix with `t = ceil(mu/2)`,
`b = 2^t` (columns), `b_row = 2^{mu-t}` (rows), so `b * b_row = N` exactly (a
square `b x b` grid for even `mu`, a rectangular `b x (b/2)` grid for odd `mu`;
see §5). Columns index the low `t` variables `u1`, rows the high `mu-t`
variables `u2`.

| Paper | Local (this crate) | Meaning |
|-------|--------------------|---------|
| `f in F^n`, `fhat` | `poly: Arc<DenseMultilinearExtension>`, `coeffs = poly.to_evaluations()` | committed multilinear; `coeffs[idx]=fhat(bits(idx))`, little-endian |
| `f(X)=sum f_{i,j} X^{i+jb}` | `coeffs[i + j*b]`, `i` low (columns), `j` high (rows) | univariate twin, little-endian variable order |
| `f_i(X)=sum_j f_{i,j} X^j` | column polynomial `col_poly(i)` | degree `< b_row` |
| `u=(u1,u2)`, `u1,u2 in F^t` | `u1 = point[0..t]` (columns/low), `u2 = point[t..s]` (rows/high) | opening point split |
| `P_{u}(X)=sum eq(i,u)X^i` | `pu_eval(u,z)=prod_k (u_k z^{2^k}+(1-u_k))` and `eq_vec=build_eq_x_r_vec(u)` | `[X^i]P_u=eq(i,u)` |
| `h(X)=sum_i eq(i,u1) f_i(X)` | `compute_h(coeffs,eq_col,b_row,b)` | restricted poly; `h(alpha)=ghat(u1)`, `hhat(u2)=v` |
| `g(X)=f mod (X^b-alpha)=sum_i f_i(alpha)X^i` | `g` from `divide_by_binomial` | folded poly, degree `< b` |
| `q(X)`, `f=(X^b-alpha)q+g` | `q` from `divide_by_binomial` | quotient, degree `< N` |
| `S(X)` (§4.1) | `make_s_polynomial_structured` (FFT-free) | batched symmetric-Laurent IPA witness |
| `D(X)=X^{b-1} g(1/X)` | `d = g reversed` | degree check for `g` |
| `pi_z=[H(x)]`, `H=(f-(z^b-α)q-g_z)/(X-z)` | `comm_quot_f` | KZG folding proof |
| `pi'` (§4 BDFG20) | `comm_w`, `comm_w_prime` | batched multi-point KZG proof |
| `[1],[x],...` and `[1]_2,[x]_2` | `g1_powers`, `g2_one`, `g2_tau` | SRS |

`z` (paper's evaluation challenge) is named `zeta` locally to match Nova and to
avoid clashing with the BDFG20 point which we call `z_bdfg`.

## 2. Nova function map (Nova → local)

| Nova (`src/provider/mercury.rs`) | Local |
|----------------------------------|-------|
| `EvaluationEngine::prove` | `mercury_core_open` |
| `EvaluationEngine::verify` | `mercury_core_verify` |
| `divide_by_binomial` | `divide_by_binomial` |
| `compute_h_poly` | `compute_h` |
| `make_s_polynomial` (FFT) | `make_s_polynomial_structured` (FFT-free, `laurent::mul_by_reciprocal_tensor`) |
| `eval_pu_poly` | `pu_eval` |
| `UniPoly::divide_by_linear_polynomial` | `divide_by_linear` |
| `d_poly = g reversed` | `reverse_coeffs` |
| `batch_evaluation::generate_batch_evaluate_arg` | `bdfg_prove` |
| `batch_evaluation::extract_pairing_to_verify_batch_evaluation` | `bdfg_verify_lhs` |
| `UniPoly::from_evals_with_xs` | `interpolate_small` (Lagrange, 1..3 pts) |
| `hyperkzg::{ProverKey,VerifierKey}` | `MercuryProverParam`, `MercuryVerifierParam` |

## 3. Transcript order (domain `mercury-mlpcs-v1`)

Prover and verifier share `absorb_statement` / the same append+challenge helper
sequence. Statement is bound **before the first challenge**.

| # | Action | label |
|---|--------|-------|
| 0 | append protocol version | `ver` |
| 0 | append `mu`, `s`, `t`, `b` (split params) | `mu`,`s`,`t`,`b` |
| 0 | append `C_f` | `cf` |
| 0 | append full point `u` | `u` |
| 0 | append claimed value `v` | `e` |
| 1 | append `comm_h` | `h` |
| 2 | **squeeze `alpha`** | `a` |
| 3 | append `comm_q`, `comm_g` | `q`,`g` |
| 4 | **squeeze `gamma`** | `gm` |
| 5 | append `comm_s`, `comm_d` | `s`,`d` |
| 6 | **squeeze `zeta`** (require `zeta!=0`, `zeta^2!=1`) | `zt` |
| 7 | append `g_zeta,g_zeta_inv,h_zeta,h_zeta_inv,s_zeta,s_zeta_inv` | `gz`,`gzi`,`hz`,`hzi`,`sz`,`szi` |
| 8 | append `comm_quot_f` | `t` |
| 9 | **squeeze `beta`** (BDFG20 batch challenge) | `b` |
| 10 | append `comm_w` | `w` |
| 11 | **squeeze `z_bdfg`** (require distinct from `alpha,zeta,zeta_inv`) | `z` |
| 12 | append `comm_w_prime` | `wp` |
| 13 | **squeeze `d_pair`** (final pairing batch challenge) | `pd` |

This matches Nova's schedule exactly (Nova squeezes `alpha` after `comm_h` and
before absorbing `comm_q/comm_g`), extended with an explicit statement binding
(version + split params) at step 0 and length checks throughout.

## 4. Proof, SRS, complexity

**Proof** (`MercuryProof`): 8 G1 elements
`comm_h, comm_g, comm_q, comm_s, comm_d, comm_quot_f, comm_w, comm_w_prime`
plus 6 field elements `g_zeta, g_zeta_inv, h_zeta, h_zeta_inv, s_zeta,
s_zeta_inv`, plus `mu` (bound into transcript, checked by verifier). This is
constant size, independent of `nv`. `h_alpha` and `d_zeta` are **not** sent —
the verifier reconstructs them from the two batched Lagrange-IPA identities and
the degree-check identity. `alpha,gamma,zeta,beta,z_bdfg,d_pair` are Fiat–Shamir
challenges, never sent. Every field/G1 in the struct is used by the verifier.

Cryptographic payload = `8*|G1| + 6*|F|`. On BLS12-381 compressed: `8*48 + 6*32
= 576` bytes (plus the tiny `mu` and serialization framing).

**SRS** (tight Mercury SRS, *not* the Claymore 2N SRS):
- G1: `[tau^0..tau^{N-1}]_1`, exactly `N = 2^mu` powers (max committed degree is
  `deg(quot_f) = N-2` and `deg(f) = N-1`; `q` has degree `N-b-1`; `g,h,s,d,w,w'`
  have degree `< b`). All fit in `N` powers.
- G2: exactly `[1]_2` and `[tau]_2`. Every pairing check has the form
  `e(L,[1]_2)=e(R,[tau]_2)`; no `[tau^2]_2` or higher is needed.

**Prover** (core, excludes trait-API recommit of `C_f`): two `N`-scalar MSMs
(`comm_q`, `comm_quot_f`) + six `O(b)=O(sqrt N)` MSMs (`comm_h,comm_g,comm_s,
comm_d,comm_w,comm_w_prime`) => `2N + O(sqrt N)` scalar multiplications. Field
work: `divide_by_binomial` `O(N)`, structured `S` `O(sqrt N * log N)`, all other
`O(sqrt N)` => `O(N)` field ops, no FFT.

**Verifier**: `O(log N)` field ops (two `P_u` evaluations at `zeta,1/zeta`),
three G1 MSMs (sizes 3, 7, 2), and **2 pairings** (constant, independent of
`nv`).

## 5. odd `nv`

Mercury's paper is stated for an even variable count `s = 2t` (`n = b^2`). This
implementation uses a **rectangular, non-padding split** that keeps the
committed polynomials at their original size and degenerates to the square case
for even `mu`:

- `t = ceil(mu/2)`, `b = 2^t` (columns = low `t` variables `u1`),
- `b_row = 2^{floor(mu/2)} = 2^{mu-t}` (rows = high `mu-t` variables `u2`),
- so `b * b_row = 2^mu = N` exactly, and `idx = i + j*b` is a genuine
  little-endian split of `[0, N)` into low `t` bits `i` and high `mu-t` bits `j`.

For even `mu` this is the square `b x b` grid (`b_row = b`). For odd `mu` it is
`b x (b/2)` (`b_row = b/2`): the column polynomials `f_i` have degree `< b/2`,
`g` still has degree `< b`, and the degree check / IPA identities hold verbatim
because Claim 5.1 only needs `g(X) = sum_{i<b} f_i(alpha) X^i` and `deg g < b`,
which is independent of `b_row <= b`. The `P_{u2}` factor uses the `mu-t`
components of `u2`; for the structured `S` helper we pad `u2` and `h` up to the
tensor length `t`/`b` with trailing zeros (a trivial factor `(0*X^{...}+1)=1`).

Consequences of odd `mu`:
- The committed `f`, `q`, `quot_f` stay `N`-sized => the two dominant prover
  MSMs stay `N`-sized and the SRS degree bound stays `N-1` (no doubling).
- The `O(sqrt N)` helper polynomials `g,h,s,d,w,w'` have length `b = 2^{ceil(mu/2)}
  = sqrt(2N)`, i.e. a `sqrt(2)` factor larger than the even case; this is a pure
  `O(sqrt N)` overhead, reported separately.

This differs from Nova, which instead prepends a zero variable and works over a
`b x b` matrix with a zero upper half (`b = sqrt(2N)`, `b_row = b/2` used only to
skip the zero rows in the big MSMs). Both are correct; the rectangular split
avoids the point-padding bookkeeping and is the "non-padding rectangular split"
option the task allows. The odd-`nv` tests (`open_verify_even_and_odd`,
`open_verify_minimum_nv`, HyperPlonk `nv=5`) exercise this path directly.

## 6. Differences vs Nova (and why)

1. **No FFT.** Nova's `make_s_polynomial` uses `best_fft` over a `2b`-point
   subgroup (requires 2-adic `ROOT_OF_UNITY`). We compute `S` FFT-free from the
   tensor structure of `P_{u1}, P_{u2}` (§7). Output and transcript are
   identical; complexity is `O(sqrt N log N)` instead of `O(sqrt N log N)` FFT
   but with no field-structure requirement. This is what keeps the scheme
   `Pairing`-generic with no `FftField` bound.
2. **arkworks types**, `IOPTranscript`, `Commitment<E>`, `PCSError` instead of
   Nova's `ff`/`serde`/`NovaError`.
3. **Explicit statement binding & input validation.** We bind protocol
   version + split params, and the verifier checks `proof.mu`, point length,
   vector lengths, and vk capacity before any shift/alloc; malformed proofs
   return `PCSError::InvalidProof` / `Ok(false)` and never panic.
4. **Degenerate-challenge rejection.** We reject `zeta=0`, `zeta^2=1`, and
   `z_bdfg in {alpha,zeta,zeta_inv}` (negligible probability) rather than
   silently substituting; both parties apply identical checks.
5. **`open_with_commitment`.** The trait `open` re-commits `C_f`; we also expose
   `open_with_commitment` to avoid the extra `N`-MSM when the caller already has
   `C_f` (the benchmark reports both).

## 7. FFT-free structured `S(X)`

`S` is defined by the symmetric Laurent identity
```
g(X)P_{u1}(1/X) + g(1/X)P_{u1}(X) + gamma(h(X)P_{u2}(1/X)+h(1/X)P_{u2}(X))
  = 2(h(alpha)+gamma v) + X S(X) + (1/X) S(1/X).
```
Because `P_u(1/X) = prod_{k<t} (u_k X^{-2^k} + (1-u_k))` is a tensor product,
`C1(X) := g(X)P_{u1}(1/X)` is computed by `t` structured shift-add passes over a
length-`(2b-1)` Laurent buffer (`laurent::mul_by_reciprocal_tensor`, the same
kernel used by ReciPCS). By symmetry `g(1/X)P_{u1}(X) = C1(1/X)`, so the
coefficient of `X^k` (`k>=1`) in the LHS is
`A_k = C1[b-1+k]+C1[b-1-k] + gamma(C2[b-1+k]+C2[b-1-k])`, and
`S(X) = sum_{k=1}^{b-1} A_k X^{k-1}` (degree `b-2`). Cost `O(b t)=O(sqrt N log
N)`, no FFT. A dense `O(b^2)` reference (`make_s_polynomial_dense_reference`)
computes the full symmetric Laurent product and is checked coefficient-by-
coefficient against the structured version for `nv=2,4,6,8` in the tests.

The kernel `laurent::mul_by_reciprocal_tensor(coeffs, m, r)` returns the Laurent
buffer of `coeffs(X) * prod_k ((1-r_k)+r_k X^{-2^k})` (length `2^{m+1}-1`,
offset `2^m-1`). ReciPCS's `compute_laurent_h` is a thin wrapper over it, so the
formula lives in exactly one place.

## 8. Security model / caveats

- Knowledge soundness holds in the Algebraic Group Model under `Q`-DLOG for
  `Q = N-1` (paper §6). This crate implements the protocol; it does not
  re-prove soundness.
- `gen_srs_for_testing` samples the trapdoor `tau` locally and is **for testing
  only**. Production use requires an `N`-power powers-of-tau ceremony. No hiding
  / zero-knowledge is provided (matching the other PCS backends here).
- `trim` returns `Result` and never panics on bad sizes; all shifts use
  `checked_shl`, all size products use `checked_mul`.

## 9. Verifier checks performed

1. Reconstruct `d_zeta = zeta^{b-1} g_zeta_inv` (degree check for `g`).
2. Reconstruct `h_alpha` from the two batched Lagrange-IPA relations.
3. Folding relation `f = (X^b-alpha)q + g` at `zeta` via `comm_quot_f`.
4. BDFG20 batched opening of `{g,h,s,d}` at `{zeta,zeta_inv,alpha}` (all seven
   scalar identities are baked into the reconstructed pairing statement).
5. The two pairing statements are combined with a fresh challenge `d_pair` and
   checked with **2 pairings**.

All of these are necessary and jointly imply `fhat(u)=v`.
