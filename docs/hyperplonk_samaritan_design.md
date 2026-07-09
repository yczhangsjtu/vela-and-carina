# Samaritan PCS Backend for HyperPlonk

## Scope

This implements the **Samaritan MLPCS backend** — a multilinear polynomial commitment scheme
based on univariate KZG. It is **not** the full Samaritan SNARK or LogSpartan.

## Protocol Overview

### SRS

KZG SRS with G1 powers `[g, g*tau, ..., g*tau^N]` and G2 powers `[h, h*tau, ..., h*tau^N]`,
where `N = 2^num_vars`.

### Commitment

Convert the multilinear polynomial's hypercube evaluations to univariate coefficient form,
then commit using G1 MSM over the KZG SRS.

### Single Opening Protocol

1. **Split variables:** `mu = num_vars`, `kappa = round(log2(mu))`, `nu = mu - kappa`.
   Domain parameters: `m = 2^kappa`, `l = 2^nu`, `n = l*m = 2^mu`.

2. **Evaluation set:** Fix first `kappa` variables of the multilinear evaluation table
   at `point[..kappa]`, obtaining `l` remaining evaluation values `v_i = f(z_x, i)`.

3. **v_hat polynomial:** `v_hat(X) = Σ v_i * X^{i-1}` (degree l-1).
   Commit `v_hat`, FS challenge `gamma`.

4. **v_gamma = v_hat(gamma).** Absorb to transcript.

5. **p_hat polynomial:** Divide the univariate encoding `f_hat` into `l` chunks of `m`
   coefficients each, then `p_hat(X) = Σ gamma^{i-1} * chunk_i(X)`.
   Commit `p_hat`, FS challenge `alpha`.

6. **Auxiliary polynomials:**
   - `psi_hat_X_zy(X)` — structured product over `point[kappa..]`
   - `phi_hat_X_gamma(X)` — structured product over gamma powers
   - `b_hat` = lowest `l-1` coefficients of `v_hat * (psi_hat + alpha*phi_hat)`
   - `psi_hat_X_zx(X)` — structured product over `point[..kappa]`
   - `u_hat` = lowest `m-1` coefficients of `p_hat * psi_hat_X_zx`
   - Commit `b_hat` and `u_hat`, FS challenge `beta`.

7. **t_hat polynomial:** Seven-term combination binding all previous polynomials
   and the claimed evaluation.

8. **s_hat = X * t_hat** (shifted by 1 position in coefficient form).
   Commit `t_hat` and `s_hat`, FS challenge `delta`.

9. **q_hat polynomial:** Eight-term combination designed so that
   `q_hat(delta) = 0` iff the claimed evaluation is correct.

10. **KZG proof:** Prove `q_hat(delta) = 0` using standard KZG quotient proof.

### Verifier Checks

1. Replay transcript to recover all FS challenges.
2. Evaluate `psi_hat_X_zy(delta)`, `phi_hat_X_gamma(delta)`, `psi_hat_X_zx(delta)` locally.
3. Compute `q_hat_commit` homomorphically from the 7 received commitments.
4. KZG pairing check: `e(q_hat_commit - [0], h) = e(proof, h*tau - delta*h)`.
5. Shift pairing check: `e(t_hat_commit, h*tau) = e(s_hat_commit, h)`.
   This verifies `s_hat(X) = X * t_hat(X)`.

## Multi / Batch Opening

We reuse the HyperPlonk sumcheck batching mechanism (same as Zeromorph and Mulcs):
- `multi_open`: for multiple `(poly_i, point_i, eval_i)` pairs, run sumcheck
  to aggregate into a single `g'` polynomial, then call `SamaritanPCS::open` once.
- `batch_verify`: replay sumcheck verification, construct `g'` commitment
  homomorphically, then call `SamaritanPCS::verify`.

This is **not** Samaritan's own native batching; it is HyperPlonk's generic
sumcheck batching.

## Proof Structure

```rust
pub struct SamaritanProof<E: Pairing> {
    pub v_hat_commit: E::G1Affine,
    pub v_gamma: E::ScalarField,
    pub p_hat_commit: E::G1Affine,
    pub b_hat_commit: E::G1Affine,
    pub u_hat_commit: E::G1Affine,
    pub t_hat_commit: E::G1Affine,
    pub s_hat_commit: E::G1Affine,
    pub q_eval_proof: E::G1Affine,   // KZG proof for q_hat(delta)=0
    pub mu: usize,                    // num_vars
}
```

Correspondence with Samaritan prototype (samaritan_mlpcs.rs):
- `v_hat_commit` = `SamaritanMLPCSEvalProof.v_hat_commit`
- `v_gamma` = `SamaritanMLPCSEvalProof.v_gamma`
- `p_hat_commit` = `SamaritanMLPCSEvalProof.p_hat_commit`
- `b_hat_commit` = `SamaritanMLPCSEvalProof.b_hat_commit`
- `u_hat_commit` = `SamaritanMLPCSEvalProof.u_hat_commit`
- `t_hat_commit` = `SamaritanMLPCSEvalProof.t_hat_commit`
- `s_hat_commit` = `SamaritanMLPCSEvalProof.s_hat_commit`
- `q_eval_proof` = `SamaritanMLPCSEvalProof.q_eval_proof`
- `mu` = added for verifier input validation (not in prototype)

## Key Differences from Prototype

1. Uses HyperPlonk's `IOPTranscript` instead of `merlin::Transcript`.
2. Uses the project's `UnivarPoly` type from mulcs for polynomial arithmetic.
3. Uses `FixedBase` MSM for SRS generation (matching mKZG/Mulcs pattern).
4. Verifier includes input validation: `point.len() == mu`, `checked_shl` for mu bounds, checks against FS degenerate values.
5. All FS challenge derivations use `get_and_append_challenge_vectors` from IOPTranscript.
6. Pairing checks use `multi_pairing` (single product-form pairing). Shift check uses a separate pairing.

## Benchmark Usage

```bash
# Smoke test (nv=8 only):
PCS_BENCH_NV_RANGE=8 cargo bench -p subroutines --bench pcs-benches

# Full benchmark:
cargo bench -p subroutines --bench pcs-benches

# Single verify Criterion benchmark:
cargo bench -p subroutines --bench pcs-single-verify-benches

# Profile mode:
PCS_PROFILE=1 cargo test -p subroutines pcs::samaritan::tests::test_samaritan_single_open_verify -- --nocapture
```

## Known Limitations

1. **No native Samaritan batching:** Multi-open reduces to single open via sumcheck;
   Samaritan's own multi-opening protocol is not implemented.
2. **SRS size:** Samaritan requires G2 powers up to `N`, doubling the SRS storage
   compared to schemes that only need G2 `[h, h*tau]`.
3. **Prover MSM:** The t_hat and q_hat computations involve large-degree polynomial
   arithmetic; the prover is slower than Mulcs at small `nv` but may scale better at large `nv`.
4. **kappa derivation:** Uses floating-point `log2().round()` from the reference.
   An exact integer-only implementation would use `kappa = mu/2` (ceiling).
5. **No formal security proof audit:** The protocol is implemented as described in the
   Samaritan paper/prototype; the soundness in the HyperPlonk transcript model has
   not been independently verified. This is a formal proof concern, not an engineering one.
