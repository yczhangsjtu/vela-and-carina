# HyperPlonk + Zeromorph PCS Design

## Reference

Zeromorph implementation adapted from [han0110/plonkish](https://github.com/han0110/plonkish) (MIT-licensed).

Key files: `plonkish_backend/src/pcs/multilinear/zeromorph.rs`, `plonkish_backend/src/pcs/multilinear.rs`.

## SRS Structure

Zeromorph uses two slices of a single univariate KZG SRS:

```
Universal params: monomial_g1[0..M], powers_of_s_g2[0..M]
  trim(poly_size=N) →
    offset = M - N
    commit_powers: G1[0..N]        — for poly + quotient + q_hat commitments
    open_powers:   G1[offset .. offset+N] — for final KZG opening proof
    s_offset_g2:   G2[offset]      — boundary G2 element for pairing
```

When `gen_srs_for_testing(nv)` generates `M = 2N = 2·2^{nv}` and `trim(N)` is called:
`offset = 2N - N = N`.

**G2 powers optimization**: Both G1 and G2 powers are generated via `FixedBase::msm` 
(Arkworks FixedBase), reusing the same field power vector. This achieves 
near-parity with mKZG/Mulcs SRS generation speed for G1, and eliminates 
the sequential G2 scalar multiplication bottleneck.

## Single Opening Protocol

1. Compute NV quotient polynomials q_i via MLE reduction (matching plonkish `quotients`)
2. Commit each q_i using `commit_powers`
3. Challenge y from transcript → form q_hat(X) = Σ y^i · X^{N-2^i} · q_i(X)
4. Commit q_hat using `commit_powers`
5. Challenges x, z from transcript
6. Compute scalars (matching plonkish `eval_and_quotient_scalars`)
7. Build f(X) = z·poly + q_hat + eval_scalar·eval + Σ q_scalars·q_i  (f(x) = 0)
8. KZG open f at x using `open_powers` (offset=N)

## Verifier

- Replay transcript, reconstruct c = q_hat_comm + z·comm + eval_scalar·eval·g + Σ q_scalars·q_comms
- Pairing: e(C, s_offset_g2) == e(π, s_g2 - x·g2)

## Batch Opening

Reuses hyperplonk-baseline's sumcheck batching framework (same as mKZG/Mulcs). Multiple openings are reduced to a single opening of g' at sumcheck point a2, then `open_with_transcript` is called on g'.

## Integration with HyperPlonk

`ZeromorphPCS<E>` implements `PolynomialCommitmentScheme<E>` with:
- `Polynomial = Arc<DenseMultilinearExtension<Fr>>`
- `BatchProof = BatchProof<E, ZeromorphPCS<E>>` (sumcheck batching)
