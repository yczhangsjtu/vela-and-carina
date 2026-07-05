# HyperPlonk + Mulcs PCS Design

## 1. Existing HyperPlonk + mKZG Call Chain

```
PolyIOP::<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>
  │
  ├─ preprocess(index, pcs_srs)
  │   ├─ PCS::trim(pcs_srs, None, Some(num_vars))  → (pk, vk)
  │   ├─ PCS::commit(&ck, perm_oracle)
  │   └─ PCS::commit(&ck, selector_oracle)
  │
  ├─ prove(pk, pub_input, witnesses)
  │   ├─ PCS::commit(&ck, witness_poly)              // commit witnesses
  │   ├─ ZeroCheck::prove(f(x), transcript)           // PIOP
  │   ├─ PermutationCheck::prove(..., transcript)     // PIOP
  │   ├─ PcsAccumulator::insert_poly_and_points(...)  // collect openings
  │   └─ PCS::multi_open(&ck, polys, points, evals, transcript)
  │
  └─ verify(vk, pub_input, proof)
      ├─ ZeroCheck::verify(...)
      ├─ PermutationCheck::verify(...)
      └─ PCS::batch_verify(&vk, comms, points, batch_proof, transcript)
```

## 2. MulcsPCS Associated Types

```rust
impl<E: Pairing> PolynomialCommitmentScheme<E> for MulcsPCS<E> {
    type ProverParam = MulcsProverParam<E>;
    type VerifierParam = MulcsVerifierParam<E>;
    type SRS = MulcsUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = MulcsProof<E>;
    type BatchProof = MulcsBatchProof<E>;
}
```

## 3. From DenseMultilinearExtension to Coefficient/Evaluation Vector

`DenseMultilinearExtension` stores evaluations on the hypercube. Mulcs uses
univariate representation. Both representations are compatible: the evaluation
vector of length N = 2^nv can be used directly as the coefficients of a
univariate polynomial f_v(X) = Σ coeffs[i] · X^i.

For commitment: `to_evaluations()` → `UnivarPoly::new(evals)` → KZG commit.

For evaluation: built-in `evaluate(&point)` from `MultilinearExtension`.

## 4. SRS, ProverParam, VerifierParam Design

### MulcsUniversalParams<E>

Wraps univariate KZG SRS: `g1_powers` up to degree 2·N, G2 elements.

```rust
struct MulcsUniversalParams<E: Pairing> {
    prover_param: MulcsProverParam<E>,
    verifier_param: MulcsVerifierParam<E>,
}
```

### MulcsProverParam<E>

G1 powers for univariate MSM, plus gamma structured randomness.

```rust
struct MulcsProverParam<E: Pairing> {
    g1_powers: Vec<E::G1Affine>,   // [1, x, x², ..., x^{max_degree}] in G1
    g2_one: E::G2Affine,
    g2_x: E::G2Affine,
    g2_x2: E::G2Affine,
    gamma: E::ScalarField,         // structured randomness for Claymore identity
    max_degree: usize,
}
```

### MulcsVerifierParam<E>

G2 elements only (no G1 powers).

```rust
struct MulcsVerifierParam<E: Pairing> {
    g1_one: E::G1Affine,
    g1_x: E::G1Affine,
    g2_one: E::G2Affine,
    g2_x: E::G2Affine,
    g2_x2: E::G2Affine,
    gamma: E::ScalarField,
    max_degree: usize,
}
```

### gen_srs_for_testing

Generate random trapdoor x, compute G1 powers, G2 elements, random gamma.
WARNING: not a real trusted setup.

### trim

For `supported_num_vars` → max_degree = 2^(num_vars+1) (need 2N for h̄ degree).
Return `(prover_param, verifier_param)` with clipped g1_powers.

## 5. multi_open / batch_verify

### multi_open (prover)

For each (poly, point) pair:
1. Compute univariate polynomial f_v from poly's evaluations
2. Compute evaluation y = poly.evaluate(&point)
3. Compute h(X) = z^{N-1} · (f_v(z) · f_T(r)(z⁻¹) - y) via Claymore identity
4. Compute h̄(X) (with random δ at vanishing position)
5. Collect all (f_v, h̄) pairs
6. Batch all quotient polynomials into a single KZG multi-point proof
7. Output MulcsBatchProof

### batch_verify (verifier)

For each (commitment, point, claimed_value):
1. Re-derive group elements from opening data
2. Single pairing check on batched KZG proof
3. Per-polynomial Claymore identity check

## 6. Fairness with mKZG Baseline

| Aspect | mKZG | Mulcs |
|--------|------|-------|
| Basis | Multilinear | Univariate |
| Commitment | MSM over 2^NV G1 | MSM over 2^NV G1 (same) |
| Open | NV MSMs (log size) | KZG quotient + 1 MSM (batch) |
| Verify | NV pairings | 1 pairing (batch) |
| Batch | Sumcheck-based | Claymore aggregation |
| SRS | 2^NV · (NV+1) G1 elem | 2·2^NV G1 elem |

First implementation uses per-opening proofs aggregated naively.
Target: single pairing check overall.

## 7. Known Limitations

- `gamma` must be such that γ^{N-1} ≠ γ^i for i ≠ N-1. Random field element
  works w.h.p.
- `delta` randomness per h̄ opening — verifier sees cm_h̄ but not δ.
- SRS size: 2·N G1 elements for prover (vs mKZG's N·(NV+1)).
- Current `multi_open` naive aggregation (not truly batched sumcheck).
- Proof size: Mulcs proofs carry G1 commitments + field elements per
  polynomial, generally larger than mKZG.
