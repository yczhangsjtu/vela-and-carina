# Vela and Carina

This repository accompanies the paper *Vela and Carina: Fast Pairing-Based
Multilinear Polynomial Commitments from Reciprocal Polynomials*. It contains
Rust implementations of two pairing-based multilinear polynomial commitment
schemes (MLPCSs), together with a HyperPlonk integration and comparable
research implementations of several published PCS baselines.

- **Vela** targets compact proofs and fast verification. Its opening proof has
  `2 G1 + 4 F` cryptographic elements.
- **Carina** targets prover efficiency while retaining a compact proof and a
  low verifier cost. Its opening proof has `4 G1 + 8 F` cryptographic
  elements.

The repository also includes mKZG, Gemini, Zeromorph, Samaritan, Mercury,
CHOPIN, and Mulcs implementations for reproducible comparisons.

## Security notice

This is research software and has not received an external security audit. It
does not provide zero knowledge or hiding. The `gen_srs_for_testing` APIs
sample trapdoors locally and are suitable only for tests and benchmarks; a
deployment must use an appropriate trusted or updatable setup ceremony.

## Build and test

Install a current stable Rust toolchain, then run from the repository root:

```bash
cargo build --workspace
cargo test --workspace
cargo fmt -- --check
```

Focused PCS and HyperPlonk integration tests are available through:

```bash
cargo test -p subroutines pcs::vela
cargo test -p subroutines pcs::carina
cargo test -p hyperplonk --test vela_backend
cargo test -p hyperplonk --test carina_backend
```

## Reproducible benchmarks

The single-opening benchmark measures setup, trimming, commitment, opening,
and verification separately. Opening is measured with a precomputed statement
commitment so it does not accidentally include a second size-`N` commitment.
Each backend and dimension is executed in its own process by the matrix runner.

```bash
cargo build --release -p subroutines --bin pcs_single_open_bench

PCS_BENCH_BACKEND=vela PCS_BENCH_NV=12 \
  target/release/pcs_single_open_bench

PCS_BENCH_BACKENDS=vela,carina,mkzg \
PCS_BENCH_NV_RANGE=8,12,16,20 \
  scripts/run_pcs_single_open_matrix.sh

PCS_VERIFY_BACKEND=vela PCS_VERIFY_NV_RANGE=8,12,16,20 \
  cargo bench -p subroutines --bench pcs-single-verify-benches
```

The matrix runner is serial and writes machine-readable CSV output. Generated
results are intentionally ignored by Git. See [docs/benchmarking.md](docs/benchmarking.md)
for timing scope, repetition policy, and the CSV schema.

## Layout

- `subroutines/`: PCS implementations, SRS types, and standalone benchmarks.
- `hyperplonk/`: HyperPlonk implementation and backend integration tests.
- `scripts/`: reproducible benchmark runner and test helpers.
- `docs/benchmarking.md`: benchmark methodology.

## Citation

Please cite the accompanying paper when using Vela or Carina:

```text
Yuncong Zhang. Vela and Carina: Fast Pairing-Based Multilinear Polynomial
Commitments from Reciprocal Polynomials. 2026.
```

## License

Licensed under the [MIT License](LICENSE).
