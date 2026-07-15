# Benchmarking

This repository benchmarks the PCS layer independently of HyperPlonk. The
authoritative single-opening entry point is
`subroutines/src/bin/pcs_single_open_bench.rs`; the matrix runner executes it
serially, once per backend and number of variables.

## Timing scope

Each run reports `srs`, `trim`, `commit`, `core_open`, and verification rows.
`core_open` receives a precomputed commitment and includes evaluation of the
input polynomial at the query point. It therefore measures the opening
algorithm without an accidental statement recommitment. Expensive operations
other than verification are run once. Verification reports both the mean and
median over 100 repetitions after proof deserialization.

The runner disables PCS profiling and records one CSV row per phase. Its output
schema is:

```text
backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,threads,proof_bytes,status,api_used,peak_rss_bytes
```

## Running the matrix

```bash
cargo build --release -p subroutines --bin pcs_single_open_bench
PCS_BENCH_BACKENDS=vela,carina,mkzg \
PCS_BENCH_NV_RANGE=8,12,16,20 \
  scripts/run_pcs_single_open_matrix.sh
```

`PCS_BENCH_SEED` fixes the benchmark inputs, and `PCS_BENCH_THREADS` fixes the
Rayon thread count. The runner writes generated CSV and metadata files locally;
they are not version-controlled.

Use the Criterion verifier benchmark for more stable verifier-only timing:

```bash
PCS_VERIFY_BACKEND=vela PCS_VERIFY_NV_RANGE=8,12,16,20 \
  cargo bench -p subroutines --bench pcs-single-verify-benches
```
