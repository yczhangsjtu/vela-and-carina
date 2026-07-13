# PCS Single-Open Benchmark Methodology

## Overview

This document describes the benchmarking methodology for single polynomial commitment
opening (one polynomial, one point, one evaluation) across all multilinear PCS
backends in the hyperplonk-baseline repository.

## CSV Schema

Fixed schema (13 columns):

```
backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,threads,proof_bytes,status,api_used,peak_rss_bytes
```

| Column | Description |
|--------|-------------|
| `backend` | Display name (e.g. `mKZG`, `Gemini`) |
| `nv` | Number of variables |
| `N` | Domain size = 2^nv |
| `phase` | See phase list below |
| `scope` | Semantic grouping (`setup`, `trim`, `commit`, `core_open_prebound`, `verify_once`, `legacy_trait_open`) |
| `elapsed_ms` | Wall-clock milliseconds |
| `heavy_invocations` | 1 for heavy phases, 0 for verify |
| `verify_iterations` | 0 for heavy phases, 100 for verify |
| `threads` | Rayon thread count |
| `proof_bytes` | Serialized proof size in bytes |
| `status` | `pass` or `failed_<reason>` |
| `api_used` | `open_with_commitment`, `trait_open_no_recommit`, `trait_open_may_recommit`, or `-` |
| `peak_rss_bytes` | Peak resident set size in bytes, or `unavailable` |

## Backends

| Display Name | Module | Core Open API | `api_used` |
|-------------|--------|---------------|-----------|
| mKZG | `multilinear_kzg` | `trait open` (no C_f recommit) | `trait_open_no_recommit` |
| Mulcs (Claymore) | `mulcs` | `trait open` (no C_f recommit) | `trait_open_no_recommit` |
| NestedGridKZG | `nested_grid_kzg` | `open_with_commitment` | `open_with_commitment` |
| Mercury | `mercury` | `open_with_commitment` | `open_with_commitment` |
| Chopin | `chopin` | `open_with_commitment` | `open_with_commitment` |
| ReciPCS | `recipcs` | `open_with_commitment` | `open_with_commitment` |
| Gemini | `gemini` | `open_with_commitment` | `open_with_commitment` |
| Samaritan | `samaritan` | `open_with_commitment` | `open_with_commitment` |
| Zeromorph | `zeromorph` | `open_with_commitment` | `open_with_commitment` |

Notes:
- `MulcsSymmetricPCS` is a re-export alias for `ReciPCS` from `mulcs/mod.rs:38`.
  It is NOT double-counted.
- Univariate KZG is excluded (not a multilinear PCS).
- mKZG and Mulcs do NOT have `open_with_commitment`. Their `trait open` has been
  audited and confirmed to NOT recompute C_f internally. The `api_used` column
  marks this as `trait_open_no_recommit`.

## Phase Definitions

### 1. SRS Generation (`srs` / `setup`)
Generate universal structured reference string. Uses system entropy for SRS
randomness. Measured once. Not included in paper comparison tables.

### 2. Trim (`trim` / `trim`)
Trim universal SRS to prover/verifier parameters. Measured once. Not included
in paper comparison tables.

### 3. Commit (`commit` / `commit`)
Compute `Commit(f)` for a deterministically-generated multilinear polynomial `f`.
This is exactly one N-size MSM. Measured once. Reported as `commit` for reference;
paper tables should use `core_open_prebound` as the prover cost (see below).

### 4. Core Open (`core_open` / `core_open_prebound`)
**This is the paper's primary prover metric.** Generate an opening proof that
`f(r) = y` given the **pre-computed commitment** `C_f`. This phase MUST NOT
recompute `C_f`:

- For backends with `open_with_commitment`: calls `open_with_commitment(pp, f, r, C_f)`
- For backends where `trait open` is audit-clean (mKZG, Mulcs): calls `trait open(pp, f, r)`

**Timing boundary (all backends)**: Includes all computations from
`(poly, point, commitment)` → `(proof, value)`, including `f(point)` evaluation.
For ReciPCS, the `poly.evaluate(point)` call is inside the timed block (unlike
the previous version).

Measured exactly once per backend/nv. `heavy_invocations` = 1. The `api_used`
column records which API path was used.

### 5. Verify (`verify_core_mean` / `verify_core_median` / `verify_once`)
Verify the proof generated in Core Open. The proof is serialized once and
deserialized **outside** each timed iteration. Only `PCS::verify()` is inside
the timing loop. 100 iterations. Mean and median reported.

The proof serialization/deserialization cost is **NOT included** in the verify
metrics. Documented explicitly.

### 6. Legacy Trait Open (`legacy_trait_open` / `legacy_trait_open`)
Full end-to-end using `trait open`: calls `Commit(f)` then `Open(f, r)`.
**WARNING**: This path may include an extra N-MSM C_f recommitment for backends
where `trait open` recomputes C_f (NestedGridKZG, Mercury, Chopin, ReciPCS,
Gemini, Samaritan, Zeromorph). This data is for **audit reference only** and
MUST NOT be used in paper comparison tables. Marked `trait_open_may_recommit`.
It is disabled by default. Set `PCS_BENCH_INCLUDE_LEGACY=1` only for a separate
audit run; final paper-data runs must leave it disabled.

## Deterministic Input Generation

- `PCS_BENCH_SEED` controls the deterministic (poly, point) generation.
- Each backend at the same nv uses the SAME polynomial evaluations and evaluation
  point, regardless of how many random bytes the SRS consumed.
- SRS generation uses a backend-local `test_rng`, separate from the deterministic
  input RNG and not controlled by `PCS_BENCH_SEED`.
- Inputs are derived solely from `(master_seed, nv)` with separate fixed domain
  separators for the polynomial and point. The backend name is deliberately not
  part of the derivation, so every backend at one `nv` receives identical inputs.

## API Audit: C_f Recomputation

| Backend | trait `open` recomputes C_f? | Evidence |
|---------|------------------------------|----------|
| mKZG | No | `open_internal` computes quotient MSMs only. No C_f in transcript. |
| Mulcs | No | `open_with_transcript` commits `cm_hbar` (protocol witness) only. |
| NestedGridKZG | Yes | `open` calls `Self::commit` then `open_with_commitment`. |
| Mercury | Yes | `open` calls `MercuryPCS::commit` then core. |
| Chopin | Yes | `open` calls `ChopinPCS::commit` then core. |
| ReciPCS | Yes | Fixed: `open_with_commitment` added; `trait open` delegates to it. |
| Gemini | Yes | Fixed: `open_with_commitment` added; uses `gemini_core_open_prebound`. |
| Samaritan | Yes | Fixed: `open_with_commitment` added; uses `samaritan_core_open_prebound`. |
| Zeromorph | Yes | Fixed: `open_with_commitment` added; uses `zeromorph_core_open_prebound`. |

## Data Accuracy Notes

- Core Open data from the old `pcs_bench.rs` was **contaminated by N-MSM C_f
  recommits** for 7 backends. Those numbers must NOT be used for publication.
- This harness (`pcs_single_open_bench.rs`) produces clean Core Open data.
- `legacy_trait_open` rows are NOT valid for paper comparison (they include
  double N-MSM for backends that recompute C_f).
- All data from single-backend, single-nv, single-process runs. Heavy phases
  run exactly once. Verifier measured separately with 100 iterations.
- `PCS_PROFILE` is rejected by the binary; profiling CSV is never intermixed.

## Platform

- Rust edition: 2021
- Curve: BLS12-381
- Profile: `--release`
- Threading: Rayon, configurable via `PCS_BENCH_THREADS`
- Peak RSS captured by runner via `/usr/bin/time -l`

## Reproducibility Artifacts

The runner writes the raw CSV and a sibling `.metadata.txt` file. The metadata
records the UTC timestamp, git revision and worktree status, Rust/Cargo versions,
OS and available machine descriptors, backend/NV matrix, thread setting, seed,
and verifier repetition count. Preserve both files with any paper table derived
from this benchmark.
