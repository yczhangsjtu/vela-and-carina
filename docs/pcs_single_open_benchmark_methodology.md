# PCS Single-Open Benchmark Methodology

## Overview

This document describes the benchmarking methodology for single polynomial commitment
opening (one polynomial, one point, one evaluation) across all multilinear PCS
backends in the hyperplonk-baseline repository.

## Backends

| Display Name | Module | Core Open API |
|-------------|--------|---------------|
| mKZG | `multilinear_kzg` | `trait open` (no C_f recommit — audit-clean) |
| Mulcs (Claymore) | `mulcs` | `trait open` (no C_f recommit — audit-clean) |
| NestedGridKZG | `nested_grid_kzg` | `open_with_commitment` |
| Mercury | `mercury` | `open_with_commitment` |
| Chopin | `chopin` | `open_with_commitment` |
| ReciPCS | `recipcs` | `open_with_commitment` (added in this work) |
| Gemini | `gemini` | `open_with_commitment` (added in this work) |
| Samaritan | `samaritan` | `open_with_commitment` (added in this work) |
| Zeromorph | `zeromorph` | `open_with_commitment` (added in this work) |

Notes:
- `MulcsSymmetricPCS` is a re-export alias for `ReciPCS` from `mulcs/mod.rs:38`.
  It is NOT double-counted.
- Univariate KZG is excluded (not a multilinear PCS).

## Phase Definitions

### 1. SRS Generation (`setup`)
Generate universal structured reference string for the given variable count `nv`.
Includes computing powers-of-tau, grid G1 elements, and G2 elements as needed by
each scheme. Measured once per process.

### 2. Trim (`trim`)
Trim universal SRS to prover/verifier parameters specialized to `nv`.
Measured once per process.

### 3. Commit (`commit`)
Compute `Commit(f)` for a randomly generated multilinear polynomial `f` of
`nv` variables. This is always exactly one N-size MSM (no per-scheme prefix).
Measured once per process.

### 4. Core Open (`core_open_precommitted`)
Generate an opening proof that `f(r) = y` given the **pre-computed commitment**
`C_f`. This phase does NOT recompute `C_f`:

- For backends with `open_with_commitment` (NestedGridKZG, Mercury, Chopin,
  ReciPCS, Gemini, Samaritan, Zeromorph): calls `open_with_commitment(pp, f, r, C_f)`.
- For backends where trait `open` is audit-clean (mKZG, Mulcs): calls
  `trait open(pp, f, r)`, confirmed to NOT contain any N-size C_f recomputation.

Measured once per process. The `api_used` CSV column records which API was used:
`open_with_commitment` or `trait_open_no_recommit`.

### 5. Commit + Open (`commit_plus_open`)
Full end-to-end: compute `Commit(f)` then `Open(f, r)` using the trait `open`
API. This is always the `trait open` path (includes commit). Measured once per
process for reference only.

### 6. Verify (`verify_once`)
Verify the proof generated in Core Open. The proof is serialized once and
deserialized in each iteration to ensure a realistic verification workload.
100 iterations are timed; mean and median are reported. The verifier loop does
NOT regenerate SRS, commitment, proof, or any prover-side data.

## API Audit: C_f Recomputation

| Backend | trait `open` recomputes C_f? | Evidence |
|---------|------------------------------|----------|
| mKZG | No | `open_internal` computes quotient MSMs only (no C_f recomputation). No transcript binding of C_f. |
| Mulcs | No | `open_with_transcript` commits `cm_hbar` (protocol witness) only. No C_f in transcript. |
| NestedGridKZG | Yes | `open` calls `Self::commit` then `open_with_commitment`. `nrg_open_statement_recommit` profiler phase. |
| Mercury | Yes | `open` calls `MercuryPCS::commit` then core. `mercury_open_statement_recommit` profiler phase. |
| Chopin | Yes | `open` calls `ChopinPCS::commit` then core. `chopin_open_statement_recommit` profiler phase. |
| ReciPCS | Yes | `open` calls `ReciPCS::commit` before `recipcs_open`. Fixed: added `open_with_commitment`. |
| Gemini | Yes | `gemini_open_with_transcript` calls `pp.try_commit(&f_hat)`. Fixed: added `open_with_commitment`. |
| Samaritan | Yes | `samaritan_open_with_transcript` calls `pp.try_commit(&coeffs)`. Fixed: added `open_with_commitment`. |
| Zeromorph | Yes | `open_with_transcript` calls `pp.commit_commit(&coeffs)`. Fixed: added `open_with_commitment`. |

## Data Accuracy Notes

- Core Open data from the old `pcs_bench.rs` was **contaminated by N-MSM C_f
  recommits** for NestedGridKZG, Mercury, Chopin, ReciPCS, Gemini, Samaritan,
  and Zeromorph. Those numbers should NOT be used for publication.
- This new benchmark harness (`pcs_single_open_bench.rs`) produces clean Core
  Open data guaranteed free of C_f recommits.
- All data comes from single-backend, single-nv, single-process runs. Heavy
  phases run exactly once. Verifier is measured separately with 100 iterations.

## Platform

- Rust edition: 2021
- Curve: BLS12-381
- Profile: `--release`
- Threading: Rayon, configurable via `PCS_BENCH_THREADS`
- macOS, Apple Silicon (M-series)
