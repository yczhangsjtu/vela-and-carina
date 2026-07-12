#!/usr/bin/env bash
# ReciPCS unified single-PCS benchmark runner.
#
# Runs each backend (mkzg, gemini, zeromorph, samaritan, recipcs) in its OWN
# process so allocator state, cache, and SRS peak memory do not leak across
# backends. Emits one CSV to the path given by $1 (default: single_pcs.csv).
#
# Usage:
#   scripts/run_recipcs_bench.sh [out.csv] [nv_list]
#   e.g. scripts/run_recipcs_bench.sh /tmp/out.csv 8,10,12,14,16,18,20
#
# Must be run from the hyperplonk-baseline workspace root.

set -euo pipefail

OUT="${1:-single_pcs.csv}"
NV="${2:-8,10,12,14,16,18,20}"
BACKENDS=(mkzg gemini zeromorph samaritan recipcs)

echo "Building recipcs benchmark (release)..." >&2
cargo build -p subroutines --release --bench recipcs-benches >&2

BIN="$(ls target/release/deps/recipcs_benches-* | grep -v '\.d$' | head -1)"
echo "Binary: $BIN" >&2

echo "backend,nv,srs_gen_ms,trim_ms,commit_ms,open_ms,verify_us_mean,verify_us_median,proof_bytes" > "$OUT"
for b in "${BACKENDS[@]}"; do
  echo ">>> $b (nv=$NV)" >&2
  RECIPCS_BACKEND="$b" RECIPCS_NV="$NV" "$BIN" 2>/dev/null \
    | grep -v '^#' | grep -v '^backend' >> "$OUT"
done

echo "Wrote $OUT" >&2
