#!/bin/bash
set -euo pipefail

# PCS single-open benchmark serial runner.
# Each combination (backend × nv) runs in its own process.
# Heavy phases (setup/trim/commit/core_open/commit_open) are measured once.
# Verifier is measured with 100 repetitions.

BIN="pcs_single_open_bench"
RELEASE_DIR="$(dirname "$0")/../target/release"
RUST_LOG=info
export PCS_PROFILE="${PCS_PROFILE:-}"

# ── config ──
backends=(mkzg mulcs nrg mercury chopin recipcs gemini samaritan zeromorph)
nvs=(8 10 12 14 16 18 20)
THREADS="${PCS_BENCH_THREADS:-}"
SEED="${PCS_BENCH_SEED:-}"

# ── build release binary once ──
echo "=== Building release binary ==="
cargo build --release -p subroutines --bin "$BIN"
BIN_PATH="$RELEASE_DIR/$BIN"
echo "Binary: $BIN_PATH"
echo ""

# ── CSV header ──
CSV_FILE="pcs_single_open_results.csv"
echo "=== Starting serial benchmark matrix ==="
echo "Output: $CSV_FILE"

# Header written by the binary itself; we collect all output.
> "$CSV_FILE"

for nv in "${nvs[@]}"; do
    for backend in "${backends[@]}"; do
        echo ""
        echo "======================================================"
        echo "  backend=$backend  nv=$nv"
        echo "======================================================"

        ENV=(
            PCS_BENCH_BACKEND="$backend"
            PCS_BENCH_NV="$nv"
        )
        if [ -n "$THREADS" ]; then
            ENV+=(PCS_BENCH_THREADS="$THREADS")
        fi
        if [ -n "$SEED" ]; then
            ENV+=(PCS_BENCH_SEED="$SEED")
        fi
        if [ -n "${PCS_PROFILE:-}" ]; then
            ENV+=(PCS_PROFILE="$PCS_PROFILE")
        fi

        LOG_FILE="pcs_bench_${backend}_nv${nv}.log"

        set +e
        /usr/bin/time -l env "${ENV[@]}" "$BIN_PATH" > "$LOG_FILE" 2> "${LOG_FILE}.stderr"
        RC=$?
        set -e

        if [ $RC -ne 0 ]; then
            echo "  FAILED (exit=$RC) — see ${LOG_FILE}.stderr"
            # Still collect CSV lines if any
            if [ -f "$LOG_FILE" ]; then
                tail -n +2 "$LOG_FILE" | while IFS= read -r line; do
                    echo "${line},failed_$RC"
                done >> "$CSV_FILE"
            fi
            continue
        fi

        # Append CSV data (skip header)
        tail -n +2 "$LOG_FILE" >> "$CSV_FILE"

        # Quick peak RSS from /usr/bin/time -l (macOS)
        PEAK_RSS=$(grep "maximum resident set size" "${LOG_FILE}.stderr" 2>/dev/null | awk '{print $1}' || echo "0")
        echo "  OK — peak RSS ${PEAK_RSS} bytes"

        # Print summary lines
        grep "core_open_precommitted" "$LOG_FILE" | head -1 || true
    done
done

echo ""
echo "=== Done. Results in $CSV_FILE ==="
