#!/bin/bash
set -euo pipefail

# PCS single-open benchmark serial runner.
# Each combination (backend × nv) runs in its own process.
#
# Env overrides:
#   PCS_BENCH_BACKENDS    comma-separated list, default: mkzg,mulcs,nrg,mercury,chopin,recipcs,gemini,samaritan,zeromorph
#   PCS_BENCH_NV_RANGE    comma-separated list, default: 8,10,12,14,16,18,20
#   PCS_BENCH_THREADS     rayon thread count, default num_cpus
#   PCS_BENCH_SEED        u64 seed, default fixed
#   PCS_BENCH_OUTPUT       output CSV path, default: <repo>/pcs_single_open_results.csv
#   PCS_BENCH_INCLUDE_LEGACY  set to 1 only for audit-only legacy trait-open rows
#   PCS_BENCH_RESUME       set to 1 to append only missing backend/NV pairs
#
# CSV schema (13 columns):
#   backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,
#   threads,proof_bytes,status,api_used,peak_rss_bytes

# ── determine repo root (relative to script location) ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BIN="pcs_single_open_bench"
RELEASE_DIR="$REPO_ROOT/target/release"

# ── config (overridable via env) ──
DEFAULT_BACKENDS="mkzg,mulcs,nrg,mercury,chopin,recipcs,gemini,samaritan,zeromorph"
DEFAULT_NVS="8,10,12,14,16,18,20"

BACKENDS_STR="${PCS_BENCH_BACKENDS:-$DEFAULT_BACKENDS}"
NVS_STR="${PCS_BENCH_NV_RANGE:-$DEFAULT_NVS}"

IFS=',' read -ra backends <<< "$BACKENDS_STR"
IFS=',' read -ra nvs <<< "$NVS_STR"

THREADS="${PCS_BENCH_THREADS:-}"
SEED="${PCS_BENCH_SEED:-}"
INCLUDE_LEGACY="${PCS_BENCH_INCLUDE_LEGACY:-0}"
RESUME="${PCS_BENCH_RESUME:-0}"

# ── reject PCS_PROFILE in benchmark mode ──
PCP="${PCS_PROFILE:-}"
if [ -n "$PCP" ] && [ "$PCP" != "0" ]; then
    echo "ERROR: PCS_PROFILE=$PCP is set. Benchmark runner refuses to run with profiling enabled."
    echo "       Profiling CSV intermixes with benchmark output. Use PCS_PROFILE=0 or unset it."
    exit 1
fi

# ── build release binary once ──
echo "=== Building release binary ==="
cargo build --release -p subroutines --bin "$BIN"
BIN_PATH="$RELEASE_DIR/$BIN"
echo "Binary: $BIN_PATH"
echo ""

# ── output CSV ──
CSV_FILE="${PCS_BENCH_OUTPUT:-${REPO_ROOT}/pcs_single_open_results.csv}"
METADATA_FILE="${CSV_FILE%.csv}.metadata.txt"
RESULT_DIR="$(dirname "$CSV_FILE")"
RESULT_STEM="$(basename "${CSV_FILE%.csv}")"
LOG_DIR="${RESULT_DIR}/${RESULT_STEM}.logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
MEMORY_BYTES="$(sysctl -n hw.memsize 2>/dev/null || true)"
if [ -z "$CPU_MODEL" ] && command -v system_profiler >/dev/null 2>&1; then
    CPU_MODEL="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/^[[:space:]]*Chip:/ {print $2; exit}')"
fi
if [ -z "$MEMORY_BYTES" ] && command -v system_profiler >/dev/null 2>&1; then
    MEMORY_BYTES="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/^[[:space:]]*Memory:/ {print $2; exit}')"
fi
CPU_MODEL="${CPU_MODEL:-unavailable}"
MEMORY_BYTES="${MEMORY_BYTES:-unavailable}"

HEADER="backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,threads,proof_bytes,status,api_used,peak_rss_bytes"
if [ "$RESUME" = "1" ] && [ -f "$CSV_FILE" ]; then
    if [ "$(head -n 1 "$CSV_FILE")" != "$HEADER" ]; then
        echo "ERROR: cannot resume: CSV header does not match current schema" >&2
        exit 1
    fi
    echo "=== Resuming existing CSV: $CSV_FILE ==="
else
    echo "$HEADER" > "$CSV_FILE"
fi
{
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git status --short | tr '\n' ';')"
    echo "rustc=$(rustc -V 2>/dev/null || echo unknown)"
    echo "cargo=$(cargo -V 2>/dev/null || echo unknown)"
    echo "os=$(uname -a 2>/dev/null || echo unknown)"
    echo "cpu=$CPU_MODEL"
    echo "memory=$MEMORY_BYTES"
    echo "backends=$BACKENDS_STR"
    echo "nv_range=$NVS_STR"
    echo "threads=${THREADS:-auto}"
    echo "seed=${SEED:-default}"
    echo "include_legacy_trait_open=$INCLUDE_LEGACY"
    echo "verify_repetitions=100"
    echo "profile=disabled"
} > "$METADATA_FILE"

echo "=== Starting serial benchmark matrix ==="
echo "Backends: ${backends[*]}"
echo "NVs:      ${nvs[*]}"
echo "Output:   $CSV_FILE"
echo "Metadata: $METADATA_FILE"
echo "Seed:     ${SEED:-default}"
echo "Threads:  ${THREADS:-auto}"
echo "Resume:   $RESUME"
echo ""

for nv in "${nvs[@]}"; do
    for backend in "${backends[@]}"; do
        case "$backend" in
            mkzg) label="mKZG" ;;
            mulcs) label="MulcsClaymore" ;;
            nrg) label="NestedGridKZG" ;;
            mercury) label="Mercury" ;;
            chopin) label="Chopin" ;;
            recipcs) label="ReciPCS" ;;
            gemini) label="Gemini" ;;
            samaritan) label="Samaritan" ;;
            zeromorph) label="Zeromorph" ;;
            *) label="$backend" ;;
        esac
        if [ "$RESUME" = "1" ] && awk -F, -v b="$label" -v v="$nv" \
            '$1 == b && $2 == v && $4 == "core_open" && $11 == "pass" { found = 1 } END { exit !found }' \
            "$CSV_FILE"; then
            echo "  SKIP completed backend=$backend nv=$nv"
            continue
        fi
        echo ""
        echo "======================================================"
        echo "  backend=$backend  nv=$nv"
        echo "======================================================"

        ENV=(
            PCS_BENCH_BACKEND="$backend"
            PCS_BENCH_NV="$nv"
            PCS_PROFILE=0
            PCS_BENCH_INCLUDE_LEGACY="$INCLUDE_LEGACY"
        )
        if [ -n "$THREADS" ]; then
            ENV+=(PCS_BENCH_THREADS="$THREADS")
        fi
        if [ -n "$SEED" ]; then
            ENV+=(PCS_BENCH_SEED="$SEED")
        fi

        LOG_FILE="${LOG_DIR}/${backend}_nv${nv}.log"
        STDERR_FILE="${LOG_FILE}.stderr"
        STATUS_FILE="${LOG_FILE}.status"
        rm -f "$LOG_FILE" "$STDERR_FILE" "$STATUS_FILE"

        set +e
        # macOS `time -l` can itself fail after a successful child process when
        # a sandbox denies sysctl(kern.clockrate). Record the child exit code in
        # a sidecar file, while making the wrapper exit successfully.
        /usr/bin/time -l env "${ENV[@]}" sh -c '
            "$1" > "$2"
            rc=$?
            printf "%s\\n" "$rc" > "$3"
            exit 0
        ' sh "$BIN_PATH" "$LOG_FILE" "$STATUS_FILE" 2> "$STDERR_FILE"
        TIME_RC=$?
        set -e

        RC=""
        if [ -f "$STATUS_FILE" ]; then
            RC=$(cat "$STATUS_FILE")
        fi

        # Extract peak RSS from /usr/bin/time -l (macOS: "maximum resident set size")
        PEAK_RSS="unavailable"
        if grep -q "maximum resident set size" "$STDERR_FILE" 2>/dev/null; then
            PEAK_RSS=$(grep "maximum resident set size" "$STDERR_FILE" 2>/dev/null | head -1 | awk '{print $1}')
        elif grep -q "maxresident" "$STDERR_FILE" 2>/dev/null; then
            # Linux /usr/bin/time -v format or alternative
            PEAK_RSS=$(grep "maxresident" "$STDERR_FILE" 2>/dev/null | head -1 | awk '{print $NF}')
        fi

        if [ -z "$RC" ] || [ "$RC" -ne 0 ]; then
            STATUS="failed_time_${TIME_RC}"
            if [ -n "$RC" ]; then
                STATUS="failed_exit_${RC}"
            fi
            echo "  FAILED (${STATUS}) — see $STDERR_FILE"
            # Emit a synthetic failure row with correct column count
            echo "${backend},${nv},0,runner,setup,0,0,0,${THREADS:-auto},0,${STATUS},-,${PEAK_RSS}" >> "$CSV_FILE"
            continue
        fi

        # Skip binary's header line, post-process peak_rss_bytes column
        # The binary emits "unavailable" in the last column — replace with actual RSS
        tail -n +2 "$LOG_FILE" | while IFS= read -r line; do
            # Replace the last "unavailable" with actual peak RSS
            echo "${line}" | sed "s/unavailable\$/${PEAK_RSS}/"
        done >> "$CSV_FILE"

        echo "  OK — peak RSS ${PEAK_RSS} bytes"

        # Print summary line
        grep "core_open_prebound" "$LOG_FILE" | head -1 || true
    done
done

echo ""
echo "=== Done. Results in $CSV_FILE ==="

# ── validate the entire CSV, not merely its first data row ──
EXPECTED_COLS=$(awk -F, 'NR == 1 { print NF }' "$CSV_FILE")
HEADER_ROWS=$(grep -Fxc "$HEADER" "$CSV_FILE" || true)
if [ "$HEADER_ROWS" -ne 1 ]; then
    echo "ERROR: expected exactly one CSV header, found $HEADER_ROWS" >&2
    exit 1
fi
if ! awk -F, -v expected="$EXPECTED_COLS" '
    NR > 1 && (NF != expected || $1 == "backend") {
        printf "invalid CSV row %d: expected %d columns, got %d\\n", NR, expected, NF > "/dev/stderr";
        exit 1;
    }
' "$CSV_FILE"; then
    exit 1
fi
DATA_ROWS=$(( $(wc -l < "$CSV_FILE") - 1 ))
if [ "$DATA_ROWS" -le 0 ]; then
    echo "ERROR: no data rows in CSV" >&2
    exit 1
fi
echo "CSV consistency check: $EXPECTED_COLS columns across $DATA_ROWS data rows"
