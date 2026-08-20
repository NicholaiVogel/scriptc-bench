#!/usr/bin/env bash
# Compare the dependency-free microGPT workload across Python, Bun, and scriptc.
# Usage: SCRIPTC=/path/to/scriptc ./bench.sh [steps] [runs]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEPS=${1:-1000}
RUNS=${2:-1}
SKIP_PYTHON=${SKIP_PYTHON:-0}
SCRIPTC=${SCRIPTC:-scriptc}
RUSTC=${RUSTC:-rustc}
CC=${CC:-cc}
ZIG=${ZIG:-zig}
TMP=${TMPDIR:-/tmp}/signet-scriptc-bench
INPUT_URL=https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
mkdir -p "$TMP"

if [[ ! -s "$ROOT/input.txt" ]]; then
	command -v curl >/dev/null 2>&1 || { echo "input.txt is missing; install curl or download the makemore names dataset manually:" >&2; echo "  $INPUT_URL" >&2; exit 1; }
	echo "input.txt missing; downloading the benchmark dataset..." >&2
	curl --fail --location --silent --show-error "$INPUT_URL" -o "$ROOT/input.txt"
fi

cd "$HERE"
bun build --compile microgpt.ts --outfile "$TMP/microgpt-bun" >/dev/null
bun build --compile microgpt-flat.ts --outfile "$TMP/microgpt-flat-bun" >/dev/null
"$SCRIPTC" build microgpt.ts --dynamic --no-keep-c -o "$TMP/microgpt-scriptc" >/dev/null
"$SCRIPTC" build microgpt-flat.ts --dynamic --no-keep-c -o "$TMP/microgpt-flat-scriptc" >/dev/null
clang -O3 -c "$HERE/native_math.c" -o "$TMP/native_math.o"
ar rcs "$TMP/libnative_math.a" "$TMP/native_math.o"
sed "s#\./libnative_math\.a#$TMP/libnative_math.a#" "$HERE/ffi.json" > "$TMP/ffi.json"
"$SCRIPTC" build microgpt-ffi.ts --ffi "$TMP/ffi.json" --no-keep-c -o "$TMP/microgpt-scriptc-ffi" >/dev/null
"$SCRIPTC" build microgpt-flat-ffi.ts --ffi "$TMP/ffi.json" --no-keep-c -o "$TMP/microgpt-flat-scriptc-ffi" >/dev/null
"$RUSTC" -O "$ROOT/microgpt.rs" -o "$TMP/microgpt-rust"
"$CC" -std=gnu11 -O2 "$ROOT/microgpt.c" -lm -o "$TMP/microgpt-c"
"$ZIG" build-exe -O ReleaseFast "$ROOT/microgpt.zig" -lc -lm -femit-bin="$TMP/microgpt-zig"

run() {
	local label=$1
	local command=$2
	for run_number in $(seq 1 "$RUNS"); do
		/usr/bin/time -f "$label run=$run_number wall=%e rss=%M" sh -c "$command" >/dev/null 2>>"$TMP/results.txt"
	done
}

: > "$TMP/results.txt"
if [[ "$SKIP_PYTHON" != "1" ]]; then
	run python "cd '$ROOT' && MICROGPT_STEPS=$STEPS python3 microgpt.py"
fi
run bun-source "cd '$HERE' && MICROGPT_STEPS=$STEPS bun microgpt.ts"
run bun-compiled "cd '$HERE' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-bun'"
run scriptc-compiled "cd '$HERE' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-scriptc'"
run scriptc-ffi "cd '$HERE' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-scriptc-ffi'"
run bun-flat-source "cd '$HERE' && MICROGPT_STEPS=$STEPS bun microgpt-flat.ts"
run bun-flat-compiled "cd '$HERE' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-flat-bun'"
run scriptc-flat "cd '$HERE' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-flat-scriptc'"
run scriptc-flat-ffi "cd '$HERE' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-flat-scriptc-ffi'"
run rust "cd '$ROOT' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-rust'"
run c "cd '$ROOT' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-c'"
run zig "cd '$ROOT' && MICROGPT_STEPS=$STEPS '$TMP/microgpt-zig'"

cat "$TMP/results.txt"
echo "binaries: $TMP"
