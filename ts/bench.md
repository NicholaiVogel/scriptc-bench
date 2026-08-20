# TypeScript runtime comparison

This benchmark compares the same scalar, object-heavy microGPT workload across Python, Rust, Bun, a Bun-compiled binary, and two scriptc-compiled binaries.

The TypeScript port is algorithmically equivalent to `microgpt.py`, but it does not reproduce Python's Mersenne Twister. Both TypeScript variants use the same seeded xorshift generator, so the runtime comparison is still between the same TypeScript program and compilation target.

## Workload

| | |
| --- | --- |
| Dataset | `input.txt`, 32,033 names |
| Model parameters | 4,192 scalar `Value` objects |
| Training steps | 1,000, matching the original `microgpt.py` default |
| Inference | 20 generated samples |
| Runs | 1 full run per variant |
| Host | Linux x86-64, Bun 1.4.0, Node 26.5.0, Python 3.12.12, Rust 1.91.1, GCC/G++ 16.1.1, Zig 0.16.0 |
| scriptc | 0.0.33 |

The original Python default is 1,000 steps. The benchmark adds an environment override, `MICROGPT_STEPS`, while retaining the 1,000-step default for ordinary use.

## Results

| Variant | Wall time | Peak RSS | Binary size |
| --- | ---: | ---: | ---: |
| Bun 1.4.0 `--compile`, fused flat tape | 0.31 s | 305 MB | 95 MB |
| Bun 1.3.11 source, fused flat tape | 0.33 s | 154 MB | runtime |
| Bun 1.3.11 `--compile`, fused flat tape | 0.36 s | 149 MB | 95 MB |
| Bun 1.4.0 source, fused flat tape | 0.36 s | 106 MB | runtime |
| C++ (`g++ -std=gnu++17 -O2`) | 0.55 s | 49.8 MB | 26 KB |
| C (`cc -std=gnu11 -O2`) | 0.58 s | 48 MB | 25 KB |
| Zig (`zig build-exe -O ReleaseFast`) | 0.59 s | 47.8 MB | 118 KB |
| scriptc, fused flat tape + native math FFI | 4.19 s | 55 MB | 615 KB |
| scriptc, fused flat tape (`--dynamic`) | 4.58 s | 55 MB | 1.6 MB |
| Rust (`rustc -O`) | 4.72 s | 4.9 MB | 3.9 MB |
| Bun 1.4.0 source | 7.34 s | 305 MB | runtime |
| Bun 1.4.0 `--compile` | 7.43 s | 280 MB | 95 MB |
| Bun 1.3.11 source | 9.15 s | 454 MB | runtime |
| Bun 1.3.11 `--compile` | 9.36 s | 413 MB | 95 MB |
| scriptc static FFI | 105.82 s | 73 MB | 625 KB |
| scriptc `--dynamic` | 110.76 s | 74 MB | 1.7 MB |
| Python | 164.64 s | 61 MB | interpreter |

These are full-step results, not a reduced smoke run. The FFI run was performed without rerunning Python, as requested. Bun's compiled binary differs little from Bun source because this workload is dominated by hot-loop execution rather than startup.

Bun was upgraded from 1.3.11 to 1.4.0 before the new Bun measurements. Both
versions remain in the table for direct comparison; RSS differences should be
treated cautiously because these are single-run measurements.

### Build commands

The Bun binary was compiled with:

```sh
bun build --compile microgpt.ts --outfile microgpt-bun
```

The scriptc binary was compiled with:

```sh
scriptc build microgpt.ts --dynamic --no-keep-c -o microgpt-scriptc
```

The static FFI variant replaces the hot `Math.*` calls with direct C ABI calls to `libm`:

```sh
clang -O3 -c native_math.c -o native_math.o
ar rcs libnative_math.a native_math.o
scriptc build microgpt-ffi.ts --ffi ffi.json --no-keep-c -o microgpt-scriptc-ffi
```

## Important scriptc qualification

`scriptc coverage microgpt.ts` reported 95% static compilation. The remaining sites were `Math.sqrt`, `Math.log`, `Math.exp`, `Math.sin`, `Math.cos`, and `Math.PI`, so the benchmark used `--dynamic`, embedding scriptc's QuickJS dynamic engine. This is not yet a fully static scriptc result.

`scriptc coverage microgpt-ffi.ts --ffi ffi.json` reported 100% static compilation. Moving those math operations into a small C shim improved the full run from 110.76 seconds to 105.82 seconds, only 4.5% faster.

That makes the result useful but bounded:

- scriptc already demonstrates a much smaller native artifact and lower RSS;
- the current compiler's math lowering is a performance and staticness limitation;
- removing the dynamic math boundary helped only modestly in this workload;
- the remaining slowdown is therefore mostly in scriptc's object-heavy `Value` graph, reference-counting, and generated runtime operations;
- no conclusion about full Signet should be drawn from this microbenchmark alone.

## Flat tape results

The flat-tape rows replace the object-heavy `Value` graph with integer IDs into parallel numeric arrays. At 11.05 seconds, scriptc is approximately **10× faster** than the object-graph version (110.76 seconds), while peak RSS falls from 74 MB to 56 MB. This confirms that the object-heavy autograd/runtime model — not ordinary `Math.*` dispatch — was the dominant bottleneck.

The latest rows use fused dot-product tape nodes for linear and attention operations, so one node represents a whole dot product and its backward pass. Python was not rerun for this experiment. The fused tape brings scriptc from 110.76 seconds to 4.58 seconds (~24× faster) and 55 MB RSS. Native math FFI improves runtime another 8% without changing RSS materially in this fused layout.

Passing `-ffast-math` through `CFLAGS` to the scriptc build produced 4.67 seconds versus the 4.58-second baseline — no improvement, and no reason to accept its relaxed floating-point semantics here.

## Implications for Signet

scriptc can deliver real gains for Signet, especially:

- Much lower memory usage
- Small standalone binaries
- Static native linking
- No production Bun runtime
- Major speedups when code avoids object-heavy patterns

But it is not a transparent "compile Signet and get faster" switch. To make it infrastructure-grade, Signet would need:

- Flat, predictable data structures
- Fused/tensor-like operations
- Coarse-grained native FFI where useful
- Minimal dynamic/runtime islands
- Explicit benchmarks for daemon workloads
- Careful treatment of numerical and database boundaries

## Precision-sensitive scripts

The three high-precision verification scripts are documented in `port-assessment.md` rather than silently downgraded to float64 ports. Their Python behavior depends on mpmath arbitrary precision, interval arithmetic, numerical quadrature, and SymPy exact symbolic algebra; a TypeScript rewrite using ordinary `number` would be a different experiment.

## Reproduce

```sh
cd /mnt/work/life/scratch/scriptc/ts
SCRIPTC=/path/to/scriptc ./bench.sh
```
