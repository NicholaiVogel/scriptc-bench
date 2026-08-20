# Signet × scriptc experiment

Small, reproducible runtime experiments for evaluating whether Signet can use
Bun for development and ship as a single `scriptc`-compiled native binary.

This is **not** a Signet production implementation. It isolates the kinds of
runtime behavior that matter: object-heavy autograd, flat numeric tapes, fused
operations, native FFI, memory use, and compiler flags.

## Layout

```text
├── input.txt             # microGPT names dataset
├── microgpt.py           # Python reference
├── microgpt.rs           # Rust comparison
└── ts/
    ├── microgpt.ts       # object-heavy TypeScript port
    ├── microgpt-flat.ts   # flat tape + fused dot-product operations
    ├── microgpt-ffi.ts    # object-heavy port with native math FFI
    ├── microgpt-flat-ffi.ts
    ├── native_math.c      # small libm FFI shim
    ├── ffi.json            # scriptc FFI manifest
    ├── bench.sh            # builds and runs every comparison
    ├── bench.md            # measured results and interpretation
    └── package.json        # Bun module-mode declaration
```

## Reproduce

Install or build `scriptc`, then run the full 1,000-step workload:

```sh
cd ts
SCRIPTC=/path/to/scriptc ./bench.sh
```

`input.txt` is intentionally gitignored. If it is absent, `bench.sh` fetches
the pinned names dataset automatically from the official `makemore` commit.
To download it manually instead:

```sh
curl -fL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt \
  -o input.txt
```

To avoid rerunning the Python baseline:

```sh
SKIP_PYTHON=1 SCRIPTC=/path/to/scriptc ./bench.sh
```

Override steps or repetitions only for exploratory runs:

```sh
SCRIPTC=/path/to/scriptc ./bench.sh 1000 3
```

The benchmark builds temporary artifacts under `/tmp/signet-scriptc-bench`;
it does not place comparison binaries in this directory intentionally.

## Current conclusion

The object-heavy scriptc port is slow, but representation-aware optimization
changes the result substantially. The current fused flat-tape scriptc build
is about 24× faster than the original object-heavy scriptc build on this
workload, while remaining a small native artifact. Native math FFI helps only
modestly after the representation is fixed. See [`ts/bench.md`](ts/bench.md)
for the exact measurements and limitations.

The result supports a qualified Signet hypothesis: scriptc is promising for a
single production binary, but adopting it requires optimizing hot paths around
flat data, fused operations, and explicit native boundaries. It is not a
transparent compile-only speedup.
