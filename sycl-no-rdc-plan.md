# Plan: `-fno-sycl-rdc` (non-relocatable device code) support for SYCL

## Context

Upstream SYCL offloading in Clang only implements a single, implicitly-RDC
compilation model. For every translation unit (TU), the driver emits device
LLVM bitcode, packages it with `llvm-offload-binary`, and embeds the raw
(unlinked) device bitcode into the host object via `-fembed-offload-object`.
Device linking is **deferred to the final `clang-linker-wrapper`**, which
collects the device images from *all* host objects, groups them by
triple/arch, and runs `clang-sycl-linker` (`--sycl-link`) **once across all
TUs** — i.e. relocatable / separate-compilation (RDC) semantics.

DPC++ (downstream) supports `-fno-sycl-rdc`, where each TU's device code is
**self-contained and finalized per-TU**: `clang-sycl-linker` runs on each TU
individually and the resulting device image is embedded into that TU's host
object as a proper OffloadBinary. This avoids the expensive, monolithic
whole-program device link at final link time and matches the CUDA/HIP
`-fno-gpu-rdc` model. This change brings that capability upstream.

Goal:
- `-fsycl-rdc` (default, `true`): current cross-TU behavior, unchanged.
- `-fno-sycl-rdc`: each SYCL TU device-links independently via a per-TU
  `clang-linker-wrapper`/`clang-sycl-linker` invocation; the finalized image
  is packed into an OffloadBinary and embedded; the final host-link wrapper
  must **not** re-link SYCL device code across TUs.

Reference model: HIP `-fno-gpu-rdc` (`HIPNoRDC`), which uses a per-TU
`LinkerWrapperJobAction` in `BuildOffloadingActions`.

## Decisions (confirmed with user)

- **Action graph**: per-TU `LinkerWrapperJobAction` (HIP-style).
- **Final link**: `clang-sycl-linker` *is* invoked per-TU; each TU's linked
  device image is packed into a proper OffloadBinary/wrapped object, and the
  final `clang-linker-wrapper` must skip cross-TU relinking of already-finalized
  SYCL images.
- **LangOpt**: introduce `SYCLRelocatableDeviceCode`, mirroring
  `GPURelocatableDeviceCode`.

## Current flow references

- `Driver::BuildOffloadingActions` — `clang/lib/Driver/Driver.cpp:5006-5247`
  (called once per input TU from `Driver.cpp:4630`; final host
  `LinkerWrapperJobAction` created once at `Driver.cpp:4683`).
  - `HIPNoRDC` computed at `Driver.cpp:5016-5018`.
  - HIP no-rdc per-TU wrapper branch: `Driver.cpp:5202-5224`
    (`OffloadPackagerJobAction` → per-TU `LinkerWrapperJobAction(AL, TY_HIP_FATBIN)`).
  - Generic SYCL (RDC) branch today: `Driver.cpp:5225-5232`.
- Host embedding: `-fembed-offload-object` — `clang/lib/Driver/ToolChains/Clang.cpp:8282-8284`
  (`IsRDCMode` at `Clang.cpp:5174` is HIP/CUDA-only; leave SYCL in the `else`).
- Final wrapper cross-TU device link: `linkAndWrapDeviceFiles` —
  `clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp:966-1053`
  (calls `linkDevice`→`generic::clang` w/ `--sycl-link` at `596-608`,
  `1015-1017`).
- SYCL bundling/wrapping: `bundleSYCL` (`792-803`), `wrapSYCLBinaries`
  (`748-755`).
- `--emit-fatbin-only` precedent for per-TU-only wrapper output:
  `LinkerWrapperOpts.td:67`, `ClangLinkerWrapper.cpp:1378-1391`,
  `Clang.cpp:10015-10019`.

## Implementation

### 1. Options + LangOpt

- `clang/include/clang/Basic/LangOptions.def`: add after line 257
  ```
  LANGOPT(SYCLRelocatableDeviceCode, 1, 1, NotCompatible, "generate relocatable SYCL device code")
  ```
  (default `1` = RDC).
- `clang/include/clang/Options/Options.td` (SYCL group, ~7560-7575): add a
  `BoolFOption` mapped to the LangOpt, default true, exposed to `CC1Option`:
  ```
  defm sycl_rdc : BoolFOption<"sycl-rdc",
    LangOpts<"SYCLRelocatableDeviceCode">, DefaultTrue,
    PosFlag<SetTrue, [], [ClangOption, CC1Option],
            "Generate relocatable SYCL device code (separate compilation, default)">,
    NegFlag<SetFalse, [], [ClangOption, CC1Option],
            "Generate non-relocatable, self-contained SYCL device code per translation unit">>;
  ```
  Marshalling to `LangOpts.SYCLRelocatableDeviceCode` is automatic via
  `BoolFOption`/`LangOpts<>`.

### 2. Driver action graph (`Driver.cpp` `BuildOffloadingActions`)

- Near `Driver.cpp:5016`, compute:
  ```cpp
  bool SYCLNoRDC =
      C.isOffloadingHostKind(Action::OFK_SYCL) &&
      !Args.hasFlag(options::OPT_fsycl_rdc, options::OPT_fno_sycl_rdc, true);
  ```
- Add a SYCL no-rdc branch before the generic `else` at `Driver.cpp:5225`,
  mirroring the HIP branch (5202-5224) but with `TY_Image` (there is no SYCL
  fatbin type):
  ```cpp
  } else if (SYCLNoRDC) {
    Action *PackagerAction =
        C.MakeAction<OffloadPackagerJobAction>(OffloadActions, types::TY_Image);
    ActionList AL{PackagerAction};
    Action *WrapperAction =
        C.MakeAction<LinkerWrapperJobAction>(AL, types::TY_Image);
    DDep.add(*WrapperAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
             /*BA=*/{}, Action::OFK_SYCL);
  } else {
    // existing RDC packager path
  }
  ```
  Because `BuildOffloadingActions` runs once per TU, this yields one
  `llvm-offload-binary` + one device-side `clang-linker-wrapper` per TU. Its
  output is added as a host dependence (5243-5246) and embedded into the host
  object.

Expected `-ccc-print-phases` under `-fno-sycl-rdc` (vs current
`sycl-offload-jit.cpp:14-20`): a new device-side `clang-linker-wrapper` node
appears between `llvm-offload-binary` and the host offload node:
```
8: llvm-offload-binary, {7}, image, (device-sycl)
9: clang-linker-wrapper, {8}, image, (device-sycl)
10: offload, "host-sycl ..." {2}, "device-sycl ..." {9}, ir
11: backend, {10}, assembler, (host-sycl)
12: assembler, {11}, object, (host-sycl)
13: clang-linker-wrapper, {12}, image, (host-sycl)
```
RDC (default) phases stay exactly as today.

### 3. Per-TU device wrapper invocation (`Clang.cpp` linker-wrapper cmd)

The per-TU device-side `LinkerWrapperJobAction` (kind `device-sycl`,
`TY_Image`) must run `clang-sycl-linker` on just that TU and produce a wrapped
image, without a host link. Model on HIP's `--emit-fatbin-only`
(`Clang.cpp:10015-10019`):
- In `Clang.cpp` `ConstructJob` for the linker-wrapper, when the job is the
  per-TU SYCL device wrapper, pass a mode flag (e.g. reuse/extend the
  existing wrapper flags) so it: extracts the single packaged image, calls
  `clang-sycl-linker` (`--sycl-link`), wraps via `wrapSYCLBinaries`, and emits
  the wrapped object **without host linking** — analogous to
  `--emit-fatbin-only` skipping `runLinker` (`ClangLinkerWrapper.cpp:1388`).
- The wrapped per-TU image is then embedded via the existing
  `-fembed-offload-object` path (`Clang.cpp:8282-8284`); no change needed
  there, and do **not** extend `IsRDCMode` to SYCL.

### 4. Final `clang-linker-wrapper`: skip cross-TU relink for no-rdc SYCL

In no-rdc mode each host object embeds an **already device-linked + wrapped**
SYCL image. The final host-link wrapper (`Driver.cpp:4683`) still runs
`getDeviceInput` and `linkAndWrapDeviceFiles`, which would otherwise merge and
re-link all SYCL images across TUs (`ClangLinkerWrapper.cpp:1001-1017`).

- Mark no-rdc SYCL images as already-finalized so the final wrapper does
  **not** call `linkDevice`/`generic::clang` again. Options:
  - Add a per-image OffloadBinary string key (e.g. `sycl-rdc=0`) written by the
    per-TU wrapper, and in `linkAndWrapDeviceFiles` (`980-1017`) detect it and
    pass the image straight through to `Images[Kind]` (bundle via `bundleSYCL`,
    wrap via `wrapSYCLBinaries`) without relinking; **or**
  - Communicate the mode via a new `clang-linker-wrapper` option
    (`LinkerWrapperOpts.td`) the driver passes on the final wrapper job.
  Prefer the per-image marker so mixed inputs behave correctly and the mode
  travels with the image.
- Keep grouping so multiple no-rdc TUs' images are each preserved (not merged
  into one device link). Ensure `wrapSYCLBinaries`/`bundleSYCL` handle multiple
  finalized images.

### 5. Symbol internalization for no-rdc

No-rdc device modules must be self-contained. Mirror the HIP approach
(`HIPAMD.cpp:245` adds `-mllvm -amdgpu-internalize-symbols`) but for SPIR-V:
- In the SYCL device `-cc1` construction (`Clang.cpp` `IsSYCLDevice` block,
  ~5308-5331), when `!SYCLRelocatableDeviceCode`, drive internalization of
  non-exported device symbols. SPIR-V has no `amdgpu-internalize-symbols`;
  use the generic internalize path in the SYCL device/post-link pipeline. The
  `SYCLRelocatableDeviceCode` LangOpt lets the front-end/device middle-end make
  this decision. Exact pass wiring belongs to the SYCL device pipeline; the
  driver's job is to propagate the mode.

## Files to modify

- `clang/include/clang/Basic/LangOptions.def` — add `SYCLRelocatableDeviceCode`.
- `clang/include/clang/Options/Options.td` — add `sycl_rdc` `BoolFOption`.
- `clang/lib/Driver/Driver.cpp` — `SYCLNoRDC` + new no-rdc branch in
  `BuildOffloadingActions` (~5016, ~5225).
- `clang/lib/Driver/ToolChains/Clang.cpp` — per-TU device wrapper mode flag
  (~10015 area), internalization flag in SYCL device cc1 (~5308).
- `clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp` — mark/route no-rdc
  SYCL images to skip cross-TU relink (`966-1053`); handle multiple finalized
  images in bundle/wrap (`748-803`).
- `clang/tools/clang-linker-wrapper/LinkerWrapperOpts.td` — new mode option if
  the marker approach needs a driver-passed flag.

## Verification

- **Phases**: extend `clang/test/Driver/sycl-offload-jit.cpp` with a
  `-fno-sycl-rdc` phases block (new `device-sycl clang-linker-wrapper` node)
  and an explicit `-fsycl-rdc` block confirming today's phases are unchanged
  (default == RDC).
- **New test** `clang/test/Driver/sycl-offload-nordc.cpp` (`-###`):
  - per-TU device `clang-linker-wrapper` is invoked in device/emit-only mode;
  - `-fembed-offload-object` present in both modes;
  - internalize `-mllvm` flag present on device `-cc1` only under
    `-fno-sycl-rdc`;
  - final wrapper receives the no-rdc marker / does not re-run `--sycl-link`
    across TUs.
- **cc1 LangOpt**: a test asserting `-fno-sycl-rdc` sets
  `SYCLRelocatableDeviceCode=0` on the device `-cc1`.
- **Build & run**:
  ```
  ninja -C build clang clang-linker-wrapper
  ./build/bin/llvm-lit -v clang/test/Driver/sycl-offload-jit.cpp \
      clang/test/Driver/sycl-offload-nordc.cpp
  ```
- **Two-TU functional smoke** (if a SYCL runtime target is available): compile
  two `.cpp` TUs with `-fno-sycl-rdc`, confirm each host object embeds a
  finalized device image and the link succeeds without a cross-TU device link.
