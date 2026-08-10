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
- `-fno-sycl-rdc`: **exact mirror of the CUDA/HIP no-rdc structure.** Each SYCL
  TU device-links (finalizes) independently via a per-TU `clang-sycl-linker`
  invocation; the finalized device image is included into that TU's host
  compilation via `-foffload-include-binary` (the generalization of
  `-fcuda-include-gpubinary` — see §0), where host CodeGen embeds + registers it
  **at compile time** by calling the shared `offloading::wrapSYCLBinaries`. The
  final `clang-linker-wrapper` then finds no SYCL device code to link and does
  **no device work** — there is no second device-link/wrap step.

This is the CUDA/HIP RDC-vs-no-RDC split applied to SYCL: RDC registers at link
(the image only exists after cross-TU device link); no-RDC registers at compile
(each TU's image exists standalone). Registration stays in the shared
`OffloadWrapper` — CodeGen *calls* it rather than reimplementing it (cleaner
than CGCUDANV, which reimplements registration).

Reference model: **CUDA** `-fno-gpu-rdc` — per-TU device finalize + host-compile
inclusion via `-fcuda-include-gpubinary` (renamed to `-foffload-include-binary`,
§0), so the final link does no device work.
(CUDA branch in `BuildOffloadingActions` at `Driver.cpp:5186-5193`; host include
at `Clang.cpp:8273-8275`; compile-time registration in `CGCUDANV`.) SYCL mirrors
this but *calls* the shared `wrapSYCLBinaries` instead of reimplementing
registration.

## Decisions (confirmed with user)

- **Action graph**: per-TU `LinkerWrapperJobAction` (HIP-style).
- **Per-TU finalize + compile-time registration** (exact CUDA/HIP mirror):
  `clang-sycl-linker` is invoked per-TU to *finalize* each TU's device image to
  a file; host CodeGen includes it via `-foffload-include-binary` and calls the
  shared `offloading::wrapSYCLBinaries` to embed + register at compile time. The
  final `clang-linker-wrapper` does **no** SYCL device work (no second
  device-link/wrap). Registration code stays shared in `OffloadWrapper.cpp`;
  CodeGen calls it (no reimplementation, no `already-linked` marker, no
  multi-image wrapper changes).
- **Flag/LangOpt**: `-f[no-]sycl-rdc` is an **alias** of `-f[no-]gpu-rdc`
  (matching intel/llvm), reusing the existing `GPURelocatableDeviceCode`
  LangOpt. No new LangOpt/marshalling. SYCL's RDC-by-default is enforced by the
  driver (see §1/§2), since the underlying `-fgpu-rdc` LangOpt is `DefaultFalse`.
- **No new SYCL-specific cc1 option for host inclusion.** Rather than adding
  `-fsycl-include-target-binary` alongside `-fcuda-include-gpubinary`, the
  existing CUDA option is generalized to `-foffload-include-binary` and shared
  (§0). Raised in review of the intel/llvm counterpart (intel/llvm#22833).
- **`-flto` is rejected for a `clang-sycl-linker` finalize link** rather than
  silently routed to `llvm-lto`, which would emit bitcode where a finalized
  device image is expected (§3).

## Current flow references

- `Driver::BuildOffloadingActions` — `clang/lib/Driver/Driver.cpp:5006-5247`
  (called once per input TU from `Driver.cpp:4630`; final host
  `LinkerWrapperJobAction` created once at `Driver.cpp:4683`).
  - `HIPNoRDC` computed at `Driver.cpp:5016-5018`.
  - HIP no-rdc per-TU wrapper branch: `Driver.cpp:5202-5224`
    (`OffloadPackagerJobAction` → per-TU `LinkerWrapperJobAction(AL, TY_HIP_FATBIN)`).
  - Generic SYCL (RDC) branch today: `Driver.cpp:5225-5232`.
(Locations in this section are as they stand **before** this work; §0 renames
`-fcuda-include-gpubinary`/`CudaGpuBinaryFileName`.)

- Host embedding today (RDC): `-fembed-offload-object` —
  `clang/lib/Driver/ToolChains/Clang.cpp:8282-8284`. CUDA/HIP no-rdc instead use
  `-fcuda-include-gpubinary` (`Clang.cpp:8273-8275`) — the model §3 mirrors.
  Note CUDA and HIP already **share** that option, which is the precedent for
  generalizing rather than duplicating it (§0).
- CUDA/HIP no-rdc host-compile registration (the reference for §3):
  `-fcuda-include-gpubinary` → `CodeGenOpts.CudaGpuBinaryFileName`
  (`Options.td:8977-8979`, `CodeGenOptions.h:383`) → `CGCUDANV.cpp:832,858`
  (reads file, embeds+registers at compile); ctor hooked in
  `CodeGenModule::Release` at `CodeGenModule.cpp:1108-1111`.
- Shared registration reused by §3: `offloading::wrapSYCLBinaries` /
  `SYCLWrapper` (`OffloadWrapper.cpp:751`, `636-708`). Clang CodeGen already
  links `FrontendOffloading` (`clang/lib/CodeGen/CMakeLists.txt:17`) and already
  calls `offloading::` helpers (`CGCUDANV.cpp:24,1279`), so calling
  `wrapSYCLBinaries` from host CodeGen needs no new dependency.
- Final wrapper cross-TU device link (RDC path, unchanged):
  `linkAndWrapDeviceFiles` — `ClangLinkerWrapper.cpp:966-1053` (calls
  `linkDevice`→`generic::clang` w/ `--sycl-link` at `596-608`, `1015-1017`);
  `bundleSYCL` (`792-803`), `wrapSYCLBinaries` wrap (`748-755`).

## Implementation

### 0. Generalize `-fcuda-include-gpubinary` → `-foffload-include-binary` (NFC)

The cc1 option that names a finalized device binary to incorporate into the host
object at compile time is not CUDA-specific — it is the generic "device code was
finalized per TU, embed it here instead of deferring to a device link" channel,
and CUDA and HIP already **share** it (`Clang.cpp:8306`:
`if ((IsCuda || IsHIP) && ...)`). SYCL no-rdc needs exactly the same channel, so
rather than adding a parallel `-fsycl-include-target-binary`, rename the existing
option and let all three models use it:

- `-fcuda-include-gpubinary` → `-foffload-include-binary`, placed next to
  `-fembed-offload-object=` (its device-link-deferred counterpart) rather than in
  the CUDA block. `-fcuda-include-gpubinary` is kept as a plain `Alias<>` so
  existing invocations and the ~19 CUDA/HIP CodeGen tests that pass it keep
  working unchanged.
- `CodeGenOptions::CudaGpuBinaryFileName` → `OffloadBinaryToEmbedFile`.
- All push sites in `Clang.cpp` emit the new spelling; read sites
  (`CGCUDANV.cpp`, `CIRGenModule.cpp`, `DeviceOffload.cpp`) use the new field.

Two preconditions were verified empirically, both required for a *single*
shared option to be sufficient:
1. **Offloading kinds cannot be mixed on one command line** —
   `err_drv_mix_offload` (`Driver.cpp:1071`) rejects e.g. `-fsycl -x cuda`, so
   there is never a need to include a CUDA binary and a SYCL binary in the same
   host compile. The offloading model already in effect (`-fcuda-is-device`,
   `-fsycl-is-host`, ...) unambiguously selects how the named binary is
   incorporated, so the option needs no per-model spelling.
2. **The option is legitimately single-valued.** It is `Separate<>` with
   `MarshallingInfoString` (a repeat overwrites, unlike the
   `MarshallingInfoStringVector` used by `-fembed-offload-object=`). This is
   correct by construction: `Driver.cpp:5227` builds **one** `LinkJobAction`
   over all `OffloadActions`, so a single finalized binary covers every
   requested arch (multi-target is handled *inside* the file), and
   `HostOffloadingInputs` is asserted to be size 1 at every include site.

This is NFC and lands as its own PR **before** the SYCL host-registration PR
(see the PR table). Suggested by review of intel/llvm#22833.

### 1. Options (alias to `-fgpu-rdc`)

No new LangOpt. Reuse `GPURelocatableDeviceCode`
(`LangOptions.def:257`, `DefaultFalse`).

- `clang/include/clang/Options/Options.td`: add named SYCL aliases inside the
  existing `let Group = sycl_Group in { ... }` block (`Options.td:7560-7575`,
  alongside `fsycl_device_only`), so they appear in SYCL `-help` output:
  ```
  def fsycl_rdc : Flag<["-"], "fsycl-rdc">,
    Visibility<[ClangOption, CC1Option]>, Alias<fgpu_rdc>,
    HelpText<"Generate relocatable SYCL device code, also known as separate "
    "compilation mode (default). Device symbols may be referenced across "
    "translation units and are resolved by a device link step.">;
  def fno_sycl_rdc : Flag<["-"], "fno-sycl-rdc">,
    Visibility<[ClangOption, CC1Option]>, Alias<fno_gpu_rdc>,
    HelpText<"Generate non-relocatable SYCL device code. Each translation "
    "unit's device code is self-contained; with -c the final device binary is "
    "produced within the generated fat object. Device code must be "
    "self-contained per translation unit; cross-translation-unit device "
    "references (e.g. via SYCL_EXTERNAL) are not resolved in this mode.">;
  ```
  Aliases carry no separate marshalling — both spellings set the one
  `GPURelocatableDeviceCode` bit. Named defs (vs the anonymous CUDA
  `def : Flag<...>` at `Options.td:1362`) are used so `HelpText`/`Visibility`
  attach cleanly. The `(default)` note reflects the driver-enforced RDC default
  (see "RDC-by-default for SYCL" below), not the underlying `fgpu_rdc`
  `DefaultFalse`.

  Text derived from intel/llvm (`Options.td:7882-7889`), but corrected: their
  `-fno-sycl-rdc` HelpText erroneously begins "Generate relocatable device
  code" (copy-paste artifact) — upstream text should describe non-relocatable
  behavior. The substantive points kept: (a) with `-c`, final device binaries
  land inside the fat object; (b) device code must be self-contained per TU;
  (c) cross-TU device references (e.g. `SYCL_EXTERNAL`) are **not resolved** in
  no-rdc — but this is a usage contract, **not** a compiler diagnostic (§5,
  matching intel/llvm).
- **RDC-by-default for SYCL**: `-fgpu-rdc` is `DefaultFalse`, but SYCL must
  default to RDC to preserve today's behavior. The driver enforces this in two
  places:
  1. Every SYCL rdc check uses `hasFlag(OPT_fgpu_rdc, OPT_fno_gpu_rdc,
     /*Default=*/true)` (note: `true`, unlike HIP's `false`), so a SYCL compile
     with no explicit flag is treated as RDC (§2).
  2. When SYCL offloading is active and neither flag is given, inject
     `-fgpu-rdc` onto the SYCL device `-cc1` line so the front-end LangOpt
     (`GPURelocatableDeviceCode`) is set for the linkage logic in §5. Do this in
     the SYCL device cc1 construction (`Clang.cpp` `IsSYCLDevice` block,
     ~5308-5331), guarded so it only fires for SYCL and doesn't perturb
     CUDA/HIP.

Trade-off accepted: `-###` will show `-fgpu-rdc` on the SYCL device `-cc1`
line. All 17 CUDA/HIP read-sites of `GPURelocatableDeviceCode` are additionally
gated on `LangOpts.CUDA`/`CUDAIsDevice`, so reusing the bit does not perturb
CUDA/HIP behavior.

### 2. Driver action graph (`Driver.cpp` `BuildOffloadingActions`)

- Near `Driver.cpp:5016`, compute:
  ```cpp
  bool SYCLNoRDC =
      C.isOffloadingHostKind(Action::OFK_SYCL) &&
      !Args.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                    /*Default=*/true);
  ```
  (`-fsycl-rdc`/`-fno-sycl-rdc` alias to these; note `Default=true` for SYCL,
  vs HIP's `false` at `Driver.cpp:5016-5018`.)
- Add a SYCL no-rdc branch before the generic `else` at `Driver.cpp:5225`,
  mirroring the **CUDA** per-TU branch (`5186-5193`) rather than the HIP wrapper
  branch. Each TU produces a **finalized device image** (via `clang-sycl-linker`,
  see §3) that is included into the host compile — not embedded as raw offload
  bitcode. Shape (one finalizer job per TU; no per-TU `LinkerWrapperJobAction`):
  ```cpp
  } else if (SYCLNoRDC) {
    // Finalize this TU's device image (clang-sycl-linker). Result is a device
    // image FILE that host CodeGen will include via -foffload-include-binary.
    Action *FinalizeAction =
        C.MakeAction<LinkJobAction>(OffloadActions, types::TY_Image);
    DDep.add(*FinalizeAction, *C.getSingleOffloadToolChain<Action::OFK_SYCL>(),
             /*BA=*/{}, Action::OFK_SYCL);
  } else {
    // existing RDC packager path (OffloadPackagerJobAction)
  }
  ```
  Because `BuildOffloadingActions` runs once per TU, this yields one finalized
  device image per TU. It is consumed by the host compile (§3) via
  `-foffload-include-binary` — the very same option CUDA's per-TU fatbin uses
  (`Clang.cpp:8306-8313`, §0) — **not** via
  `-fembed-offload-object`. (Exact `LinkJobAction` vs a dedicated finalize
  action type is an implementation detail; the point is: device-only finalize,
  no host-side linker-wrapper node.)

Expected `-ccc-print-phases` under `-fno-sycl-rdc`: the device chain ends in a
`clang-sycl-linker`/finalize node whose output is a host-compile input; there is
**no** device-side `clang-linker-wrapper` node and the final host
`clang-linker-wrapper` does no device work. Sketch:
```
6: backend, {5}, ir, (device-sycl)
7: offload, "device-sycl (spirv64-unknown-unknown)" {6}, ir
8: linker (clang-sycl-linker), {7}, image, (device-sycl)   ; per-TU finalize
9: offload, "host-sycl ..." {2}, "device-sycl ..." {8}, ir
10: backend, {9}, assembler, (host-sycl)                   ; host includes image
11: assembler, {10}, object, (host-sycl)                   ; registration emitted here
12: clang-linker-wrapper, {11}, image, (host-sycl)         ; no SYCL device work
```
RDC (default) phases stay exactly as today (`sycl-offload-jit.cpp:14-20`).

### 3. Per-TU device finalize + host-compile inclusion (CUDA/HIP mirror)

**Device finalize.** The per-TU device action (§2) invokes `clang-sycl-linker`
to fully finalize that TU's device code (bitcode → SPIR-V device image),
producing an image file. This is device-only; no wrapping, no host link. It
reuses the existing `--sycl-link` path (`ClangLinkerWrapper.cpp:596-608` is the
*RDC* route; here the driver calls `clang-sycl-linker` directly for the per-TU
job, or a device-only linker-wrapper mode that stops after `linkDevice`).

The driver must also supply the device target triple and arch (`-triple=`,
`-arch=`) when it invokes `clang-sycl-linker` directly, deriving them from the
offloading job, so the tool can tag the image correctly. The RDC route forwards
these via `-Xlinker`, so already-forwarded values must not be duplicated.

**`-flto` is diagnosed, not silently mis-routed.** `SPIRV::Linker::ConstructJob`
checks `isUsingLTO` *before* deciding on `clang-sycl-linker`, so `-flto` would
substitute `llvm-lto` and emit LLVM bitcode into `.sycl_fatbin` with exit code 0
— a finalized-image slot silently holding un-finalized bitcode. The
`clang-sycl-linker` predicate is therefore computed **before** the LTO check and
the existing `err_drv_no_linker_llvm_support` diagnostic is emitted for that
case, the same diagnostic used when LTO is unsupported outright. This also fixes
`--sycl-link -flto`, which had the same latent bug independently of no-rdc.

**Host inclusion uses the shared `-foffload-include-binary`** (§0) — no new
SYCL-specific option:
- Driver: in no-rdc, pass `-foffload-include-binary <finalized-image>` on the
  **host** SYCL `-cc1` line, in the same `Clang.cpp` block CUDA/HIP use
  (`Clang.cpp:8306-8313`). (RDC keeps `-fembed-offload-object`; the two are
  mutually exclusive per TU.)
- The filename lands in `CodeGenOptions::OffloadBinaryToEmbedFile`.

**Host-CodeGen registration via the shared wrapper.** In host SYCL compilation,
when `OffloadBinaryToEmbedFile` is set, read the file and call the **shared**
`llvm::offloading::wrapSYCLBinaries(CGM.getModule(), Buffer, Options)`
(`OffloadWrapper.cpp:751`) to embed the image + emit the `__sycl_register_lib`
ctor / `__sycl_unregister_lib` dtor directly into the host module.
- This is feasible with no new dependency: clang CodeGen already links
  `FrontendOffloading` (`clang/lib/CodeGen/CMakeLists.txt:17`) and already calls
  `llvm::offloading::` helpers (`CGCUDANV.cpp:24,1279`).
- **Reuse over reimplementation:** unlike `CGCUDANV` (which *reimplements*
  registration), SYCL host CodeGen *calls* the shared `wrapSYCLBinaries`. Add a
  minimal hook — `CodeGenModule::embedSYCLTargetBinary()`, invoked from
  `CodeGenModule::Release` next to the CUDA hook (`CodeGenModule.cpp:1166-1170`),
  gated on `LangOpts.SYCLIsHost && !OffloadBinaryToEmbedFile.empty()`. The hook is
  thin glue (read file → call shared wrapper); all IR-emitting logic stays in
  `OffloadWrapper.cpp`.
- **The hook body lives in `CodeGenSYCL.cpp`, not `CodeGenModule.cpp`.** Only the
  one-line gated call sits in `Release()`; SYCL-specific logic belongs in the
  SYCL CodeGen file so the common module does not accumulate per-offload-model
  code. (Raised in review of intel/llvm#22833.)

**Why this is the CUDA/HIP mirror (and why no second wrapper call):** CUDA/HIP
no-rdc finalize the device image first, then host CodeGen embeds+registers it at
compile time via `-fcuda-include-gpubinary` (`CGCUDANV.cpp:832,858`), so the
final `clang-linker-wrapper` finds nothing in `.llvm.offloading` and does zero
device work. SYCL now does exactly the same, differing only in that it *calls*
the shared registration code instead of reimplementing it. `-fno-sycl-rdc -c`
self-containment holds: the single host `.o` contains the finalized device image
together with its registration.

### 4. Final `clang-linker-wrapper`: no SYCL device work in no-rdc

No changes required in `clang-linker-wrapper` itself for no-rdc, but the
embedding **section** matters. Because the finalized image is included at
host-compile (§3) rather than embedded via `-fembed-offload-object`, there must
be **no** SYCL entry in `.llvm.offloading` for the final wrapper to extract.
`getDeviceInput` then returns nothing for SYCL and `linkAndWrapDeviceFiles` runs
no `clang-sycl-linker` — the final wrapper is effectively a plain host link (same
as CUDA/HIP no-rdc). No marker, no `--sycl-device-link-only` mode, and **no
multi-image `wrapSYCLBinaries` change** (each host TU includes exactly one
finalized image, so the existing single-image `wrapSYCLBinaries`/`SYCLWrapper` is
sufficient).

**Section separation is required (found during E2E bring-up).** `wrapSYCLBinaries`
originally always emitted into `.llvm.offloading`, and `OffloadBinary.cpp:114`
matches that name by **prefix**, so the wrapper treated the already-finalized
per-TU image as device code still needing a link and handed it back to
`clang-sycl-linker`, failing with `unsupported file type: '...o'`. The fix is an
`IsFinalizedImage` flag on `wrapSYCLBinaries` that places compile-time-embedded
images in `.sycl_fatbin` instead; it defaults to false, so the RDC path
(wrapper-internal call) and `llvm-offload-wrapper` are unchanged. This mirrors
CUDA/HIP, which keep compile-time-embedded fatbins in `.nv_fatbin`/`.hip_fatbin`
(`OffloadWrapper.cpp:293-305`) precisely so the wrapper never re-consumes them.
Unlike CUDA there is no second `.nvFatBinSegment`-style section, since SYCL
passes `(ptr, size)` straight to `__sycl_register_lib` and has no wrapper
descriptor to place. This change lives in **PR C** (it touches the shared
`llvm/lib/Frontend/Offloading/OffloadWrapper.cpp`), making C a prerequisite for
PR E working end to end.

**Registration is per-TU, never aggregated across TUs** — matching every other
model's no-rdc path. Each host `.o` emits its own single-blob registration
(`SYCLWrapper::createRegisterFatbinFunction`, one `__sycl_register_lib` ctor +
`__sycl_unregister_lib` dtor, `OffloadWrapper.cpp:660-702`). With N TUs the final
executable has N independent register ctors, each registering its own image at
startup — exactly how the runtime is meant to be driven, and structurally
identical to:
- **HIP/CUDA no-rdc**: `CGCUDANV::finalizeModule` → `makeModuleCtorFunction`
  runs per host TU (`CGCUDANV.cpp:1443-1474`), one `.cuda.fatbin_reg`/atexit
  unreg per fatbin (`OffloadWrapper.cpp:560-635`).
- **intel/llvm old-model no-rdc**: device-link + `clang-offload-wrapper` run
  **under `llvm-foreach`** (`SYCL.cpp:308 shouldDoPerObjectFileLinking`, `:221`),
  i.e. one wrapper/registration per TU. (`-batch` multi-image wrapping is the
  *RDC* path, not no-rdc.)
Aggregating multiple blobs into one registration structure only happens when
images are linked together first (RDC / old-model batch) — never in per-TU
no-rdc.

RDC mode is unchanged: it still embeds via `-fembed-offload-object` and the
final wrapper device-links across TUs and wraps via the same shared
`wrapSYCLBinaries`.

### 5. Self-contained device code for no-rdc (front-end linkage)

No-rdc device modules must be self-contained. The mechanism is **front-end
linkage control** in `getLLVMLinkageForDeclarator`, adopting the shape intel/llvm
ships (`CodeGenModule.cpp:7345-7356`) rather than a `-mllvm` internalize pass
(SPIR-V has no `amdgpu-internalize-symbols` equivalent). Upstream currently has
**no** SYCL device block here — SYCL device functions fall through to generic
C++ linkage (weak_odr/external) — so this adds SYCL linkage semantics for both
RDC and no-RDC.

- **Add a SYCL-device block early in `getLLVMLinkageForDeclarator`**
  (`CodeGenModule.cpp`, before the generic C++ tail ~6737-6753), gated on
  `LangOpts.SYCLIsDevice`. For a device symbol that is **not** an exported/entry
  symbol, return:
  - `GPURelocatableDeviceCode` (RDC) → `llvm::Function::LinkOnceODRLinkage`
    (dedup across TUs at device link);
  - else (no-RDC) → `llvm::Function::InternalLinkage` (self-contained TU, enables
    IPO).
  Intel/llvm reference:
  ```cpp
  if (getLangOpts().SYCLIsDevice && !D->hasAttr<SYCLKernelAttr>() &&
      !D->hasAttr<SYCLDeviceAttr>() && !D->hasAttr<SYCLExternalAttr>() &&
      !SemaSYCL::isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(
          D->getType()))
    return getLangOpts().GPURelocatableDeviceCode
               ? llvm::Function::LinkOnceODRLinkage
               : llvm::Function::InternalLinkage;
  ```
- **Carve-out set — adapt to upstream's available attributes.** Not all of
  intel/llvm's attrs exist upstream (verified): upstream **has** `SYCLKernelAttr`
  (`Attr.td:1670`) and `SYCLExternalAttr` (`Attr.td:1716`); it **lacks**
  `SYCLDeviceAttr`, `SYCLDeviceGlobalAttr`/`device_global`, and the
  `isTypeDecoratedWithDeclAttribute` helper. So the upstream carve-out is the
  subset that exists: keep externally-visible linkage for `SYCLKernelAttr` and
  `SYCLExternalAttr`; internalize/`linkonce_odr` everything else. Add the
  `device_global`/`SYCLDeviceAttr` exclusions only if/when those land upstream.
- **`SYCL_EXTERNAL` is carved out, NOT diagnosed** (matches intel/llvm — verified
  it emits no no-rdc diagnostic; `CheckSYCLExternalFunctionDecl` keeps only its
  linkage/deleted checks). Under `-fno-sycl-rdc`, `SYCLExternalAttr` functions
  **retain external linkage**; using one across TUs simply won't resolve at
  device link (the "self-contained per TU" contract is a documented usage rule,
  not a compiler error). **No `err_sycl_external_no_rdc` diagnostic is added.**
- **Kernel entry point** is unaffected regardless: `sycl_kernel_entry_point`
  functions are not emitted on the normal path (`CodeGenModule.cpp:3958-3970`);
  `EmitSYCLKernelCaller` creates the real entry point directly with
  `ExternalLinkage` (`CodeGenSYCL.cpp:72`), bypassing
  `getLLVMLinkageForDeclarator`. The `SYCLKernelAttr` carve-out additionally
  covers `sycl_kernel`-attributed template functions that do reach this path.
- **RDC-behavior caution:** because upstream has no SYCL block today, adding the
  `LinkOnceODRLinkage` (RDC) arm changes current RDC linkage (from
  weak_odr/external to linkonce_odr for non-exported device symbols). Verify
  existing RDC device output / cross-TU dedup still behaves (this matches
  intel/llvm, so it is the intended target state, but it is a change to today's
  upstream output).
- No `-cc1` `-mllvm` internalize flag is added; the driver injects `-fgpu-rdc`
  on the SYCL device `-cc1` line (or omits it for no-rdc), setting the
  `GPURelocatableDeviceCode` LangOpt (§1). The device
  middle-end/`clang-sycl-linker` module-split (`--module-split-mode`, default
  `source`) then operates on already self-contained modules.
- Audit the other CUDA/HIP LangOpt use sites for SYCL analogs (most are
  CUDA-specific and need no change): externalized-static-var mangling
  (`CodeGenModule.cpp:2388`), `__CLANG_RDC__`-style macro
  (`InitPreprocessor.cpp:581` — decide whether SYCL wants a predefined macro),
  and the CIR mirrors (`CIRGenModule.cpp`, `CIRGenCUDANV.cpp`,
  `LoweringPrepare.cpp`) if/when SYCL uses the CIR path.

## Implementation as independent PRs

The feature splits into **5 PRs**, each ≤~170 LOC. Key idea: the customer-facing
switch (`-f[no-]sycl-rdc`) is also the *activation* switch — until the driver has
both the alias and the `BuildOffloadingActions` no-rdc branch, no-rdc is
unreachable. So all enabling machinery lands **dormant/internal** first (PRs
A–D), and the final PR E adds the user-facing flags and wiring that turn it on.

| PR | Scope | ~LOC | Depends on | Customer-facing? |
|---|---|---|---|---|
| **A. Device linkage + `-fgpu-rdc` injection** | §5 SYCL block in `getLLVMLinkageForDeclarator` **and** (unified) inject `-fgpu-rdc` on the SYCL device `-cc1` line by default. Bundled because the linkage block reads `GPURelocatableDeviceCode`; the injection makes default SYCL langopt=true → `LinkOnceODR` (preserves RDC). | ~120 | — | No (internal cc1 flag + codegen) |
| **B. `-foffload-include-binary` (NFC)** | §0: rename `-fcuda-include-gpubinary` → `-foffload-include-binary` (keeping the old spelling as an `Alias<>`), `CudaGpuBinaryFileName` → `OffloadBinaryToEmbedFile`. Pure refactor, no SYCL code. | ~40 | — | No (cc1-only rename, old spelling still accepted) |
| **C. Host registration** | §3 host side: `embedSYCLTargetBinary` in `CodeGenSYCL.cpp` honoring `-foffload-include-binary` for `SYCLIsHost`, calling shared `wrapSYCLBinaries`, plus the §4 `IsFinalizedImage`/`.sycl_fatbin` section split in `OffloadWrapper.cpp`. Dormant (nothing passes the flag for SYCL yet). | ~130 | B | No (internal) |
| **D. Device finalize** | §3 device side: per-TU `clang-sycl-linker` finalize path (`-triple=`/`-arch=` derivation, `-flto` diagnostic). Dormant (no action graph invokes it yet). | ~90 | — | No (internal; the `-flto` fix does change `--sycl-link -flto` from silent bitcode to an error) |
| **E. Activation (last)** | `-f[no-]sycl-rdc` **aliases** (§1); `SYCLNoRDC` no-rdc branch in `BuildOffloadingActions` (§2); make the `-fgpu-rdc` injection conditional (omit in no-rdc); wire the finalized image to host `-foffload-include-binary`; phases + section tests. | ~150 | A, C, D | **Yes — all user-facing surface** |

Dependency DAG (A, B→C, D parallel; E integrates last):
```
PR A ──────┐
PR B ─► C ─┼──► PR E
PR D ──────┘
```

Notes:
- **B is a prerequisite of C only** (C reads the renamed option/field). B is NFC
  and independently landable — it is worth review on its own merits since it
  removes a CUDA-specific name from a mechanism CUDA, HIP, and now SYCL share.
- **A is the one behavior-changing PR.** It changes existing RDC SYCL linkage
  (non-exported device symbols: external/weak_odr → `linkonce_odr`) and may
  require updating existing `CodeGenSYCL` lit tests. This is the intended target
  state (matches intel/llvm) and preserves RDC cross-TU semantics, but it is a
  real change to today's output — isolate it so reviewers can scrutinize it.
- **The `-fgpu-rdc` injection is unified into A (per design), not a separate
  step**, because the linkage block is incorrect without it (default SYCL would
  internalize everything). E only *narrows* the injection (omit it in no-rdc).
- **D avoids dead code:** implement the finalize as an extension of the existing
  `clang-sycl-linker`/`SPIRV.cpp` linker path with its own lit test driving it
  directly (via `--sycl-link`), so behavior is reviewed even before E references
  it.
- All of A–D are safe to land in any order (subject to B before C): `-fgpu-rdc`
  is passed for SYCL only after A, and all 17 `GPURelocatableDeviceCode`
  read-sites are CUDA/HIP-gated, so none fire for SYCL until A's own block.

## Files to modify

- `clang/include/clang/Options/Options.td` — add `fsycl_rdc`/`fno_sycl_rdc`
  named aliases of `fgpu_rdc`/`fno_gpu_rdc` with `Visibility` + `HelpText`, in
  the `sycl_Group` block (~7567). No new LangOpt.
- `clang/lib/Driver/Driver.cpp` — `SYCLNoRDC` (with `Default=true`) + new
  no-rdc branch in `BuildOffloadingActions` (~5016, ~5225): per-TU device
  finalize action (mirror CUDA branch `5186-5193`).
- `clang/lib/Driver/ToolChains/Clang.cpp` — inject `-fgpu-rdc` on SYCL device
  cc1 by default (~5308); pass `-foffload-include-binary <finalized-image>` on
  the host SYCL `-cc1` in no-rdc, as a new arm of the existing CUDA/HIP block
  (~8306-8317). (No internalize `-mllvm` flag — see §5.)
- `clang/include/clang/Options/Options.td` — §0: `-foffload-include-binary`
  (`Separate`, `MarshallingInfoString<CodeGenOpts<"OffloadBinaryToEmbedFile">>`)
  next to `fembed_offload_object_EQ`; `fcuda_include_gpubinary` becomes an
  `Alias<>` of it.
- `clang/include/clang/Basic/CodeGenOptions.h` — §0: rename
  `CudaGpuBinaryFileName` → `OffloadBinaryToEmbedFile` (~398).
- `clang/lib/CodeGen/CGCUDANV.cpp`, `clang/lib/CIR/CodeGen/CIRGenModule.cpp`,
  `clang/lib/Interpreter/DeviceOffload.cpp` — §0: read the renamed field.
- `clang/lib/CodeGen/CodeGenSYCL.cpp` — `CodeGenModule::embedSYCLTargetBinary()`:
  read the file named by `OffloadBinaryToEmbedFile` and call shared
  `offloading::wrapSYCLBinaries(getModule(), ..., /*IsFinalizedImage=*/true)`.
  Declared in `CodeGenModule.h`; the gated one-line call goes in
  `CodeGenModule::Release` next to the CUDA hook (`CodeGenModule.cpp:~1169`).
  No registration logic reimplemented, and no SYCL logic in `CodeGenModule.cpp`.
- `clang/lib/CodeGen/CodeGenModule.cpp` — add a SYCL-device block in
  `getLLVMLinkageForDeclarator` (before the generic C++ tail ~6737), gated on
  `SYCLIsDevice`, carving out `SYCLKernelAttr`/`SYCLExternalAttr` (upstream's
  available subset); non-exported device symbols → `LinkOnceODR` (RDC) /
  `Internal` (no-RDC). See §5. (No `SemaSYCL.cpp` / diagnostic changes —
  `SYCL_EXTERNAL` is carved out, not rejected, matching intel/llvm.)
- `clang/lib/Driver/ToolChains/SPIRV.cpp` — §3 device side: route a SYCL
  device-offloading link to `clang-sycl-linker`, derive `-triple=`/`-arch=` from
  the offloading job (skipping values already forwarded via `-Xlinker`), and
  diagnose `-flto` with `err_drv_no_linker_llvm_support` instead of falling
  through to `llvm-lto`.
- `clang/lib/Driver/ToolChains/SYCL.{h,cpp}` — give `SYCLToolChain` a device
  linker (reusing the SPIR-V toolchain's `clang-sycl-linker` path) and native
  LLVM bitcode support, so the per-TU finalize link can consume device bitcode.
- `clang-linker-wrapper`: **no changes for no-rdc** (final wrapper sees no SYCL
  device input). Only if the per-TU finalize is implemented as a device-only
  linker-wrapper mode (vs a direct `clang-sycl-linker` driver job) would a small
  mode flag be needed — prefer the direct `clang-sycl-linker` job to avoid it.
- `llvm/lib/Frontend/Offloading/OffloadWrapper.cpp` +
  `llvm/include/llvm/Frontend/Offloading/OffloadWrapper.h` — add an
  `IsFinalizedImage` parameter to `wrapSYCLBinaries` (defaulted false) and have
  `SYCLWrapper::embedBinary` emit into `.sycl_fatbin` when set, `.llvm.offloading`
  otherwise. No multi-image support needed: still one finalized image per host TU.
  See §4 — without this the final wrapper re-links the finalized image and the
  build fails.

## Verification

- **Phases**: extend `clang/test/Driver/sycl-offload-jit.cpp` with a
  `-fno-sycl-rdc` phases block (device chain ends in a per-TU finalize node
  feeding the host compile; **no** device-side `clang-linker-wrapper` node) and
  an explicit `-fsycl-rdc` block confirming today's phases are unchanged
  (default == RDC).
- **Driver `-###`** (in `sycl-offload-jit.cpp`):
  - no-rdc: host SYCL `-cc1` receives `-foffload-include-binary`, and
    `clang-sycl-linker` is invoked per TU with `-triple=`/`-arch=`;
  - rdc/default: `-fembed-offload-object=` present;
  - each direction asserts the *absence* of the other's option with
    `--implicit-check-not=`, not a trailing `-NOT` — a trailing `CHECK-NOT` only
    covers the region after the last positive match, so it would not catch the
    option appearing earlier on the command line. (Raised in review of
    intel/llvm#22833.)
  - final `clang-linker-wrapper` performs no SYCL device link in no-rdc.
- **`-flto` rejection**: assert `-fsycl -fno-sycl-rdc -flto` (with and without
  `-c`) errors with `unable to pass LLVM bit-code files to linker` and never
  invokes `llvm-lto`; likewise for `--sycl-link -flto` in
  `sycl-link-spirv-target.cpp`. Without this the driver exits 0 having written
  LLVM bitcode into `.sycl_fatbin`.
- **Linkage** (`clang/test/CodeGenSYCL/...`): device-codegen test asserting, for
  a non-exported device helper function:
  - `-fno-sycl-rdc` → `internal` linkage;
  - `-fsycl-rdc` (default) → `linkonce_odr`;
  and that in **both** modes a `SYCL_EXTERNAL` (`sycl_external`) function keeps
  external linkage (carved out), and the kernel-caller entry point stays external
  (emitted via `EmitSYCLKernelCaller`, not this path).
- **No SYCL_EXTERNAL diagnostic**: assert `SYCL_EXTERNAL` compiles **without
  error** under `-fno-sycl-rdc` (matches intel/llvm; the contract is enforced by
  link-time resolution, not a Sema error).
- **Flag aliasing / cc1**: assert `-fsycl-rdc` maps to `-fgpu-rdc` on the SYCL
  device `-cc1` line and that a default SYCL compile (no explicit flag) also
  emits `-fgpu-rdc` (RDC default), while `-fno-sycl-rdc` omits it.
- **Build & run**:
  ```
  ninja -C build clang clang-sycl-linker clang-linker-wrapper
  ./build/bin/llvm-lit -v clang/test/Driver clang/test/CodeGenSYCL \
      clang/test/CodeGenCUDA clang/test/CodeGenHIP
  ```
  (CUDA/HIP CodeGen tests are included because §0 touches their option; they pass
  unchanged through the `-fcuda-include-gpubinary` alias, while the CUDA/HIP
  *driver* tests that check the emitted spelling are updated.)
- **Registration codegen** (`clang/test/CodeGenSYCL/offload-include-binary.cpp`):
  with `-foffload-include-binary <file>` on the host `-cc1`, assert the host module
  contains the embedded `.sycl_offloading.binary` global and a
  `__sycl_register_lib` ctor (i.e. the shared `wrapSYCLBinaries` ran at compile
  time), **in section `.sycl_fatbin` and not `.llvm.offloading`** (§4).
- **Section contract** (`clang/test/Driver/sycl-nordc-fatbin-section.cpp`,
  `REQUIRES: spirv-registered-target`): do a real `-c` compile of both modes and
  check the host object's sections with `llvm-readelf -S` — no-rdc has
  `.sycl_fatbin` and no `.llvm.offloading`; rdc is the reverse. This is the
  GPU-free regression guard for the §4 bug; it fails if `IsFinalizedImage` is
  dropped.
- **Two-TU functional smoke** (if a SYCL runtime target is available): compile
  two `.cpp` TUs with `-fno-sycl-rdc`, confirm each host object embeds+registers
  its own finalized device image and the link succeeds with no cross-TU device
  link.

### E2E result (verified on PVC)

`run/e2e_nordc/run_e2e.sh` builds 3 TUs (`tu1`, `tu2`, kernel-less `main`) both
ways, links, runs on GPU, and counts `__sycl_register_lib` call sites:

| mode | device images | register calls | run |
|---|---|---|---|
| `-fsycl-rdc` | 1 | 1 | exit 0 |
| `-fno-sycl-rdc` | 3 | 3 | exit 0 |

**Note the no-rdc count is 3, not 2.** `BuildOffloadingActions` builds the
offload graph per-TU from the presence of `-fsycl` alone, before parsing, so a TU
with **no kernels** (`main.cpp`) still gets a device compile, a finalize, and its
own registration — same as CUDA/HIP. Separately, `clang-sycl-linker`'s default
split mode is `SPLIT_PER_TU` in **both** modes, so the rdc-vs-no-rdc difference
is in the number of OffloadBinary blobs and `__sycl_register_lib` calls, not in
the split granularity.

## Review feedback from intel/llvm#22833

intel/llvm#22833 implements no-rdc downstream, inspired by this design. Points
from its review that apply upstream, and where each is handled:

| Point | Disposition |
|---|---|
| `-flto` silently routes a `clang-sycl-linker` finalize link to `llvm-lto`, writing bitcode where a device image is expected | Fixed in §3 / PR D. Reproduced upstream: `-fsycl -fno-sycl-rdc -flto -c` exited 0 with `BC C0 DE` in `.sycl_fatbin`. Also fixed the same latent bug on `--sycl-link -flto`. |
| Detect the `clang-sycl-linker` route precisely | §3: use `JA.isDeviceOffloading(OFK_SYCL)` rather than a broader `isOffloading` query (PR E, where that arm is added), and compute the predicate before the LTO check so LTO can be diagnosed (PR D). |
| A trailing `CHECK-NOT` does not prove absence across the whole command line | §Verification: converted to `--implicit-check-not=` in both SYCL driver tests. |
| Don't add a SYCL-specific option duplicating `-fcuda-include-gpubinary` | §0 / PR B: generalized to the shared `-foffload-include-binary`; both preconditions (no mixed offloading, single-valued) verified. |
| Keep SYCL-specific logic out of common CodeGen | §3 / PR C: `embedSYCLTargetBinary` body lives in `CodeGenSYCL.cpp`; only the gated call remains in `CodeGenModule::Release`. |
| Trim over-explanatory comments | Applied across the implementation; comments state the contract, not the narrative. |

**Convergence.** intel/llvm#22833 independently landed the same unification
after the same review discussion, choosing `-foffload-include-binary` (identical
option spelling) with the field named `OffloadBinaryFileName`. Upstream uses
`OffloadBinaryToEmbedFile`, which follows the `…File` suffix convention that
dominates `clang/include/clang/Basic/` and keeps the "binary *to embed*" role
explicit next to the neighbouring `-fembed-offload-object=` list. Two other
deliberate divergences: upstream keeps `-fcuda-include-gpubinary` as a bare
`Alias<>` without "Deprecated alias" `HelpText` (it is cc1-only, so there is no
user-facing spelling to deprecate), and upstream **deletes**
`-fsycl-include-target-binary` outright rather than aliasing it, since that
option never shipped upstream and has no invocations to keep working.

**Known gap versus intel/llvm.** Their tests assert that
`-fno-sycl-rdc -flto -c` *succeeds*; upstream currently rejects `-flto` on a
finalize link (`err_drv_no_linker_llvm_support`, §3). Rejecting is the correct
interim behaviour — the alternative observed upstream was exit 0 with bitcode in
`.sycl_fatbin` — but making LTO actually work on the per-TU finalize path is a
follow-up beyond this stack.

Points from that review that do **not** apply upstream (different code base):
the `SYCL_EXTERNAL` diagnostic discussion (upstream carves it out rather than
diagnosing, §5), `llvm-foreach`/old-offload-model plumbing, downstream-only
driver options, `sycl-post-link` staging, and DPC++-specific device-library
handling.

## Appendix: no-RDC flow — CUDA vs HIP vs designed SYCL

Step-by-step comparison of the non-relocatable-device-code flow, with source
locations. SYCL mirrors the **CUDA** model (per-TU finalize → host-compile
inclusion → final link does no device work); the one deliberate improvement is
step 6–7, where SYCL *calls* the shared `OffloadWrapper` instead of
reimplementing registration in CodeGen.

| Step | CUDA `-fno-gpu-rdc` | HIP `-fno-gpu-rdc` | SYCL `-fno-sycl-rdc` (designed) |
|---|---|---|---|
| 1. Device compile (per TU/arch) | `-fcuda-is-device` → PTX/obj | `-fcuda-is-device` → obj/bc | `-fsycl-is-device -emit-llvm-bc` → device bitcode |
| 2. Self-containment (front-end linkage) | `StrongODR`→internal, `CUDAGlobalAttr` kernels external — `CodeGenModule.cpp:6730-6733` | same — `CodeGenModule.cpp:6730-6733` | new SYCL block: non-exported device symbols → `Internal` (no-RDC) / `LinkOnceODR` (RDC); carve out `SYCLKernelAttr`+`SYCLExternalAttr` — `CodeGenModule.cpp:~6737` (§5) |
| 3. Device finalize (per TU) | assemble PTX → fatbin: `LinkJobAction(TY_CUDA_FATBIN)` — `Driver.cpp:5190-5191` | link → fatbin via `LinkerWrapperJobAction(TY_HIP_FATBIN)` — `Driver.cpp:5220-5221` | `clang-sycl-linker` finalizes → device image: per-TU finalize action — `Driver.cpp:~5194` (§2) |
| 4. Pass image to host compile | `-foffload-include-binary <fatbin>` — `Clang.cpp:8306-8308` | `-foffload-include-binary <fatbin>` — `Clang.cpp:8310-8313` | **same option** `-foffload-include-binary <image>` — `Clang.cpp:8314-8320` (§0, §3) |
| 5. Store filename in CodeGenOpts | `OffloadBinaryToEmbedFile` — `Options.td` (next to `fembed_offload_object_EQ`), `CodeGenOptions.h:398` | same | same (§0) |
| 6. Host compile: embed image | `CGCUDANV` reads file → `.nv_fatbin` section — `CGCUDANV.cpp:858-868,895+` | reads file → `.hip_fatbin` — `CGCUDANV.cpp:898-916` | call shared `offloading::wrapSYCLBinaries` → `.sycl_offloading.binary` global in `.sycl_fatbin` section — `OffloadWrapper.cpp:751` + thin host hook (§3) |
| 7. Host compile: emit registration | `CGNVCUDARuntime` builds register ctor (reimplemented) — `CGCUDANV.cpp:828,1443` | HIP register ctor (reimplemented) | `__sycl_register_lib` ctor + `__sycl_unregister_lib` dtor via shared `SYCLWrapper` — `OffloadWrapper.cpp:660-702` |
| 8. Hook ctor into module | `CodeGenModule::Release` → `CUDARuntime->finalizeModule()` → `AddGlobalCtor` — `CodeGenModule.cpp:1163-1167` | same | gated one-line call next to it (`CodeGenModule.cpp:1169`); body in `CodeGenSYCL.cpp` (§3) |
| 9. Final link (`clang-linker-wrapper`) | runs; `getDeviceInput` finds nothing → no device work, plain host link | same | same — no clang-linker-wrapper changes (§4) |

Notes:
- Steps 4–9 are the CUDA/HIP no-rdc model; SYCL matches it. The only functional
  difference is steps 6–7: CUDA/HIP **reimplement** embed+registration in
  `CGCUDANV.cpp`; SYCL **calls the shared** `OffloadWrapper.cpp` (no third copy
  of registration logic).
- Step 9 is identical for all three and is why there is **no** second
  `clang-linker-wrapper` device pass: the image was already included at host
  compile.
- Step 3: all three finalize per TU and each may pack **multiple** device
  images (from multiple `--offload-arch` targets, and for SYCL also from
  `clang-sycl-linker` module splitting). They differ only in *which tool* packs:
  CUDA uses a dedicated `fatbinary` tool (`Cuda.cpp:574`) behind a plain
  `LinkJobAction`; HIP reuses `clang-linker-wrapper --emit-fatbin-only`
  (`Driver.cpp:5220`). SYCL follows the plain-`LinkJobAction` shape (like CUDA)
  because `clang-sycl-linker` **already emits a single packed blob per TU**
  (`ClangLinkerWrapper.cpp:796-797`: "clang-sycl-linker packs outputs into one
  binary blob") — so no `clang-linker-wrapper` layer is needed at finalize, and
  exactly one blob per TU is handed to `-foffload-include-binary`. The images
  packed are always within one TU; cross-TU merging happens only in RDC. This is
  also why `-foffload-include-binary` can be single-valued (§0).
- Step 2: SYCL adopts intel/llvm's linkage block (`CodeGenModule.cpp:7345-7356`),
  carving out `SYCLKernelAttr` + `SYCLExternalAttr` (upstream's available subset;
  intel/llvm also excludes `SYCLDeviceAttr`/`device_global`, which upstream
  lacks). `SYCL_EXTERNAL` stays external and is **not** diagnosed. The kernel
  entry point is external regardless — emitted on a separate path
  (`EmitSYCLKernelCaller`, `CodeGenSYCL.cpp:72`).
