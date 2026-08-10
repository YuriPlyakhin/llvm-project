/// Verify which ELF section carries the SYCL device image in the host object.
///
/// clang-linker-wrapper scans sections whose name starts with ".llvm.offloading"
/// and hands what it finds to the device linker as code still needing a link. In
/// non-RDC mode the per-TU image is already finalized at compile time, so it must
/// live in a section the wrapper does not scan, mirroring how CUDA/HIP use
/// ".nv_fatbin"/".hip_fatbin".

// REQUIRES: spirv-registered-target

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc \
// RUN:   -c %s -o %t.nordc.o
// RUN: llvm-readelf -S %t.nordc.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'
// NORDC: .sycl_fatbin

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fsycl-rdc \
// RUN:   -c %s -o %t.rdc.o
// RUN: llvm-readelf -S %t.rdc.o \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.sycl_fatbin'
// RDC: .llvm.offloading

void f() {}
