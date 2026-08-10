// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target

// Test the SYCL non-RDC route through clang-linker-wrapper: it finalizes the
// device images the driver packaged for one translation unit and writes them
// out as a fat binary, to be embedded into the host object rather than linked
// into an image.

// The device code is compiled as OpenCL C, so that the kernels a SYCL device
// image is split by are real SPIR-V kernels.
__kernel void kernel_a(void) {}
__kernel void kernel_b(void) {}

// RUN: %clang -cc1 %s -triple spirv64-unknown-unknown -emit-llvm-bc -x cl -o %t.spirv.bc

//
// Every architecture is finalized on its own, and the architecture each image
// was requested for is what reaches clang-sycl-linker, which the driver turns
// into the -arch= the image is tagged with. "generic" asks for no specific
// device, which getLinkerArgs() spells as an empty architecture, so no -march=
// is forwarded for it; the input file following the target directly is what
// checks that.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=generic \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=bmg_g21
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld --emit-fatbin-only %t.out -o %t.syclfb 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCHS

// ARCHS-DAG: clang{{.*}} --target=spirv64-unknown-unknown {{[^ ]*}}.o --sycl-link{{$}}
// ARCHS-DAG: clang{{.*}} --target=spirv64-unknown-unknown -march=bmg_g21 {{[^ ]*}}.o --sycl-link{{$}}

//
// The finalized images of all the architectures end up in one fat binary, each
// still tagged with the architecture it was built for. clang-sycl-linker emits
// an offload binary of its own per architecture, so this is only true if their
// entries are merged rather than handed on as they are: were they concatenated,
// a reader would stop after the first header and see image [0] twice.
//
// The two architectures below are deliberately not ones an offload arch name
// maps to, which keeps both of them on the SPIR-V path so that the test needs
// no ahead-of-time compiler for an actual device.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=arch_one \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=arch_two
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --emit-fatbin-only %t.out -o %t.syclfb
// RUN: llvm-objdump --offloading %t.syclfb | FileCheck %s --check-prefix=FATBIN

// FATBIN:      OFFLOADING IMAGE [0]:
// FATBIN:      kind{{ *}}spir-v
// FATBIN:      arch{{ *}}arch_two
// FATBIN:      producer{{ *}}sycl
// FATBIN:      OFFLOADING IMAGE [1]:
// FATBIN:      kind{{ *}}spir-v
// FATBIN:      arch{{ *}}arch_one
// FATBIN:      producer{{ *}}sycl

//
// Splitting the device code of an architecture gives it more than one image, so
// the merged binary holds an entry per image per architecture.
//
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --emit-fatbin-only \
// RUN:   --device-linker=spirv64-unknown-unknown=--module-split-mode=kernel \
// RUN:   %t.out -o %t.split.syclfb
// RUN: llvm-objdump --offloading %t.split.syclfb | FileCheck %s --check-prefix=SPLIT

// SPLIT-COUNT-4: OFFLOADING IMAGE [{{[0-3]}}]:
// SPLIT-NOT: OFFLOADING IMAGE
