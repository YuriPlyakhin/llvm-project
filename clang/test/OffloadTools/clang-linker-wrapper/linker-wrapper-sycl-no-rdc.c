// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target

// Test the SYCL non-RDC route through clang-linker-wrapper.

// The device code is compiled as OpenCL C, so that the kernels a SYCL device
// image is split by are real SPIR-V kernels.
__kernel void kernel_a(void) {}
__kernel void kernel_b(void) {}

// Create device binaries and package them. The named architecture is
// not one an offload arch name maps to so that the test needs no
// ahead-of-time compiler for an actual device.
// RUN: %clang -cc1 %s -triple spirv64-unknown-unknown -emit-llvm-bc -x cl -o %t.spirv.bc
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=generic \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=arch_one

// Test that every architecture is finalized on its own and that all the images
// it was finalized into end up in one fat binary, each still tagged with the
// architecture it was built for. Splitting the device code by kernel gives an
// architecture an image per kernel, so the binary holds two entries for each of
// the two architectures.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --emit-fatbin-only \
// RUN:   --device-linker=spirv64-unknown-unknown=--module-split-mode=kernel \
// RUN:   %t.out -o %t.syclfb
// RUN: llvm-objdump --offloading %t.syclfb | FileCheck %s

// CHECK:      OFFLOADING IMAGE [0]:
// CHECK:      kind{{ *}}spir-v
// CHECK:      arch{{ *}}arch_one
// CHECK:      producer{{ *}}sycl
// CHECK:      OFFLOADING IMAGE [1]:
// CHECK:      arch{{ *}}arch_one
// CHECK:      OFFLOADING IMAGE [2]:
// CHECK:      arch{{ *$}}
// CHECK:      OFFLOADING IMAGE [3]:
// CHECK:      arch{{ *$}}
// CHECK-NOT:  OFFLOADING IMAGE
