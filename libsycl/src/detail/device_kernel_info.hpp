//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the class that aggregates information
/// specific to device kernels (i.e. information that is uniform between
/// different submissions of the same kernel).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_KERNEL_INFO
#define _LIBSYCL_DEVICE_KERNEL_INFO

#include <sycl/__impl/detail/config.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <OffloadAPI.h>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class ProgramAndKernelManager;

// TODO: Pointers to instances of this class are supported to be stored in
// header function templates as a static variable to avoid repeated runtime
// lookup overhead.
class DeviceKernelInfo {
public:
  /// Constructs a device kernel info instance.
  ///
  /// \param KernelName the name of the kernel.
  /// \param DeviceImage a device image containing device code of this kernel.
  DeviceKernelInfo(std::string_view KernelName, DeviceImageManager &DeviceImage)
      : MName(KernelName), MDeviceImages({&DeviceImage}) {}

  /// \return the name of this kernel.
  std::string_view getName() { return MName; }

  /// A kernel can be offered by more than one device image, because a fat
  /// binary holds an image per architecture it was built for, and the device
  /// code of an architecture can be split into several images. Which of them
  /// to use is only known once the device is.
  /// \return the device images containing the device code of this kernel.
  llvm::ArrayRef<DeviceImageManager *> getDeviceImages() const {
    return MDeviceImages;
  }

private:
  std::unordered_map<ol_device_handle_t, ol_symbol_handle_t> MBuiltKernels;

  std::string_view MName;
  llvm::SmallVector<DeviceImageManager *, 1> MDeviceImages;

  /// Searches for the existing kernel handle compatible with the specified
  /// device.
  /// \param Device the device the kernel must be compatible with.
  /// \return a liboffload kernel handle if a built kernel was found; otherwise
  /// returns nullptr.
  ol_symbol_handle_t getKernel(ol_device_handle_t Device) const {
    auto KernelIt = MBuiltKernels.find(Device);
    if (KernelIt == MBuiltKernels.end())
      return nullptr;
    return KernelIt->second;
  }

  /// Records another device image that offers this kernel.
  /// \param DeviceImage the device image to add.
  void addDeviceImage(DeviceImageManager &DeviceImage) {
    MDeviceImages.push_back(&DeviceImage);
  }

  /// Attaches a liboffload kernel handle to this device kernel info object.
  /// \param Device the device the kernel symbol was created for.
  /// \param Kernel the liboffload kernel symbol to attach.
  void addKernel(ol_device_handle_t Device, ol_symbol_handle_t Kernel) {
    assert(Kernel && "Invalid liboffload kernel handle");
    assert(Device && "Invalid liboffload device handle");
    assert((MBuiltKernels.find(Device) == MBuiltKernels.end()) &&
           "Kernel is being managed already");
    MBuiltKernels.insert({Device, Kernel});
  }

  /// Kernel info update is intended to be done only by ProgramAndKernelManager.
  friend class ProgramAndKernelManager;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_KERNEL_INFO
