//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager.hpp>

#include <sycl/__impl/exception.hpp>

#include <detail/device_impl.hpp>
#include <detail/offload/offload_utils.hpp>

#include <llvm/Frontend/Offloading/Utility.h>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

DeviceKernelInfo &_LIBSYCL_EXPORT
getDeviceKernelInfo(std::string_view KernelName) {
  return ProgramAndKernelManager::getInstance().getDeviceKernelInfo(KernelName);
}

DeviceKernelInfo &
ProgramAndKernelManager::getDeviceKernelInfo(std::string_view KernelName) {
  auto It = MDeviceKernelInfoMap.find(KernelName);
  assert(It != MDeviceKernelInfoMap.end());
  return It->second;
}

void ProgramAndKernelManager::releaseResources() {
  MDeviceKernelInfoMap.clear();
  MDeviceImageManagers.clear();
}

static inline bool
checkDeviceImageValidity(const llvm::object::OffloadBinary &OB) {
  return (OB.getOffloadKind() == llvm::object::OFK_SYCL) &&
         (OB.getImageKind() == llvm::object::IMG_SPIRV);
}

void ProgramAndKernelManager::registerFatBin(const void *BinaryStart,
                                              const void *BinaryEnd) {
  assert(BinaryStart && BinaryEnd && "Binary pointers can't be nullptr");

  llvm::MemoryBufferRef MBR(
      llvm::StringRef(static_cast<const char *>(BinaryStart),
                      static_cast<const char *>(BinaryEnd) -
                          static_cast<const char *>(BinaryStart)),
      /*Identifier=*/"");
  auto BinOrErr = llvm::object::OffloadBinary::create(MBR);
  if (!BinOrErr || BinOrErr->empty())
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Failed to parse OffloadBinary");

  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  for (std::unique_ptr<llvm::object::OffloadBinary> &OB : *BinOrErr) {
    if (!checkDeviceImageValidity(*OB))
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Incompatible device image.");

    llvm::StringRef Symbols = OB->getString("symbols");
    const auto &Header =
        *reinterpret_cast<const llvm::offloading::sycl::SymbolTableHeader *>(
            Symbols.data());

    const void *ImageStart = OB->getImage().data();
    auto [DevImgIt, Inserted] = MDeviceImageManagers.emplace(
        ImageStart, std::make_unique<DeviceImageManager>(std::move(OB)));
    assert(Inserted && "Device image registered twice");
    DeviceImageManager &NewImageWrapper = *DevImgIt->second;

    llvm::offloading::sycl::forEachSymbol(Header, [&](llvm::StringRef Name) {
      auto It = MDeviceKernelInfoMap.find(std::string_view(Name));
      if (It == MDeviceKernelInfoMap.end()) {
        [[maybe_unused]] auto [Iterator, EmplaceSucceeded] =
            MDeviceKernelInfoMap.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(std::string_view(Name)),
                std::forward_as_tuple(std::string_view(Name), NewImageWrapper));
        assert(EmplaceSucceeded && "Kernel name found in multiple images");
      }
    });
  }
}

void ProgramAndKernelManager::unregisterFatBin(const void *BinaryStart,
                                               const void *BinaryEnd) {
  assert(BinaryStart && BinaryEnd && "Binary pointers can't be nullptr");

  llvm::MemoryBufferRef MBR(
      llvm::StringRef(static_cast<const char *>(BinaryStart),
                      static_cast<const char *>(BinaryEnd) -
                          static_cast<const char *>(BinaryStart)),
      /*Identifier=*/"");
  auto BinOrErr = llvm::object::OffloadBinary::create(MBR);
  if (!BinOrErr || BinOrErr->empty())
    return;

  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  for (std::unique_ptr<llvm::object::OffloadBinary> &OB : *BinOrErr) {
    const void *ImageStart = OB->getImage().data();
    auto DevImageIt = MDeviceImageManagers.find(ImageStart);
    if (DevImageIt == MDeviceImageManagers.end())
      continue;

    llvm::StringRef Symbols =
        DevImageIt->second->getOffloadBinary().getString("symbols");
    const auto &Header =
        *reinterpret_cast<const llvm::offloading::sycl::SymbolTableHeader *>(
            Symbols.data());
    llvm::offloading::sycl::forEachSymbol(Header, [&](llvm::StringRef Name) {
      if (auto KernelIt = MDeviceKernelInfoMap.find(std::string_view(Name));
          KernelIt != MDeviceKernelInfoMap.end()) {
        // Programs are attached to the image and will be released with image
        // destruction. Clear only kernel specific data by destroying its kernel
        // info object.
        MDeviceKernelInfoMap.erase(KernelIt);
      }
    });

    MDeviceImageManagers.erase(DevImageIt);
  }
}

static bool isImageCompatible(const DeviceImageManager &Image,
                              const DeviceImpl &Device) {
  const llvm::object::OffloadBinary &OB = Image.getOffloadBinary();
  if (!(OB.getTriple() == DeviceBinaryTripleSPIRV64 &&
        Device.getBackend() == sycl::backend::level_zero))
    return false;

  bool IsValid{};
  llvm::StringRef ImageBytes = OB.getImage();
  callAndThrow(olIsValidBinary, Device.getOLHandle(), ImageBytes.data(),
               ImageBytes.size(), &IsValid);
  return IsValid;
}

ol_symbol_handle_t
ProgramAndKernelManager::getOrCreateKernel(DeviceKernelInfo &KernelInfo,
                                           DeviceImpl &Device) {

  std::lock_guard<std::mutex> KernelGuard(MDataCollectionMutex);

  if (auto Kernel = KernelInfo.getKernel(Device.getOLHandle()))
    return Kernel;

  auto &DeviceImage = KernelInfo.getDeviceImage();

  if (!isImageCompatible(DeviceImage, Device))
    throw exception(make_error_code(errc::runtime),
                    std::string("No compatible image for ") +
                        KernelInfo.getName().data() + " was found");

  auto DeviceHandle = Device.getOLHandle();
  auto Program = DeviceImage.getOrCreateProgram(DeviceHandle);

  ol_symbol_handle_t Kernel{};
  callAndThrow(olGetSymbol, Program, KernelInfo.getName().data(),
               OL_SYMBOL_KIND_KERNEL, &Kernel);
  KernelInfo.addKernel(DeviceHandle, Kernel);
  return Kernel;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

extern "C" _LIBSYCL_EXPORT void
__sycl_register_lib(const void *BinaryStart, const void *BinaryEnd) {
  sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(BinaryStart,
                                                                    BinaryEnd);
}

extern "C" _LIBSYCL_EXPORT void
__sycl_unregister_lib(const void *BinaryStart, const void *BinaryEnd) {
  sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
      BinaryStart, BinaryEnd);
}
