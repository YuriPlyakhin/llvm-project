//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests that a kernel offered by more than one device image of a fat binary,
/// as is the case when its device code was built for several architectures, is
/// built from an image the device it is launched on can use.
///
//===----------------------------------------------------------------------===//

#include <common/device_images.hpp>
#include <mock/helpers.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager.hpp>

#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/exception.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>

#include <llvm/ADT/SmallVector.h>

using namespace sycl;
using namespace ::testing;

namespace {

// The device code of these two images is what tells them apart in the
// liboffload calls the manager makes for them.
constexpr llvm::StringLiteral FirstCode = "code of the first architecture";
constexpr llvm::StringLiteral SecondCode = "code of the second architecture";
constexpr llvm::StringLiteral KernelName = "kernel";

struct MockProgramAndKernelManager : public detail::ProgramAndKernelManager {
  using detail::ProgramAndKernelManager::MDeviceImageManagers;
  using detail::ProgramAndKernelManager::MDeviceKernelInfoMap;
};

/// Builds a fat binary of a kernel that was built for two architectures, as the
/// device code of a translation unit is when it is built for several of them.
llvm::SmallString<0> createMultiArchBinary() {
  std::array<llvm::StringRef, 1> KernelNames = {KernelName};
  llvm::SmallString<0> SymbolsBlob;
  llvm::offloading::sycl::writeSymbolTable(KernelNames, SymbolsBlob);

  llvm::SmallVector<llvm::object::OffloadBinary::OffloadingImage, 2> Images;
  Images.push_back(unittests::createSYCLImage(
      SymbolsBlob, llvm::object::IMG_SPIRV, llvm::object::OFK_SYCL, "arch_one",
      FirstCode));
  Images.push_back(unittests::createSYCLImage(
      SymbolsBlob, llvm::object::IMG_SPIRV, llvm::object::OFK_SYCL, "arch_two",
      SecondCode));

  return llvm::object::OffloadBinary::write(Images);
}

class ImageSelectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    MBinary = createMultiArchBinary();
    ASSERT_NO_THROW(MManager.registerFatBin(MBinary.data(), MBinary.size()));
    // Both images offer the kernel, so both must be candidates for it.
    ASSERT_THAT(MManager.MDeviceKernelInfoMap, SizeIs(1u));
    ASSERT_THAT(MManager.getDeviceKernelInfo(KernelName).getDeviceImages(),
                SizeIs(2u));
  }

  void TearDown() override {
    MManager.unregisterFatBin(MBinary.data(), MBinary.size());
  }

  /// Accepts the images whose device code is the given one, as a device
  /// accepts an image built for it and rejects one that was not.
  void acceptOnly(llvm::StringRef Code) {
    EXPECT_CALL(MMock.get(), olIsValidBinary(_, _, _, _))
        .WillRepeatedly([Code](ol_device_handle_t /*Device*/,
                               const void *ProgData, size_t ProgDataSize,
                               bool *Valid) -> ol_result_t {
          *Valid = llvm::StringRef(static_cast<const char *>(ProgData),
                                   ProgDataSize) == Code;
          return OL_SUCCESS;
        });
  }

  /// Builds the kernel for the default device.
  /// \return the liboffload kernel handle.
  static ol_symbol_handle_t buildKernel(MockProgramAndKernelManager &Manager) {
    sycl::device Device;
    detail::DeviceImpl &DevImpl = *sycl::detail::getSyclObjImpl(Device);
    return Manager.getOrCreateKernel(
        Manager.getDeviceKernelInfo(KernelName),
        DevImpl.getPlatformImpl().getDefaultContext(), DevImpl);
  }

  /// \return the device code the program the kernel was built from was created
  /// from.
  std::string buildKernelFrom() {
    std::string CreatedFrom;
    EXPECT_CALL(MMock.get(), olCreateProgram(_, _, _, _, _))
        .WillOnce([&CreatedFrom](ol_context_handle_t /*Context*/,
                                 ol_device_handle_t Device,
                                 const void *ProgData, size_t ProgDataSize,
                                 ol_program_handle_t *Program) -> ol_result_t {
          CreatedFrom.assign(static_cast<const char *>(ProgData), ProgDataSize);
          *Program = mock::createDummyHandleWithData<ol_program_handle_t>(
              reinterpret_cast<unsigned char *>(&Device), sizeof(Device));
          return OL_SUCCESS;
        });

    EXPECT_NE(buildKernel(MManager), nullptr);
    return CreatedFrom;
  }

  mock::MockWrapper MMock;
  MockProgramAndKernelManager MManager;
  llvm::SmallString<0> MBinary;
};

TEST_F(ImageSelectionTest, TheImageTheDeviceAcceptsIsUsed) {
  // Of the two images offering the kernel, this device can only run the second
  // one, so building the kernel must fall to that one rather than to the image
  // that happens to be found first.
  acceptOnly(SecondCode);

  EXPECT_EQ(buildKernelFrom(), SecondCode);
}

TEST_F(ImageSelectionTest, NoCompatibleImage) {
  acceptOnly("device code of no image of this binary");

  EXPECT_CALL(MMock.get(), olCreateProgram(_, _, _, _, _)).Times(0);

  EXPECT_THAT([&]() { buildKernel(MManager); },
              Throws<sycl::exception>(AllOf(
                  Property(&sycl::exception::what,
                           HasSubstr("No compatible image for kernel")),
                  Property(&sycl::exception::code, Eq(sycl::errc::runtime)))));
}

TEST_F(ImageSelectionTest, UnregisterDropsAllCandidates) {
  MManager.unregisterFatBin(MBinary.data(), MBinary.size());

  // All the images a kernel can be built from belong to one fat binary and go
  // away with it.
  EXPECT_THAT(MManager.MDeviceImageManagers, IsEmpty());
  EXPECT_THAT(MManager.MDeviceKernelInfoMap, IsEmpty());
}

TEST(ImageSelection, ImagesOfAnotherBinaryAreNotCandidates) {
  mock::MockWrapper Mock;
  MockProgramAndKernelManager Manager;

  // Two fat binaries offering the same kernel, as two translation units built
  // separately do when the kernel is defined in a header they share. Only the
  // images of the binary that got there first can be built from, so that the
  // images a kernel keeps are exactly the ones that go away with it.
  llvm::SmallString<0> First = createMultiArchBinary();
  llvm::SmallString<0> Second = createMultiArchBinary();
  ASSERT_NO_THROW(Manager.registerFatBin(First.data(), First.size()));
  ASSERT_NO_THROW(Manager.registerFatBin(Second.data(), Second.size()));

  EXPECT_THAT(Manager.MDeviceImageManagers, SizeIs(2u));
  EXPECT_THAT(Manager.getDeviceKernelInfo(KernelName).getDeviceImages(),
              SizeIs(2u));

  Manager.unregisterFatBin(Second.data(), Second.size());
  Manager.unregisterFatBin(First.data(), First.size());
  EXPECT_THAT(Manager.MDeviceKernelInfoMap, IsEmpty());
}

} // namespace
