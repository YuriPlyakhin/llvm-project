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
/// built from the image that suits the device it is launched on.
///
//===----------------------------------------------------------------------===//

#include <common/device_images.hpp>
#include <mock/helpers.hpp>

#include <detail/device_impl.hpp>
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
constexpr llvm::StringLiteral SPIRVCode = "spir-v code";
constexpr llvm::StringLiteral AOTCode = "device binary code";
constexpr llvm::StringLiteral KernelName = "kernel";

struct MockProgramAndKernelManager : public detail::ProgramAndKernelManager {
  using detail::ProgramAndKernelManager::MDeviceImageManagers;
  using detail::ProgramAndKernelManager::MDeviceKernelInfoMap;
};

/// Builds a fat binary of a kernel that was built both ahead of time, for one
/// architecture, and as SPIR-V to be compiled at run time for any other. The
/// SPIR-V comes first, so that a test of which image is picked cannot pass by
/// the wanted one happening to be the one that is found first.
llvm::SmallString<0> createMultiArchBinary() {
  std::array<llvm::StringRef, 1> KernelNames = {KernelName};
  llvm::SmallString<0> SymbolsBlob;
  llvm::offloading::sycl::writeSymbolTable(KernelNames, SymbolsBlob);

  llvm::SmallVector<llvm::object::OffloadBinary::OffloadingImage, 2> Images;
  Images.push_back(
      unittests::createSYCLImage(SymbolsBlob, llvm::object::IMG_SPIRV,
                                 llvm::object::OFK_SYCL, "generic", SPIRVCode));
  Images.push_back(
      unittests::createSYCLImage(SymbolsBlob, llvm::object::IMG_Object,
                                 llvm::object::OFK_SYCL, "bmg_g21", AOTCode));

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

  /// \return the device code the program the kernel was built from was created
  /// from.
  std::string buildKernelFrom() {
    std::string CreatedFrom;
    EXPECT_CALL(MMock.get(), olCreateProgram(_, _, _, _))
        .WillOnce([&CreatedFrom](ol_device_handle_t Device,
                                 const void *ProgData, size_t ProgDataSize,
                                 ol_program_handle_t *Program) -> ol_result_t {
          CreatedFrom.assign(static_cast<const char *>(ProgData), ProgDataSize);
          *Program = mock::createDummyHandleWithData<ol_program_handle_t>(
              reinterpret_cast<unsigned char *>(&Device), sizeof(Device));
          return OL_SUCCESS;
        });

    sycl::device Device;
    EXPECT_NE(
        MManager.getOrCreateKernel(MManager.getDeviceKernelInfo(KernelName),
                                   *sycl::detail::getSyclObjImpl(Device)),
        nullptr);
    return CreatedFrom;
  }

  mock::MockWrapper MMock;
  MockProgramAndKernelManager MManager;
  llvm::SmallString<0> MBinary;
};

TEST_F(ImageSelectionTest, DeviceBinaryIsPreferredOverSPIRV) {
  // Both images can be used on this device, so the one that was already built
  // for it must win over the SPIR-V that would have to be compiled first.
  EXPECT_CALL(MMock.get(), olIsValidBinary(_, _, _, _))
      .WillRepeatedly([](ol_device_handle_t /*Device*/, const void * /*Data*/,
                         size_t /*Size*/, bool *Valid) -> ol_result_t {
        *Valid = true;
        return OL_SUCCESS;
      });

  EXPECT_EQ(buildKernelFrom(), AOTCode);
}

TEST_F(ImageSelectionTest, SPIRVIsUsedWhenDeviceBinaryDoesNotFitTheDevice) {
  // This is a device other than the one the kernel was built ahead of time
  // for, which leaves the SPIR-V as the only image it can run.
  acceptOnly(SPIRVCode);

  EXPECT_EQ(buildKernelFrom(), SPIRVCode);
}

TEST_F(ImageSelectionTest, NoCompatibleImage) {
  acceptOnly("device code of no image of this binary");

  EXPECT_CALL(MMock.get(), olCreateProgram(_, _, _, _)).Times(0);

  sycl::device Device;
  EXPECT_THAT(
      [&]() {
        MManager.getOrCreateKernel(MManager.getDeviceKernelInfo(KernelName),
                                   *sycl::detail::getSyclObjImpl(Device));
      },
      Throws<sycl::exception>(
          AllOf(Property(&sycl::exception::what,
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
