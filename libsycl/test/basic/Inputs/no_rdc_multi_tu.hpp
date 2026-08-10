#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

// Defined in Inputs/no_rdc_multi_tu_second.cpp, which contributes a kernel of
// its own to the program.
void runSecondTuKernel(sycl::queue &Q, int *Data, std::size_t N);
