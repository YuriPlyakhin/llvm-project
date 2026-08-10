// Second translation unit for basic/no_rdc_multi_tu.cpp.

#include "no_rdc_multi_tu.hpp"

void runSecondTuKernel(sycl::queue &Q, int *Data, std::size_t N) {
  Q.parallel_for<class SecondTuKernel>(N,
                                       [=](sycl::item<1> I) { Data[I] *= 3; });
}
