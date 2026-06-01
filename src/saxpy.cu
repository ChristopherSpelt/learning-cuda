#include "kernels/saxpy.h"
#include "saxpy_harness.h"

#include <cstdlib>
#include <iostream>

using namespace cul;

int main(int argc, char **argv) {
  const auto args = bench::parse_args(
      argc, argv, {1 << 10, 1 << 13, 1 << 16, 1 << 19, 1 << 22, 1 << 25, 1 << 27});

  constexpr SaxpyKernel registry[] = {
      {"cuBLAS", &kernels::saxpy::cublas},
      {"naive", &kernels::saxpy::naive},
      {"grid stride", &kernels::saxpy::grid_stride},
      {"vec loads", &kernels::saxpy::vec_loads},
  };

  if (args.csv)
    bench::print_csv_header(std::cout);

  for (int n : args.sizes) {
    SaxpyHarness harness(n);
    for (const auto &k : registry) {
      const auto rec = harness.run(k);
      args.csv ? bench::print_csv_row(std::cout, rec) : bench::print_table_row(std::cout, rec);
    }
  }
  return EXIT_SUCCESS;
}
