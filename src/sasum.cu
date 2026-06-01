#include "kernels/sasum.h"
#include "sasum_harness.h"

#include <cstdlib>
#include <iostream>

using namespace cul;

int main(int argc, char **argv) {
  const auto args = bench::parse_args(
      argc, argv, {1 << 10, 1 << 13, 1 << 16, 1 << 19, 1 << 22, 1 << 25, 1 << 27});

  constexpr SasumKernel registry[] = {
      {"cuBLAS", &kernels::sasum::cublas},
      {"naive", &kernels::sasum::naive},
      {"block reduce", &kernels::sasum::block_reduce},
      {"grid stride", &kernels::sasum::grid_stride},
      {"warp reduce", &kernels::sasum::warp_reduce},
  };

  if (args.csv)
    bench::print_csv_header(std::cout);

  for (int n : args.sizes) {
    SasumHarness harness(n);
    for (const auto &k : registry) {
      const auto rec = harness.run(k);
      args.csv ? bench::print_csv_row(std::cout, rec) : bench::print_table_row(std::cout, rec);
    }
  }
  return EXIT_SUCCESS;
}
