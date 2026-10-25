#include "kernels/saxpy.h"
#include "saxpy_harness.h"

#include <cstdlib>
#include <iostream>

using namespace cul;

int main(int argc, char **argv) {
  const auto args = bench::parse_args(
      argc, argv, {1 << 10, 1 << 13, 1 << 16, 1 << 19, 1 << 22, 1 << 25, 1 << 27});

  std::ostream &log = args.csv ? std::cerr : std::cout; // commentary never pollutes CSV stdout
  bench::print_device_banner(log);

  constexpr SaxpyKernel registry[] = {
      {"cuBLAS", &kernels::saxpy::cublas},
      {"naive", &kernels::saxpy::naive},
      {"grid stride", &kernels::saxpy::grid_stride},
      {"vec loads", &kernels::saxpy::vec_loads},
  };

  const int width = bench::name_width(registry);
  if (args.csv)
    bench::print_csv_header(std::cout);

  bool ok = true;
  for (int n : args.sizes) {
    SaxpyHarness harness(n);
    harness.print_reference(log);
    for (const auto &k : registry) {
      const auto rec = harness.run(k);
      ok = ok && rec.passed;
      args.csv ? bench::print_csv_row(std::cout, rec)
               : bench::print_table_row(std::cout, rec, width);
    }
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
