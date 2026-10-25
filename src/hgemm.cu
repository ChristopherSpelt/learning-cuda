#include "hgemm_harness.h"
#include "kernels/hgemm.h"

#include <cstdlib>
#include <iostream>

using namespace cul;

int main(int argc, char **argv) {
  // Square sweep M=N=K=n. O(n^3): pass other sizes explicitly, e.g. `./hgemm 16384`.
  const auto args = bench::parse_args(argc, argv, {1024, 2048, 4096, 8192});

  std::ostream &log = args.csv ? std::cerr : std::cout; // commentary never pollutes CSV stdout
  bench::print_device_banner(log);

  constexpr HgemmKernel registry[] = {
      {"cuBLAS (default)", &kernels::hgemm::cublas_default},
  };

  const int width = bench::name_width(registry);
  if (args.csv)
    bench::print_csv_header(std::cout);

  bool ok = true;
  for (int n : args.sizes) {
    HgemmHarness harness(n, n, n);
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
