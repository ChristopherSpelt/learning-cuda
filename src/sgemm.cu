#include "kernels/sgemm.h"
#include "sgemm_harness.h"

#include <cstdlib>
#include <iostream>

using namespace cul;

int main(int argc, char **argv) {
  // Square sweep M=N=K=n. O(n^3): pass larger sizes explicitly, e.g. `./sgemm 8192`.
  const auto args = bench::parse_args(argc, argv, {256, 512, 1024, 2048, 4096});

  std::ostream &log = args.csv ? std::cerr : std::cout; // commentary never pollutes CSV stdout
  bench::print_device_banner(log);

  constexpr SgemmKernel registry[] = {
      {"cuBLAS (default)", &kernels::sgemm::cublas_default},
      {"naive", &kernels::sgemm::naive},
      {"block tile", &kernels::sgemm::block_tile},
      {"thread tile 1D", &kernels::sgemm::thread_tile_1d},
      {"thread tile 2D", &kernels::sgemm::thread_tile_2d},
      {"vec loads", &kernels::sgemm::vec_loads},
      {"warp tile", &kernels::sgemm::warp_tile},
      {"resolve bank", &kernels::sgemm::resolve_bank},
      {"double buffer", &kernels::sgemm::double_buffer},
      {"double buffer prefetch", &kernels::sgemm::double_buffer_prefetch},
      {"double buffer prefetch bank", &kernels::sgemm::double_buffer_prefetch_bank},
  };

  const int width = bench::name_width(registry);
  if (args.csv)
    bench::print_csv_header(std::cout);

  bool ok = true;
  for (int n : args.sizes) {
    SgemmHarness harness(n, n, n);
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
