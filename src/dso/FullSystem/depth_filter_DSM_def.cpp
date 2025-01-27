#include "depth_filter_DSM_def.h"

namespace dso {
void DF_Frame::InitSeedDepth(Seed &seed, const bool &has_init_depth,
                             const number_t &init_depth) {
  if (has_init_depth) {
    seed.rho = 1.0 / init_depth;
    if (init_depth > 0.5) {
      seed.z_range = 4.0 / (init_depth - 0.2);
    } else {
      seed.z_range = 4.0 / init_depth;
    }
    //    point.z_range = 4.0 / (init_depth);
  } else {
    seed.rho = 1.0 / 2.0;
    seed.z_range = 25.0;
    //    point.z_range = 10.0 / (0.5 * 2.5);
  }
  seed.sigma2 = seed.z_range * seed.z_range / 36.0;

  //  number_t rho_min = seed.rho - std::sqrt(seed.sigma2);
  //  number_t rho_max = seed.rho + std::sqrt(seed.sigma2);
  //  printf("depth init %f, z_range %f, sigma2 %f, depth min %f, depth max
  //  %f\n", seed.rho, seed.z_range, seed.sigma2,
  //         1.0 / rho_max, 1.0 / rho_min);
}
} // namespace dso