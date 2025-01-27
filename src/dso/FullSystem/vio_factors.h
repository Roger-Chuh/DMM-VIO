#pragma once

#include "../camera_model/camera_base.h"
#include "../camera_model/vio_def.h"
#include "opt_nodes_def.h"
#include "patch.h"

namespace dso {
struct Point;

class DirectVisualFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DirectFactorRes
  Evaluate(const Mat4 &T10, const number_t &idp,
           const std::shared_ptr<AlgsImage> &target_image, const Patch &patch,
           CameraBase *camera_ptr, const int &border, Patch::ArrayV &r_vec,
           number_t &ws2, number_t &r2, Vec2 *p_uv = nullptr,
           Vec3 *p_target_dir = nullptr, number_t *p_cos_theta = nullptr,
           Vec3 *p_dp_didp = nullptr, number_t *p_zncc = nullptr,
           const number_t &pass_threshold = 0.8);

  DirectFactorRes Evaluate_UV(const Patch::Matrix2P &target_uv,
                              const int &border,
                              const std::shared_ptr<AlgsImage> &target_image,
                              const Patch &patch,
                              const number_t &inlier_threshold,
                              Patch::ArrayV &r_vec, number_t &zncc);

  bool check_depth_ = true;

  bool dir_reject_ = false;
  bool disparity_reject = false;
  number_t rad_step_ = 0.7 * 0.2438 / 180 * 3.14159257;
  number_t cos_grad_epipolar_dir_threshold_ = 0.0871557; // 0.0871557
  number_t cos_max_dist_angle_ =
      0.34202; // cos(70)
               //  number_t cos_max_dist_angle_ = 0.642788;

  number_t epipolar_grad_cos_theta_;

  number_t depth_in_target_camera_;

  bool calc_R_theta_;
  Mat2 R_theta_;

  size_t cur_pid_ = kInvalid_uint64_t;

private:
  bool has_OOB_;
};

} // namespace dso
