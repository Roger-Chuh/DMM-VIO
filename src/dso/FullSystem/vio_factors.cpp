#include "vio_factors.h"
#include "../camera_model/vio_math_0.h"
#include "opt_nodes_def.h"
//#include <vector>

namespace dso {
DirectFactorRes DirectVisualFactor::Evaluate(
    const Mat4 &T10, const number_t &idp,
    const std::shared_ptr<AlgsImage> &target_image, const Patch &patch,
    CameraBase *camera_ptr, const int &border, Patch::ArrayV &r_vec,
    number_t &ws2, number_t &r2, Vec2 *p_uv, Vec3 *p_target_dir,
    number_t *p_cos_theta, Vec3 *p_dp_didp, number_t *p_zncc,
    const number_t &pass_threshold) {
  const size_t &img_cols = target_image->width;
  const size_t &img_rows = target_image->height;

  dir_reject_ = false;

  //  patch.TransformScaled(T10, idp, target_scaled_points_);

  Vec3 P = T10.block<3, 3>(0, 0) * patch.dir0 + idp * T10.block<3, 1>(0, 3);
  //  std::cout << "T10:\n" << T10 << std::endl;
  //  std::cout << "dir0: " << patch.dir0.transpose() << std::endl;
  //  std::cout << "idp: " << idp << std::endl;
  number_t depth_scale = P.norm();
  Vec3 nt = P / depth_scale;
  //  if (check_depth_ && depth_scale >= 2 /*|| depth_scale <= 0.5*/) {
  //    return kOOB;
  //  }

  depth_in_target_camera_ = depth_scale / idp;

  const Vec3 &host_dir = patch.dir0;
  const Vec3 &target_dir = P;
  const Vec3 &host_dir_in_target = T10.block<3, 3>(0, 0) * host_dir;
  number_t cos_theta = nt.dot(host_dir_in_target);
  if (cos_theta <= cos_max_dist_angle_) {
    disparity_reject = true;
    return kOOB;
  }
  disparity_reject = false;
  if (p_cos_theta) {
    *p_cos_theta = cos_theta;
  }

  if (p_target_dir) {
    *p_target_dir = P.normalized();
  }

  if (P.hasNaN()) {
    printf("DirectVisualFactor Evaluate Error: target scaled point has nan");
    exit(-1);
  }

  Vec2 uv0;
  Mat23 Jproj_target;
  camera_ptr->Project(P, uv0, &Jproj_target);

  Mat2 duv_target_duv_host =
      Jproj_target * T10.block<3, 3>(0, 0) * patch.J_unproj;

  Patch::Matrix2P target_uvs = duv_target_duv_host * pattern2_def;
  target_uvs = target_uvs.colwise() + uv0;

  if (p_uv) {
    (*p_uv) = target_uvs.col(0);
  }

  number_t zncc;

  DirectFactorRes res = Evaluate_UV(target_uvs, border, target_image, patch,
                                    pass_threshold, r_vec, zncc);

  if (res == kOOB || res == kWithoutSigma) {
    return res;
  }

  if (p_zncc) {
    *p_zncc = zncc;
  }

  r2 = 2 - 2 * zncc;
  if (p_dp_didp) {
    (*p_dp_didp) = T10.block<3, 3>(0, 0).transpose() * T10.block<3, 1>(0, 3);
  }

  if (res == kOutlier) {
    return res;
  }
  {
    Vec3 t10_in_host =
        T10.block<3, 3>(0, 0).transpose() * T10.block<3, 1>(0, 3);
    Vec3 epipolar_plane_dir_test = Skew(patch.dir0) * t10_in_host;
    epipolar_plane_dir_test.normalize();

    epipolar_grad_cos_theta_ =
        std::abs(patch.grad_plane_dir.dot(epipolar_plane_dir_test));
  }

  //    number_t epipolar_grad_cos_theta_ = epipolar_dir.dot(grad_cur);
  if (epipolar_grad_cos_theta_ < cos_grad_epipolar_dir_threshold_) {
    dir_reject_ = true;
  }

  ws2 = 1.0;

  return kInlier;
}

DirectFactorRes DirectVisualFactor::Evaluate_UV(
    const Patch::Matrix2P &target_uvs, const int &border,
    const std::shared_ptr<AlgsImage> &target_image, const Patch &patch,
    const number_t &inlier_threshold, Patch::ArrayV &r_vec, number_t &zncc) {
  const size_t &img_cols = target_image->width;
  const size_t &img_rows = target_image->height;

  Patch::ArrayP x_coords = target_uvs.row(0);
  Patch::ArrayP y_coords = target_uvs.row(1);

  has_OOB_ = !(
      (x_coords >= border).all() && (x_coords < img_cols - border - 1).all() &&
      (y_coords >= border).all() && (y_coords < img_rows - border - 1).all());

  if (has_OOB_) {
    return kOOB;
  }

  patch.GetPatchValues(target_uvs, target_image, r_vec, nullptr, nullptr);

  number_t mean1 = r_vec.sum() / (number_t)Patch::PATCH_SIZE;
  r_vec = r_vec - mean1;
  number_t sigma1 = std::sqrt(r_vec.square().sum());

  if (sigma1 * sigma1 < patch.sigma2_threshold_vec[0] ||
      sigma1 * sigma1 > patch.sigma2_threshold_vec[1]) {
    return kWithoutSigma;
  }

  r_vec = r_vec / sigma1;
  zncc = patch.normalized_vals.transpose().matrix() * r_vec.matrix();

  //  std::cout << "norm0 " << patch.normalized_vals.transpose() << "   norm1  "
  //  << r_vec.transpose() << std::endl;
  r_vec = patch.normalized_vals - r_vec;

  if (zncc < inlier_threshold) {
    return kOutlier;
  } else {
    return kInlier;
  }
}
} // namespace dso