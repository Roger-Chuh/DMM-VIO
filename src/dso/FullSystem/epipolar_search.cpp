#include "epipolar_search.h"
#include "../camera_model/camera_base.h"
#include "../camera_model/vio_math_0.h"
#include "estimator_config.h"

#include <iostream>

#include <opencv2/opencv.hpp>

namespace dso {
EpipolarSearch::EpipolarSearch(MultiCamera* cameras, const int& img_width, const int& img_height,
                               const EstimatorConfig* p_estimator_config) {
  level_cid_to_camera_ = cameras;
  img_width_ = img_width;
  img_height_ = img_height;

  fov_z_threshold_ = p_estimator_config->z_threshold;

  direct_visual_factor_.check_depth_ = false;
};

/*
Mat2 EpipolarSearch::GetAffineMatrix(const Mat4 &T_cur_ref, const Patch &patch,
const size_t &host_cid, const size_t &target_cid, CameraBase::Ptr target_camera,
const Vec2 &cur_px, const number_t &idp) { const Vec3 &xyz_du_ref =
patch.xyz_du_ref; const Vec3 &xyz_dv_ref = patch.xyz_dv_ref;

  Vec2 px_du, px_dv, px_middle;
  target_camera->Project(T_cur_ref.block<3, 3>(0, 0) * xyz_du_ref +
T_cur_ref.block<3, 1>(0, 3) * idp, px_du);

  target_camera->Project(T_cur_ref.block<3, 3>(0, 0) * xyz_dv_ref +
T_cur_ref.block<3, 1>(0, 3) * idp, px_dv);

  Mat2 A_cur_ref;
  A_cur_ref.col(0) = (px_du - cur_px) / Patch::HALF_PATCH_SIZE_B;
  A_cur_ref.col(1) = (px_dv - cur_px) / Patch::HALF_PATCH_SIZE_B;
  return A_cur_ref;
}
*/

/*
int EpipolarSearch::GetPatchZNCCScore(const Patch::Matrix2P_B &target_patch_uv,
const Patch &patch, std::shared_ptr<AlgsImage> target_img, number_t &zncc, const
Vec2 &host_affine, const Vec2 &target_affine, const number_t &photometry_scale,
number_t &r2) { bool in_range_row, in_range_col; in_range_row =
target_patch_uv.row(0).minCoeff() >= 1 && target_patch_uv.row(0).maxCoeff() <
img_width_ - 1; in_range_col = target_patch_uv.row(1).minCoeff() >= 1 &&
target_patch_uv.row(1).maxCoeff() < img_height_ - 1; Patch::ArrayV_B target_val;
  if (!in_range_row || !in_range_col) {
    return -1;
  }

  patch.GetBigPatchValues(target_patch_uv, target_img, target_val);

  number_t mean = target_val.sum() / (number_t)Patch::PATCH_SIZE_B;

  r2 = ((target_val - target_affine[1]) -
        photometry_scale * std::exp(target_affine[0] - host_affine[0]) *
(patch.vals_Big - host_affine[1])) .matrix() .squaredNorm();

  target_val -= mean;
  number_t patch_sigma = std::sqrt(target_val.square().sum());
  if (patch_sigma < 5) {
    return -2;
  }
  target_val /= patch_sigma;
  zncc = patch.normalized_vals_Big.transpose().matrix() * target_val.matrix();
  return 1;
}
*/

int EpipolarSearch::TriangulateWithoutCheckTheta(number_t& idp, const Mat4& T01, const Vec3& v0, const Vec3& v1) {
  Mat63 A = Mat63::Zero();
  Vec6 b = Vec6::Zero();
  A.block<3, 3>(0, 0) = Skew(v0);
  A.block<3, 3>(3, 0) = Skew(v1) * T01.block<3, 3>(0, 0).transpose();
  b.segment<3>(3) = Skew(v1) * T01.block<3, 3>(0, 0).transpose() * T01.block<3, 1>(0, 3);
  Vec3 s = A.transpose() * b;
  Mat3 AA = A.transpose() * A;
  Vec3 xyz = AA.ldlt().solve(s);

  if (xyz.z() < 0) {
    return -1;
  }
  idp = 1.0 / xyz.norm();

  return 1;
}

std::vector<size_t> EpipolarSearch::FindLocalMaxima(const std::vector<SearchRes>& search_result_vec) {
  std::vector<size_t> maxima_index_vec;
  if (search_result_vec[0].zncc > search_result_vec[1].zncc) {
    maxima_index_vec.emplace_back(0);
  }

  for (int i = 1; i < search_result_vec.size() - 1; ++i) {
    if (search_result_vec[i].zncc > search_result_vec[i - 1].zncc &&
        search_result_vec[i].zncc > search_result_vec[i + 1].zncc) {
      maxima_index_vec.emplace_back(i);
    }
  }

  size_t last_index = search_result_vec.size() - 1;
  if (search_result_vec[last_index].zncc > search_result_vec[last_index - 1].zncc) {
    maxima_index_vec.emplace_back(last_index);
  }

  return maxima_index_vec;
}

}  // namespace dso