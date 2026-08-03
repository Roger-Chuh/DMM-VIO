#include "epipolar_match_DSM.h"

namespace dso {

EpipolarMatchDSM::EpipolarMatchDSM(MultiCamera* p_level_to_multi_camera, const number_t& fov_threshold,
                                   const number_t& epipolar_grad_threshold)
    : p_level_to_multi_camera_(p_level_to_multi_camera) {
  direct_visual_factor_.check_depth_ = false;

  OOB_check_cos_theta_threshold_ = fov_threshold;
  cos_grad_epipolar_dir_theta_threshold_ = epipolar_grad_threshold;
}

bool EpipolarMatchDSM::InFrame(const Vec2& uv, const std::shared_ptr<AlgsImage>& p_img, int border) {
  if (uv[0] >= border && uv[0] < p_img->width - border && uv[1] >= border && uv[1] < p_img->height - border) {
    return true;
  }
  return false;
}

EpipolarMatchDSM::TriResult EpipolarMatchDSM::TriangulateWithCheckTheta(number_t& idp, const Mat4& T01, const Vec3& v0,
                                                                        const Vec3& v1) {
  Mat63 A = Mat63::Zero();
  Vec6 b = Vec6::Zero();
  A.block<3, 3>(0, 0) = Skew(v0);
  A.block<3, 3>(3, 0) = Skew(v1) * T01.block<3, 3>(0, 0).transpose();
  b.segment<3>(3) = Skew(v1) * T01.block<3, 3>(0, 0).transpose() * T01.block<3, 1>(0, 3);
  Vec3 s = A.transpose() * b;
  Mat3 AA = A.transpose() * A;
  Vec3 xyz = AA.ldlt().solve(s);

  if (xyz.z() < 0) {
    return EpipolarMatchDSM::kNegIDP;
  }
  idp = 1.0 / xyz.norm();

  Vec3 v1_0 = T01.block<3, 3>(0, 0) * v1;
  number_t cos_theta = v1_0.dot(v0);
  if (cos_theta > 0.9999) {
    return EpipolarMatchDSM::kParallel;
  }

  return EpipolarMatchDSM::kSuccess;
}

std::vector<size_t> EpipolarMatchDSM::FindLocalMaxima(const std::vector<SearchResult>& search_result_vec) {
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