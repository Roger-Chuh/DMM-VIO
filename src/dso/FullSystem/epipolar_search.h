#pragma once
#include "opt_nodes_def.h"
#include "vio_factors.h"

#define _SHOW_EPIPOLAR_SEARCH_DETAIL_

#ifdef _SHOW_EPIPOLAR_SEARCH_DETAIL_
#include <opencv2/opencv.hpp>
#endif
namespace dso {

class MultiCamera;
class DirectVisualFactor;
struct EstimatorConfig;

class EpipolarSearch {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum EpipolarSearchStatus { Reject, Fail, Success };

  struct SearchRes {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec2 px;
    Vec2 grad;
    Vec3 dir;
    number_t idp;
    number_t zncc;
    number_t r2;
  };

  EpipolarSearch(MultiCamera *cameras, const int &img_width,
                 const int &img_height,
                 const EstimatorConfig *p_estimator_config);

private:
  //  Mat2 GetAffineMatrix(const Mat4& T_cur_ref, const Patch& patch, const
  //  size_t& host_cid, const size_t& target_cid,
  //                       CameraBase::Ptr target_camera, const Vec2& cur_px,
  //                       const number_t& idp);

  //  int GetPatchZNCCScore(const Patch::Matrix2P_B& target_patch_uv, const
  //  Patch& patch,
  //                        std::shared_ptr<AlgsImage> target_img, number_t&
  //                        zncc, const Vec2& host_affine, const Vec2&
  //                        target_affine, const number_t& photometry_scale,
  //                        number_t& r2);

  int TriangulateWithoutCheckTheta(number_t &idp, const Mat4 &T01,
                                   const Vec3 &v0, const Vec3 &v1);

  std::vector<size_t>
  FindLocalMaxima(const std::vector<SearchRes> &search_result_vec);

  bool IsInFrame(const Vec2 &uv, const size_t &boundary, const size_t &level) {
    if (uv[0] >= boundary && uv[0] < img_width_ / (1 << level) - boundary &&
        uv[1] >= boundary && uv[1] < img_height_ / (1 << level) - boundary) {
      return true;
    }
    return false;
  }

  size_t img_width_ = 0;
  size_t img_height_ = 0;

  MultiCamera *level_cid_to_camera_;
  DirectVisualFactor direct_visual_factor_;

  number_t fov_z_threshold_;

  int half_search_step_ = 5;
  number_t grad_check_threshold_ = 0.34202; // cos(70)
  number_t rad_step_ = 0.7 * 0.2438 / 180 * M_PI;
  number_t zncc_threshold_ = 0.8;
};

} // namespace dso