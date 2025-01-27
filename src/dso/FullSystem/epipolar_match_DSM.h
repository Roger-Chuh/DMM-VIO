#pragma once
#include "../camera_model//vio_math_0.h"
#include "../camera_model/vio_def.h"
#include "opt_nodes_def.h"
#include "vio_factors.h"
//#include "../vio_basic/vio_factors.h"

#include <opencv2/opencv.hpp>

namespace dso {

class EpipolarMatchDSM {
public:
  enum MatchResult {
    kMatchGood = 0,
    kMatchDuplicate,
    kMatchFail,
    kMatchReject
  };

  enum TriResult { kSuccess = 0, kParallel, kNegIDP };

  struct SearchResult {
    Vec3 target_dir;
    Vec2 target_uv;
    number_t idp;
    number_t zncc;
    TriResult tri_result;
  };

  EpipolarMatchDSM(MultiCamera *p_level_to_multi_camera,
                   const number_t &fov_threshold,
                   const number_t &epipolar_grad_threshold);

  bool print_detail_ = false;

  bool is_end_match_ = false;

private:
  TriResult TriangulateWithCheckTheta(number_t &idp, const Mat4 &T01,
                                      const Vec3 &v0, const Vec3 &v1);

  std::vector<size_t>
  FindLocalMaxima(const std::vector<SearchResult> &search_result_vec);

  bool InFrame(const Vec2 &uv, const std::shared_ptr<AlgsImage> &p_img,
               int border);

  int maxSteps_ = 25;
  number_t rad_step_ = 0.7 * 0.2438 / 180 * M_PI;
  number_t OOB_check_cos_theta_threshold_ = -1; // FOV Threshold cos(80)
  number_t cos_grad_epipolar_dir_theta_threshold_ =
      -1; // duplicate checking threshold cos(80)
  MultiCamera *p_level_to_multi_camera_;
  DirectVisualFactor direct_visual_factor_;
};

} // namespace dso