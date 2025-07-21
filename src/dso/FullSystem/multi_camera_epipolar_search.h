#pragma once
#include "../camera_model/camera_base.h"
#include "../camera_model/vio_math_0.h"
#include "opt_nodes_def.h"
#include "vio_factors.h"

namespace dso {

class MultiCamera;
class DirectVisualFactor;
struct EstimatorConfig;

class MultiCameraEpipolarSearch {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum State { kVisible, kParallel, kFail, kSuccess, kReject, kUnVisible };
  struct CamData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t target_level;

    Vec3 dir_mid, dir_left, dir_right;
    Vec2 uv_mid, uv_left, uv_right;
    number_t epipolar_px_length;

    Mat4 T10;
    Mat4 T01;

    State state;
  };

  struct MatchRes {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 target_dir;
    Vec2 target_uv;
    Mat4 T01;
    Mat2 R_theta;
    number_t zncc = 0;
    number_t epipolar_grad_cos_theta = 0;
    number_t disparity_cos_theta = 0;
    bool match_success = false;
    bool dir_reject = false;
  };
  struct MultiCamMatchRes {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::array<MatchRes, kCameraNumUsed> cid_to_match_res;
    number_t idp = 0;
    bool tri_success = false;
    size_t match_success_cam_num = 0;
    number_t avg_zncc = 0;
  };

  MultiCameraEpipolarSearch(MultiCamera *cameras,
                            const number_t &OOB_check_cos_theta_threshold,
                            const number_t &cos_grad_epipolar_dir,
                            const EstimatorConfig *estimator_config);

  State FindEpipolarMatch(
      const Point &point, const int &host_cid,
      std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_img,
      const size_t &pid, const number_t &init_rho, const number_t &rho_sigma2,
      const size_t &target_fid,
      std::array<MatchRes, kCameraNumUsed> &cid_to_output, number_t &idp,
      const number_t &search_length_threshold, const bool &is_same_fid,
      const int &intr_level, Mat4 *T10 = nullptr);

  std::array<CamData, kCameraNumUsed> cid_to_cam_data_;
  number_t rad_step_;

private:
  bool InFrame(const Vec2 &uv, const size_t &img_width,
               const size_t &img_height, const int &border);
  bool Triangulate(number_t &idp, const Mat4 &T01, const Vec3 &v0,
                   const Vec3 &v1);

  size_t FindSearchedCid(const number_t *cid_to_epipolar_length);

  std::vector<size_t> FindLocalMaxima(const std::vector<number_t> &zncc_vec);

  void GetTheBestAndSecondScore(const std::vector<number_t> &zncc_vec,
                                const std::vector<size_t> &maxima_index_vec,
                                number_t &best_zncc, number_t &second_zncc);

  // maxSteps may cause the search unable to reach the min and max depth
  int maxSteps_ = 125;
  number_t OOB_check_cos_theta_threshold_ =
      -1; // FOV Threshold cos(75)
          //  number_t cos_grad_epipolar_dir_theta_threshold_ = -1;  // cos(75)

  //  number_t epipolar_length_threshold_ = 10.0;

  //  int search_target_level_ = 1;
  int search_target_level_;
  int opt_target_level_ = 0;

  //  number_t pixel_step_ = 1.0;
  // number_t rad_step_;
  number_t search_zncc_threshold_ = setting_outlierTh_zncc; // 0.7; // 0.9;
  number_t opt_zncc_threshold_ = setting_outlierTh_zncc;    // 0.6;    // 0.8;

  size_t max_iter_ = 5;

  MultiCamera *p_level_cid_to_camera_;
  DirectVisualFactor direct_visual_factor_;
};

} // namespace dso