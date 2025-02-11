#include "depth_filter_DSM.h"
#include "../camera_model//vio_math_0.h"
#include "../camera_model/camera_base.h"
#include "../util/settings.h"
#include "epipolar_match_DSM.h"
#include "estimator_config.h"
#include "multi_camera_epipolar_search.h"
#include "patch.h"
#include <opencv2/opencv.hpp>

//#define _PRINT_DEPTH_FILTER_DETAIL_
#ifdef _PRINT_DEPTH_FILTER_DETAIL_
const size_t target_point_pid = yvr::yvr_vio::kInvalid;
#endif

namespace dso {

DepthFilterDSM::DepthFilterDSM(MultiCamera *p_multi_camera,
                               const EstimatorConfig *p_estimator_config)
    : p_level_to_multi_camera_(p_multi_camera),
      p_estimator_config_(p_estimator_config) {
  px_err_angle_vec_.resize(kCameraNumUsed);
  for (size_t i = 0; i < kCameraNumUsed; ++i) {
    const number_t fx =
        p_level_to_multi_camera_->cid_to_cam[0]->GetParamByIndex(0);
    px_err_angle_vec_[i] = std::atan(px_noise_ / fx);
  }

  p_epipolar_match_dsm_ = new EpipolarMatchDSM(
      p_multi_camera, p_estimator_config_->z_threshold, 1.0);
  p_multi_cam_epipolar_search_ = new MultiCameraEpipolarSearch(
      p_multi_camera, p_estimator_config_->z_threshold, 1.0,
      p_estimator_config);

  cell_size_ = p_estimator_config->detect_cell_size / 2;
  row_cell_num_ = 480 / cell_size_;
  col_cell_num_ = 640 / cell_size_;

  Patch::PatchInit(p_estimator_config_->z_threshold);

  if (480 % cell_size_ != 0) {
    row_cell_num_++;
  }

  if (640 % cell_size_ != 0) {
    col_cell_num_++;
  }

  const size_t &cell_size_per_cam = row_cell_num_ * col_cell_num_;

  cid_to_new_frame_mask_mat_.resize(
      kCameraNumUsed, std::vector<Seed *>(cell_size_per_cam, nullptr));
}

DepthFilterDSM::~DepthFilterDSM() {
  delete p_multi_cam_epipolar_search_;
  delete p_epipolar_match_dsm_;
}

void DepthFilterDSM::ProcessDepthFilter(
    const size_t &cur_fid, const bool &is_first_frame,
    const aligned_vector<aligned_vector<Vec2>> &edge_features,
    const aligned_vector<aligned_vector<Vec2>> &corner_features,
    InitDepthData *p_init_depth_data) {
  // InsertNewFrame(cur_fid, is_first_frame);
  std::vector<Seed *> seed_vec;
  std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_img;

  GetSeeds(frame_vec_, seed_vec);

  //  UpdateSeeds(seed_vec);
  UpdateSeedMultiCam(cid_to_img, seed_vec, false, is_first_frame);

  // int total_new_features = GenerateNewPoints(edge_features, corner_features,
  // p_init_depth_data);

  // reset mask data
  for (size_t cid = 0; cid < kCameraNumUsed; ++cid) {
    for (Seed *&p_seed : cid_to_new_frame_mask_mat_[cid]) {
      p_seed = nullptr;
    }
  }

  seed_vec.clear();

  GetSeeds(frame_vec_.back(), seed_vec);

  //  UpdateSeeds(seed_vec);
  UpdateSeedMultiCam(cid_to_img, seed_vec, false, is_first_frame);
}

void DepthFilterDSM::GetSeeds(std::vector<DF_Frame> &frames,
                              std::vector<Seed *> &seeds_vec) {
  seeds_vec.clear();
  for (size_t i = 0; i < frames.size(); ++i) {
    GetSeeds(frames[i], seeds_vec);
  }
}

void DepthFilterDSM::GetSeeds(DF_Frame &frame, std::vector<Seed *> &seeds_vec) {
  for (size_t i = 0; i < frame.seed_vec.size(); ++i) {
    Seed &seed = frame.seed_vec[i];
    if (seed.state == Seed::kSeedInvalid || seed.state == Seed::kSeedConverge) {
      continue;
    }
    seeds_vec.emplace_back(&(frame.seed_vec[i]));
  }
}

void DepthFilterDSM::UpdateSeedMultiCam(
    std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_img,
    std::vector<Seed *> &seeds_vec, bool mask_cur_frame, bool is_first_frame) {
  aligned_vector<Vec3> cid_to_twc(kCameraNumUsed);
  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    cid_to_twc[target_cid] = Vec3::Zero();
  }

#ifdef _VIO_PRINT_DEBUG_
  size_t total_seed_num = seeds_vec.size();
  size_t masked_seed_num = 0;
#endif

  for (int i = 0; i < seeds_vec.size(); ++i) {
    Seed *seed = seeds_vec[i];
    if (seed->state == Seed::kSeedInvalid ||
        seed->state == Seed::kSeedConverge) {
      printf("seed is invalid or converge in UpdateSeedMultiCam\n");
      continue;
    }

    Point point;
    size_t cid = 0;
    Vec2 uv;
    bool is_corner;
    bool success = point.pyramid_patch.SetFromImg(
        cid_to_img[cid], uv, cid, is_corner, p_level_to_multi_camera_);

    number_t res_idp;
    std::array<MultiCameraEpipolarSearch::MatchRes, kCameraNumUsed>
        cid_to_output;
    Point pt;
    MultiCameraEpipolarSearch::State state =
        p_multi_cam_epipolar_search_->FindEpipolarMatch(
            pt, 1, cid_to_img, seed->pid, seed->rho, seed->sigma2, cur_fid_,
            cid_to_output, res_idp,
            is_first_frame ? -1 : p_estimator_config_->search_length_threshold,
            true);

    if (state == MultiCameraEpipolarSearch::kReject) {
      continue;
    } else if (state == MultiCameraEpipolarSearch::kUnVisible) {
    } else if (state == MultiCameraEpipolarSearch::kFail) {
    } else if (state == MultiCameraEpipolarSearch::kSuccess) {
    } else {
      printf("you should never see this\n");
      std::exit(1);
    }
  }
}

} // namespace dso