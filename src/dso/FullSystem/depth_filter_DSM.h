#pragma once
#include "../camera_model/vio_def.h"
#include "../camera_model/vio_math_0.h"
#include "depth_filter_DSM_def.h"
#include "estimator_config.h"
#include "opt_nodes_def.h"
#include <vector>
//#include "grid_cell.h"

namespace dso {

// class OptDataBase;
// class DataManager;
class MultiCamera;
class EpipolarMatchDSM;
class MultiCameraEpipolarSearch;
struct EstimatorConfig;
struct InitDepthData;

class DepthFilterDSM {
public:
  friend class EstimatorInterface;

  DepthFilterDSM(MultiCamera *p_multi_camera,
                 const EstimatorConfig *p_estimator_config);

  ~DepthFilterDSM();

  void ProcessDepthFilter(
      const size_t &cur_fid, const bool &is_first_frame,
      const aligned_vector<aligned_vector<Vec2>> &edge_features,
      const aligned_vector<aligned_vector<Vec2>> &corner_features,
      InitDepthData *p_init_depth_data = nullptr);

  void Reset();
  void DeleteFrame(const size_t &fid_to_delete);

  void RotateSeedPreFrameVM(const Mat3 &Rw0);

private:
  void InsertNewFrame(const size_t &fid, const bool &is_first_frame);
  void GetSeeds(std::vector<DF_Frame> &frames, std::vector<Seed *> &seeds_vec);
  void GetSeeds(DF_Frame &frame, std::vector<Seed *> &seeds_vec);
  //  void UpdateSeeds(std::vector<Seed*>& seeds_vec);
  //  void UpdateSeed(Seed* seed, const size_t& target_cid);
  void UpdateSeedMultiCam(
      std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_img,
      std::vector<Seed *> &seeds_vec, bool mask_cur_frame = false,
      bool is_first_frame = false);
  int GenerateNewPoints(
      const aligned_vector<aligned_vector<Vec2>> &edgelet_features,
      const aligned_vector<aligned_vector<Vec2>> &corner_features,
      InitDepthData *p_init_depth_data = nullptr);
#if CODE_ACC_SETFROMIMG
  int GenerateNewPoints_ACC(
      const aligned_vector<aligned_vector<Vec2>> &edgelet_features,
      const aligned_vector<aligned_vector<Vec2>> &corner_features,
      InitDepthData *p_init_depth_data = nullptr);
#endif

  //  std::vector<std::pair<size_t, number_t>> CalCosAngle(Seed* p_seed);

  MultiCamera *p_level_to_multi_camera_;
  const EstimatorConfig *p_estimator_config_;

  bool is_first_frame_ = false;
  bool is_keyframe_ = false;
  bool is_visual_keyframe_ = false;

  size_t total_init_seed_num_ = 0;
  size_t total_converged_seed_num_ = 0;

  number_t px_noise_ = 0.5;
  number_t vm_err2_threshold_ = 2.5 / 235.0 * 2.5 / 235.0;
  number_t vm_inlier_ratio_threshold_ = 0.6;
  number_t avg_vm_err2_threshold_ = 0.5 / 235.0 * 0.5 / 235.0;
  std::vector<number_t> px_err_angle_vec_;

  number_t seed_convergence_sigma2_threshold_ = 200.0;

  std::vector<Seed *> cur_frame_converged_seed_vec_;

  size_t cur_fid_;
  std::vector<DF_Frame> frame_vec_;

  EpipolarMatchDSM *p_epipolar_match_dsm_;
  MultiCameraEpipolarSearch *p_multi_cam_epipolar_search_;

  size_t cell_size_;
  size_t row_cell_num_;
  size_t col_cell_num_;
  std::vector<std::vector<Seed *>> cid_to_new_frame_mask_mat_;

  size_t removed_fid_in_depth_filter_ = kInvalid;
  std::vector<size_t> deleted_pid_vec_;
};

} // namespace dso
