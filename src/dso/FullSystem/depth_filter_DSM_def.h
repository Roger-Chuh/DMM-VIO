#pragma once
#include "../camera_model//vio_def.h"
#include <vector>
//#include "../vio_opt/opt_nodes_def.h"
#include "epipolar_match_DSM.h"

namespace dso {
// class DepthGrid;

struct DuplicateVM {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  size_t target_frame_timestamp;
  size_t target_frame_cid;
  std::vector<number_t> idp_vec;
  aligned_vector<Vec3> vm_dir_vec;
  std::vector<number_t> tau_inverse_vec;
};

struct PreFrameVM {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  size_t target_frame_timestamp;
  size_t target_cid;
  Vec3 dir;
  number_t zncc;
  Vec3 dir_in_world;  // for outlier check
  Vec3 twc;           // for outlier check

  Mat3 vm_sigma;
};

struct Seed {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum SeedState { kSeedInvalid, kSeedInited, kSeedUpdate, kSeedConverge };
  size_t pid;
  Vec2 detect_uv;  // host uv in level0
  SeedState state;

  number_t sigma2 = 1;
  number_t z_range = -1;
  number_t a = 10;
  number_t b = 10;
  number_t rho;
  size_t fail_count = 0;

  int OOB_count = 0;

  std::vector<DuplicateVM> duplicate_vm_vec;
  std::vector<PreFrameVM> pre_keyframe_vm_vec;
  std::vector<PreFrameVM> pre_normalframe_vm_vec;
};

struct DF_Frame {
  size_t fid;
  std::vector<Seed> seed_vec;

  static void InitSeedDepth(Seed& seed, const bool& has_init_depth, const number_t& init_depth);
};
}  // namespace dso