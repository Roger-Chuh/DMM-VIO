#pragma once
#include "../camera_model/vio_def.h"
#include "patch.h"
#include <memory>

namespace dso {
struct InitDepthData {
  struct BlockDepthData {
    int total_point_num = 0;
    number_t total_depth = 0;
    number_t avg_depth = -1;
  };

  struct CamDepthData {
    int col_block_size = 0;
    int row_block_size = 0;
    std::vector<BlockDepthData> bid_to_depth_data;
  };

  std::vector<CamDepthData> cid_to_block_data;
};
struct Point {
public:
  enum TYPE { kEdgelet, kCorner };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  size_t host_vid = kInvalid;
  Vec3 n = Vec3::Constant(std::nan("")); // bearing vector in host camera frame
  number_t rho = 0; //!< Inverse distance in host camera frame
  number_t rho_align1d = 0;
  Vec3 xyz = Vec3::Zero();
  PyramidPatch pyramid_patch;

  //  int inlier_vm_num = 0;
  int outlier_vm_num = 0;

  TYPE type;

  bool has_valid_vm = false;

  number_t rho_prior = 0;
  number_t H22_prior = 0;
  number_t b2_prior = 0;

private:
  // backup
  number_t backup_rho = 0; //!< Inverse depth in host frame
  number_t backup_rho_align1d = 0;
  Vec3 backup_xyz = Vec3::Constant(std::nan(""));
};

} // namespace dso