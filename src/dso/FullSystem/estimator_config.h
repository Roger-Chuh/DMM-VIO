#pragma once
#include "../camera_model/vio_def.h"
namespace dso {
struct EstimatorConfig {
  enum MODE { kVIO, kOnlineCalib };
  size_t kImgWidth = kInvalid;
  size_t kImgHeight = kInvalid;

  MODE mode = kVIO;

  // voxel map
  const number_t voxel_size = 0.16;
  const number_t max_search_dis = 500;
  const int max_points_in_one_voxel = 2;

  //  const number_t z_threshold = 0.422618262;  // cos(65) degree
  const number_t z_threshold = 0.1;
  const number_t point_vm_entropy_threshold = 5.0;

  // max sw size
  const int max_sw_num_init = 15;
  const int continuous_sw_num = 5; // >= 2
  const int vkf_sw_num = 2;
  const int max_sw_num = continuous_sw_num + vkf_sw_num;

  const int max_pre_keyframe_num_online_calib = 100;
  const int max_pre_keyframe_num_vio = 51;
  int max_pre_keyframe_num = max_pre_keyframe_num_vio;

  // skip first frame when visual init failure
  const uint8_t skip_first_frame_when_visual_init_fail = 10;

  // for feature detect
  const int detect_min_grad = 8;
  //  const int detect_min_grad = 8;
  const int detect_max_grad = 64;
  //  const number_t detect_min_ratio = 0.16;
  //  const number_t detect_max_ratio = 0.32;
  const number_t detect_min_ratio = 0.4;
  const number_t detect_max_ratio = 0.6;
  //  const number_t znssd_grad_check_pixel_radius = 2.0;
  //  const number_t znssd_grad_check_threshold = 0.9;  // zncc score

  const uint8_t detect_cell_size = 25; // 16;

  const uint8_t edgelet_detect_level = 1;
  const int desired_edgelet_per_cam = 75;
  //  const number_t grad_ratio_threshold = 0.15;

  const uint8_t corner_detect_level = 0;

  const int desired_corner_per_cam_online_calib = 200;
  const int desired_corner_per_cam_vio = 75;
  int desired_corner_per_cam = desired_corner_per_cam_vio;

  const number_t outlier_threshold_in_align = 1.5; // pixel
  const number_t outlier_threshold_in_align_dir =
      outlier_threshold_in_align / 235;
  const number_t outlier_threshold2_in_align_dir =
      outlier_threshold_in_align_dir * outlier_threshold_in_align_dir;

  // DSM detector
  const int num_blocks = 50;
  const number_t grad_threshold = 7.0;
  //  const int desired_point_num_each_cam = 150;

  // motion model
  const number_t motion_model_alpha = 0.5;

  // Threshold for motion blur
  const number_t vkf_blur_pixel_threshold = 6.0; // pixel

  // Epipolar search length threshold
  const int search_level = 1;
  const number_t pixel_step = 1.0;
  const number_t search_length_threshold = 20.0;

  // Depth Init Block
  const int init_depth_col_block_num = 4;
  const int init_depth_row_block_num = 3;
};
} // namespace dso