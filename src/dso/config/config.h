#pragma once

#include <dirent.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
//#include <toml/parser.hpp>
//#include <toml/get.hpp>
#include "../camera_model//vio_def.h"
#include "../camera_model/calib_def.h"
#include "toml.hpp"
#include <Eigen/Core>
namespace dso::CalibIO {

enum BoardType { DOT = 0, APRILTAG = 1 };
enum CalibStage { GRAY_VI = 0, RGB_VI = 1, GRAY_RGB = 2 };

struct DotConfig {
  std::string dot_bin_file = "dotInfo.bin";
  std::string dot_bin_file_gray = "dotInfo.bin";
  std::string dot_bin_file_rgb = "dotInfo.bin";
  bool show_detect_picture = false;
  bool black_dot = false;
  double adaptive_thresh = 0.9;
  int window_ratio = 10;
  double grid_spacing = 0.03;
  Eigen::Vector2i grid_size{30, 90};
  int grid_seed = 92;
  int unique_size = 6;
  double conic_min_area = 4.0;
  double conic_symmetry = 0.25;
  double conic_min_aspect = 0.3;
  int min_area_point_num = 25;
  bool skip_detection = false;
};

struct ApriltagConfig {
  bool show_tag_detect = false;
  // same size board; continue id; 0-99, 100-199, 200-299;
  std::string apriltag_bin_file = "aptriltag.bin";
  std::string apriltag_bin_file_gray = "aptriltag.bin";
  std::string apriltag_bin_file_rgb = "aptriltag.bin";
  int black_board = 1;
  int rows = 10;
  int cols = 10;
  int apriltag_start_id = 0;

  double tag_size = 0.08;  // unit: m
  double tag_gap = 0.2;    // ratio

  // not config, need init
  int pattern_num = 3;
  int one_board_tags = rows * cols;
  double min_border_distance = 4.0;
  int black_tag_border = 1;
  int min_tags_for_valid_obs = 2;
  int min_points_obsetved = 4;
  double max_subpix_displacement2 = 1;
  int board_offset = 0;
  bool do_sub_pixel_refine = true;
};

class ConfigData {
 public:
  // main config
  std::vector<int> calib_cid = {0, 1, 2, 3};
  std::vector<int> gray_cid = {0, 1, 2, 3};
  std::vector<int> rgb_cid = {0, 1, 2, 3};
  int64_t frame_mini_gap = 100000;  // ns
  int64_t frame_gap = 30000000;     // ns
  int log_level = 0;
  std::string save_folder = "results";
  std::string intr_folder = "intrinsic";
  std::string extr_folder = "extrinsic";
  int image_width = 640;
  int image_height = 480;
  bool is_rgb = false;
  bool is_gray_rgb = false;
  CalibStage calib_stage = GRAY_VI;
  bool do_detection_only = false;
  // dataset config
  std::string dataSet = "path of dataset";
  std::string cam_folder_prefix = "Camera";
  std::string cam_folder_subfix = "images";
  std::string cam_json = "data.json";
  BoardType board_type = DOT;
  int plate_num = 3;
  int detect_thread_num = 5;
  std::string imu_folder_name = "IMU0";
  std::string imu_json_name = "data.json";
  // imu config

  // dot && apriltag config
  DotConfig dot_config;
  ApriltagConfig apriltag_config;

  // calib check config
  std::vector<int> calib_check_cid = {0, 1, 2, 3};
  std::string calib_file_name;
  std::string calib_check_result_folder;
  bool show_calib_check_img = false;

  // vi_calib config
  VI_Config vi_config;

  std::string self_calib_bin_file_points;
  std::string self_calib_bin_file_lines;
  bool skip_detection;
  bool use_prior_marker_map;
  std::string prior_marker_map_path;
  std::string marker_map_vm_bin_file;
  // generate by init
  std::unordered_map<int, std::string> camPaths;
  std::unordered_map<int, std::string> camJsonPaths;
  std::string imuJsonPath;
  std::string resultPath;
  std::string extrinsicPath;
  std::string intrinsicPath;
  std::string binFilePath;
  std::string binFilePath_gray;
  std::string binFilePath_rgb;

  ConfigData() = default;
  explicit ConfigData(const std::string& config_file_path);
  VI_Config GetVIConfig();

  void InitConfigs();

 private:
  bool LoadConfig(const std::string& config_path);
  void CheckConfig();
};

}  // namespace dso::CalibIO
