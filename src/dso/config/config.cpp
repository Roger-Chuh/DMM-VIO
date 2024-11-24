#include "config.h"
//#include "macro_define.h"
#include <stdexcept>

namespace dso::CalibIO {

ConfigData::ConfigData(const std::string &config_path) {
  LoadConfig(config_path);
  InitConfigs();
  CheckConfig();
}

VI_Config ConfigData::GetVIConfig() { return vi_config; }
bool ConfigData::LoadConfig(const std::string &config_path) {
  try {
    const toml::value &data = toml::parse(config_path);

    // main config
    const toml::value &main_config = toml::find(data, "main");
    calib_cid.clear();
    auto cid_value_list = toml::find(main_config, "calib_cid").as_array();
    for (auto cid_value : cid_value_list) {
      calib_cid.emplace_back(cid_value.as_integer());
    }
    gray_cid.clear();
    auto gray_cid_value_list = toml::find(main_config, "gray_cid").as_array();
    for (auto gray_value : gray_cid_value_list) {
      gray_cid.emplace_back(gray_value.as_integer());
    }
    rgb_cid.clear();
    auto rgb_cid_value_list = toml::find(main_config, "rgb_cid").as_array();
    for (auto rgb_value : rgb_cid_value_list) {
      rgb_cid.emplace_back(rgb_value.as_integer());
    }
    std::sort(calib_cid.begin(), calib_cid.end());
    std::sort(gray_cid.begin(), gray_cid.end());
    std::sort(rgb_cid.begin(), rgb_cid.end());
    frame_mini_gap = toml::find<int64_t>(main_config, "frame_mini_gap");
    frame_gap = toml::find<int64_t>(main_config, "frame_gap");
    log_level = toml::find<int>(main_config, "log_level");
    save_folder = toml::find<std::string>(main_config, "save_folder");
    intr_folder = toml::find<std::string>(main_config, "intr_folder");
    extr_folder = toml::find<std::string>(main_config, "extr_folder");
    image_width = toml::find<int>(main_config, "image_width");
    image_height = toml::find<int>(main_config, "image_height");
    is_rgb = toml::find<bool>(main_config, "is_rgb");
    is_gray_rgb = toml::find<bool>(main_config, "is_gray_rgb");
    calib_stage =
        static_cast<CalibStage>(toml::find<int>(main_config, "calib_stage"));
    do_detection_only = toml::find<bool>(main_config, "do_detection_only");
    vi_config.is_rgb = is_rgb;
    vi_config.is_gray_rgb = is_gray_rgb;

    if (calib_stage == GRAY_VI) {
      if (!rgb_cid.empty() || gray_cid.empty() ||
          gray_cid.size() != calib_cid.size()) {
        std::cout << "cid_list conflicts with calib_stage" << std::endl;
        std::exit(-1);
      }
    }
    if (calib_stage == RGB_VI) {
      if (!rgb_cid.empty() || gray_cid.empty() ||
          gray_cid.size() != calib_cid.size()) {
        std::cout << "cid_list conflicts with calib_stage" << std::endl;
        std::exit(-1);
      }
    }
    if (calib_stage == GRAY_RGB) {
      if (rgb_cid.empty() || gray_cid.empty()) {
        std::cout << "cid_list conflicts with calib_stage" << std::endl;
        std::exit(-1);
      }
    }
    // dataset config
    const toml::value &dataset_config = toml::find(data, "dataset");
    dataSet = toml::find<std::string>(dataset_config, "dataSet");
    self_calib_bin_file_points =
        toml::find<std::string>(dataset_config, "self_calib_bin_file_points");
    self_calib_bin_file_lines =
        toml::find<std::string>(dataset_config, "self_calib_bin_file_lines");
    skip_detection = toml::find<bool>(dataset_config, "skip_detection");
    use_prior_marker_map =
        toml::find<bool>(dataset_config, "use_prior_marker_map");
    prior_marker_map_path =
        toml::find<std::string>(dataset_config, "prior_marker_map_path");
    marker_map_vm_bin_file =
        toml::find<std::string>(dataset_config, "marker_map_vm_bin_file");
    cam_folder_prefix =
        toml::find<std::string>(dataset_config, "cam_folder_prefix");
    cam_folder_subfix =
        toml::find<std::string>(dataset_config, "cam_folder_subfix");
    cam_json = toml::find<std::string>(dataset_config, "cam_json");
    board_type =
        static_cast<BoardType>(toml::find<int>(dataset_config, "board_type"));
    plate_num = toml::find<int>(dataset_config, "plate_num");
    detect_thread_num = toml::find<int>(dataset_config, "detect_thread_num");

    // imu config
    imu_folder_name =
        toml::find<std::string>(dataset_config, "imu_folder_name");
    imu_json_name = toml::find<std::string>(dataset_config, "imu_json_name");

    // board config
    if (board_type == DOT) {
      std::cout << "use dot plate\n" << std::endl;
      dot_config.skip_detection = skip_detection;
      const toml::value &dot_plate_config = toml::find(data, "dot_plate");
      dot_config.dot_bin_file =
          toml::find<std::string>(dot_plate_config, "dot_bin_file");
      dot_config.dot_bin_file_gray =
          toml::find<std::string>(dot_plate_config, "dot_bin_file_gray");
      dot_config.dot_bin_file_rgb =
          toml::find<std::string>(dot_plate_config, "dot_bin_file_rgb");
      dot_config.show_detect_picture =
          toml::find<bool>(dot_plate_config, "show_detect_picture");
      dot_config.black_dot = toml::find<bool>(dot_plate_config, "black_dot");
      dot_config.adaptive_thresh =
          toml::find<double>(dot_plate_config, "adaptive_thresh");
      dot_config.window_ratio =
          toml::find<int>(dot_plate_config, "window_ratio");
      dot_config.grid_spacing =
          toml::find<double>(dot_plate_config, "grid_spacing");
      auto grid_value = toml::find(dot_plate_config, "grid_size").as_array();
      if (grid_value.size() != 2) {
        std::cout << "dot plate grid_size isn't 2 value\n" << std::endl;
        std::abort();
      }
      dot_config.grid_size = Eigen::Vector2i(grid_value[0].as_integer(),
                                             grid_value[1].as_integer());
      dot_config.grid_seed = toml::find<int>(dot_plate_config, "grid_seed");
      dot_config.unique_size = toml::find<int>(dot_plate_config, "unique_size");
      dot_config.conic_min_area =
          toml::find<double>(dot_plate_config, "conic_min_area");
      dot_config.conic_symmetry =
          toml::find<double>(dot_plate_config, "conic_symmetry");
      dot_config.conic_min_aspect =
          toml::find<double>(dot_plate_config, "conic_min_aspect");
      dot_config.conic_min_aspect =
          toml::find<double>(dot_plate_config, "conic_min_aspect");
      dot_config.min_area_point_num =
          toml::find<int>(dot_plate_config, "min_area_point_num");
    } else if (board_type == APRILTAG) {
      std::cout << "use aprilTag plate\n" << std::endl;
      const toml::value &apriltag_plate_config =
          toml::find(data, "apriltag_plate");
      apriltag_config.show_tag_detect =
          toml::find<bool>(apriltag_plate_config, "show_tag_detect");
      apriltag_config.apriltag_bin_file =
          toml::find<std::string>(apriltag_plate_config, "apriltag_bin_file");
      apriltag_config.apriltag_bin_file_gray = toml::find<std::string>(
          apriltag_plate_config, "apriltag_bin_file_gray");
      apriltag_config.apriltag_bin_file_rgb = toml::find<std::string>(
          apriltag_plate_config, "apriltag_bin_file_rgb");
      apriltag_config.black_board =
          toml::find<int>(apriltag_plate_config, "black_board");
      apriltag_config.rows = toml::find<int>(apriltag_plate_config, "rows");
      apriltag_config.cols = toml::find<int>(apriltag_plate_config, "cols");
      apriltag_config.apriltag_start_id =
          toml::find<int>(apriltag_plate_config, "apriltag_start_id");
      apriltag_config.tag_size =
          toml::find<double>(apriltag_plate_config, "tag_size");
      apriltag_config.tag_gap =
          toml::find<double>(apriltag_plate_config, "tag_gap");
      apriltag_config.pattern_num =
          toml::find<int>(apriltag_plate_config, "pattern_num");
      apriltag_config.one_board_tags =
          apriltag_config.rows * apriltag_config.cols;
    } else {
      std::cout << "unknown board type, abort!\n" << std::endl;
      std::abort();
    }

    // calib check config
    const toml::value &calib_check_config = toml::find(data, "calib_check");
    calib_check_cid.clear();
    auto calib_check_cid_value_list =
        toml::find(calib_check_config, "calib_check_cid").as_array();
    for (auto cid_value : calib_check_cid_value_list) {
      calib_check_cid.emplace_back(cid_value.as_integer());
    }
    calib_file_name =
        toml::find<std::string>(calib_check_config, "calib_file_name");
    calib_check_result_folder = toml::find<std::string>(
        calib_check_config, "calib_check_result_folder");
    show_calib_check_img =
        toml::find<bool>(calib_check_config, "show_calib_check_img");

    // vi_calib config
    const toml::value &vi_calib_config = toml::find(data, "vi_calib");
    vi_config.dt = toml::find<double>(vi_calib_config, "spline_dt");
    vi_config.dt_rgb = toml::find<double>(vi_calib_config, "spline_dt_rgb");
    vi_config.shift_imu_data =
        toml::find<bool>(vi_calib_config, "shift_imu_data");
    vi_config.skip_first_frames_num =
        toml::find<int>(vi_calib_config, "skip_first_frames_num");
    vi_config.high_weight_knot_ratio =
        toml::find<double>(vi_calib_config, "high_weight_knot_ratio");
    vi_config.power = toml::find<double>(vi_calib_config, "power");
    vi_config.g_norm = toml::find<double>(vi_calib_config, "g_norm");
    vi_config.use_sim = toml::find<bool>(vi_calib_config, "use_sim");
    vi_config.noise_level = toml::find<double>(vi_calib_config, "noise_level");
    vi_config.disturb_level =
        toml::find<double>(vi_calib_config, "disturb_level");
    vi_config.fix_extr = toml::find<bool>(vi_calib_config, "fix_extr");
    vi_config.use_intr_rot_factor =
        toml::find<bool>(vi_calib_config, "use_intr_rot_factor");
    vi_config.intr_rot_factor_weight =
        toml::find<double>(vi_calib_config, "intr_rot_factor_weight");
    vi_config.resolution_difference_weight =
        toml::find<double>(vi_calib_config, "resolution_difference_weight");
    vi_config.res_amplify_ratio =
        toml::find<double>(vi_calib_config, "res_amplify_ratio");
    vi_config.disable_reproj_factor =
        toml::find<bool>(vi_calib_config, "disable_reproj_factor");
    vi_config.use_magic_number =
        toml::find<bool>(vi_calib_config, "use_magic_number");
    vi_config.window_size = toml::find<int>(vi_calib_config, "window_size");
    vi_config.gyro_std = toml::find<double>(vi_calib_config, "gyro_std");
    vi_config.gyro_std2 = toml::find<double>(vi_calib_config, "gyro_std2");
    vi_config.gyro_std_high =
        toml::find<double>(vi_calib_config, "gyro_std_high");
    vi_config.acc_std = toml::find<double>(vi_calib_config, "acc_std");
    vi_config.acc_std2 = toml::find<double>(vi_calib_config, "acc_std2");
    vi_config.acc_std_high =
        toml::find<double>(vi_calib_config, "acc_std_high");
    vi_config.g_std = toml::find<double>(vi_calib_config, "g_std");
    vi_config.g_std2 = toml::find<double>(vi_calib_config, "g_std2");
    vi_config.pose_std = toml::find<double>(vi_calib_config, "pose_std");
    vi_config.pose_std2 = toml::find<double>(vi_calib_config, "pose_std2");
    vi_config.comp_imu = toml::find<bool>(vi_calib_config, "comp_imu");
    vi_config.verbose = toml::find<bool>(vi_calib_config, "verbose");
    vi_config.opt_rolling_shutter =
        toml::find<bool>(vi_calib_config, "opt_rolling_shutter");
    vi_config.opt_visual_scale =
        toml::find<bool>(vi_calib_config, "opt_visual_scale");
    vi_config.fix_imu_intr = toml::find<bool>(vi_calib_config, "fix_imu_intr");
    vi_config.opt_camera_intr =
        toml::find<bool>(vi_calib_config, "opt_camera_intr");
    vi_config.disable_imu_factor =
        toml::find<bool>(vi_calib_config, "disable_imu_factor");
    vi_config.is_four_as_one =
        toml::find<bool>(vi_calib_config, "is_four_as_one");
    vi_config.opt_Tc0ci = toml::find<bool>(vi_calib_config, "opt_Tc0ci");
    vi_config.fix_imu_bias = toml::find<bool>(vi_calib_config, "fix_imu_bias");
    vi_config.four_cam_calib_file_name =
        toml::find<std::string>(vi_calib_config, "four_cam_calib_file_name");
    vi_config.is_imu_calibrated =
        toml::find<bool>(vi_calib_config, "is_imu_calibrated");
    vi_config.use_trifocal_tensor_factor =
        toml::find<bool>(vi_calib_config, "use_trifocal_tensor_factor");
    vi_config.vm_step_reproj =
        toml::find<int>(vi_calib_config, "vm_step_reproj");
    vi_config.vm_step_line = toml::find<int>(vi_calib_config, "vm_step_line");
    vi_config.detect_only = toml::find<bool>(vi_calib_config, "detect_only");

  } catch (const std::exception &e) {
    printf("parse toml err, file_path:[%s]  err:[%s]\n", config_path.c_str(),
           e.what());
    std::abort();
  }
  vi_config.fixed_camera_id = calib_cid[0];
  printf("fixed_camera_id: %d\n", vi_config.fixed_camera_id);
  return true;
}

void ConfigData::InitConfigs() {
  camPaths.clear();
  camJsonPaths.clear();
  for (int camId : calib_cid) {
    camPaths.emplace(camId, dataSet + "/" + cam_folder_prefix +
                                std::to_string(camId) + "/" +
                                cam_folder_subfix);
    camJsonPaths.emplace(camId, dataSet + "/" + cam_folder_prefix +
                                    std::to_string(camId) + "/" + cam_json);
  }
  for (int camId : rgb_cid) {
    camPaths.emplace(camId, dataSet + "/" + cam_folder_prefix +
                                std::to_string(camId) + "/" +
                                cam_folder_subfix);
    camJsonPaths.emplace(camId, dataSet + "/" + cam_folder_prefix +
                                    std::to_string(camId) + "/" + cam_json);
  }
  imuJsonPath = dataSet + "/" + imu_folder_name + "/" + imu_json_name;
  resultPath = dataSet + "/" + save_folder;
  extrinsicPath = resultPath + "/" + extr_folder;
  intrinsicPath = resultPath + "/" + intr_folder;

  if ((opendir(resultPath.c_str())) == nullptr) {
    mkdir(resultPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
  }

  if ((opendir(extrinsicPath.c_str())) == nullptr) {
    mkdir(extrinsicPath.c_str(),
          S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
  }

  if ((opendir(intrinsicPath.c_str())) == nullptr) {
    mkdir(intrinsicPath.c_str(),
          S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
  }

  switch (board_type) {
  case DOT:
    binFilePath = resultPath + "/" + dot_config.dot_bin_file;
    binFilePath_gray = resultPath + "/" + dot_config.dot_bin_file_gray;
    binFilePath_rgb = resultPath + "/" + dot_config.dot_bin_file_rgb;
    break;
  case APRILTAG:
    binFilePath = resultPath + "/" + apriltag_config.apriltag_bin_file;
    binFilePath_gray =
        resultPath + "/" + apriltag_config.apriltag_bin_file_gray;
    binFilePath_rgb = resultPath + "/" + apriltag_config.apriltag_bin_file_rgb;
    break;
  default:
    printf("NOT SUPPORT BOARD!! abort\n");
    std::abort();
  }
}

void ConfigData::CheckConfig() {}

} // namespace dso::CalibIO