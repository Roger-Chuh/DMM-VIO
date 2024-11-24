#include "vio_math_0.h"
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#pragma once

namespace dso {

class CameraBase;

struct CalibBoardPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<int> line_index_vec;
  Vec3 p_w; // point xyz in corresponding calibration board
  int calib_board_id = -1;
};

struct CalibBoardLine {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<int> point_index_vec;
  int line_type = -1; // 0: row line, 1: col line, 2: co-dir slash(0.5,0.5), 3:
                      // oppo-dir slash(-0.5,0.5)
  int calib_board_id = -1;
};

struct PointVM {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec2 uv = Vec2::Ones();
  Vec3 xyz = Vec3::Ones(); // by Unproj
  bool valid_point = true;
  bool is_valid_projection = true;
  bool is_updated = false;
  Eigen::Matrix<number_t, 3, Eigen::Dynamic> d_xyz_param;
  int iter_count = 0;
  bool is_outlier = false;
  number_t max_err = -1;
  Vec3 line_normal = Vec3::Ones(); // for trifocal line factor
  Vec6 start_end = Vec6::Ones();   // for trifocal line factor
  bool is_3dof = false;
  bool is_5dof = true;
};

struct LineVM {
  std::vector<int> visible_point_index_vec;
};

struct CalibFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CalibFrame() {
    v_Tcm_calibboardj.clear();
    v_Tcm_calibboardj.resize(10, Mat4::Identity());
    v_Tcm_calibboard0.clear();
    v_Tcm_calibboard0.resize(10, Mat4::Identity());
  }
  number_t timestamp;
  int64_t timestamp_ns;
  bool visual_valid = true;
  aligned_map<int, aligned_map<int, PointVM>>
      cid_pid_to_point_vm; // cam_id-> (id,point)
  aligned_map<int, aligned_map<int, LineVM>> cid_line_id_to_line_vm;
  aligned_map<int, number_t> cid_to_gain;
  aligned_map<int, number_t> cid_to_exposure_time;

  Mat4 Tc0_calibboard0;                   // camera rig to calib_board rig
  aligned_vector<Mat4> v_Tcm_calibboardj; // boardj to cameram
  aligned_vector<Mat4> v_Tcm_calibboard0; // boardj to cameram
  std::set<int> visible_cids = {};
  int is_used = 0;
  bool is_gray_frame = true;
  // todo: imu data

  // for debug and show
  aligned_map<int, std::string> cid_to_img_file_path; // full path
};

struct CalibBoards {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void GenerateAprilTagCalibBoards(
      const int &board_num, const int &tag_row, const number_t &tag_size,
      const number_t &tag_gap,
      const aligned_vector<Mat4> &calib_board_id_to_T01);

  void
  GenerateDotCalibBoards(const int board_num, const int &dot_per_row,
                         const number_t &dot_distance,
                         const aligned_vector<Mat4> &calib_board_id_to_T01);

  void
  GenerateLineVMBYPointVM(aligned_vector<CalibFrame> *p_input_frame_data_vec);

  int calib_board_num;
  aligned_vector<CalibBoardPoint> pid_to_CalibBoardPoint;
  aligned_vector<CalibBoardLine> line_id_to_CalibBoardLine;

  aligned_vector<Mat4> id_to_T01; // Calib Board Extrinsic
  aligned_vector<Vec3> line_type_to_dir_vec;

  std::vector<int> id_to_T01_opt; // calib board id to Extrinsic is opt
};

// struct MultiCamera {
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//  int cam_num = -1;
//  std::vector<int> cids;  // size must be equal to cam_num
//  std::map<int, CameraBase *> cid_to_cam;
//  aligned_map<int, Mat4> cid_to_T01;  // Camera Extrinsic
//
//  std::map<int, bool> cid_to_intrinsic_opt;  // cid to Camera Intrinsic is opt
//  std::map<int, bool> cid_to_extrinsic_opt;  // cid to Camera Extrinsic is opt
//};

struct AccelData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int knot_id = -1;
  int id;
  int64_t timestamp_ns;
  double timestamp;
  Vec3 data;
  Vec3 data_sim;
  bool is_useful = true;
  bool is_high_weight = false;
};

struct GyroData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int knot_id = -1;
  int id;
  int64_t timestamp_ns;
  double timestamp;
  Vec3 data;
  Vec3 data_sim;
  bool is_useful = true;
  bool is_high_weight = false;
};

struct ImuData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int knot_id = -1;
  int id;
  int64_t timestamp_ns;
  double timestamp;
  Vec3 data;
  Vec3 data_sim;
  bool is_useful = true;
  bool is_high_weight = false;
};
struct ImuDataSingle {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int64_t timestamp_ns;
  Vec3 acc;
  Vec3 gyro;
  ImuDataSingle(){};
  ImuDataSingle(const Vec3 &acc_, const Vec3 &gyro_,
                const int64_t &timestamp_ns_)
      : acc(acc_), gyro(gyro_), timestamp_ns(timestamp_ns_){};
};

struct IMUState {
  Mat4 Tbc0;
  Vec3 acc_bias;
  Vec3 w_bias;
  Vec3 ka;
  Vec3 kg;
  Vec3 na;
  Vec3 ng;
  Vec3 ombg;
  number_t time_delay;
  number_t time_delay_rgb = -1;
  number_t rolling_shutter_time = -1;
};
struct VI_Config {
  number_t apriltagSize = 0.08;
  number_t apriltagInterval = 0.2;
  int apriltag_rows = 10;
  std::set<int> high_weight_knot_ids;

  number_t gyro_std = 0.1;
  number_t gyro_std2 = 0.01;
  number_t gyro_std_high = 0.0025;

  number_t acc_std = 0.5;
  number_t acc_std2 = 0.05;
  number_t acc_std_high = 0.01;

  number_t g_std = 1.0;
  number_t g_std2 = 1.0;

  number_t pose_std = 0.01;
  number_t pose_std2 = 0.01;

  number_t high_weight_knot_ratio = 0.1;
  number_t power = 2;
  number_t dt = 0.0033;
  number_t dt_rgb = 0.1;
  bool shift_imu_data = true;
  int skip_first_frames_num = 30;
  number_t g_norm = 9.7946;
  bool use_sim = false;
  number_t disturb_level = 0.3;
  number_t noise_level = 0;
  bool fix_extr = true;
  bool use_intr_rot_factor = true;
  number_t intr_rot_factor_weight = 10;
  number_t res_amplify_ratio = 1;
  number_t resolution_difference_weight = 10;
  bool disable_reproj_factor = false;
  bool use_magic_number = true;
  IMUState imu_state_calibreted;
  int window_size = -2;
  bool comp_imu = true;
  bool sim_set = false;
  int fixed_camera_id = 0;
  int fixed_plate_id = 0;
  bool verbose = true;

  bool opt_visual_scale = false;
  bool opt_rolling_shutter = false;
  bool opt_rolling_shutter_for_trifocal_tensor = false;
  bool fix_imu_intr = false;
  bool fix_imu_bias = false;
  bool opt_camera_intr = false;
  bool disable_imu_factor = false;
  bool is_rgb = false;
  bool is_gray_rgb = false;
  bool is_four_as_one = true;
  std::string four_cam_calib_file_name;
  bool is_imu_calibrated;
  bool fix_all_T_c0ci = false;
  bool opt_Tc0ci = false;
  int max_cid = 100;
  std::set<int> useful_cids;
  std::set<int> opt_cids;
  std::map<int /*cid*/, bool> cid_to_opt_tr;
  std::map<int, double> fid_to_gray_timestamp;
  std::map<int, double> fid_to_rgb_timestamp;
  std::map<uint64_t, int> gray_timestamp_to_fid;
  std::map<uint64_t, int> rgb_timestamp_to_fid;
  std::vector<double> gray_timestamps;
  std::map<int, int> rgb_fid_to_closest_gray_fid;
  std::map<
      int /*rgb_fid*/,
      std::pair<
          int, /*gray_fid*/ std::map<
              int /*pid*/,
              std::vector<int /*cids that obverved this pid at this fid*/>>>>
      fid_to_pid_to_cid;
  std::map<
      int /*rgb_fid*/,
      std::pair<
          int, /*gray_fid*/ std::map<
              int /*pid*/,
              std::vector<int /*cids that obverved this pid at this fid*/>>>>
      fid_to_pid_to_cid_full;
  bool use_trifocal_tensor_factor = false;
  int gray_fid_offset = 5;
  bool fix_spline_kont_pose = false;
  bool fix_spline_kont_pose_for_trifocal_tensor = false;
  int vm_step_reproj = 1;
  int vm_step_line = 1;
  bool detect_only = false;
};
struct SelfCalib_Config {};
struct ThreadsGyroStruct {
  std::vector<GyroData *> sub_factors;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  std::unordered_map<long, int> parameter_block_size; // global size
  std::unordered_map<long, int> parameter_block_idx;  // local size
};
struct ThreadsAccelStruct {
  std::vector<AccelData *> sub_factors;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  std::unordered_map<long, int> parameter_block_size; // global size
  std::unordered_map<long, int> parameter_block_idx;  // local size
};
struct AprilgridCornersData {
  int64_t timestamp_ns;
  double timestamp;
  // outCornerObserved 啥意思??
  std::vector<bool> outCornerObserved;
  std::vector<int> corner_id;
  std::vector<cv::Point2f> corners;
  std::vector<cv::Point2f> corners_undistorted;
  std::vector<cv::Point3f> tagpoints;
  cv::Mat image;
  std::string image_path;
  std::vector<std::vector<int>> row_ids; // TODO 属于同一行的所有grid id
  std::vector<std::vector<int>> col_ids; // TODO 属于同一列的所有grid id
  std::map<int, cv::Point2f> grid_id_to_uv;
};
struct ImageImuData {
  bool imu_set = false;
  bool image_set = false;
  uint64_t timestamp;
  AccelData acc;
  GyroData gyro;
  //  std::map<int /*camera id*/, std::map<int /*palte id*/,
  //  AprilgridCornersData>> imageData;
  int imageData;
};
struct HandEyeInfo {
  Mat4 Twc = Mat4::Identity();
};
} // namespace dso