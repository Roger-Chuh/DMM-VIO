//
// Created by zk on 24-4-7.
//

#ifndef CAMERADETECTION_H
#define CAMERADETECTION_H

#include <set>
#include <string>
#include <vector>

#include "../camera_model/calib_def.h"
#include "../config/config.h"
#include "FrameData.h"
#include "apriltags/TagDetector.h"
#include "dotdetect/ImageProcessing.h"

namespace dso::CamCalib {

class CameraDetection {
private:
  void dotDetection();

  static int computeAprilTagPatternID(
      int tagId, int pattern_num,
      const std::vector<std::pair<int, int>> &startID_endIDs);

  static void
  dotSingleThreadDetect(const int &camId, const CalibIO::ConfigData &config,
                        const DotDetect::ParamsImageProcessing &curParams,
                        const DotDetect::ImageProcessing &image_processing,
                        std::vector<CalibIO::FrameData *> &picJson,
                        std::vector<CalibIO::CurFrameRes *> &picRes);

  static void aprilTagThreadDetection(
      const int &camId, const CalibIO::ConfigData &config,
      const AprilTags::TagDetector &detector,
      std::vector<CalibIO::FrameData *> &picJson,
      std::vector<CalibIO::CurFrameRes *> &picRes,
      const std::vector<std::pair<int, int>> &startID_endIDs,
      const Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor> &grid_points);

  void multiThreadDetect();

  void loadCameraData(bool skip_half_data = false);

  void loadImuData(CalibIO::ImuJsonData *p_imuData);

  bool hasResBin();

  bool loadBinFile();

  void transRes(dso::aligned_vector<dso::CalibFrame> &res, bool is_rgb = false);

  static bool FileEndsWith(const std::string &str, const std::string &suffix);
  static void GetBmpNames(const std::string &path,
                          std::vector<std::string> &filenames);

  void drawPointsRangePic() const;

public:
  explicit CameraDetection(const CalibIO::ConfigData &config)
      : m_config(config) {
    int tag_num_per_pattern =
        m_config.apriltag_config.rows * m_config.apriltag_config.cols;
    m_startID_endIDs.resize(m_config.apriltag_config.pattern_num);
    for (int i = 0; i < m_config.apriltag_config.pattern_num; i++) {
      m_startID_endIDs[i] = (std::make_pair(
          tag_num_per_pattern * i + m_config.apriltag_config.board_offset,
          tag_num_per_pattern * (i + 1) - 1 +
              m_config.apriltag_config.board_offset));
    }
    m_grid_points.resize(4 * m_config.apriltag_config.one_board_tags, 3);
    for (unsigned r = 0; r < m_config.apriltag_config.rows * 2; r++) {   // 12
      for (unsigned c = 0; c < m_config.apriltag_config.cols * 2; c++) { // 12
        Eigen::Matrix<double, 1, 3> point;

        point(0) = (int)(c / 2) * (1 + m_config.apriltag_config.tag_gap) *
                       m_config.apriltag_config.tag_size +
                   (c % 2) * m_config.apriltag_config.tag_size;
        point(1) = (int)(r / 2) * (1 + m_config.apriltag_config.tag_gap) *
                       m_config.apriltag_config.tag_size +
                   (r % 2) * m_config.apriltag_config.tag_size;
        point(2) = 0.0;

        m_grid_points.row(r * m_config.apriltag_config.cols * 2 + c) = point;
      }
    }
  }

  void showDetection();

  bool pipeline(dso::aligned_vector<dso::CalibFrame> &res,
                CalibIO::ImuJsonData *p_imuData = nullptr,
                bool skip_half_data = false,
                dso::aligned_vector<dso::CalibFrame> *p_rgb_data = nullptr) {
    loadCameraData(skip_half_data);
    if (hasResBin()) {
      loadBinFile();
    } else {
      multiThreadDetect();
    }
    if (p_imuData) {
      loadImuData(p_imuData);
    }
    if (m_config.calib_stage != CalibIO::GRAY_RGB) {
      drawPointsRangePic();
    }
    transRes(res);
    if (m_config.calib_stage == CalibIO::GRAY_RGB) {
      m_multiCamFrames.clear();
      m_multiCamFrames = m_multiCamFrames_rgb;
      transRes(*p_rgb_data, true);
    }
    return true;
  }

  void dotDetect(const std::string &imgFolder, const std::string &saveFolder);

  const auto &GetCamJsonData() const { return m_multiCamJsonRes; }

public:
  std::unordered_map<int /*camId*/, CalibIO::CamFrames> m_multiCamFrames;
  std::unordered_map<int /*camId*/, CalibIO::CamFrames> m_multiCamFrames_rgb;

private:
  const CalibIO::ConfigData &m_config;
  std::map<int, CalibIO::CameraJsonData> m_multiCamJsonRes;
  CalibIO::ImuJsonData m_imuJsonRes;
  std::vector<std::pair<int, int>> m_startID_endIDs;
  Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      m_grid_points;
};

} // namespace dso::CamCalib

#endif // CAMERADETECTION_H
