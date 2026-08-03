#pragma once
#include "calib_def.h"
#include "vio_def.h"
#include <string>

namespace dso {

void LoadXML(const std::string& file_path, MultiCamera& multi_camera, IMUState& imu_state);
void SaveXML(const std::string& file_path, const MultiCamera& multi_camera, const IMUState& imu_state,
             const std::string& device_sn);
void SaveIPD(const std::string& file_path, const MultiCamera& multi_camera, const IMUState& imu_state);
void SaveBoardExtrinsic(const std::string& file_path, const CalibBoards& calib_boards);
void LoadBoardExtrinsic(const std::string& file_path, CalibBoards& calib_boards);

}  // namespace dso
