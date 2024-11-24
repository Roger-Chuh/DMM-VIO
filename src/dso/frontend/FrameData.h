#ifndef FRAMEDATA_H
#define FRAMEDATA_H

#include "../camera_model/calib_def.h"
#include "../camera_model/vio_def.h"
#include "../rapidjson/document.h"
#include "frontend_define.h"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

namespace dso {

namespace CalibIO {

struct FrameData {
  int64_t timestamp = 0;
  std::string filename;
  int64_t exposure_time = 0;
  int gain = 0;
  int slam_image = 0;
  int frame_id = 0;
};

struct CameraInfo {
  struct Camera {
    std::string sensor;
    std::string name;
    int width;
    int height;
  } camera;
  int width;
  int height;
  std::string name;
  bool interleaved;
};

// Define struct to hold sequence data
struct CameraJsonData {
  int camId;
  CameraInfo cameraInfo;
  std::string version;
  std::vector<FrameData> frames;

  int dropCount = 0;
};

struct ImuJsonData {
  std::vector<AccelData> acc_data;
  std::vector<GyroData> gyro_data;
};

inline bool camJsonLoad(const std::string &jsonPath, int camId,
                        CameraJsonData &camJson) {
  std::ifstream ifs(jsonPath);
  if (!ifs) {
    LOG_FRONT_ERROR("Failed to open file: %s\n", jsonPath.c_str());
    return false;
  }

  camJson.camId = camId;

  // 读取文件内容到 std::string 中
  std::string json_str((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());

  // 解析 JSON
  rapidjson::Document doc;
  doc.Parse(json_str.c_str());

  // 检查解析是否成功
  if (doc.HasParseError()) {
    LOG_FRONT_ERROR("JSON parse error...\n");
    return false;
  }

  const rapidjson::Value &sequence = doc["Sequence"];
  const rapidjson::Value &camera_info = sequence["CameraInfo"];
  const rapidjson::Value &frameset = sequence["Frameset"];

  // CameraInfo
  const rapidjson::Value &camera = camera_info["Camera"];
  camJson.cameraInfo.camera.sensor = camera["sensor"].GetString();
  camJson.cameraInfo.camera.name = camera["name"].GetString();
  camJson.cameraInfo.camera.width = camera["width"].GetInt();
  camJson.cameraInfo.camera.height = camera["height"].GetInt();
  camJson.cameraInfo.width = camera_info["width"].GetInt();
  camJson.cameraInfo.height = camera_info["height"].GetInt();
  camJson.cameraInfo.name = camera_info["name"].GetString();
  //  camJson.cameraInfo.interleaved = camera_info["interleaved"].GetBool();

  // Frames
  const rapidjson::Value &frame_array = frameset["Frame"];
  for (rapidjson::SizeType i = 0; i < frame_array.Size(); i++) {
    const rapidjson::Value &frame = frame_array[i];
    FrameData f;
    f.timestamp = frame["timestamp"].GetInt64();
    f.filename = frame["filename"].GetString();
    f.exposure_time = frame["exposure_time"].GetInt64();
    f.gain = frame["gain"].GetInt();
    f.slam_image = frame["slam_image"].GetInt();
    f.frame_id = frame["frame_id"].GetInt();
    camJson.frames.push_back(f);
  }

  return true;
}

inline bool imuJsonLoad(const std::string &jsonPath, ImuJsonData *imuJson) {
  std::ifstream ifs(jsonPath);
  if (!ifs) {
    LOG_FRONT_ERROR("Failed to open file: %s\n", jsonPath.c_str());
    return false;
  }
  // 读取文件内容到 std::string 中
  std::string json_str;
  std::getline(ifs, json_str);

  // 解析 JSON
  rapidjson::Document imu_doc;
  imu_doc.Parse(json_str.c_str());

  // 检查解析是否成功
  if (imu_doc.HasParseError()) {
    LOG_FRONT_ERROR("JSON parse error...\n");
    return false;
  }
  rapidjson::Value &IMUData = imu_doc["Sequence"]["Dataset"]["Data"];
  for (std::size_t i = 0; i < IMUData.Size() - 0; i++) {
    int64_t timestamp = IMUData[i]["timestamp"].GetUint64();
    number_t ax = IMUData[i]["a_x"].GetFloat();
    number_t ay = IMUData[i]["a_y"].GetFloat();
    number_t az = IMUData[i]["a_z"].GetFloat();
    number_t wx = IMUData[i]["g_x"].GetFloat();
    number_t wy = IMUData[i]["g_y"].GetFloat();
    number_t wz = IMUData[i]["g_z"].GetFloat();

    imuJson->acc_data.emplace_back();
    imuJson->acc_data.back().id = i;
    imuJson->acc_data.back().timestamp_ns = timestamp;
    imuJson->acc_data.back().timestamp =
        static_cast<number_t>(timestamp) * 1e-9;
    imuJson->acc_data.back().data = Vec3(ax, ay, az);

    imuJson->gyro_data.emplace_back();
    imuJson->gyro_data.back().id = i;
    imuJson->gyro_data.back().timestamp_ns = timestamp;
    imuJson->gyro_data.back().timestamp =
        static_cast<number_t>(timestamp) * 1e-9;
    imuJson->gyro_data.back().data = Vec3(wx, wy, wz);
  }
  ifs.close();
  return true;
}

inline bool camJsonAlign(std::map<int, CalibIO::CameraJsonData> &allCamJson,
                         int64_t minGap = 1000000,
                         int64_t frameGap = 30000000) { // unit ns
  if (allCamJson.size() < 2)
    return true;

  int idx = 0;
  for (auto it = allCamJson.begin()->second.frames.begin();
       it != allCamJson.begin()->second.frames.end();) {
    int64_t cam0Time = allCamJson.begin()->second.frames[idx].timestamp;

    std::vector<std::pair<int, int64_t>> timeDiff; // t_cami - t_cam0
    bool dataOver = false;                         // ending align
    for (auto map_it = std::next(allCamJson.begin());
         map_it != allCamJson.end(); ++map_it) {
      int camId = map_it->first;
      if (idx < map_it->second.frames.size()) {
        timeDiff.emplace_back(camId,
                              map_it->second.frames[idx].timestamp - cam0Time);
      } else {
        dataOver = true;
        break;
      }
    }

    if (dataOver) {
      for (auto &[camId, oneCamJson] : allCamJson) {
        if (oneCamJson.frames.size() > idx) {
          LOG_FRONT_WARN("Drop other cam frame: %s, cam %d\n",
                         oneCamJson.frames[idx].filename.c_str(),
                         oneCamJson.camId);
          ++oneCamJson.dropCount;
          oneCamJson.frames.erase(oneCamJson.frames.begin() + idx,
                                  oneCamJson.frames.end());
        }
      }
      return true;
    }

    if (timeDiff.size() < allCamJson.size() - 1) {
      ++idx;
      ++it;
      continue;
    }

    std::vector<int> inCount, outCount;
    std::vector<std::pair<int, int64_t>> wrongCount;
    bool deletaCam0 = false;
    for (auto &i : timeDiff) {
      int temp = i.first;
      int64_t data = i.second;
      if (std::abs(data) <= minGap)
        inCount.emplace_back(temp);
      if (std::abs(data) > minGap && std::abs(data) <= frameGap / 3)
        outCount.emplace_back(temp);
      if (std::abs(data) > frameGap / 3) {
        if (data > 0) {
          deletaCam0 = true;
          break;
        }
        wrongCount.emplace_back(temp, data);
      }
    }

    if (deletaCam0) {
      LOG_FRONT_WARN("Drop base cam frame: %s, cam %d\n", it->filename.c_str(),
                     allCamJson.begin()->second.camId);
      ++allCamJson.begin()->second.dropCount;
      it = allCamJson.begin()->second.frames.erase(it);
      if (allCamJson.begin()->second.frames.empty()) {
        LOG_FRONT_ERROR("Camera has not match frame, cam %d\n",
                        allCamJson.begin()->second.camId);
        for (auto &[_, oneCam] : allCamJson) {
          oneCam.frames.clear();
        }
        return false;
      }
      continue;
    }

    if (wrongCount.empty()) {
      if (!outCount.empty()) {
        LOG_FRONT_WARN("Drop can't align frame: %s, cam %d\n",
                       it->filename.c_str(), allCamJson.begin()->second.camId);
        ++allCamJson.begin()->second.dropCount;
        it = allCamJson.begin()->second.frames.erase(it);
        for (auto map_it = std::next(allCamJson.begin());
             map_it != allCamJson.end(); ++map_it) {
          LOG_FRONT_WARN("Drop other can't align camera frame: %s, cam %d\n",
                         map_it->second.frames[idx].filename.c_str(),
                         map_it->second.camId);
          map_it->second.frames.erase(map_it->second.frames.begin() + idx);
        }
        continue;
      }
      ++idx;
      ++it;
      continue;
    } else {
      for (const std::pair<int, int64_t> &oneCam : wrongCount) {
        LOG_FRONT_WARN("Drop wrong align frame: %s, cam %d\n",
                       allCamJson[oneCam.first].frames[idx].filename.c_str(),
                       allCamJson[oneCam.first].camId);
        ++allCamJson[oneCam.first].dropCount;
        allCamJson[oneCam.first].frames.erase(
            allCamJson[oneCam.first].frames.begin() + idx);
      }
    }
  }

  for (auto &[_, oneCam] : allCamJson) {
    if (oneCam.frames.size() < idx) {
      LOG_FRONT_ERROR("Cam json align size wired, check json size: %d\n",
                      oneCam.frames.size());
      std::exit(-1);
    }
    if (oneCam.frames.size() > idx) {
      LOG_FRONT_INFO("Remove cam tail json, target: %d, cam %d, frames %d\n",
                     idx, oneCam.camId, oneCam.frames.size());
      oneCam.frames.erase(oneCam.frames.begin() + idx, oneCam.frames.end());
    }
  }

  return true;
}

struct CurFrameRes {
  double timestamp = 0;
  double exposure = 0;
  int gain = 0;

  aligned_unordered_map<int, std::vector<Eigen::Vector3d>> mObjectPointSets;
  aligned_unordered_map<int, std::vector<Eigen::Vector2d>> mImagePointSets;
  aligned_unordered_map<int, std::vector<int>> mGridId;
  bool enough_points = false;
  std::string frameName; // xx.bmp

  CurFrameRes()
      : mObjectPointSets(), mImagePointSets(), mGridId(), enough_points(false),
        frameName() {}
  ~CurFrameRes() = default;
};

struct CamFrames {
  std::set<int> obPlates;
  int camID = -1;
  std::string filePath; // rootpath
  std::vector<CurFrameRes> eachFrameInfo;
};

// Function to save CamFrames data to a binary file
inline void
saveCamFrames(const std::unordered_map<int, CamFrames> &multiCamFrames,
              const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    LOG_FRONT_ERROR("Failed to open file for writing: %s\n", filename.c_str());
    return;
  }

  // Save the size of the map
  size_t mapSize = multiCamFrames.size();
  file.write(reinterpret_cast<const char *>(&mapSize), sizeof(size_t));

  for (const auto &oneCam : multiCamFrames) {
    int key = oneCam.first;
    const CamFrames &camFrames = oneCam.second;
    // Save the key
    file.write(reinterpret_cast<const char *>(&key), sizeof(int));
    // Save camID
    file.write(reinterpret_cast<const char *>(&camFrames.camID), sizeof(int));

    // Save filePath
    size_t filePathSize = camFrames.filePath.size();
    file.write(reinterpret_cast<const char *>(&filePathSize), sizeof(size_t));
    file.write(camFrames.filePath.c_str(), filePathSize);

    // Save eachFrameInfo vector size
    size_t numFrames = camFrames.eachFrameInfo.size();
    file.write(reinterpret_cast<const char *>(&numFrames), sizeof(size_t));

    // Save each CurFrameRes in eachFrameInfo vector
    for (const auto &frame : camFrames.eachFrameInfo) {
      // Save frameInfo
      file.write(reinterpret_cast<const char *>(&frame.timestamp),
                 sizeof(double));
      file.write(reinterpret_cast<const char *>(&frame.exposure),
                 sizeof(double));
      file.write(reinterpret_cast<const char *>(&frame.gain), sizeof(int));

      // Save frameName
      size_t frameNameSize = frame.frameName.size();
      file.write(reinterpret_cast<const char *>(&frameNameSize),
                 sizeof(size_t));
      file.write(frame.frameName.c_str(), frameNameSize);

      // Save enough_points
      file.write(reinterpret_cast<const char *>(&frame.enough_points),
                 sizeof(bool));

      // Save mObjectPointSets
      size_t mObjectPointSetsSize = frame.mObjectPointSets.size();
      file.write(reinterpret_cast<const char *>(&mObjectPointSetsSize),
                 sizeof(size_t));
      for (const auto &pair : frame.mObjectPointSets) {
        int boardid = pair.first;
        size_t vecSize = pair.second.size();
        file.write(reinterpret_cast<const char *>(&boardid), sizeof(int));
        file.write(reinterpret_cast<const char *>(&vecSize), sizeof(size_t));
        for (const auto &vec : pair.second) {
          file.write(reinterpret_cast<const char *>(vec.data()),
                     sizeof(double) * vec.size());
        }
      }

      // Save mImagePointSets
      size_t mImagePointSetsSize = frame.mImagePointSets.size();
      file.write(reinterpret_cast<const char *>(&mImagePointSetsSize),
                 sizeof(size_t));
      for (const auto &onePair : frame.mImagePointSets) {
        int boardid = onePair.first;
        size_t vecSize = onePair.second.size();
        file.write(reinterpret_cast<const char *>(&boardid), sizeof(int));
        file.write(reinterpret_cast<const char *>(&vecSize), sizeof(size_t));
        for (const auto &vec : onePair.second) {
          file.write(reinterpret_cast<const char *>(vec.data()),
                     sizeof(double) * vec.size());
        }
      }

      // Save mGridId
      size_t mGridIdSize = frame.mGridId.size();
      file.write(reinterpret_cast<const char *>(&mGridIdSize), sizeof(size_t));
      for (const auto &onePair : frame.mGridId) {
        int boardid = onePair.first;
        size_t vecSize = onePair.second.size();
        file.write(reinterpret_cast<const char *>(&boardid), sizeof(int));
        file.write(reinterpret_cast<const char *>(&vecSize), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(onePair.second.data()),
                   sizeof(int) * vecSize);
      }
    }
  }

  file.close();
}

// Function to load CamFrames data from a binary file
inline bool loadCamFrames(const std::string &filename,
                          std::unordered_map<int, CamFrames> &multiCamFrames) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  multiCamFrames.clear();
  // Save the size of the map
  size_t mapSize;
  file.read(reinterpret_cast<char *>(&mapSize), sizeof(size_t));

  for (size_t idx = 0; idx < mapSize; ++idx) {
    int key;
    CamFrames camFrames;
    // Read the key
    file.read(reinterpret_cast<char *>(&key), sizeof(int));
    // Load camID
    file.read(reinterpret_cast<char *>(&camFrames.camID), sizeof(int));

    // Load filePath
    size_t filePathSize;
    file.read(reinterpret_cast<char *>(&filePathSize), sizeof(size_t));
    camFrames.filePath.resize(filePathSize);
    file.read(&camFrames.filePath[0], filePathSize);

    // Load eachFrameInfo vector size
    size_t numFrames;
    file.read(reinterpret_cast<char *>(&numFrames), sizeof(size_t));
    camFrames.eachFrameInfo.resize(numFrames);

    // Load each CurFrameRes in eachFrameInfo vector
    for (size_t i = 0; i < numFrames; ++i) {
      auto &frame = camFrames.eachFrameInfo[i];

      // Load frameInfo
      file.read(reinterpret_cast<char *>(&frame.timestamp), sizeof(double));
      file.read(reinterpret_cast<char *>(&frame.exposure), sizeof(double));
      file.read(reinterpret_cast<char *>(&frame.gain), sizeof(int));

      // Load frameName
      size_t frameNameSize;
      file.read(reinterpret_cast<char *>(&frameNameSize), sizeof(size_t));
      frame.frameName.resize(frameNameSize);
      file.read(&frame.frameName[0], frameNameSize);

      // Load enough_points
      file.read(reinterpret_cast<char *>(&frame.enough_points), sizeof(bool));

      // Load mObjectPointSets
      size_t mObjectPointSetsSize;
      file.read(reinterpret_cast<char *>(&mObjectPointSetsSize),
                sizeof(size_t));
      for (size_t j = 0; j < mObjectPointSetsSize; ++j) {
        int boardid;
        size_t vecSize;
        file.read(reinterpret_cast<char *>(&boardid), sizeof(int));
        file.read(reinterpret_cast<char *>(&vecSize), sizeof(size_t));
        std::vector<Eigen::Vector3d> vec(vecSize);
        for (size_t k = 0; k < vecSize; ++k) {
          Eigen::Vector3d v;
          file.read(reinterpret_cast<char *>(v.data()),
                    sizeof(double) * v.size());
          vec[k] = v;
        }
        frame.mObjectPointSets[boardid] = vec;
      }

      // Load mImagePointSets
      size_t mImagePointSetsSize;
      file.read(reinterpret_cast<char *>(&mImagePointSetsSize), sizeof(size_t));
      for (size_t j = 0; j < mImagePointSetsSize; ++j) {
        int boardid;
        size_t vecSize;
        file.read(reinterpret_cast<char *>(&boardid), sizeof(int));
        file.read(reinterpret_cast<char *>(&vecSize), sizeof(size_t));
        std::vector<Eigen::Vector2d> vec(vecSize);
        for (size_t k = 0; k < vecSize; ++k) {
          Eigen::Vector2d v;
          file.read(reinterpret_cast<char *>(v.data()),
                    sizeof(double) * v.size());
          vec[k] = v;
        }
        frame.mImagePointSets[boardid] = vec;
      }

      // Load mGridId
      size_t mGridIdSize;
      file.read(reinterpret_cast<char *>(&mGridIdSize), sizeof(size_t));
      for (size_t j = 0; j < mGridIdSize; ++j) {
        int boardid;
        size_t vecSize;
        file.read(reinterpret_cast<char *>(&boardid), sizeof(int));
        file.read(reinterpret_cast<char *>(&vecSize), sizeof(size_t));
        std::vector<int> vec(vecSize);
        file.read(reinterpret_cast<char *>(vec.data()), sizeof(int) * vecSize);
        frame.mGridId[boardid] = vec;
      }
    }

    multiCamFrames[key] = camFrames;
  }

  file.close();

  return true;
}
} // namespace CalibIO

} // namespace dso
#endif // FRAMEDATA_H
