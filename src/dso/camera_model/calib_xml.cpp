#include "calib_xml.h"
#include "../../../thirdparty/tinyxml2/tinyxml2.h"
#include "../frontend/eigen_utils.h"
#include "../frontend/wlog.h"
#include "double_sphere_camera.h"
#include "kannala_brandt_camera.h"
#include "kb16_camera.h"
#include "kb20_camera.h"
#include "macro_define.h"
#include "pinhole_camera.h"
#include "radial_tangential_camera.h"
#include "ucm_rtp_camera.h"

#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>

namespace dso {
using namespace tinyxml2;

static inline std::string to_string_with_row_data(number_t value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<number_t>::digits10) << value;
  return oss.str();
}

struct CameraInfo {
  std::string camera_type;
  int width;
  int height;
  number_t parameters_[50];
  int parameters_size;
  Mat4 T01;
};

void LoadXML(const std::string& file_path, MultiCamera& multi_camera, IMUState& imu_state) {
  std::unordered_map<int, CameraInfo> cid_to_camera;
  Vec3 ombc, tbc, aBias, wBias, ka, kg, na, ng, ombg;
  Mat3 Rbc;
  number_t accelDelta, delta;

  tinyxml2::XMLDocument caliXML;
  if (caliXML.LoadFile(file_path.c_str()) == 0) {
    auto firstEle = caliXML.GetDocument()->RootElement()->FirstChildElement("Camera");
    const char* stream = "";
    const char* model = "";
    int width, height;
    number_t fx, fy, cx, cy, tcc[3], rcc[9];
    CameraInfo t_camera;
    std::istringstream dataStream(stream);

    while (firstEle != nullptr) {
      int id;
      firstEle->QueryAttribute("id", &id);
      const auto& curCalib = firstEle->FirstChildElement("Calibration");

      curCalib->QueryAttribute("size", &stream);
      dataStream.clear();
      dataStream.str(stream);
      dataStream >> width >> height;

      curCalib->QueryAttribute("principal_point", &stream);
      dataStream.clear();
      dataStream.str(stream);
      dataStream >> cx >> cy;

      curCalib->QueryAttribute("focal_length", &stream);
      dataStream.clear();
      dataStream.str(stream);
      dataStream >> fx >> fy;

      curCalib->QueryAttribute("model", &model);
      std::string camera_model(model);

      curCalib->QueryAttribute("radial_distortion", &stream);
      dataStream.clear();
      dataStream.str(stream);

      t_camera.camera_type = camera_model;
      t_camera.width = width;
      t_camera.height = height;
      t_camera.parameters_[0] = fx;
      t_camera.parameters_[1] = fy;
      t_camera.parameters_[2] = cx;
      t_camera.parameters_[3] = cy;
      t_camera.parameters_size = 4;
      while (dataStream >> t_camera.parameters_[t_camera.parameters_size]) {
        t_camera.parameters_size++;
      }
      // ucmrtp parameters: fx, fy, cx, cy, alpha, k1, k2, k3, k4, k5, k6, p1,
      // p2, s1, s2, s3, s4 kb16 parameters: fx, fy, cx, cy, k1, k2, k3, k4, k5,
      // k6, p1, p2, s1, s2, s3, s4 kb8 parameters: fx, fy, cx, cy, k1, k2, k3,
      // k4

      const auto& curExPose = firstEle->FirstChildElement("Rig");
      curExPose->QueryAttribute("translation", &stream);
      dataStream.clear();
      dataStream.str(stream);
      dataStream >> tcc[0] >> tcc[1] >> tcc[2];

      curExPose->QueryAttribute("rowMajorRotationMat", &stream);
      dataStream.clear();
      dataStream.str(stream);
      dataStream >> rcc[0] >> rcc[1] >> rcc[2] >> rcc[3] >> rcc[4] >> rcc[5] >> rcc[6] >> rcc[7] >> rcc[8];
      Mat3 R10;
      R10 << rcc[0], rcc[1], rcc[2], rcc[3], rcc[4], rcc[5], rcc[6], rcc[7], rcc[8];
      Vec3 t10(tcc[0], tcc[1], tcc[2]);
      t_camera.T01.setIdentity();
      t_camera.T01.block<3, 3>(0, 0) = R10.transpose();
      t_camera.T01.block<3, 1>(0, 3) = -t_camera.T01.block<3, 3>(0, 0) * t10;
      cid_to_camera.insert(std::make_pair(id, t_camera));
      firstEle = firstEle->NextSiblingElement("Camera");
    }

    const auto& imuInfo = caliXML.GetDocument()->RootElement()->FirstChildElement("SFConfig");

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("ombc", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> ombc[0] >> ombc[1] >> ombc[2];
    LinearAlgebraLib::AngleAxis<number_t> aa(ombc.norm(), ombc.normalized());
    Rbc = aa.toRotationMatrix();

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("tbc", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> tbc[0] >> tbc[1] >> tbc[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("aBias", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> aBias[0] >> aBias[1] >> aBias[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("wBias", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> wBias[0] >> wBias[1] >> wBias[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("ka", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> ka[0] >> ka[1] >> ka[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("kg", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> kg[0] >> kg[1] >> kg[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("na", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> na[0] >> na[1] >> na[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("ng", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> ng[0] >> ng[1] >> ng[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("ombg", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> ombg[0] >> ombg[1] >> ombg[2];

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("accelDelta", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> accelDelta;

    imuInfo->FirstChildElement("Stateinit")->QueryAttribute("delta", &stream);
    dataStream.clear();
    dataStream.str(stream);
    dataStream >> delta;
  } else {
    printf("can't load calibration results: %s\n", file_path.c_str());
    std::abort();
  }

  multi_camera.cam_num = cid_to_camera.size();
  std::cout << "cam_num in xml: " << multi_camera.cam_num << std::endl;
  for (const auto& cid_cam : cid_to_camera) {
    int cid = cid_cam.first;
    const CameraInfo& camera = cid_cam.second;
    CameraBase* p_cam;
    if (camera.camera_type == "KB8" || camera.camera_type == "KANNALA_BRANDT") {
      p_cam = new KB8Camera(cid, camera.width, camera.height, camera.parameters_);
    } else if (camera.camera_type == "KB16") {
      p_cam = new KB16Camera(cid, camera.width, camera.height, camera.parameters_);
    } else if (camera.camera_type == "KB20") {
      p_cam = new KB20Camera(cid, camera.width, camera.height, false, true);
      VecX intr;
      intr.resize(p_cam->kParamLength);
      intr.setZero();
      intr = Eigen::Map<const VecX>(camera.parameters_, p_cam->kParamLength);
      p_cam->SetIntrinsic(intr);
    } else if (camera.camera_type == "UcmRTP" || camera.camera_type == "UCMRTP") {
      p_cam = new UCMRTPCamera(cid, camera.width, camera.height, camera.parameters_, 0, false);
    } else if (camera.camera_type == "Pinhole") {
      p_cam = new PinholeCamera(cid, camera.width, camera.height, camera.parameters_);
    } else if (camera.camera_type == "RT") {
      p_cam = new RadtanCamera(cid, camera.width, camera.height, camera.parameters_);
    } else if (camera.camera_type == "DS") {
      p_cam = new DoubleSphereCamera(cid, camera.width, camera.height, camera.parameters_);
    } else {
      printf("unknown camera model, abort %s\n", camera.camera_type.c_str());
      std::abort();
    }

    if (camera.parameters_size < p_cam->kParamLength) {
      printf("camera parameter size don't match, abort\n");
      std::abort();
    }

    multi_camera.cid_to_cam[cid] = p_cam;
    multi_camera.cid_to_T01[cid] = camera.T01;
    multi_camera.cids.push_back(cid);
  }

  std::sort(multi_camera.cids.begin(), multi_camera.cids.end());

  imu_state.Tbc0.setIdentity();
  imu_state.Tbc0.block<3, 3>(0, 0) = Rbc;
  imu_state.Tbc0.block<3, 1>(0, 3) = tbc;
  imu_state.acc_bias = aBias;
  imu_state.w_bias = wBias;
  imu_state.ka = ka;
  imu_state.kg = kg;
  imu_state.na = na;
  imu_state.ng = ng;
  imu_state.ombg = ombg;
  imu_state.time_delay = delta;
}

void SaveXML(const std::string& file_path, const MultiCamera& multi_camera, const IMUState& imu_state,
             const std::string& device_sn) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm* now_tm = std::localtime(&now_c);
  std::ostringstream cur_time;
  cur_time << std::put_time(now_tm, "%Y-%m-%d.%H:%M:%S");

  XMLDocument doc;

  // 创建并添加 XML 声明,兼容旧的代码
  XMLDeclaration* decl = doc.NewDeclaration("xml version=\"1.0\" encoding=\"utf-8\"");
  doc.InsertFirstChild(decl);

  XMLElement* DeviceConfiguration = doc.NewElement("DeviceConfiguration");
  doc.InsertEndChild(DeviceConfiguration);
  DeviceConfiguration->SetAttribute("deviceSN", device_sn.c_str());
  DeviceConfiguration->SetAttribute("calibration_time", cur_time.str().c_str());
  DeviceConfiguration->SetAttribute("imu_type", "need to set");

  // save tracking camera in default order
  // std::vector<int> cid_order = {0, 3, 2, 1};
  // if (multi_camera.cid_to_cam.size() > 4) {
  //   for (int cid = 4; cid < multi_camera.cid_to_cam.size(); cid++) {
  //     cid_order.emplace_back(cid);
  //   }
  // }

  std::string delta_str = to_string_with_row_data(imu_state.time_delay);
  for (int cid : multi_camera.cids) {
    // save camera info
    XMLElement* Camera = doc.NewElement("Camera");
    DeviceConfiguration->InsertEndChild(Camera);
    const auto& p_cam = multi_camera.cid_to_cam.at(cid);
    const Mat4& T01 = multi_camera.cid_to_T01.at(cid);
    const number_t* parameters_ptr = p_cam->parameters_ptr();
    std::string cam_name, name, id;
    if (cid == 0 || cid == 3) {
      cam_name = "TrackingMaster";
      name = "TrackingMaster";
    } else if (cid == 1 || cid == 2) {
      cam_name = "TrackingSlave";
      name = "TrackingSlave";
    } else if (cid == 4 || cid == 5) {
      cam_name = "TrackingAux";
      name = "TrackingAux";
    } else if (cid == 6 || cid == 7) {
      cam_name = "VST";
      name = "VST";
    } else if (cid == 10) {
      cam_name = "TOF";
      name = "TOF";
    } else {
      cam_name = "EXTRA_CAM";
      name = "EXTRA_CAM";
      //      LOG_Calib_ERROR("unknown cid %d\n", cid);
      //      std::abort();
    }

    id = std::to_string(cid);
    Camera->SetAttribute("cam_name", cam_name.c_str());
    Camera->SetAttribute("name", name.c_str());
    Camera->SetAttribute("id", id.c_str());

    // save camera intrinsic
    XMLElement* Calibration = doc.NewElement("Calibration");
    Camera->InsertEndChild(Calibration);

    std::string size = std::to_string(p_cam->width()) + " " + std::to_string(p_cam->height());
    std::string focal_length =
        to_string_with_row_data(parameters_ptr[0]) + " " + to_string_with_row_data(parameters_ptr[1]);
    std::string principal_point =
        to_string_with_row_data(parameters_ptr[2]) + " " + to_string_with_row_data(parameters_ptr[3]);
    std::string model, radial_distortion;
    model = dso::CameraBase::ModelAsString(p_cam->camera_model());

    if (reinterpret_cast<UCMRTPCamera*>(p_cam)->use_exp_ &&
        model == dso::CameraBase::ModelAsString(CameraBase::CameraModel::kUcmRTP)) {
      reinterpret_cast<UCMRTPCamera*>(p_cam)->SetExpAlpha();
    }

    if (model == dso::CameraBase::ModelAsString(CameraBase::CameraModel::kUcmRTP)) {
      model = "UCMRTP";
    } else if (model == dso::CameraBase::ModelAsString(CameraBase::CameraModel::kKB8)) {
      model = "KANNALA_BRANDT";
    }

    for (int i = 4; i < kMaxIntrSize; i++) {
      if (i < p_cam->kParamLength) {
        radial_distortion += to_string_with_row_data(parameters_ptr[i]);
      } else {
        radial_distortion += "0";
      }

      if (i != kMaxIntrSize - 1) {
        radial_distortion += " ";
      }
    }
    Calibration->SetAttribute("size", size.c_str());
    Calibration->SetAttribute("principal_point", principal_point.c_str());
    Calibration->SetAttribute("focal_length", focal_length.c_str());
    Calibration->SetAttribute("model", model.c_str());
    Calibration->SetAttribute("radial_distortion", radial_distortion.c_str());
    Calibration->SetAttribute("distortion_limit", "9.000000");
    Calibration->SetAttribute("undistortion_limit", "1.577340");

    // save camera extrinsic
    XMLElement* Rig = doc.NewElement("Rig");
    Camera->InsertEndChild(Rig);
    std::string translation, rowMajorRotationMat;
    Mat4 T10 = InversePose(T01);
    Vec3 t10 = T10.block<3, 1>(0, 3);
    Mat3 R10 = T10.block<3, 3>(0, 0);
    for (int i = 0; i < 3; i++) {
      translation += to_string_with_row_data(t10[i]);
      if (i != 2) {
        translation += " ";
      }
    }
    for (int i = 0; i < 9; i++) {
      rowMajorRotationMat += to_string_with_row_data(R10(i / 3, i % 3));
      if (i != 8) {
        rowMajorRotationMat += " ";
      }
    }
    Rig->SetAttribute("translation", translation.c_str());
    Rig->SetAttribute("rowMajorRotationMat", rowMajorRotationMat.c_str());

    // save TimeAlignment
    XMLElement* TimeAlignment = doc.NewElement("TimeAlignment");
    Camera->InsertEndChild(TimeAlignment);
    if (cid == 7 || cid == 6) {
      // TODO temp use magic value
      delta_str = "100.011423";
    }
    TimeAlignment->SetAttribute("delta", delta_str.c_str());
  }  // loop for tracking camera

  // save imu_state
  XMLElement* SFConfig = doc.NewElement("SFConfig");
  DeviceConfiguration->InsertEndChild(SFConfig);
  XMLElement* Stateinit = doc.NewElement("Stateinit");
  SFConfig->InsertEndChild(Stateinit);
  std::string ombc_str, tbc_str, aBias_str, wBias_str, ka_str, kg_str, na_str, ng_str, ombg_str, accelDelta_str;
  LinearAlgebraLib::AngleAxis<number_t> aa(imu_state.Tbc0.block<3, 3>(0, 0));
  Vec3 ombc = aa.angle() * aa.axis();
  Vec3 tbc(imu_state.Tbc0.block<3, 1>(0, 3));
  for (int i = 0; i < 3; i++) {
    ombc_str += to_string_with_row_data(ombc[i]);
    tbc_str += to_string_with_row_data(tbc[i]);
    aBias_str += to_string_with_row_data(imu_state.acc_bias[i]);
    wBias_str += to_string_with_row_data(imu_state.w_bias[i]);
    ka_str += to_string_with_row_data(imu_state.ka[i]);
    kg_str += to_string_with_row_data(imu_state.kg[i]);
    na_str += to_string_with_row_data(imu_state.na[i]);
    ng_str += to_string_with_row_data(imu_state.ng[i]);
    ombg_str += to_string_with_row_data(imu_state.ombg[i]);
    if (i != 2) {
      ombc_str += " ";
      tbc_str += " ";
      aBias_str += " ";
      wBias_str += " ";
      ka_str += " ";
      kg_str += " ";
      na_str += " ";
      ng_str += " ";
      ombg_str += " ";
    }
  }
  accelDelta_str = "0.00";
  Stateinit->SetAttribute("ombc", ombc_str.c_str());
  Stateinit->SetAttribute("tbc", tbc_str.c_str());
  Stateinit->SetAttribute("aBias", aBias_str.c_str());
  Stateinit->SetAttribute("wBias", wBias_str.c_str());
  Stateinit->SetAttribute("ka", ka_str.c_str());
  Stateinit->SetAttribute("kg", kg_str.c_str());
  Stateinit->SetAttribute("na", na_str.c_str());
  Stateinit->SetAttribute("ng", ng_str.c_str());
  Stateinit->SetAttribute("ombg", ombg_str.c_str());
  Stateinit->SetAttribute("accelDelta", accelDelta_str.c_str());
  Stateinit->SetAttribute("delta", delta_str.c_str());
  doc.SaveFile(file_path.c_str());
}

void SaveIPD(const std::string& file_path, const MultiCamera& multi_camera, const IMUState& imu_state) {
  // 创建一个unordered_map来存储转换矩阵
  //  std::unordered_map<int, Eigen::Matrix4d> T_b_ci;
  //  Eigen::Matrix4d T_b_c0 = imu_state.Tbc0;
  //  for (const auto &[cid, T_c0ci] : multi_camera.cid_to_T01) {
  //    Eigen::Matrix4d T_bci = T_b_c0 * T_c0ci;
  //    T_b_ci[cid] = T_bci;
  //  }
  //
  //  if (T_b_ci.find(6) == T_b_ci.end() || T_b_ci.find(7) == T_b_ci.end()) {
  //    return;
  //  }
  //
  //  Eigen::Matrix4d TIc6 = T_b_ci.at(6);
  //  Eigen::Matrix4d TIc7 = T_b_ci.at(7);
  //
  //  Eigen::Vector3d t_middle = (TIc6.block(0, 3, 3, 1) + TIc7.block(0, 3, 3,
  //  1)) * 0.5;
  //
  //  Eigen::Vector3d t_left = t_middle - TIc6.block(0, 3, 3, 1);
  //  Eigen::Vector3d t_right = t_middle - TIc7.block(0, 3, 3, 1);
  //
  //  std::ofstream file(file_path);
  //  if (file.is_open()) {
  //    file << t_middle[0] << "," << t_middle[1] << "," << t_middle[2] <<
  //    std::endl; file << t_left[0] << "," << t_left[1] << "," << t_left[2] <<
  //    std::endl; file << t_right[0] << "," << t_right[1] << "," << t_right[2]
  //    << std::endl; file.close();
  //  } else {
  //    YLOG_ERROR("无法打开文件: %s", file_path.c_str());
  //  }
}

void SaveBoardExtrinsic(const std::string& file_path, const CalibBoards& calib_boards) {
  std::ofstream file(file_path);
  if (!file.is_open()) {
    YLOG_ERROR("Failed to open file for writing:%s", file_path.c_str());
    return;
  }

  for (const auto& mat : calib_boards.id_to_T01) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        file << mat(i, j);
        if (j < 3) file << " ";
      }
      file << "\n";
    }
    file << "\n";
  }

  file.close();
}
void LoadBoardExtrinsic(const std::string& file_path, CalibBoards& calib_boards) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << file_path << std::endl;
    return;
  }

  calib_boards.id_to_T01.clear();
  Mat4 mat;
  while (file) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        file >> mat(i, j);
      }
    }
    if (file) {  // To avoid pushing back an incomplete matrix at the end
      calib_boards.id_to_T01.push_back(mat);
    }
  }

  file.close();
}
}  // namespace dso