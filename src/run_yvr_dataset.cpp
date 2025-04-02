/**
 * This file is based on the file main_dso_pangolin.cpp of the project DSO
 * written by Jakob Engel. It has been modified by Lukas von Stumberg for the
 * inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
 *
 * Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

// Main file for running on datasets, based on the main file of DSO.

#include "util/MainSettings.h"
#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"

#include "dso/util/DatasetReader.h"
#include "dso/util/globalCalib.h"
#include "dso/util/globalFuncs.h"
#include "dso/util/settings.h"
#include "util/TimeMeasurement.h"
#include <boost/thread.hpp>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "dso/util/NumType.h"

#include <util/SettingsUtil.h>

#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include "../camera_model/camera_base.h"
#include "../camera_model/pinhole_camera.h"
#include "dso/FullSystem/algs_tools_images_buffer.h"
#include "dso/camera_model/calib_xml.h"
#include "dso/config/config.h"
#include "dso/frontend/CameraDetection.h"
#include <Eigen/Dense> // Eigen库的头文件
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/eigen.hpp> // OpenCV与Eigen的桥接头文件
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> // OpenCV的核心头文件，或者只包含<opencv2/core.hpp>
#include <thread>

std::string gtFile = "";
std::string source = "";
std::string imuFile = "";

bool is_reverse = false;
int start = 0;
int ending = 100000;
int maxPreloadImages =
    0; // If set we only preload if there are less images to be loade.
bool useSampleOutput = false;
MultiCamera multi_camera_calibed;
aligned_vector<CalibFrame> frameInfo_bak;
using namespace dso;

dmvio::MainSettings mainSettings;
dmvio::IMUCalibration imuCalibration;
dmvio::IMUSettings imuSettings;
std::array<std::pair<cv::Mat, cv::Mat>, kCameraNumUsed> cid_to_undist_map;
Mat3 K, Kinv;
void GenUndistortionMap(MultiCamera &multi_camera, const int &width,
                        const int &height, const int &cam_num) {
  cv::Size image_size = cv::Size(width, height);
  number_t fov_rad = 120.0 * kOur_PI / 180.0;
  number_t focal =
      static_cast<number_t>(width) / (2.0 * std::tan(fov_rad / 2.0));
  K << focal, 0, 0.5 * static_cast<number_t>(width), 0, focal,
      0.5 * static_cast<number_t>(height), 0, 0, 1;
  Kinv = K.inverse();
  Vec2 proj;
  for (size_t cid = 0; cid < cam_num; ++cid) {
    cid_to_undist_map[cid].first.create(image_size, CV_32FC1);
    cid_to_undist_map[cid].second.create(image_size, CV_32FC1);
    for (size_t col = 0; col < width; ++col) {
      for (size_t row = 0; row < height; ++row) {
        Vec3 uv =
            Vec3(static_cast<number_t>(col), static_cast<number_t>(row), 1);
        Vec3 bearing = Kinv * uv;
        multi_camera.cid_to_cam.at(cid)->Project(bearing, proj);
        cid_to_undist_map[cid].first.at<float>(row, col) =
            static_cast<float>(proj.x());
        cid_to_undist_map[cid].second.at<float>(row, col) =
            static_cast<float>(proj.y());
      }
    }
  }
}
void VigCorrection(
    cv::Mat &image,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &vig_mat) {
  uint8_t raw_val;
  float viged_val;
  cv::Mat img_cv_after_vig = cv::Mat(image.rows, image.cols, CV_8UC1);
  for (size_t col = 0; col < img_cv_after_vig.cols; ++col) {
    for (size_t row = 0; row < img_cv_after_vig.rows; ++row) {
      float vig = vig_mat(row, col);
      // vig = 1.0;
      raw_val = image.at<uint8_t>(row, col);

      if (vig < 0.15) {
        viged_val = 0;
      } else {
        viged_val = static_cast<float>(raw_val) / vig;
        if (viged_val >= 255) {
          viged_val = 255;
        }
      }
      //      std::cout << "raw_val: " << static_cast<int>(raw_val) << ",
      //      viged_val: " << viged_val << ", vig: " << vig
      //                << std::endl;
      img_cv_after_vig.at<uint8_t>(row, col) = static_cast<uint8_t>(viged_val);
    }
  }

  image = img_cv_after_vig.clone();
}
static ImuDataSingle interpolate_data(
    const ImuDataSingle &imu_1, /// imu at begining of interpolation interval
    const ImuDataSingle &imu_2, /// imu at end of interpolation interval
    int64_t timestamp)          /// Timestamp being interpolated to
{
  /// time-distant lambda
  double lambda = static_cast<double>(timestamp - imu_1.timestamp_ns) /
                  static_cast<double>(imu_2.timestamp_ns - imu_1.timestamp_ns);
  /// interpolate between the two times
  ImuDataSingle data;
  data.timestamp_ns = timestamp;
  data.acc = (1 - lambda) * imu_1.acc + lambda * imu_2.acc;
  data.gyro = (1 - lambda) * imu_1.gyro + lambda * imu_2.gyro;
  return data;
}
std::vector<ImuDataSingle>
select_IMU_readings(const CalibIO::ImuJsonData
                        &imu_data, /// IMU data we will select measurements from
                    int64_t time0, /// Start timestamp
                    int64_t time1) { /// End timestamp
  /// Vector IMU Readings
  std::vector<ImuDataSingle> prop_data;
  /// Ensure we have some measurements
  if (imu_data.acc_data.empty()) {
    printf("Propagator::select_IMU_readings(): No IMU measurements.\n");
    return prop_data;
  }
  /// Loop through and find all the needed measurements to propagate with
  /// The measurement is split based on the given state time and the update
  /// timestamp

  for (size_t i = 0; i < imu_data.acc_data.size() - 1; i++) {
    /// Start of the integration period

    /// If the next timestamp is greater than our current state time
    /// And the current is not greater than it
    /// Then we should split our current IMU measurement
    /// The cond below is t_{m_i} < t_{s} < t_{m_{i+1}}
    /// Here we determine the imu measurement corresponding to the start of the
    /// integration time stamp i.e. time0
    if (imu_data.acc_data[i].timestamp_ns < time0 &&
        time0 < imu_data.acc_data[i + 1].timestamp_ns) {
      // printf("!!!!! interpolate front\n");
      ImuDataSingle data = interpolate_data(
          ImuDataSingle(imu_data.acc_data[i].data, imu_data.gyro_data[i].data,
                        imu_data.acc_data[i].timestamp_ns),
          ImuDataSingle(imu_data.acc_data[i + 1].data,
                        imu_data.gyro_data[i + 1].data,
                        imu_data.acc_data[i + 1].timestamp_ns),
          time0);
      prop_data.push_back(data);
      //            printf("!!!!! interpolate front, before: [%f %f]. after,:
      //            [%f]\n", imu_data.acc_data[i].timestamp_ns,
      //                   imu_data.acc_data[i + 1].timestamp_ns,
      //                   static_cast<double>(prop_data.back().timestamp_ns) *
      //                   1e-9);
      continue;
    }

    /// Middle of the integration period
    /// If our IMU measurement is in between our propagation period
    /// Then we should just append the whole measurement time to our propagation
    /// vector
    if (time0 <= imu_data.acc_data[i].timestamp_ns &&
        imu_data.acc_data[i + 1].timestamp_ns <= time1) {
      prop_data.push_back(ImuDataSingle(imu_data.acc_data[i].data,
                                        imu_data.gyro_data[i].data,
                                        imu_data.acc_data[i].timestamp_ns));
      continue;
    }

    /// End of the integration period
    /// If the current timestamp is greater than the update time
    /// We should split the next IMU measurement to the update time
    if (imu_data.acc_data[i + 1].timestamp_ns > time1) {
      // printf("!!!!! interpolate back\n");
      if (imu_data.acc_data[i].timestamp_ns > time1) {
        // break;
        ImuDataSingle data = interpolate_data(
            ImuDataSingle(imu_data.acc_data[i - 1].data,
                          imu_data.gyro_data[i - 1].data,
                          imu_data.acc_data[i - 1].timestamp_ns),
            ImuDataSingle(imu_data.acc_data[i].data, imu_data.gyro_data[i].data,
                          imu_data.acc_data[i].timestamp_ns),
            time1);
        prop_data.push_back(data);
      } else {
        prop_data.push_back(ImuDataSingle(imu_data.acc_data[i].data,
                                          imu_data.gyro_data[i].data,
                                          imu_data.acc_data[i].timestamp_ns));
      }
      if (prop_data.at(prop_data.size() - 1).timestamp_ns != time1) {
        ImuDataSingle data = interpolate_data(
            ImuDataSingle(imu_data.acc_data[i].data, imu_data.gyro_data[i].data,
                          imu_data.acc_data[i].timestamp_ns),
            ImuDataSingle(imu_data.acc_data[i + 1].data,
                          imu_data.gyro_data[i + 1].data,
                          imu_data.acc_data[i + 1].timestamp_ns),
            time1);
        prop_data.push_back(data);
      }
      //            printf("!!!!! interpolate back, before: [%f %f]. after,:
      //            [%f]\n", imu_data.acc_data[i].timestamp,
      //                   imu_data.acc_data[i + 1].timestamp,
      //                   static_cast<double>(prop_data.back().timestamp_ns) *
      //                   1e-9);
      break;
    }
  }

  /// Check that we have at least one measurement to propagate with
  if (prop_data.empty()) {
    printf("Propagator::select_imu_readings(): No IMU measurement to propgate "
           "with (%d of 2) \n",
           (int)prop_data.size());
    return prop_data;
  }

  /// Loop through and ensure we do not have an zero dt values
  /// This would cause the noise covariance to be Infinity
  for (size_t i = 0; i < prop_data.size() - 1; i++) {
    if (std::abs(prop_data.at(i + 1).timestamp_ns -
                 prop_data.at(i).timestamp_ns) < 10) {
      printf("Propagator::select_imu_readings(): Zero DT between IMU reading "
             "%d and %d, removing it!\n",
             (int)i, (int)(i + 1));
      prop_data.erase(prop_data.begin() + i);
      i--;
    }
  }

  /// Check that we have at least one measurement to propagate with
  if (prop_data.size() < 2) {
    printf("Propagator::select_imu_readings(): No IMU measurements to "
           "propagate with (%d of 2).\n",
           (int)prop_data.size());
    return prop_data;
  }

  return prop_data;
}
void CorrectImuReadings(dso::CalibIO::ImuJsonData &imu_data,
                        const IMUState &imu_state) {
  Eigen::Matrix3d inv_acc_scale;
  Eigen::Matrix3d inv_gyro_scale;

  inv_acc_scale.setZero();
  inv_gyro_scale.setZero();

  inv_acc_scale(0, 0) = imu_state.ka[0];
  inv_acc_scale(1, 1) = imu_state.ka[1];
  inv_acc_scale(2, 2) = imu_state.ka[2];
  inv_acc_scale(0, 1) = imu_state.na[0];
  inv_acc_scale(0, 2) = imu_state.na[1];
  inv_acc_scale(1, 2) = imu_state.na[2];

  inv_gyro_scale(0, 0) = imu_state.kg[0];
  inv_gyro_scale(1, 1) = imu_state.kg[1];
  inv_gyro_scale(2, 2) = imu_state.kg[2];
  inv_gyro_scale(0, 1) = imu_state.ng[0];
  inv_gyro_scale(0, 2) = imu_state.ng[1];
  inv_gyro_scale(1, 2) = imu_state.ng[2];

  inv_acc_scale = inv_acc_scale + Eigen::Matrix3d::Identity();
  inv_gyro_scale = inv_gyro_scale + Eigen::Matrix3d::Identity();

  Mat3 acc_scale = inv_acc_scale.inverse();   // - Eigen::Matrix3d::Identity();
  Mat3 gyro_scale = inv_gyro_scale.inverse(); // - Eigen::Matrix3d::Identity();
  Mat3 gyro_rotation = ExpSO3(imu_state.ombg);

  std::cout << "acc_scale:\n" << acc_scale << std::endl;
  std::cout << "gyro_scale:\n" << gyro_scale << std::endl;
  std::cout << "gyro_rotation:\n" << gyro_rotation << std::endl;
  std::cout << "imu_state.time_delay: " << imu_state.time_delay << std::endl;

  for (int id = 0; id < imu_data.acc_data.size(); ++id) {
    (imu_data.acc_data)[id].timestamp -= (imu_state.time_delay - 100);
    (imu_data.acc_data)[id].timestamp_ns =
        static_cast<uint64_t>(imu_data.acc_data[id].timestamp * 1e9);

    (imu_data.gyro_data)[id].timestamp = (imu_data.acc_data)[id].timestamp;
    (imu_data.gyro_data)[id].timestamp_ns =
        (imu_data.acc_data)[id].timestamp_ns;

    (imu_data.acc_data)[id].data =
        acc_scale * ((imu_data.acc_data)[id].data - imu_state.acc_bias);
    (imu_data.gyro_data)[id].data =
        gyro_rotation * gyro_scale *
        ((imu_data.gyro_data)[id].data - imu_state.w_bias);
  }
}
std::map<int64_t, ImuDataSingle>
StackImuReadings(const CalibIO::ImuJsonData &imu_data,
                 aligned_vector<CalibFrame> &frameInfo) {
  // std::vector<ImuDataSingle> imu_stack;
  std::map<int64, ImuDataSingle> imu_stack;
  frameInfo[0].timestamp_ns =
      static_cast<int64_t>(frameInfo[0].timestamp * 1e9);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    frameInfo[0].cid_to_exposure_time[cid] *= 1000.0;
  }
  for (int i = 0; i < frameInfo.size() - 1; ++i) {
    frameInfo[i + 1].timestamp_ns =
        static_cast<int64_t>(frameInfo[i + 1].timestamp * 1e9);
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      frameInfo[i + 1].cid_to_exposure_time[cid] *= 1000.0;
    }
    std::vector<ImuDataSingle> imus = select_IMU_readings(
        imu_data, frameInfo[i].timestamp_ns, frameInfo[i + 1].timestamp_ns);
    // printf("i: %d\n", i);
    for (int id = 0; id < imus.size(); ++id) {
      if (imu_stack.find(imus[id].timestamp_ns) == imu_stack.end()) {
        imu_stack.insert(std::make_pair(imus[id].timestamp_ns, imus[id]));
      }
      if (id < imus.size() - 1) {
        //                std::cout << "time: " << imus[id].timestamp_ns << ",
        //                acc: " << imus[id].acc.transpose() << ", gyro: "
        //                          << imus[id].gyro.transpose() << ", dt: " <<
        //                          static_cast<double>(imus[id+1].timestamp_ns
        //                          - imus[id].timestamp_ns) * 1e-6<<std::endl;
      }
    }
    // imu_stack.insert(imu_stack.end(), imus.begin(), imus.end());
  }

  for (int i = 0; i < frameInfo.size(); ++i) {
    int hit = 0;
    for (const std::pair<const int64_t, ImuDataSingle> &pair : imu_stack) {
      if (frameInfo[i].timestamp_ns == pair.second.timestamp_ns) {
        hit++;
      }
    }
    // std::cout <<"i: " << i << ", hit: " << hit << std::endl;
    assert(hit == 1);
  }
  return imu_stack;
}
void my_exit_handler(int s) {
  printf("Caught signal %d\n", s);
  exit(1);
}

void exitThread() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  while (true)
    pause();
}

void run(ImageFolderReader *reader, IOWrap::PangolinDSOViewer *viewer) {
  //    MultiCamera multi_camera = multi_camera_calibed;
  //    multi_camera.cam_num = kCameraNumUsed;
  //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //        multi_camera.cid_to_T01[cid].setIdentity();
  //        multi_camera.cid_to_T01_SE3[cid].setRotationMatrix(Mat3::Identity());
  //        multi_camera.cid_to_T01_SE3[cid].translation().setZero();
  //        multi_camera.cid_to_T01_SE3_inv[cid] =
  //                multi_camera.cid_to_T01_SE3[cid].inverse();
  //        multi_camera.cid_to_T01_inv_Adj[cid] =
  //                multi_camera.cid_to_T01_SE3_inv[cid].Adj();
  //    }

  if (setting_photometricCalibration > 0 &&
      reader->getPhotometricGamma() == 0 && false) {
    printf("ERROR: dont't have photometric calibation. Need to use commandline "
           "options mode=1 or mode=2 ");
    exit(1);
  }

  int lstart = start;
  int lend = ending;
  int linc = 1;
  if (is_reverse) {
    assert(!setting_useIMU); // Reverse is not supported with IMU data at the
    // moment!
    printf("REVERSE!!!!");
    lstart = ending - 1;
    if (lstart >= frameInfo_bak.size())
      lstart = frameInfo_bak.size() - 1;
    lend = start;
    linc = -1;
  }

  bool linearizeOperation = (mainSettings.playbackSpeed == 0);

  if (linearizeOperation && setting_minFramesBetweenKeyframes < 0) {
    setting_minFramesBetweenKeyframes = -setting_minFramesBetweenKeyframes;
    std::cout << "Using setting_minFramesBetweenKeyframes="
              << setting_minFramesBetweenKeyframes
              << " because of non-realtime mode." << std::endl;
  }

  FullSystem *fullSystem = new FullSystem(linearizeOperation, imuCalibration,
                                          imuSettings, &multi_camera_calibed);
  fullSystem->setGammaFunction(reader->getPhotometricGamma());

  if (viewer != 0) {
    fullSystem->outputWrapper.push_back(viewer);
  }

  std::unique_ptr<IOWrap::SampleOutputWrapper> sampleOutPutWrapper;
  if (useSampleOutput) {
    sampleOutPutWrapper.reset(new IOWrap::SampleOutputWrapper());
    fullSystem->outputWrapper.push_back(sampleOutPutWrapper.get());
  }

  std::vector<int> idsToPlay;
  std::vector<double> timesToPlayAt;
  for (int i = lstart;
       i >= 0 && i < frameInfo_bak.size() && linc * i < linc * lend;
       i += linc) {
    idsToPlay.push_back(i);
    if (timesToPlayAt.size() == 0) {
      timesToPlayAt.push_back((double)0);
      // timesToPlayAt.push_back(reader->getTimestamp(0));
    } else {
      double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
      double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
      timesToPlayAt.push_back(timesToPlayAt.back() +
                              fabs(tsThis - tsPrev) /
                                  mainSettings.playbackSpeed);
    }
  }

  if (mainSettings.preload && maxPreloadImages > 0) {
    if (frameInfo_bak.size() > maxPreloadImages) {
      printf("maxPreloadImages EXCEEDED! NOT PRELOADING!\n");
      mainSettings.preload = false;
    }
  }

  std::vector<ImageAndExposure *> preloadedImages;
  if (mainSettings.preload) {
    printf("LOADING ALL IMAGES!\n");
    for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {
      int i = idsToPlay[ii];
      preloadedImages.push_back(reader->getImage(i));
    }
  }

  struct timeval tv_start;
  gettimeofday(&tv_start, NULL);
  clock_t started = clock();
  double sInitializerOffset = 0;

  bool gtDataThere = false; // reader->loadGTData(gtFile);

  bool imuDataSkipped = false;
  dmvio::IMUData skippedIMUData;
  for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {
    if (!fullSystem->initialized) // if not initialized: reset start time.
    {
      gettimeofday(&tv_start, NULL);
      started = clock();
      sInitializerOffset = timesToPlayAt[ii];
    }

    int i = idsToPlay[ii];

    ImageAndExposure *img;
    if (mainSettings.preload)
      img = preloadedImages[ii];
    else
      img = reader->getImage2(i);

    bool skipFrame = false;
    if (mainSettings.playbackSpeed != 0) {
      struct timeval tv_now;
      gettimeofday(&tv_now, NULL);
      double sSinceStart =
          sInitializerOffset +
          ((tv_now.tv_sec - tv_start.tv_sec) +
           (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

      if (sSinceStart < timesToPlayAt[ii])
        usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
      else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
        printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii,
               timesToPlayAt[ii], sSinceStart);
        skipFrame = true;
      }
    }

    dmvio::GTData data;
    bool found = false;
    if (gtDataThere) {
      data = reader->getGTData(i, found);
    }

    std::unique_ptr<dmvio::IMUData> imuData;
    if (setting_useIMU) {
      imuData = std::make_unique<dmvio::IMUData>(reader->getIMUData(i));
      printf("!!!!!! get imu i: %d, imu_size: %d\n", i, imuData->size());
      //            for (int id = 0; id < imuData->size(); ++id) {
      //                std::cout <<"time: " <<
      //                (*imuData)[id].getIntegrationTime() << std::endl;
      //            }
      // std::exit(8);
    }
    if (!skipFrame) {
      if (imuDataSkipped && imuData) {
        imuData->insert(imuData->begin(), skippedIMUData.begin(),
                        skippedIMUData.end());
        skippedIMUData.clear();
        imuDataSkipped = false;
      }
      // TODO entrance
      fullSystem->addActiveFrame(img, i, imuData.get(),
                                 (gtDataThere && found) ? &data : 0);
      if (gtDataThere && found && !disableAllDisplay) {
        viewer->addGTCamPose(data.pose);
      }
    } else if (imuData) {
      imuDataSkipped = true;
      skippedIMUData.insert(skippedIMUData.end(), imuData->begin(),
                            imuData->end());
    }

    delete img;

    if (fullSystem->initFailed || setting_fullResetRequested) {
      if (ii < 250 || setting_fullResetRequested) {
        printf("RESETTING!\n");
        std::vector<IOWrap::Output3DWrapper *> wraps =
            fullSystem->outputWrapper;
        delete fullSystem;
        for (IOWrap::Output3DWrapper *ow : wraps)
          ow->reset();

        fullSystem = new FullSystem(linearizeOperation, imuCalibration,
                                    imuSettings, &multi_camera_calibed);
        fullSystem->setGammaFunction(reader->getPhotometricGamma());
        fullSystem->outputWrapper = wraps;

        setting_fullResetRequested = false;
      }
    }

    if (viewer != nullptr && viewer->shouldQuit()) {
      std::cout << "User closed window -> Quit!" << std::endl;
      break;
    }

    if (fullSystem->isLost) {
      printf("LOST!!\n");
      break;
    }
  }
  fullSystem->blockUntilMappingIsFinished();
  clock_t ended = clock();
  struct timeval tv_end;
  gettimeofday(&tv_end, NULL);

  //    fullSystem->printResult(imuSettings.resultsPrefix + "result.txt", false,
  //    false, true); fullSystem->printResult(imuSettings.resultsPrefix +
  //    "resultKFs.txt", true, false, false);
  fullSystem->printResult(imuSettings.resultsPrefix + "resultScaled.txt", false,
                          true, true);

  dmvio::TimeMeasurement::saveResults(imuSettings.resultsPrefix +
                                      "timings.txt");

  int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
  double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) -
                                    reader->getTimestamp(idsToPlay.back()));
  double MilliSecondsTakenSingle =
      1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
  double MilliSecondsTakenMT =
      sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                            (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
  printf("\n======================"
         "\n%d Frames (%.1f fps)"
         "\n%.2fms per frame (single core); "
         "\n%.2fms per frame (multi core); "
         "\n%.3fx (single core); "
         "\n%.3fx (multi core); "
         "\n======================\n\n",
         numFramesProcessed, numFramesProcessed / numSecondsProcessed,
         MilliSecondsTakenSingle / numFramesProcessed,
         MilliSecondsTakenMT / (float)numFramesProcessed,
         1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
         1000 / (MilliSecondsTakenMT / numSecondsProcessed));
  fullSystem->printFrameLifetimes();
  if (setting_logStuff) {
    std::ofstream tmlog;
    tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
    tmlog << 1000.0f * (ended - started) /
                 (float)(CLOCKS_PER_SEC * frameInfo_bak.size())
          << " "
          << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
              (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                 (float)frameInfo_bak.size()
          << "\n";
    tmlog.flush();
    tmlog.close();
  }

  for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) {
    ow->join();
  }

  printf("DELETE FULLSYSTEM!\n");
  delete fullSystem;

  printf("DELETE READER!\n");
  delete reader;

  printf("EXIT NOW!\n");
}

int main(int argc, char **argv) {

  std::string config_path =
      "/home/roger/work/dm-vio/dm-vio/src/dso/config/calibconfig_stage0.toml";
  CalibIO::ConfigData configParams(config_path);
  IMUState imu_state_temp, imu_state;
  MultiCamera multi_camera, multi_camera_vi;
  LoadXML(configParams.dataSet + "/results/device_calibration_gray_vi_5.xml",
          multi_camera, imu_state_temp);
  multi_camera.cids = {0, 1, 2, 3};
  multi_camera.cam_num = kCameraNumUsed;
  for (const int &cid : multi_camera.cids) {
    if (cid < kCameraNumUsed) {
      multi_camera.cid_to_cam.at(cid)->PrintIntri();
    }
  }
  multi_camera_calibed = multi_camera;

  multi_camera_calibed.Tbc0.setRotationMatrix(
      imu_state_temp.Tbc0.topLeftCorner<3, 3>());
  multi_camera_calibed.Tbc0.translation() =
      imu_state_temp.Tbc0.topRightCorner<3, 1>();
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    // multi_camera.cid_to_T01[cid].setIdentity();
    multi_camera_calibed.cid_to_T01_SE3[cid].setRotationMatrix(
        multi_camera_calibed.cid_to_T01[cid].topLeftCorner<3, 3>());
    multi_camera_calibed.cid_to_T01_SE3[cid].translation() =
        multi_camera_calibed.cid_to_T01[cid].topRightCorner<3, 1>();
    multi_camera_calibed.cid_to_T01_SE3_inv[cid] =
        multi_camera_calibed.cid_to_T01_SE3[cid].inverse();
    multi_camera_calibed.cid_to_T01_inv_Adj[cid] =
        multi_camera_calibed.cid_to_T01_SE3_inv[cid].Adj();
  }
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    multi_camera_calibed.cid_to_Tbc_SE3[cid] =
        multi_camera_calibed.Tbc0 * multi_camera_calibed.cid_to_T01_SE3[cid];
  }
  aligned_vector<CalibFrame> frameInfo, frameInfo_rgb;
  aligned_vector<std::unordered_map<
      int /*cid*/, std::unordered_map<int /*bid*/, aligned_vector<PointVM>>>>
      frameInfoImageDataArranged;
  CamCalib::CameraDetection detect(configParams);
  CalibIO::ImuJsonData imuData;

  if (configParams.calib_stage != dso::CalibIO::GRAY_RGB) {
    detect.pipeline(frameInfo, &imuData, false);
  } else {
    printf("doesn't support this stage, please check\n");
    std::exit(-1);
  }
  for (int id = 10; id < frameInfo.size() - 10; ++id) {
    frameInfo_bak.emplace_back(frameInfo[id]);
  }
  CorrectImuReadings(imuData, imu_state_temp);
  int w = 640 * 1;
  int h = 480 * 1;
  int cid = 0;

  bool show = false;

  GenUndistortionMap(multi_camera_calibed, w, h, kCameraNumUsed);

  dso::ImagesBuffer::Initial(40, w, h);
  std::array<std::array<std::vector<number_t>, kCameraNumUsed>, PYR_LEVELS>
      level_cid_to_param;
  for (int level = 0; level < PYR_LEVELS; ++level) {
    int ww = w >> level;
    int hh = h >> level;

    float fx = K(0, 0) * std::pow(2, -level);
    float fy = K(1, 1) * std::pow(2, -level);
    float cx = (K(0, 2) + 0.5) / ((int)1 << level) - 0.5;
    float cy = (K(1, 2) + 0.5) / ((int)1 << level) - 0.5;

    for (int cam = 0; cam < kCameraNumUsed; ++cam) {
      multi_camera_calibed.level_cid_to_K_temp[level][cam].setIdentity();
      multi_camera_calibed.level_cid_to_K_temp[level][cam](0, 0) = fx;
      multi_camera_calibed.level_cid_to_K_temp[level][cam](1, 1) = fy;
      multi_camera_calibed.level_cid_to_K_temp[level][cam](0, 2) = cx;
      multi_camera_calibed.level_cid_to_K_temp[level][cam](1, 2) = cy;
      level_cid_to_param[level][cam] = {fx, fy, cx, cy};
      multi_camera_calibed.level_cid_to_Kinv_temp[level][cam] =
          multi_camera_calibed.level_cid_to_K_temp[level][cam].inverse();
      multi_camera_calibed.level_cid_to_cam_pinhole[level][cam] =
          new PinholeCamera(cam, ww, hh, level_cid_to_param[level][cam].data());
    }
  }

  std::string path_to_vig_img =
      "/home/roger/work/smartgit/dot001/yvrcalibration_dot/"
      "vignette_0.png"; // "../../vignette_0.png";
  cv::Mat vignette_img = cv::imread(path_to_vig_img, -1);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> vig_mat;
  vig_mat.resize(480, 640);
  if (vignette_img.rows != 480 || vignette_img.cols != 640) {
    printf("Vignette img size not equal to img size\n");
    exit(-1);
  }
  for (int col = 0; col < 640; ++col) {
    for (int row = 0; row < 480; ++row) {
      vig_mat(row, col) = vignette_img.at<uint16_t>(row, col) / 65535.0;
    }
  }

  std::map<int64_t, ImuDataSingle> imu_stack =
      StackImuReadings(imuData, frameInfo_bak);

  // std::exit(2);

  std::array<cv::Mat, 4> show_mat_vec;
#if 0
    for (size_t i = 0; i < frameInfo_bak.size() /*&& key != 27*/; i++) {
        //    cerr <<
        //    "######################################################################################################"
        //            "##################################################### FRAME: "
        //         << i << ", cam_id: " << cam_id << endl;
        for (int cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
            std::string image_path = frameInfo_bak[i].cid_to_img_file_path.at(cam_id);
            // cerr << "Reading..." << image_path << endl;
            // if (files[i].back() == '.') continue;  // skip . and ..
            cv::Mat image = cv::imread(image_path, 0);
            cv::Mat image_before = image.clone();
            //VigCorrection(image, vig_mat);
            cv::remap(image, image, cid_to_undist_map[cam_id].first, cid_to_undist_map[cam_id].second, cv::INTER_CUBIC);
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
            show_mat_vec[cam_id] = image.clone();
        }
        cv::Mat img1, img2, img_show;
        cv::hconcat(show_mat_vec[1], show_mat_vec[2], img1);
        cv::hconcat(show_mat_vec[0], show_mat_vec[3], img2);
        cv::vconcat(img1, img2, img_show);
        cv::imshow("Cam", img_show);

        cv::waitKey(0);
    }
#endif
  // std::exit(-1);

  setlocale(LC_ALL, "C");

#ifdef DEBUG
  std::cout << "DEBUG MODE!" << std::endl;
#endif

  bool use16Bit = false;

  auto settingsUtil = std::make_shared<dmvio::SettingsUtil>();

  // Create Settings files.
  imuSettings.registerArgs(*settingsUtil);
  imuCalibration.registerArgs(*settingsUtil);
  mainSettings.registerArgs(*settingsUtil);

  // Dataset specific arguments. For other commandline arguments check out
  // MainSettings::parseArgument, MainSettings::registerArgs, IMUSettings.h and
  // IMUInitSettings.h
  settingsUtil->registerArg("files", source);
  settingsUtil->registerArg("start", start);
  settingsUtil->registerArg("end", ending);
  settingsUtil->registerArg("imuFile", imuFile);
  settingsUtil->registerArg("gtFile", gtFile);
  settingsUtil->registerArg("sampleoutput", useSampleOutput);
  settingsUtil->registerArg("reverse", is_reverse);
  settingsUtil->registerArg("use16Bit", use16Bit);
  settingsUtil->registerArg("maxPreloadImages", maxPreloadImages);

  // This call will parse all commandline arguments and potentially also read a
  // settings yaml file if passed.
  mainSettings.parseArguments(argc, argv, *settingsUtil);

  //    if (mainSettings.imuCalibFile != "") {
  imuCalibration.loadFromFile2(imu_state_temp);
  //    }

  // Print settings to commandline and file.
  std::cout << "Settings:\n";
  settingsUtil->printAllSettings(std::cout);
  {
    std::ofstream settingsStream;
    settingsStream.open(imuSettings.resultsPrefix + "usedSettingsdso.txt");
    settingsUtil->printAllSettings(settingsStream);
  }

  // hook crtl+C.
  boost::thread exThread = boost::thread(exitThread);

  use16Bit = false;
  ImageFolderReader *reader =
      new ImageFolderReader(source, mainSettings.calib, mainSettings.gammaCalib,
                            mainSettings.vignette, use16Bit, true, w, h,
                            &cid_to_undist_map, &vig_mat);
  reader->loadIMUData2(imu_stack, frameInfo_bak);
  reader->setGlobalCalibration2(K.cast<float>(), w, h);
  // std::exit(2);
  if (!disableAllDisplay) {
    IOWrap::PangolinDSOViewer *viewer = new IOWrap::PangolinDSOViewer(
        wG[0], hG[0], false, settingsUtil, nullptr, &multi_camera_calibed);

    boost::thread runThread = boost::thread(boost::bind(run, reader, viewer));

    viewer->run();

    delete viewer;

    // Make sure that the destructor of FullSystem, etc. finishes, so all log
    // files are properly flushed.
    runThread.join();
  } else {
    run(reader, 0);
  }

  return 0;
}
