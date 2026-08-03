/**
 * This file is part of DSO, written by Jakob Engel.
 * It has been modified by Lukas von Stumberg for the inclusion in DM-VIO
 * (http://vision.in.tum.de/dm-vio).
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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/CoarseTracker.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"
#include "util/ImageAndExposure.h"
#include <boost/mpl/print.hpp>
#include <cmath>

#include "../camera_model/pinhole_camera.h"
#include "GTSAMIntegration/ExtUtils.h"
#include "IOWrapper/ImageRW.h"
#include "algs_tools_images_buffer.h"
#include "util/TimeMeasurement.h"

using dmvio::GravityInitializer;

namespace dso {
// Hessian矩阵计数, 有点像 shared_ptr
int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

std::ofstream output_ostr_ = std::ofstream("/home/roger/work/dm-vio/dm-vio/build/output_dmvio.txt");

boost::mutex FrameShell::shellPoseMutex{};

/********************************
 * @ function: 构造函数
 *
 * @ param:
 *
 * @ note:
 *******************************/
FullSystem::FullSystem(bool linearizeOperationPassed, const dmvio::IMUCalibration& imuCalibration,
                       dmvio::IMUSettings& imuSettings, MultiCamera* p_multi_camera)
    : linearizeOperation(linearizeOperationPassed),
      imuIntegration(&Hcalib, imuCalibration, imuSettings, linearizeOperation),
      secondKeyframeDone(false),
      gravityInit(imuSettings.numMeasurementsGravityInit, imuCalibration),
      shellPoseMutex(FrameShell::shellPoseMutex) {
  setting_useGTSAMIntegration = setting_useIMU;
  baIntegration = imuIntegration.getBAGTSAMIntegration().get();

  Hcalib.setMultiCamera(p_multi_camera);
  int retstat = 0;
  if (setting_logStuff) {
    retstat += system("rm -rf logs");
    retstat += system("mkdir logs");

    retstat += system("rm -rf mats");
    retstat += system("mkdir mats");

    calibLog = new std::ofstream();
    calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
    calibLog->precision(12);

    numsLog = new std::ofstream();
    numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
    numsLog->precision(10);

    coarseTrackingLog = new std::ofstream();
    coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
    coarseTrackingLog->precision(10);

    eigenAllLog = new std::ofstream();
    eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
    eigenAllLog->precision(10);

    eigenPLog = new std::ofstream();
    eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
    eigenPLog->precision(10);

    eigenALog = new std::ofstream();
    eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
    eigenALog->precision(10);

    DiagonalLog = new std::ofstream();
    DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
    DiagonalLog->precision(10);

    variancesLog = new std::ofstream();
    variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
    variancesLog->precision(10);

    nullspacesLog = new std::ofstream();
    nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
    nullspacesLog->precision(10);
  } else {
    nullspacesLog = 0;
    variancesLog = 0;
    DiagonalLog = 0;
    eigenALog = 0;
    eigenPLog = 0;
    eigenAllLog = 0;
    numsLog = 0;
    calibLog = 0;
  }

  poseLog = new std::ofstream();
  poseLog->open("output_dm_vio.txt", std::ios::trunc | std::ios::out);
  poseLog->precision(12);

  rmseLog = new std::ofstream();
  rmseLog->open("output_rmse.txt", std::ios::trunc | std::ios::out);
  rmseLog->precision(12);

  frameEnergyThLog = new std::ofstream();
  frameEnergyThLog->open("output_frameEnergyTh.txt", std::ios::trunc | std::ios::out);
  frameEnergyThLog->precision(12);

  assert(retstat != 293847);  // shell正常执行结束返回这么个值,填充8~15位bit, 有趣

  selectionMap = new float[wG[0] * hG[0] * kCameraNumUsed];

  coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
  coarseTracker = new CoarseTracker(wG[0], hG[0], imuIntegration);
  coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0], imuIntegration);
  coarseInitializer = new CoarseInitializer(wG[0], hG[0], Hcalib.p_multi_camera);
  pixelSelector = new PixelSelector(wG[0], hG[0]);

  statistics_lastNumOptIts = 0;
  statistics_numDroppedPoints = 0;
  statistics_numActivatedPoints = 0;
  statistics_numCreatedPoints = 0;
  statistics_numForceDroppedResBwd = 0;
  statistics_numForceDroppedResFwd = 0;
  statistics_numMargResFwd = 0;
  statistics_numMargResBwd = 0;

  lastCoarseRMSE.setConstant(100);  // 5维向量都=100

  currentMinActDist = 2;
  initialized = false;

  ef = new EnergyFunctional(*baIntegration);
  ef->red = &this->treadReduce;

  isLost = false;
  initFailed = false;

  needNewKFAfter = -1;
  runMapping = true;
  mappingThread = boost::thread(&FullSystem::mappingLoop, this);  // 建图线程单开
  lastRefStopID = 0;

  minIdJetVisDebug = -1;
  maxIdJetVisDebug = -1;
  minIdJetVisTracker = -1;
  maxIdJetVisTracker = -1;

  assert(Hcalib.p_multi_camera = p_multi_camera);
  p_depth_filter_DSM_ = new DepthFilterDSM(p_multi_camera, &estimator_config_);
}

FullSystem::~FullSystem() {
  blockUntilMappingIsFinished();
  // 删除new的ofstream
  if (setting_logStuff) {
    calibLog->close();
    delete calibLog;
    numsLog->close();
    delete numsLog;
    coarseTrackingLog->close();
    delete coarseTrackingLog;
    // errorsLog->close(); delete errorsLog;
    eigenAllLog->close();
    delete eigenAllLog;
    eigenPLog->close();
    delete eigenPLog;
    eigenALog->close();
    delete eigenALog;
    DiagonalLog->close();
    delete DiagonalLog;
    variancesLog->close();
    delete variancesLog;
    nullspacesLog->close();
    delete nullspacesLog;
  }

  poseLog->close();
  delete poseLog;

  rmseLog->close();
  delete rmseLog;

  frameEnergyThLog->close();
  delete frameEnergyThLog;

  delete[] selectionMap;

  for (FrameShell* s : allFrameHistory) delete s;
  for (FrameHessian* fh : unmappedTrackedFrames) delete fh;

  delete coarseDistanceMap;
  delete coarseTracker;
  delete coarseTracker_forNewKF;
  delete coarseInitializer;
  delete pixelSelector;
  delete ef;
  delete p_depth_filter_DSM_;
}

void FullSystem::setOriginalCalib(const VecXf& originalCalib, int originalW, int originalH) {}

//* 设置相机响应函数
void FullSystem::setGammaFunction(float* BInv) {
  if (BInv == 0) return;

  // copy BInv.
  memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

  // invert.
  for (int i = 1; i < 255; i++) {
    // find val, such that Binv[val] = i.
    // I dont care about speed for this, so do it the stupid way.

    for (int s = 1; s < 255; s++) {
      if (BInv[s] <= i && BInv[s + 1] >= i) {
        Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
        break;
      }
    }
  }
  Hcalib.B[0] = 0;
  Hcalib.B[255] = 255;
}

void FullSystem::getMetricScaleTwc(FrameShell* fs, SE3 Tbc0) {
  //        boost::unique_lock<boost::mutex> lock(trackMutex);
  //        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
  Sophus::SE3 camToWorld = fs->camToWorld;
  // Sophus::SE3 camToFirst = firstPose.inverse() * camToWorld;
  SE3 Twb_first = Sophus::SE3d(imuIntegration.getTransformDSOToIMU().transformPose(camToWorld.inverse().matrix()));
  SE3 Twc0_first = Twb_first * Tbc0;
  Eigen::Vector3f p = Twc0_first.translation().cast<float>();
  Eigen::Quaternionf q = Twc0_first.unit_quaternion().cast<float>();
  (*poseLog) << std::fixed << static_cast<double>(fs->timestamp_eval) << " " << p.x() << " " << p.y() << " " << p.z()
             << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

void FullSystem::printResult(std::string file, bool onlyLogKFPoses, bool saveMetricPoses, bool useCamToTrackingRef) {
  boost::unique_lock<boost::mutex> lock(trackMutex);
  boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

  std::ofstream myfile;
  myfile.open(file.c_str());  //, std::ios::binary | std::ios::app | std::ios::in
                              //| std::ios::out);
  myfile << std::setprecision(6);

  for (FrameShell* s : allFrameHistory) {
    if (!s->poseValid) continue;

    if (onlyLogKFPoses && s->marginalizedAt == s->id) continue;

    // firstPose is transformFirstToWorld. We actually want camToFirst here ->
    Sophus::SE3 camToWorld = s->camToWorld;

    // Use camToTrackingReference for nonKFs and the updated camToWorld for KFs.
    if (useCamToTrackingRef && s->keyframeId == -1) {
      camToWorld = s->trackingRef->camToWorld * s->camToTrackingRef;
    }
    Sophus::SE3 camToFirst = firstPose.inverse() * camToWorld;

    if (saveMetricPoses) {
      // Transform pose to IMU frame.
      // not actually camToFirst any more...
      camToFirst = Sophus::SE3d(imuIntegration.getTransformDSOToIMU().transformPose(camToWorld.inverse().matrix()));
    }
    Eigen::Vector3f p = camToFirst.translation().cast<float>();
    Eigen::Quaternionf q = camToFirst.unit_quaternion().cast<float>();
    myfile << std::fixed << static_cast<float>(s->timestamp) << " " << p.x() << " " << p.y() << " " << p.z() << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";  // std::endl;
  }
  myfile.close();
}

//@ 使用确定的运动模型对新来的一帧进行跟踪, 得到位姿和光度参数
std::pair<Vec10, bool> FullSystem::trackNewCoarse(FrameHessian* fh, Sophus::SE3* referenceToFrameHint, Mat33 dRwb) {
  dmvio::TimeMeasurement timeMeasurement(referenceToFrameHint ? "FullSystem::trackNewCoarse"
                                                              : "FullSystem::trackNewCoarseNoIMU");
  assert(allFrameHistory.size() > 0);
  // set pose initialization.

  for (IOWrap::Output3DWrapper* ow : outputWrapper) ow->pushLiveFrame(fh);

  FrameHessian* lastF = coarseTracker->lastRef;  // 参考帧

  AffLight aff_last_2_l = AffLight(0, 0);
  //  std::array<AffLight, kCameraNumUsed> a_aff_last_2_l;
  //  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //      a_aff_last_2_l[cid] = AffLight(0, 0);
  //  }
  //[ ***step 1*** ] 设置不同的运动状态
  // Seems to contain poses reference_to_newframe.
  std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

  if (referenceToFrameHint) {
    // We got a hint (typically from IMU) where our pose is, so we don't need
    // the random initializations below.
    lastF_2_fh_tries.push_back(*referenceToFrameHint);
    {
      // lock on global pose consistency (probably we don't need this for
      // AffineLight, but just to make sure).
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      // Set Affine light to last frame, where tracking was good!:
      for (int i = allFrameHistory.size() - 2; i >= 0; i--) {
        FrameShell* slast = allFrameHistory[i];
        if (slast->trackingWasGood) {
          aff_last_2_l = slast->aff_g2l;
          //          a_aff_last_2_l = slast->cid_to_aff_g2l;
          break;
        }
        if (slast->trackingRef != lastF->shell) {
          std::cout << "WARNING: No well tracked frame with the same tracking "
                       "ref available!"
                    << std::endl;
          aff_last_2_l = lastF->aff_g2l();
          //          for (int cid = 0; cid < kCameraNumUsed; ++cid) {
          //              a_aff_last_2_l[cid] = lastF->aff_g2l(cid);
          //          }
          break;
        }
      }
    }
  }

  if (!referenceToFrameHint) {
    printf("allFrameHistory.size(): %d\n", allFrameHistory.size());
    // std::exit(1);
    if (allFrameHistory.size() == 2) {
      if (setting_useIMU) {
#if 1
        /// idx = 1 must be a vkf, tracking ref is identity pose;
        // lastF_2_fh_tries.push_back(coarseTracker->thisToNext);
        SE3 T_th = SE3();
#if 0
        T_th.setRotationMatrix(
            Hcalib.p_multi_camera->cid_to_Tbc_SE3[0]
                .rotationMatrix()
                .transpose() *
            dRwb.transpose() *
            Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix());
#else
        SE3 dTwb = SE3();
        dTwb.setRotationMatrix(dRwb);
        T_th = Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].inverse() * dTwb.inverse() *
               Hcalib.p_multi_camera->cid_to_Tbc_SE3[0];
#endif
        lastF_2_fh_tries.push_back(T_th);
#else
        FrameShell* slast = allFrameHistory[allFrameHistory.size() - 2];  // 上一帧
        SE3 lastF_2_slast;
        lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;  // 参考帧到上一帧运动
        SE3 dTwb = SE3();
        dTwb.setRotationMatrix(dRwb);
        SE3 T_th = Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].inverse() * dTwb.inverse() *
                   Hcalib.p_multi_camera->cid_to_Tbc_SE3[0] * lastF_2_slast;
        //          T_th.setRotationMatrix(
        //                  Hcalib.p_multi_camera->cid_to_Tbc_SE3[0]
        //                          .rotationMatrix()
        //                          .transpose() *
        //                  dRwb.transpose() *
        //                  Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix()
        //                  * lastF_2_slast.rotationMatrix());
        lastF_2_fh_tries.push_back(T_th);
#endif
      }
      for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
        lastF_2_fh_tries.push_back(SE3());  //? 这个size()不应该是0么
      }
    } else {
      FrameShell* slast = allFrameHistory[allFrameHistory.size() - 2];     // 上一帧
      FrameShell* sprelast = allFrameHistory[allFrameHistory.size() - 3];  // 大上一帧
      SE3 slast_2_sprelast;
      SE3 lastF_2_slast;
      {  // lock on global pose consistency!
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;   // 上一帧和大上一帧的运动
        lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;  // 参考帧到上一帧运动
        aff_last_2_l = slast->aff_g2l;
        //        a_aff_last_2_l = slast->cid_to_aff_g2l;
      }
      /// 匀速模型
      SE3 fh_2_slast = slast_2_sprelast;  // assumed to be the same as fh_2_slast.

      //! 尝试不同的运动
      // get last delta-movement.
      Mat33 dRcc = Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix().transpose() * dRwb.transpose() *
                   Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix();
      SE3 fh_2_slast2_inv = fh_2_slast.inverse();
      fh_2_slast2_inv.setRotationMatrix(dRcc);
      SE3 fh_2_slast2 = fh_2_slast;
      fh_2_slast2.setRotationMatrix(Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix().transpose() * dRwb *
                                    Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix());
      if (setting_useIMU) {
#if 0
        SE3 T_th = fh_2_slast.inverse() * lastF_2_slast;
        T_th.setRotationMatrix(dRcc * lastF_2_slast.rotationMatrix());
#else
        // SE3 T_th = fh_2_slast2_inv * lastF_2_slast;
        SE3 T_th = fh_2_slast2.inverse() * lastF_2_slast;
#endif
        lastF_2_fh_tries.push_back(T_th);
      }
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() *
                                 lastF_2_slast);  // assume double motion (frame skipped)
      lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast);  // assume half motion.
      lastF_2_fh_tries.push_back(lastF_2_slast);                                               // assume zero motion.
      lastF_2_fh_tries.push_back(SE3());  // assume zero motion FROM KF.

      //! 尝试不同的旋转变动
      // just try a TON of different initializations (all rotations). In the
      // end, if they don't work they will only be tried on the coarsest level,
      // which is super fast anyway. also, if tracking rails here we loose, so
      // we really, really want to avoid that.
      for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            fh_2_slast.inverse() * lastF_2_slast *
            SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
      }

      if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
        lastF_2_fh_tries.clear();
        lastF_2_fh_tries.push_back(SE3());
      }
    }
  }

  Vec3 flowVecs = Vec3(100, 100, 100);
  SE3 lastF_2_fh = SE3();
  AffLight aff_g2l = AffLight(0, 0);

  //  std::array<AffLight, kCameraNumUsed> a_aff_g2l;
  //  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //      a_aff_g2l[cid] = AffLight(0, 0);
  //  }

  // as long as maxResForImmediateAccept is not reached, I'll continue through
  // the options. I'll keep track of the so-far best achieved residual for each
  // level in achievedRes.
  //! 把到目前为止最好的残差值作为每一层的阈值
  // If on a coarse level, tracking is WORSE than achievedRes, we will not
  // continue to save time.
  //! 粗层的能量值大, 也不继续优化了, 来节省时间

  bool trackingGoodRet = false;

  Vec5 achievedRes = Vec5::Constant(NAN);
  bool haveOneGood = false;
  int tryIterations = 0;
  //! 逐个尝试
  disable_kf_real = false;
  if (referenceToFrameHint != 0) {
#if 1
    lastF_2_fh_tries.clear();
    lastF_2_fh_tries.push_back(*referenceToFrameHint);
#endif
    disable_kf_real = disable_kf == 1;
  }
  // disable_kf_real = false;
  for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
    //[ ***step 2*** ] 尝试不同的运动状态, 得到跟踪是否良好
    AffLight aff_g2l_this = aff_last_2_l;  // 上一帧的赋值当前帧
    //    std::array<AffLight, kCameraNumUsed> a_aff_g2l_this = a_aff_last_2_l;
    SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
    bool trackingIsGood = coarseTracker->trackNewestCoarse(disable_kf_real, frameHessians, allFrameHistory.size(),
                                                           lastF, fh, lastF_2_fh_this, aff_g2l_this, pyrLevelsUsed - 1,
                                                           achievedRes);  // in each level has to be at least as good as
                                                                          // the last try.
    if (disable_kf_real) {
      disable_kf = 1;
    }
    tryIterations++;

    if (trackingIsGood) {
      trackingGoodRet = true;
    }
    if (!trackingIsGood && setting_useIMU) {
      std::cout << "WARNING: Coarse tracker thinks that tracking was not good!" << std::endl;
      // In IMU mode we can still estimate the pose sufficiently, even if vision
      // is bad.
      trackingIsGood = true;
    }

    if (i != 0) {
      printf(
          "RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f "
          "%f): %f %f %f %f %f -> %f %f %f %f %f \n",
          i, i, pyrLevelsUsed - 1, aff_g2l_this.a, aff_g2l_this.b, achievedRes[0], achievedRes[1], achievedRes[2],
          achievedRes[3], achievedRes[4], coarseTracker->lastResiduals[0], coarseTracker->lastResiduals[1],
          coarseTracker->lastResiduals[2], coarseTracker->lastResiduals[3], coarseTracker->lastResiduals[4]);
    }

    //[ ***step 3*** ] 如果跟踪正常, 并且0层残差比最好的还好留下位姿,
    //保存最好的每一层的能量值
    // do we have a new winner?
    if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) &&
        !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
      // printf("take over. minRes %f -> %f!\n", achievedRes[0],
      // coarseTracker->lastResiduals[0]);
      // TODO average optical flow
      flowVecs = coarseTracker->lastFlowIndicators;
      aff_g2l = aff_g2l_this;
      //      a_aff_g2l = a_aff_g2l_this;
      lastF_2_fh = lastF_2_fh_this;
      haveOneGood = true;
    }

    // take over achieved res (always).
    if (haveOneGood) {
      for (int i = 0; i < 5; i++) {
        if (!std::isfinite((float)achievedRes[i]) ||
            achievedRes[i] > coarseTracker->lastResiduals[i])  // take over if achievedRes is
                                                               // either bigger or NAN.
          achievedRes[i] = coarseTracker->lastResiduals[i];    // 里面保存的是各层得到的能量值
      }
    }

    //[ ***step 4*** ] 小于阈值则暂停, 并且为下次设置阈值
    if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold) break;
  }

  if (!haveOneGood) {
    printf(
        "BIG ERROR! tracking failed entirely. Take predicted pose and hope "
        "we may somehow recover.\n");
    flowVecs = Vec3(0, 0, 0);
    aff_g2l = aff_last_2_l;
    lastF_2_fh = lastF_2_fh_tries[0];
    std::cout << "Predicted pose:\n" << lastF_2_fh.matrix() << std::endl;
    if (lastF_2_fh.translation().norm() > 100000 || lastF_2_fh.matrix().hasNaN()) {
      std::cout << "TRACKING FAILED ENTIRELY, NO HOPE TO RECOVER" << std::endl;
      std::cerr << "TRACKING FAILED ENTIRELY, NO HOPE TO RECOVER" << std::endl;
      exit(1);
    }
  }
  //! 把这次得到的最好值给下次用来当阈值
  coarseTracker->thisToNext = lastF_2_fh;
  lastCoarseRMSE = achievedRes;
  //[ ***step 5*** ] 此时shell在跟踪阶段, 没人使用, 设置值
  // no lock required, as fh is not used anywhere yet.
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef = lastF->shell;
  fh->shell->aff_g2l = aff_g2l;
  fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
  fh->shell->trackingWasGood = trackingGoodRet;
  printf(
      "start !!!!! res_num_ratio, coarseTracker->firstCoarseRMSE: %f, "
      "achievedRes[0]: %f\n",
      coarseTracker->firstCoarseRMSE, achievedRes[0]);
  if (coarseTracker->firstCoarseRMSE < 0) {
    coarseTracker->firstCoarseRMSE = achievedRes[0];
    coarseTracker->firstCoarseResNum = coarseTracker->lastResidualNum[0];
    coarseTracker->firstSaturatedRatio = coarseTracker->lastSaturatedRatio[0];
  }

  if (!setting_debugout_runquiet)
    printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure,
           achievedRes[0]);

  if (setting_logStuff) {
    (*coarseTrackingLog) << std::setprecision(16) << fh->shell->id << " " << fh->shell->timestamp << " "
                         << fh->ab_exposure << " " << fh->shell->camToWorld.log().transpose() << " " << aff_g2l.a << " "
                         << aff_g2l.b << " " << achievedRes[0] << " " << tryIterations << "\n";
  }

  Vec10 ret = Vec10::Zero();
  ret.head(4) = Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
  return std::make_pair(ret, trackingGoodRet);
}

void FullSystem::convert_to_ImageData(cv::Mat& data, ImageDataAM& image_data, uint8_t camera_id) {
  image_data.exposure_ts = 1;
  image_data.camera_id = camera_id;
  image_data.width = data.cols;
  image_data.height = data.rows;
  // 999 as default tuning index
  image_data.tuning_index = 999;
  image_data.shutter_speed_ns = 1;
  image_data.frame_id = 1;
  image_data.step = data.step;
  image_data.data = data.data;
}
//@ 利用新的帧 fh 对关键帧中的ImmaturePoint进行更新
/// multi-small-baseline-stereo, update idepth
#define SHOW_DEPTH_FILTER
#if 1  // ndef USE_EDGE_ALIGN
#define USE_DSM_DEPTH_FILTER
#endif
void FullSystem::traceNewCoarse(FrameHessian* fh, bool is_first_frame) {
  dmvio::TimeMeasurement timeMeasurement("traceNewCoarse");
  boost::unique_lock<boost::mutex> lock(mapMutex);

  int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0,
      trace_uninitialized = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0) = Hcalib.fxl();
  K(1, 1) = Hcalib.fyl();
  K(0, 2) = Hcalib.cxl();
  K(1, 2) = Hcalib.cyl();
#ifdef SHOW_DEPTH_FILTER
  MinimalImageB3* img_df;
  img_df = new MinimalImageB3(wG[0], hG[0]);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    const Eigen::Vector3f* dIl_gray = fh->dI + wG[0] * hG[0] * cid;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      float colL = dIl_gray[i][0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_df->at(i, cid) = Vec3b(colL, colL, colL);
    }
  }
#endif
#ifdef USE_DSM_DEPTH_FILTER
  int lvl = 0;
  int wl = wG[lvl], hl = hG[lvl];
  // dso::ImagesBuffer::Initial(40, wl, hl);
  std::array<ImageDataAM, kCameraNumUsed> cid_to_image_data;

  Point point;
  bool is_corner;
  std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_target_img;
  std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_host_image_use;
  std::array<cv::Mat, kCameraNumUsed> cid_to_cv_img;
  MinimalImageB* img_target;
  img_target = new MinimalImageB(wl, hl);
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    //    fh->p_multi_camera->cid_to_K_temp[cam].setIdentity();
    //    fh->p_multi_camera->cid_to_K_temp[cam](0, 0) = fxG[lvl];
    //    fh->p_multi_camera->cid_to_K_temp[cam](1, 1) = fyG[lvl];
    //    fh->p_multi_camera->cid_to_K_temp[cam](0, 2) = cxG[lvl];
    //    fh->p_multi_camera->cid_to_K_temp[cam](1, 2) = cyG[lvl];
    //    std::vector<number_t> param = {fxG[lvl], fyG[lvl], cxG[lvl],
    //    cyG[lvl]}; fh->p_multi_camera->cid_to_Kinv_temp[cam] =
    //        fh->p_multi_camera->cid_to_K_temp[cam].inverse();
    //    fh->p_multi_camera->cid_to_cam_pinhole[cam] =
    //        new PinholeCamera(cam, wl, hl, param.data());
    // for (size_t i = 0; i < kCameraNumUsed; ++i) {
    p_depth_filter_DSM_->px_err_angle_vec_[cam] = std::atan(p_depth_filter_DSM_->px_noise_ / fxG[lvl]);
    // }

    assert(estimator_config_.search_level == 0);
    number_t search_level_focal_length = fxG[lvl] * std::pow(2.0f, -estimator_config_.search_level);

    p_depth_filter_DSM_->p_multi_cam_epipolar_search_->rad_step_ =
        estimator_config_.pixel_step * std::asin(1.0 / search_level_focal_length / 2.0) * 2.0;

    Eigen::Vector3f* colorCur = fh->dIp[lvl] + cam * wl * hl;
    for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorCur + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_target->at(i, cam) = static_cast<unsigned char>(colL);
    }
    cid_to_cv_img[cam] =
        cv::Mat(img_target->h, img_target->w, CV_8UC1, img_target->data + img_target->w * img_target->h * cam);

    convert_to_ImageData(cid_to_cv_img[cam], cid_to_image_data[cam], static_cast<uint8_t>(cam));

    cid_to_target_img[cam] = dso::ImagesBuffer::Acquire(wl, hl);
    cid_to_target_img[cam]->DangerouslyCopyFrom(cid_to_image_data[cam].width, cid_to_image_data[cam].height,
                                                cid_to_image_data[cam].step, cid_to_image_data[cam].data,
                                                cid_to_image_data[cam].exposure_ts, cid_to_image_data[cam].tuning_index,
                                                cid_to_image_data[cam].shutter_speed_ns, cid_to_image_data[cam].gain);
    // cv::imshow("img", cid_to_target_img[cam]);
    // cv::waitKey(0);
    cid_to_host_image_use[cam] = dso::ImagesBuffer::Acquire(wl, hl);
  }
  MinimalImageB* img_host;
  img_host = new MinimalImageB(wl, hl);
  FrameHessian* host_frame = nullptr;
  std::array<cv::Mat, kCameraNumUsed> cid_to_host_cv_img;
  std::array<ImageDataAM, kCameraNumUsed> cid_to_host_image_data_am;

  // for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //   cid_to_host_image_use[cid] = dso::ImagesBuffer::Acquire(wl, hl);
  // }
  float dso_search_success_pid_count = 0;
  float dsm_search_success_pid_count = 0;
  Vec2 uv_in_target;
#endif
  // 遍历关键帧
  float hw_sum = 0, hw_count = 0;
  for (FrameHessian* host : frameHessians)  // go through all active frames
  {
    SE3 hostToNew_ = fh->PRE_worldToCam * host->PRE_camToWorld;
    //    SE3 hostToNew =
    //              fh->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
    //                      hostToNew_ *
    //                      fh->p_multi_camera->cid_to_T01_SE3[host_cid];
    //    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() *
    //    K.inverse(); Vec3f Kt = K * hostToNew.translation().cast<float>();
    //
    Vec2f aff =
        AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();
    /// loop across all points on frameHessian
    assert(host != fh);
#ifdef USE_DSM_DEPTH_FILTER
    for (int cid_ = 0; cid_ < kCameraNumUsed; ++cid_) {
      Eigen::Vector3f* colorHost = host->dIp[lvl] + cid_ * wl * hl;
      for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(colorHost + i))[0];
        if (colL < 0) colL = 0;
        if (colL > 255) colL = 255;
        img_host->at(i, cid_) = static_cast<unsigned char>(colL);
      }
      cv::Mat host_cv_img =
          cv::Mat(img_host->h, img_host->w, CV_8UC1, img_host->data + img_host->w * img_host->h * cid_);
      cid_to_host_cv_img[cid_] = host_cv_img.clone();
      convert_to_ImageData(cid_to_host_cv_img[cid_], cid_to_host_image_data_am[cid_], static_cast<uint8_t>(cid_));

      // cid_to_host_image_use[cid_] = dso::ImagesBuffer::Acquire(wl, hl);
      cid_to_host_image_use[cid_]->DangerouslyCopyFrom(
          cid_to_host_image_data_am[cid_].width, cid_to_host_image_data_am[cid_].height,
          cid_to_host_image_data_am[cid_].step, cid_to_host_image_data_am[cid_].data,
          cid_to_host_image_data_am[cid_].exposure_ts, cid_to_host_image_data_am[cid_].tuning_index,
          cid_to_host_image_data_am[cid_].shutter_speed_ns, cid_to_host_image_data_am[cid_].gain);
    }
#endif
    for (ImmaturePoint* ph : host->immaturePoints) {
      ph->idp = -1.f;
      /// trace on epipole line
      // TODO entrance
      /// deoutlier, update immaturePoints epipolar search interval [idepth_min
      /// and idepth_max]
      int covisible_cid = 0;
      std::array<bool, kCameraNumUsed> visible_flag;
      SE3 hostToNew = SE3();
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        //        SE3 hostToNew_ = fh->PRE_worldToCam * host->PRE_camToWorld;
        visible_flag[target_cid] = false;
        hostToNew = fh->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() * hostToNew_ *
                    fh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
        Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
        Vec3f Kt = K * hostToNew.translation().cast<float>();

        //            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure,
        //            fh->ab_exposure,
        //                                                    host->aff_g2l(),
        //                                                    fh->aff_g2l())
        //                    .cast<float>();
        ph->traceOn(target_cid, fh, KRKi, Kt, aff, &Hcalib, false, 0, false, true);
#ifdef SHOW_DEPTH_FILTER
        Vec3f pr = KRKi * Vec3f(ph->u, ph->v, 1);
        Vec3f proj = pr + Kt * 0.5f * (ph->idepth_max + ph->idepth_min);
        proj /= proj[2];
        img_df->setPixelCirc(proj[0], proj[1], Vec3b(0, 0, 0), target_cid);
#endif
        if (ph->lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_GOOD) {
          hw_sum += ph->hw_use[target_cid];
          hw_count++;
#ifdef SHOW_DEPTH_FILTER
          img_df->setPixelCirc(proj[0], proj[1], Vec3b(0, 255, 0), target_cid);
#endif
          covisible_cid++;
          trace_good++;
        }
        if (ph->lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_BADCONDITION) {
#ifdef SHOW_DEPTH_FILTER
          // yellow
          img_df->setPixelCirc(proj[0], proj[1], Vec3b(0, 255, 255), target_cid);
#endif
          covisible_cid++;
          trace_badcondition++;
        }
        if (ph->lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_OOB) {
#ifdef SHOW_DEPTH_FILTER
          img_df->setPixelCirc(proj[0], proj[1], Vec3b(255, 0, 0), target_cid);
#endif
          trace_oob++;
        }
        if (ph->lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_OUTLIER) {
#ifdef SHOW_DEPTH_FILTER
          img_df->setPixelCirc(proj[0], proj[1], Vec3b(0, 0, 255), target_cid);
#endif
          covisible_cid++;
          trace_out++;
        }
        if (ph->lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_SKIPPED) {
#ifdef SHOW_DEPTH_FILTER
          // Cyan
          img_df->setPixelCirc(proj[0], proj[1], Vec3b(255, 255, 0), target_cid);
#endif
          covisible_cid++;
          trace_skip++;
        }
        if (ph->lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_UNINITIALIZED) {
#ifdef SHOW_DEPTH_FILTER
          // pink / Magenta
          img_df->setPixelCirc(proj[0], proj[1], Vec3b(255, 0, 255), target_cid);
#endif
          covisible_cid++;
          trace_uninitialized++;
        }
        trace_total++;
      }
      // TODO do extra round of multi-camera dsm depth filter
      assert(covisible_cid <= kCameraNumUsed);
      if (covisible_cid >= 1) {
        // printf("covisible_cid: %d\n", covisible_cid);
      }
#ifdef USE_DSM_DEPTH_FILTER
      if (covisible_cid > 1 && std::isfinite(ph->idepth_min) && std::isfinite(ph->idepth_max)) {
        int host_cid = ph->host_cid;
        number_t x = ph->u, y = ph->v;
        Vec2 px = Vec2(x, y);
        Vec3 xyz_pinhole = fh->p_multi_camera->level_cid_to_Kinv_temp.at(lvl).at(host_cid) * Vec3(x, y, 1.);
        xyz_pinhole /= xyz_pinhole[2];
        float mean_idp;
        if (ph->idp > 0) {
          mean_idp = ph->idp;
        } else {
          mean_idp = 0.5f * (ph->idepth_max + ph->idepth_min);
        }
        Vec3 xyz_host = xyz_pinhole / (mean_idp);
        point.n = xyz_pinhole.normalized();
        assert(ph->host != nullptr);
        assert(ph->host == host);
        if (ph->host != host_frame && false) {
          for (int cid_ = 0; cid_ < kCameraNumUsed; ++cid_) {
            Eigen::Vector3f* colorHost = ph->host->dIp[lvl] + cid_ * wl * hl;
            for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
              // BRIGHTNESS TRANSFER
              float colL = (*(colorHost + i))[0];
              if (colL < 0) colL = 0;
              if (colL > 255) colL = 255;
              img_host->at(i, cid_) = static_cast<unsigned char>(colL);
            }
            cv::Mat host_cv_img =
                cv::Mat(img_host->h, img_host->w, CV_8UC1, img_host->data + img_host->w * img_host->h * cid_);
            cid_to_host_cv_img[cid_] = host_cv_img.clone();
            convert_to_ImageData(cid_to_host_cv_img[cid_], cid_to_host_image_data_am[cid_], static_cast<uint8_t>(cid_));

            // cid_to_host_image_use[cid_] = dso::ImagesBuffer::Acquire(wl, hl);
            cid_to_host_image_use[cid_]->DangerouslyCopyFrom(
                cid_to_host_image_data_am[cid_].width, cid_to_host_image_data_am[cid_].height,
                cid_to_host_image_data_am[cid_].step, cid_to_host_image_data_am[cid_].data,
                cid_to_host_image_data_am[cid_].exposure_ts, cid_to_host_image_data_am[cid_].tuning_index,
                cid_to_host_image_data_am[cid_].shutter_speed_ns, cid_to_host_image_data_am[cid_].gain);
          }
          // printf("aaa\n");
        } else {
          // printf("bbb\n");
        }
        //        cv::Mat host_cv_img =
        //            cv::Mat(img_host->h, img_host->w, CV_8UC1,
        //                    img_host->data + img_host->w * img_host->h *
        //                    host_cid);
        //        ImageDataAM host_image_data_am;
        //        convert_to_ImageData(cid_to_host_cv_img[cid_],
        //        host_image_data_am,
        //                             static_cast<uint8_t>(host_cid));
        //
        //        std::shared_ptr<AlgsImage> host_image_use;
        //        host_image_use = dso::ImagesBuffer::Acquire(wl, hl);
        //        host_image_use->DangerouslyCopyFrom(
        //            host_image_data_am.width, host_image_data_am.height,
        //            host_image_data_am.step, host_image_data_am.data,
        //            host_image_data_am.exposure_ts,
        //            host_image_data_am.tuning_index,
        //            host_image_data_am.shutter_speed_ns,
        //            host_image_data_am.gain);
        bool is_success = point.pyramid_patch.SetFromImg(cid_to_host_image_use[host_cid], px, host_cid, is_corner,
                                                         fh->p_multi_camera, lvl);
        // printf("is_success: %d, is_corner: %d\n", is_success,
        // is_corner); std::exit(1);
        if (is_success) {
          // printf("idepth_max: %f, idepth_min: %f\n", ph->idepth_max,
          // ph->idepth_min);
          if (ph->idepth_max <= ph->idepth_min) {
            printf("idepth_max: %f, idepth_min: %f, sth wrong\n", ph->idepth_max, ph->idepth_min);
            std::exit(4);
          }
          dso_search_success_pid_count += 1.f;
          // number_t init_idp = 0.5;
          Seed seed;

          DF_Frame::InitSeedDepth(seed, false, xyz_host.norm());
          seed.rho = 1 / xyz_host.norm();
          number_t res_idp;
          std::array<MultiCameraEpipolarSearch::MatchRes, kCameraNumUsed> cid_to_output;
          Mat4 T10 = hostToNew_.matrix();
          MultiCameraEpipolarSearch::State state = p_depth_filter_DSM_->p_multi_cam_epipolar_search_->FindEpipolarMatch(
              point, host_cid, cid_to_target_img, 1, seed.rho, seed.sigma2, 1, cid_to_output, res_idp, -1, false, lvl,
              &T10);
          if (state == MultiCameraEpipolarSearch::kSuccess) {
            dsm_search_success_pid_count += 1.f;
            float delta_idp = 0.5f * (ph->idepth_max - ph->idepth_min);
            assert(delta_idp > 0);
            Vec3 xyz_host_new = point.n / res_idp;
            float idp = 1.f / static_cast<float>(xyz_host_new[2]);
            ph->idp = idp;
            //            printf("idepth_min: %f idp: %f, idepth_max: %f\n",
            //            ph->idepth_min,
            //                   idp, ph->idepth_max);

            assert(idp > 0);
            float idepth_min_new = idp - delta_idp;
            ph->idepth_min = idepth_min_new > 0 ? idepth_min_new : (0.9f * idp);
            ph->idepth_max = idp + delta_idp;
            for (int cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
              if (ph->lastTracePixelInterval[cam_id] > 0) {
                assert(ph->lastTraceUV[cam_id][0] > 0 && ph->lastTraceUV[cam_id][1] > 0);
                Mat4 Tth = InversePose(Hcalib.p_multi_camera->cid_to_T01[cam_id]) * T10 *
                           Hcalib.p_multi_camera->cid_to_T01[host_cid];
                Vec3 xyz_in_target = Tth.topLeftCorner<3, 3>() * xyz_host_new + Tth.topRightCorner<3, 1>();
                Hcalib.p_multi_camera->level_cid_to_cam_pinhole[lvl][cam_id]->Project(xyz_in_target, uv_in_target);
                if (cid_to_output[cam_id].match_success) {
                  number_t uv_diff_dsm = (uv_in_target - cid_to_output[cam_id].target_uv).norm();
                  number_t uv_diff_dso = (uv_in_target - ph->lastTraceUV[cam_id].cast<number_t>()).norm();
                  // printf("uv_diff_dsm: %f, uv_diff_dso: %f\n", uv_diff_dsm,
                  // uv_diff_dso);
                  assert(uv_diff_dsm < 0.0001);
                }
                ph->lastTraceUV[cam_id] = uv_in_target.cast<float>();
              }
            }
          }
        }
        // host_image_use.reset();
      }
#endif
    }
  }
  printf("hw_trace_on: %f\n", hw_sum / hw_count);
#ifdef USE_DSM_DEPTH_FILTER
  printf("[dsm dso] traceOn stats: [%.1f %.1f], traceOn_ratio: %.3f\n", dsm_search_success_pid_count,
         dso_search_success_pid_count, dsm_search_success_pid_count / dso_search_success_pid_count);
  // dso::ImagesBuffer::SetInitial(false);
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    cid_to_target_img[cam].reset();
    cid_to_host_image_use[cam].reset();
  }
  delete img_host;
  delete img_target;
#endif
#ifdef SHOW_DEPTH_FILTER
  IOWrap::displayImage("depth filter", img_df);
#ifdef SAVE_IMAGES
  if (fh && fh->shell) {
    char buf[100];
    snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/depth_filter_%015lu.png",
             (uint64_t)(fh->shell->timestamp_eval * 1e9));
    IOWrap::writeImage(buf, img_df);
  }
#endif
  IOWrap::waitKey(1);
  delete img_df;
#endif
  //	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip.
  //%'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%)
  // uninit.\n", 			trace_total, trace_good,
  // 100*trace_good/(float)trace_total, 			trace_skip,
  // 100*trace_skip/(float)trace_total, trace_badcondition,
  // 100*trace_badcondition/(float)trace_total, trace_oob,
  // 100*trace_oob/(float)trace_total, 			trace_out,
  // 100*trace_out/(float)trace_total, 			trace_uninitialized,
  // 100*trace_uninitialized/(float)trace_total);
}

//@ 处理挑选出来待激活的点
void FullSystem::activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,
                                           std::vector<ImmaturePoint*>* toOptimize, int min, int max, Vec10* stats,
                                           int tid) {
  printf("do pose only point opt, toOptimize: %d, frameHessians: %d\n", toOptimize->size(), frameHessians.size());
  ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size() * kCameraNumUsed];
  /// normally min = 0, max = toOptimize.size()
  int minObs;
#ifdef USE_MULTI_CAM
  int num_kf = 6;
  if (frameHessians.size() < num_kf) {
    minObs = 1;
  } else {
    minObs = num_kf / 2;
  }
  assert(frameHessians.size() >= 2);
  minObs = ((frameHessians.size() - 1) >= 5) ? 5 : (frameHessians.size() - 1);
#else
  minObs = 1;
#endif
  minObs = 1;
  for (int k = min; k < max; k++) {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], minObs /*1*/, tr, true, false);
  }
  delete[] tr;
}

//@ 激活未成熟点, 加入优化
#define SHOW_NEWLY_ACTIVATED_POINTS
//#define SHOW_DISTANCE_MAP
void FullSystem::activatePointsMT() {
  dmvio::TimeMeasurement timeMeasurement("activatePointsMT");
  //[ ***step 1*** ] 阈值计算, 通过距离地图来控制数目
  // currentMinActDist 初值为 2
  //* 这太牛逼了.....参数
  if (ef->nPoints < setting_desiredPointDensity * 0.66) currentMinActDist -= 0.8;
  if (ef->nPoints < setting_desiredPointDensity * 0.8)
    currentMinActDist -= 0.5;
  else if (ef->nPoints < setting_desiredPointDensity * 0.9)
    currentMinActDist -= 0.2;
  else if (ef->nPoints < setting_desiredPointDensity)
    currentMinActDist -= 0.1;

  if (ef->nPoints > setting_desiredPointDensity * 1.5) currentMinActDist += 0.8;
  if (ef->nPoints > setting_desiredPointDensity * 1.3) currentMinActDist += 0.5;
  if (ef->nPoints > setting_desiredPointDensity * 1.15) currentMinActDist += 0.2;
  if (ef->nPoints > setting_desiredPointDensity) currentMinActDist += 0.1;

  if (currentMinActDist < 0) currentMinActDist = 0;
#if 1  // ndef USE_MULTI_CAM
  if (currentMinActDist > 4) currentMinActDist = 4;
#else
  if (currentMinActDist > 10) currentMinActDist = 10;
#endif
  if (!setting_debugout_runquiet)
    printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n", currentMinActDist,
           (int)(setting_desiredPointDensity), ef->nPoints);

  /// newest keyframe
  FrameHessian* newestFh = frameHessians.back();

  // make dist map.
  coarseDistanceMap->makeK(&Hcalib);
  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    coarseDistanceMap->makeDistanceMap(frameHessians, newestFh, target_cid);
  }

  // coarseTracker->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint*> toOptimize;
  toOptimize.reserve(20000);  // 待激活的点
  int plot_step = 4;
#ifdef SHOW_DISTANCE_MAP
  MinimalImageB3* img_dist;

  img_dist = new MinimalImageB3(wG[0], hG[0]);
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec3f* colorRef = newestFh->dIp[0] + wG[0] * hG[0] * cam;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorRef + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_dist->at(i, cam) = Vec3b(colL, colL, colL);
    }
    float nid = 0, sid = 0;
    for (int col = 0; col < wG[1]; col += plot_step) {
      for (int row = 0; row < hG[1]; row += plot_step) {
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[col + wG[1] * row + wG[1] * hG[1] * cam];
        if (dist < 0.0001 || dist > 999) {
          continue;
        }
        nid++;
        sid += dist;
      }
    }
    float fac = nid / sid;
    for (int col = 5; col < wG[1] - 5; col += plot_step) {
      for (int row = 5; row < hG[1] - 5; row += plot_step) {
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[col + wG[1] * row + wG[1] * hG[1] * cam];
        if (dist < 0.0001 || dist > 999) {
          continue;
        }
        img_dist->setPixelCirc(2 * col, 2 * row, makeRainbow3B(dist * fac), cam);
      }
    }
  }
  IOWrap::displayImage("dist map in target before", img_dist);
  IOWrap::waitKey(1);
  delete img_dist;
#endif
  //[ ***step 2*** ] 处理未成熟点, 激活/删除/跳过
#ifdef SHOW_NEWLY_ACTIVATED_POINTS
  MinimalImageB3* img_target;

  img_target = new MinimalImageB3(wG[0], hG[0]);

  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec3f* colorRef = newestFh->dI + wG[0] * hG[0] * cam;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorRef + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_target->at(i, cam) = Vec3b(colL, colL, colL);
    }
  }
#endif
  int fid = 0;
  int can_activate_count = 0;
  ;
  int pid_to_opt = 0;
  printf("start, frameHessians size: %d\n", frameHessians.size());
  for (FrameHessian* host : frameHessians)  // go through all active frames
  {
    if (host == newestFh) continue;
    //    // TODO 老帧上的点往最新关键帧投影，构造残差
    SE3 fhToNew_ = newestFh->PRE_worldToCam * host->PRE_camToWorld;
    //    // 第0层到1层
    //    /// old[0] to new[1]
    //
    //    // TODO roger, try all possible target cids
    //    Mat33f KRKi =
    //        (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>()
    //        *
    //         coarseDistanceMap->Ki[0]);
    //    Vec3f Kt = (coarseDistanceMap->K[1] *
    //    fhToNew.translation().cast<float>());
    printf("activating points, fid: %d, immature points: %d\n", fid, host->immaturePoints.size());
    fid++;
    for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
      ImmaturePoint* ph = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;
      int in_valid_count = 0;
      int large_distance_count = 0;
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        // delete points that have never been traced successfully, or that are
        // outlier on the last trace.
        if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus[target_cid] == IPS_OUTLIER) {
          //				immature_invalid_deleted++;
          // remove point.
          in_valid_count++;
          //                      delete ph;
          //                      host->immaturePoints[i] = 0; // 指针赋零
          continue;
        }
#if 1
        // TODO 老帧上的点往最新关键帧投影，构造残差
        SE3 fhToNew = newestFh->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() * fhToNew_ *
                      newestFh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
        // 第0层到1层
        /// old[0] to new[1]

        // TODO roger, try all possible target cids
        Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
        Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());
        // see if we need to activate point due to distance map.
        float mean_idp;
        if (ph->idp > 0) {
          mean_idp = ph->idp;
        } else {
          mean_idp = 0.5f * (ph->idepth_max + ph->idepth_min);
        }
        Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (mean_idp);
        /// reproject from old[0] to new[1]
        int u = ptp[0] / ptp[2] + 0.5f;
        int v = ptp[1] / ptp[2] + 0.5f;
        if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
        }
#endif
        //* 未成熟点的激活条件
        // can activate only if this is true.
        bool canActivate =
            (ph->lastTraceStatus[target_cid] == IPS_GOOD || ph->lastTraceStatus[target_cid] == IPS_SKIPPED ||
             ph->lastTraceStatus[target_cid] == IPS_BADCONDITION || ph->lastTraceStatus[target_cid] == IPS_OOB) &&
            ph->lastTracePixelInterval[target_cid] < 8 && ph->quality[target_cid] > setting_minTraceQuality &&
            (ph->idepth_max + ph->idepth_min) > 0;

        // if I cannot activate the point, skip it. Maybe also delete it.
#ifdef SHOW_NEWLY_ACTIVATED_POINTS
        img_target->setPixelCirc(2 * u, 2 * v, Vec3b(0, 0, 255), target_cid);
        if (ph->quality[target_cid] < setting_minTraceQuality) {
          // cyan
          img_target->setPixelCirc(2 * u, 2 * v, Vec3b(255, 255, 0), target_cid);
        } else if (ph->lastTracePixelInterval[target_cid] > 8) {
          // pink
          img_target->setPixelCirc(2 * u, 2 * v, Vec3b(255, 0, 255), target_cid);
        } else if ((ph->idepth_max + ph->idepth_min) < 0) {
          img_target->setPixelCirc(2 * u, 2 * v, Vec3b(255, 255, 255), target_cid);
        }
#endif
        if (!canActivate) {
          //* 删除被边缘化帧上的, 和OOB点
          // if point will be out afterwards, delete it instead.
          if (ph->host->flaggedForMarginalization || ph->lastTraceStatus[target_cid] == IPS_OOB) {
            //					immature_notReady_deleted++;
            in_valid_count++;
            //                          delete ph;
            //                          host->immaturePoints[i] = 0;
          }
          //				immature_notReady_skipped++;
          continue;
        }
#if 0
        // TODO 老帧上的点往最新关键帧投影，构造残差
        SE3 fhToNew =
            newestFh->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
            fhToNew_ * newestFh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
        // 第0层到1层
        /// old[0] to new[1]

        // TODO roger, try all possible target cids
        Mat33f KRKi =
            (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() *
             coarseDistanceMap->Ki[0]);
        Vec3f Kt =
            (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());
        // see if we need to activate point due to distance map.
        float mean_idp;
        if (ph->idp > 0) {
          mean_idp = ph->idp;
        } else {
          mean_idp = 0.5f * (ph->idepth_max + ph->idepth_min);
        }
        Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (mean_idp);
        /// reproject from old[0] to new[1]
        int u = ptp[0] / ptp[2] + 0.5f;
        int v = ptp[1] / ptp[2] + 0.5f;
#endif
        if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
#ifdef SHOW_NEWLY_ACTIVATED_POINTS
          // yellow
          img_target->setPixelCirc(2 * u, 2 * v, Vec3b(0, 255, 255), target_cid);
#endif
          can_activate_count++;
          // 距离地图 + 小数点
          // TODO
          // TODO could we use KDTree instead?
          // TODO we need calculate pair-wise distance between any reprojected
          // points in 8 directions
          float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v + wG[1] * hG[1] * target_cid] +
                       (ptp[0] - floorf((float)(ptp[0])));
          /// remember fwdWarpedIDDistFinal is updated in makeDistanceMap, which
          /// stores all the search distances (index k in the loop [0..40]).
          /// search distance plus the decimal part of ptp[0] which is x in
          /// world cordinate w.r.t last frame.
          // TODO roger,
          // 这个阈值是4目共用的，可能某一目的数目会比较少，但整体点肯定是够的
          if (dist >= currentMinActDist * ph->my_type
#if 0  // def USE_EDGE_ALIGN
                          || true
#endif
              )  /// 点越多, 距离阈值越大 [my_type 1 2 4]
          {
            /// 每新activate一个点，那与这个点相关的distanceMap也要更新，很严谨
            coarseDistanceMap->addIntoDistFinal(u, v, target_cid);
            large_distance_count++;
            // toOptimize.push_back(ph);
          } else {
          }
        } else {
          in_valid_count++;
          //                      delete ph;
          //                      host->immaturePoints[i] = 0;
        }
      }
      assert(in_valid_count <= kCameraNumUsed);
      if (in_valid_count == kCameraNumUsed) {
        delete ph;
        host->immaturePoints[i] = 0;
      } else {
        if (large_distance_count > 0) {
          toOptimize.push_back(ph);
        }
      }
    }
  }
#ifdef SHOW_DISTANCE_MAP
  MinimalImageB3* img_dist_after;

  img_dist_after = new MinimalImageB3(wG[0], hG[0]);
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec3f* colorRef = newestFh->dIp[0] + wG[0] * hG[0] * cam;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorRef + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_dist_after->at(i, cam) = Vec3b(colL, colL, colL);
    }
    float nid = 0, sid = 0;
    for (int col = 0; col < wG[1]; col += plot_step) {
      for (int row = 0; row < hG[1]; row += plot_step) {
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[col + wG[1] * row + wG[1] * hG[1] * cam];
        if (dist < 0.0001 || dist > 999) {
          continue;
        }
        nid++;
        sid += dist;
      }
    }
    float fac = nid / sid;
    for (int col = 5; col < wG[1] - 5; col += plot_step) {
      for (int row = 5; row < hG[1] - 5; row += plot_step) {
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[col + wG[1] * row + wG[1] * hG[1] * cam];
        if (dist < 0.0001 || dist > 999) {
          continue;
        }
        img_dist_after->setPixelCirc(2 * col, 2 * row, makeRainbow3B(dist * fac), cam);
      }
    }
  }
  IOWrap::displayImage("dist map in target after", img_dist_after);
  IOWrap::waitKey(1);
  delete img_dist_after;
#endif
  //	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip
  //%d)\n", 			(int)toOptimize.size(), immature_deleted,
  // immature_notReady, immature_needMarg, immature_want, immature_margskip); [
  // ***step 3*** ] 优化上一步挑出来的未成熟点, 进行逆深度优化,
  //并得到pointhessian
  std::vector<PointHessian*> optimized;
  optimized.resize(toOptimize.size());
  /// triangulate immature points to active points
  if (multiThreading) {
    treadReduce.reduce(
        boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0,
        toOptimize.size(), 50);
  } else {
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
  }
  //[ ***step 4*** ] 把PointHessian加入到能量函数, 删除收敛的未成熟点,
  //或不好的点
  int optimized_points = 0;
  float hw_sum = 0, hw_count = 0;
  for (unsigned k = 0; k < toOptimize.size(); k++) {
    PointHessian* newpoint = optimized[k];
    // newpoint->idepth_backup = newpoint->idepth;
    ImmaturePoint* ph = toOptimize[k];
    int oob_count = 0;
    for (int id = 0; id < kCameraNumUsed; ++id) {
      if (ph->lastTraceStatus[id] == IPS_OOB) {
        oob_count++;
      }
    }

    if (newpoint != 0 && newpoint != (PointHessian*)((long)(-1))) {
#ifdef USE_MULTI_CAM
      if (allKeyFramesHistory.size() <= 5) {
        newpoint->hasDepthPrior = true;
      }
#endif
      // TODO roger, 即使是lastTrackingStatus是oob也可以尝试激活，
      // 万一它上上次，上上次是好点呢？上一次可能只是被遮挡了
      // printf("oob_count: %d\n", oob_count);
      // assert(oob_count < kCameraNumUsed);
      assert(newpoint->host == ph->host);
#ifdef SHOW_NEWLY_ACTIVATED_POINTS
      // assert(newpoint->host == ph->host);
      SE3 fhToNew_ = newestFh->PRE_worldToCam * newpoint->host->PRE_camToWorld;
      for (int target_cam = 0; target_cam < kCameraNumUsed; ++target_cam) {
        SE3 fhToNew = newestFh->p_multi_camera->cid_to_T01_SE3[target_cam].inverse() * fhToNew_ *
                      newestFh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
        Mat33f KRKi = (coarseDistanceMap->K[0] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
        Vec3f Kt = (coarseDistanceMap->K[0] * fhToNew.translation().cast<float>());
        // see if we need to activate point due to distance map.
        Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * newpoint->idepth;
        float mean_idp;
        if (ph->idp > 0) {
          mean_idp = ph->idp;
        } else {
          mean_idp = 0.5f * (ph->idepth_max + ph->idepth_min);
        }
        Vec3f xyz_pinhole = coarseDistanceMap->Ki[0] * Vec3f(ph->u, ph->v, 1);
        xyz_pinhole /= xyz_pinhole[2];
        Vec3f xyz_cur =
            fhToNew.rotationMatrix().cast<float>() * (xyz_pinhole / (mean_idp)) + fhToNew.translation().cast<float>();
        /// reproject from old[0] to new[1]
        int u = ptp[0] / ptp[2] + 0.5f;
        int v = ptp[1] / ptp[2] + 0.5f;
        Vec3f proj = coarseDistanceMap->K[0] * xyz_cur;
        proj /= proj[2];
        if ((u > 10 && v > 10 && u < wG[0] - 10 && v < hG[0] - 10) && xyz_cur[2] > 0.05 &&
            (proj[0] > 10 && proj[1] > 10 && proj[0] < wG[0] - 10 && proj[1] < hG[0] - 10)) {
          img_target->setPixelCirc(proj[0] + 0.5, proj[1] + 0.5, Vec3b(255, 0, 0), target_cam);
          img_target->setPixelCirc(u + 0.5, v + 0.5, Vec3b(0, 255, 0), target_cam);
        }
      }
#endif
      hw_sum += newpoint->hw_use;
      hw_count += 1;
      optimized_points++;
      newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      /// 自己push_back到自己里面？
      // TODO 把newpoint push到该point所host的帧的收敛点列表里去
      newpoint->host->pointHessians.push_back(newpoint);
      ef->insertPoint(newpoint);  // 能量函数中插入点 //TODO
                                  // 相当于正式把这个3d点加入到大优化中了
      /// pattern of 8 ? nah, it's usually 2 or 3
      // printf("newpoint->residuals: %d\n",newpoint->residuals.size());

      std::map<int, int> fid_to_res_hit_count;
      for (PointFrameResidual* r : newpoint->residuals) {
        if (fid_to_res_hit_count.find(r->target->idx) == fid_to_res_hit_count.end()) {
          fid_to_res_hit_count.emplace(r->target->idx, 0);
        }
        // fid_to_res_count.at(r->target->idx)++;
      }

      for (PointFrameResidual* r : newpoint->residuals) {
        // TODO roger, 真细，之前花了力气算的factor是一个也不落下,
        // 把历史帧上面的factor也存下来了
#ifdef DISABLE_CROSS_CID_ALIGN
        if (ph->host_cid != r->target_cid) {
          continue;
        }
#endif
        ef->insertResidual(r, Hcalib.p_multi_camera, fid_to_res_hit_count.at(r->target->idx) == 0);
        fid_to_res_hit_count.at(r->target->idx)++;
      }
      assert(newpoint->efPoint != 0);
      delete ph;
    } else if (newpoint == (PointHessian*)((long)(-1)) ||
               oob_count == kCameraNumUsed /*ph->lastTraceStatus[target_cid] == IPS_OOB*/) {
      // bug: 原来的顺序错误
      ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      delete ph;
    } else {
      // printf("will it reach here?\n");
      // TODO roger, 这里不着急delete这个 immature point，
      // 留给下次出发了上面两个if else时再彻底删除这个点，
      // 毕竟有可能这个seed只是暂时被遮挡了，后面还是有机会被成功优化至收敛的
      assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
    }
  }
  printf(
      "hw_activate: %f, [success / toOpt/ canActivate]: [%d / %d / %d] "
      "points in activatePointsMT\n",
      hw_sum / hw_count, optimized_points, toOptimize.size(), can_activate_count);
#ifdef SHOW_NEWLY_ACTIVATED_POINTS
  IOWrap::displayImage("newly activated in target", img_target);
#ifdef SAVE_IMAGES
  if (newestFh && newestFh->shell) {
    char buf[100];
    snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/newly_activate_%015lu.png",
             (uint64_t)(newestFh->shell->timestamp_eval * 1e9));
    IOWrap::writeImage(buf, img_target);
  }
#endif
  IOWrap::waitKey(1);
  delete img_target;
#endif
  //[ ***step 5*** ] 把删除的点丢掉
  for (FrameHessian* host : frameHessians) {
    for (int i = 0; i < (int)host->immaturePoints.size(); i++) {
      if (host->immaturePoints[i] == 0) {
        /// 这次没激活成功的immature点可以不删，留给下次关键帧激活，
        /// 但是这次判定成outlier的immature点要删掉，不给他机会了
        // bug 如果back的也是空的呢
        host->immaturePoints[i] = host->immaturePoints.back();  // 没有顺序要求, 直接最后一个给空的
        host->immaturePoints.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::activatePointsOldFirst() { assert(false); }

//@ 标记要移除点的状态, 边缘化or丢掉
void FullSystem::flagPointsForRemoval() {
  assert(EFIndicesValid);

  std::vector<FrameHessian*> fhsToKeepPoints;
  std::vector<FrameHessian*> fhsToMargPoints;

  // if(setting_margPointVisWindow>0)
  {
    // bug 又是不用的一条语句
    for (int i = ((int)frameHessians.size()) - 1; i >= 0 && i >= ((int)frameHessians.size()); i--)
      if (!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

    for (int i = 0; i < (int)frameHessians.size(); i++)
      if (frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
  }

  // ef->setAdjointsF();
  // ef->setDeltaF(&Hcalib);
  int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

  for (FrameHessian* host : frameHessians)  // go through all active frames
  {
    for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
      PointHessian* ph = host->pointHessians[i];
      if (ph == 0) continue;

      //* 丢掉相机后面, 没有残差的点
      if (ph->idepth_scaled < setting_minIdepth || ph->residuals.size() == 0) {
        host->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        host->pointHessians[i] = 0;
        flag_nores++;
      }
      //* 把边缘化的帧上的点, 以及受影响较大的点标记为边缘化or删除
      else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization) {
        flag_oob++;
        // TODO* 如果是一个内点, 则把残差在当前状态线性化, 并计算到零点残差
        // (pose和内参用的fej，idp，ab和梯度用的最新状态的雅可比 )
        if (ph->isInlierNew()) {
          flag_in++;
          int ngoodRes = 0;
          for (PointFrameResidual* r : ph->residuals) {
            for (int cid_ = 0; cid_ < kCameraNumUsed; ++cid_) {
              r->resetOOB(cid_);
              // TODO roger, recalc jac
              r->linearize(&Hcalib, cid_);  // TODO
              // (pose和内参用的fej，idp，ab和梯度用的最新状态的雅可比
              // )
              r->efResidual->isLinearized[cid_] = false;
              r->applyRes(true, cid_);
              // 如果是激活(可参与优化)的残差, 则给fix住, 计算res_toZeroF //TODO
              // 雅可比不包含逆深度的部分 dim = 8, 6 dof pose + 2 dof affine
              if (r->efResidual->isActive(cid_))  // TODO 只有是内点时才会继续
              {
                r->efResidual->fixLinearizationF(ef, cid_);
                ngoodRes++;
              }
            }
          }
          //* 如果逆深度的协方差很大直接扔掉, 小的边缘化掉
          if (ph->idepth_hessian > setting_minIdepthH_marg) {
            flag_inin++;  // the bigger the hessian, the better the point
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
            host->pointHessiansMarginalized.push_back(ph);
          } else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
          }

        }
        //* 不是内点直接扔掉
        else {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

          // printf("drop point in frame %d (%d goodRes, %d activeRes)\n",
          // ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
        }

        host->pointHessians[i] = 0;
      }
    }

    //* 删除边缘化或者删除的点
    //!< contains all ACTIVE points.
    for (int i = 0; i < (int)host->pointHessians.size(); i++) {
      if (host->pointHessians[i] == 0) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }
  }
}
/********************************
 * @ function:
 *
 * @ param: 	image		标定后的辐照度和曝光时间
 * @			id
 *
 * @ note: start from here
 *******************************/
// The function is passed the IMU-data from the previous frame until the current
// frame.
void FullSystem::addActiveFrame(ImageAndExposure* image, int id, dmvio::IMUData* imuData, dmvio::GTData* gtData) {
  // Measure Time of the time measurement.
  dmvio::TimeMeasurement timeMeasurementMeasurement("timeMeasurement");
  dmvio::TimeMeasurement timeMeasurementZero("zero");
  timeMeasurementZero.end();
  timeMeasurementMeasurement.end();
  //[ ***step 1*** ] track线程锁
  dmvio::TimeMeasurement timeMeasurement("addActiveFrame");
  boost::unique_lock<boost::mutex> lock(trackMutex);

  dmvio::TimeMeasurement measureInit("initObjectsAndMakeImage");
  //[ ***step 2*** ] 创建FrameHessian和FrameShell, 并进行相应初始化,
  //并存储所有帧
  // =========================== add into allFrameHistory
  // =========================
  FrameHessian* fh = new FrameHessian(Hcalib.p_multi_camera);
  FrameShell* shell = new FrameShell();
  shell->camToWorld = SE3();  // no lock required, as fh is not used anywhere yet.
  shell->aff_g2l = AffLight(0, 0);
  // TODO 这一帧是在这个时间戳被新建的，marg也要在这个时刻marg？？[scratch that]
  // id等于当当前keyfraem size
  shell->marginalizedAt = shell->id = allFrameHistory.size();
  shell->timestamp = image->timestamp;
#ifdef USE_MULTI_CAM
  shell->timestamp_eval = image->timestamp_eval;
#endif
  shell->incoming_id = id;
  fh->shell = shell;
  allFrameHistory.push_back(shell);

  //[ ***step 3*** ] 得到曝光时间, 生成金字塔, 计算整个图像梯度
  // =========================== make Images / derivatives etc.
  // =========================
  fh->ab_exposure = image->exposure_time;
  fh->timestamp = shell->timestamp;
  if (shell) {
    printf("======================= processing frame %f ==========\n", shell->timestamp_eval);
  }
  fh->makeImages(image->image,
                 &Hcalib);  // TODO generate pyraid, gamma correction, generate gradient
  disable_kf = (fh->mean_gray_val < -25.f || fh->mean_gray_val > 1130.f) ? 1 : 0;
  disable_kf = (fh->mean_gray_val < 30.0f /*40.f*/ || fh->mean_gray_val > 1130.f) ? 1 : 0;
  float last_kf_mean_gray_val = 0;
  if (allKeyFramesHistory.empty()) {
  } else {
    last_kf_mean_gray_val = allKeyFramesHistory.back()->mean_gray_val;
    // disable_kf = disable_kf || std::abs(last_kf_mean_gray_val -
    // fh->mean_gray_val) > 20.f;
  }
  printf("disable_kf: %d, cur_mean_gray_val: %f, last_kf_mean_gray_val: %f\n", disable_kf, fh->mean_gray_val,
         last_kf_mean_gray_val);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    printf("each mean_gray_val: %f\n", fh->mean_gray_val_each[cid]);
  }
  measureInit.end();
  //[ ***step 4*** ] 进行初始化
  if (!initialized) {
    disable_kf = -1;
    disable_kf_real = false;
    // use initializer!
    //[ ***step 4.1*** ] 加入第一帧
    if (coarseInitializer->frameID < 0)  // first frame set. fh is kept by coarseInitializer.
    {
      // Only in this case no IMU-data is accumulated for the BA as this is the
      // first frame.
      dmvio::TimeMeasurement initMeasure("InitializerFirstFrame");
      if (kCameraNumUsed == 1) {
        coarseInitializer->setFirst(&Hcalib, fh);
      } else {
        coarseInitializer->setFirstStereo(&Hcalib, fh);
      }
      if (setting_useIMU) {
        gravityInit.addMeasure(*imuData,
                               Sophus::SE3d());  // TODO imu读数均值，作为重力方向初值，Rw0
      }
      if (kCameraNumUsed > 1) {
        // traceNewCoarse(fh, true);
        // initializeFromInitializer(fh);
      }
      for (IOWrap::Output3DWrapper* ow : outputWrapper) ow->publishSystemStatus(dmvio::VISUAL_INIT);
    } else {
      dmvio::TimeMeasurement initMeasure("InitializerOtherFrames");
      // TODO roger, provide initial rotation with imu
      Mat33 dRwb = Mat33::Identity();
      if (setting_useIMU) {
        for (int id = 0; id < imuData->size(); ++id) {
          const double& dt = (*imuData)[id].getIntegrationTime();
          if (dt == 0.0) {
            continue;
          }
          const Vec3& gyro = (*imuData)[id].getGyrData();
          const Vec3 rot_vec = dt * gyro;
          if (false) {
            const Mat3 dRwb = ExpSO3(rot_vec);
            coarseInitializer->thisToNext.setRotationMatrix(coarseInitializer->thisToNext.rotationMatrix() * dRwb);
          } else {
            dRwb *= ExpSO3(rot_vec);
          }
        }
        // coarseInitializer->thisToNext.rotationMatrix() = dRwb;
        coarseInitializer->thisToNext.setRotationMatrix(
            Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix().transpose() * dRwb.transpose() *
            Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix() * coarseInitializer->thisToNext.rotationMatrix());
      } else {
        // Rwb = Mat33::Constant(std::nan(""));
      }
      // TODO roger, 会用到iR这个先验，很迷，尽量不用，因为把控不住
      bool initDone = coarseInitializer->trackFrame(fh, outputWrapper, Mat33::Identity());
      coarseTracker->thisToNext = coarseInitializer->thisToNext;
      if (setting_useIMU) {
        imuIntegration.addIMUDataToBA(*imuData);
        Sophus::SE3 imuToWorld = gravityInit.addMeasure(*imuData, Sophus::SE3d());
        std::cout << "imuToWorld: \n" << imuToWorld.matrix3x4() << std::endl;
        if (initDone) {  // cam pose in first frame;
          firstPose = imuToWorld * imuIntegration.TS_cam_imu.inverse();
        }
      }
      // visual init
      if (initDone)  // if SNAPPED
      {
        //[ ***step 4.2*** ] 跟踪成功, 完成初始化
        initializeFromInitializer(fh);
        if (setting_useIMU && linearizeOperation) {
          imuIntegration.setGTData(gtData, fh->shell->id);
        }
        lock.unlock();
        initMeasure.end();
        for (IOWrap::Output3DWrapper* ow : outputWrapper) ow->publishSystemStatus(dmvio::VISUAL_ONLY);
        deliverTrackedFrame(fh, true, false, false);
      } else {
        // if still initializing

        // Maybe change first frame.
        double timeBetweenFrames = fh->shell->timestamp - coarseInitializer->firstFrame->shell->timestamp;
        std::cout << "InitTimeBetweenFrames: " << timeBetweenFrames << std::endl;
        if (timeBetweenFrames > imuIntegration.getImuSettings().maxTimeBetweenInitFrames) {
          // Do full reset so that the next frame becomes the first initializer
          // frame.
          setting_fullResetRequested = true;
        } else {
          fh->shell->poseValid = false;
          delete fh;
        }
      }
    }
    return;
  } else  // do front-end operation.
  {
    // --------------------------  Coarse tracking (after visual initializer
    // succeeded). --------------------------
    dmvio::TimeMeasurement coarseTrackingTime("fullCoarseTracking");
    int lastFrameId = -1;
    //[ ***step 5*** ] 对新来的帧进行跟踪, 得到位姿光度, 判断跟踪状态
    // =========================== SWAP tracking reference?.
    // =========================
    bool trackingRefChanged = false;
    if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
      dmvio::TimeMeasurement referenceSwapTime("swapTrackingRef");
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
      CoarseTracker* tmp = coarseTracker;
      coarseTracker = coarseTracker_forNewKF;
      coarseTracker_forNewKF = tmp;

      if (dso::setting_useIMU) {
        // BA for new keyframe has finished and we have a new tracking
        // reference.
        if (!setting_debugout_runquiet) {
          std::cout << "New ref frame id: " << coarseTracker->refFrameID
                    << " prepared keyframe id: " << imuIntegration.getPreparedKeyframe() << std::endl;
        }

        lastFrameId = coarseTracker->refFrameID;

        assert(coarseTracker->refFrameID == imuIntegration.getPreparedKeyframe());
        SE3 lastRefToNewRef = imuIntegration.initCoarseGraph();

        trackingRefChanged = true;
      }
    }

    SE3* referenceToFramePassed = 0;
    SE3 referenceToFrame;
    Mat33 R_th = Mat33::Identity();
    if (dso::setting_useIMU) {
      // TODO 预积分连续普通帧，并做一次inertial only的优化？用来为direct image
      // alignment提供初值
      /*SE3*/ referenceToFrame =
          imuIntegration.addIMUData(*imuData, fh->shell->id, fh->shell->timestamp, trackingRefChanged, lastFrameId);
      // If initialized we use the prediction from IMU data as initialization
      // for the coarse tracking.
      referenceToFramePassed = &referenceToFrame;
      if (!imuIntegration.isCoarseInitialized()) {
        referenceToFramePassed = nullptr;
        R_th = referenceToFrame.rotationMatrix();
      }
      imuIntegration.addIMUDataToBA(*imuData);
    }
    // TODO 使用旋转和位移对像素移动的作用比来判断运动状态
    Mat33 dRwb = Mat33::Identity();
    if (setting_useIMU) {
      if (!referenceToFramePassed) {
        for (int id = 0; id < imuData->size(); ++id) {
          const double& dt = (*imuData)[id].getIntegrationTime();
          if (dt == 0.0) {
            continue;
          }
          const Vec3& gyro = (*imuData)[id].getGyrData();
          const Vec3 rot_vec = dt * gyro;
          if (false) {
            const Mat3 dRwb = ExpSO3(rot_vec);
            coarseTracker->thisToNext.setRotationMatrix(coarseTracker->thisToNext.rotationMatrix() * dRwb);
          } else {
            dRwb *= ExpSO3(rot_vec);
          }
        }
        // coarseTracker->thisToNext.rotationMatrix() = dRwb;
        coarseTracker->thisToNext.setRotationMatrix(
            Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix().transpose() * dRwb.transpose() *
            Hcalib.p_multi_camera->cid_to_Tbc_SE3[0].rotationMatrix() * coarseTracker->thisToNext.rotationMatrix());
      } else {
        // coarseTracker->thisToNext.setRotationMatrix(R_th);
      }
    }
    std::pair<Vec10, bool> pair = trackNewCoarse(fh, referenceToFramePassed, dRwb);
    {
      //            fh->shell->camToWorld;
      //            fh->shell->timestamp;

      Eigen::Quaterniond q;
      Eigen::Vector3d p;
      q = Eigen::Quaterniond(fh->shell->camToWorld.rotationMatrix());
      p = fh->shell->camToWorld.translation();
      output_ostr_ << std::fixed << fh->shell->timestamp << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x()
                   << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

#ifdef USE_MULTI_CAM
      // printf("exposure: %f\n", image->exposure_time);
      if (!setting_useIMU) {
        printf("print vo pose...\n");
        (*poseLog) << std::fixed << static_cast<double>(fh->shell->timestamp_eval) << " " << p.x() << " " << p.y()
                   << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
      } else {
        printf("imuUsedBefore: %d, print vio pose...\n", imuUsedBefore);
        if (imuUsedBefore) {
          getMetricScaleTwc(fh->shell, (*fh->p_multi_camera).Tbc0);
        }
      }
#endif
    }
    dso::Vec10 tres = std::move(pair.first);
    bool forceNoKF = !pair.second;  // If coarse tracking was bad don't make KF.
    bool forceKF = false;
    if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) ||
        !std::isfinite((double)tres[3])) {
      if (setting_useIMU) {
        // If completely Nan, don't force noKF!
        forceNoKF = false;
        forceKF = true;  // actually we force a KF in that situation as there are
                         // no points to track.
      } else {
        printf("Initial Tracking failed: LOST!\n");
        isLost = true;
        return;
      }
    }

    double timeSinceLastKeyframe = fh->shell->timestamp - allKeyFramesHistory.back()->timestamp;
    Vec2 refToFh = Vec2::Zero();
    //[ ***step 6*** ] 判断是否插入关键帧
    bool needToMakeKF = false, needToMakeKF_tracker = false;
    if (setting_keyframesPerSecond > 0)  // 每隔多久插入关键帧
    {
      needToMakeKF = allFrameHistory.size() == 1 || (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
                                                        0.95f / setting_keyframesPerSecond;
    } else {
      refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                            coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);
      float extra_ratio = 1.0;
      if (static_cast<int>(allKeyFramesHistory.size()) <= static_cast<int>(-3)) {
        extra_ratio = 10.0;
      }
      // BRIGHTNESS CHECK
      needToMakeKF =
          allFrameHistory.size() == 1 ||
          (extra_ratio * setting_kfGlobalWeight) * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                  (extra_ratio * setting_kfGlobalWeight) * setting_maxShiftWeightR * sqrtf((double)tres[2]) /
                      (wG[0] + hG[0]) +
                  (extra_ratio * setting_kfGlobalWeight) * setting_maxShiftWeightRT * sqrtf((double)tres[3]) /
                      (wG[0] + hG[0]) +
                  (extra_ratio * setting_kfGlobalWeight) * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) >
              1 ||
          // #ifdef USE_MULTI_CAM
          //           1.2
          // #else
          //           2
          // #endif
          //                   * coarseTracker->firstCoarseRMSE <
          //               tres[0] ||

          coarseTracker->lastResidualNum[0] / coarseTracker->firstCoarseResNum > 1.2 ||
          (coarseTracker->lastRS[0][9] < 0.2 /*0.7*/ &&
           (
#ifdef USE_MULTI_CAM
               1.2
#else
               2
#endif
                       * coarseTracker->firstCoarseRMSE <
                   tres[0] ||
               coarseTracker->lastResidualNum[0] / coarseTracker->firstCoarseResNum < 0.4 /*0.2*/
               || coarseTracker->lastSaturatedRatio[0] > 0.3 /*0.6*/ ||
               coarseTracker->lastResidualNum[0] < 1000 /*3000 /*1000*/)) ||
          (setting_maxTimeBetweenKeyframes > 0 && timeSinceLastKeyframe > setting_maxTimeBetweenKeyframes) || forceKF;
      needToMakeKF_tracker = needToMakeKF;
      printf(
          "time: %0.4f, thr: %0.3f, res_num_ratio: [%0.3f], pt_wo_edges_ratio: "
          "%0.3f, last_res_num: %d, lastSaturatedRatio: %0.3f, [cur_res / "
          "first_res]: "
          "[%0.3f / %0.3f] = %0.3f, affine_ratio: %0.3f, needToMakeKF: %d, "
          "affine_part_thr: "
          "%0.3f, affine_abs_log: %0.3f\n",
          image->timestamp_eval,
          setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
              setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
              setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
              setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])),
          coarseTracker->lastResidualNum[0] / coarseTracker->firstCoarseResNum, coarseTracker->lastRS[0][9],
          (int)coarseTracker->lastResidualNum[0], coarseTracker->lastSaturatedRatio[0], tres[0],
          coarseTracker->firstCoarseRMSE, tres[0] / coarseTracker->firstCoarseRMSE,
          std::exp(fabs(logf((float)refToFh[0]))), needToMakeKF,
          setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])),
          fabs(logf((float)refToFh[0])));

      if (needToMakeKF && !setting_debugout_runquiet) {
        std::cout << "Time since last keyframe: " << timeSinceLastKeyframe << std::endl;
      }
    }
    double transNorm = fh->shell->camToTrackingRef.translation().norm() * imuIntegration.getCoarseScale();
    if (imuIntegration.isCoarseInitialized() && transNorm < setting_forceNoKFTranslationThresh) {
      forceNoKF = true;
    }
    if (forceNoKF) {
      std::cout << "Forcing NO KF!" << std::endl;
      needToMakeKF = false;
    }

#if 1
    // disable_kf = (fh->mean_gray_val < 35.f || fh->mean_gray_val > 130.f) ? 1
    // : 0; float last_kf_mean_gray_val = 0; if (allKeyFramesHistory.empty()) {
    // } else {
    //   last_kf_mean_gray_val = allKeyFramesHistory.back()->mean_gray_val;
    //   // disable_kf = disable_kf || std::abs(last_kf_mean_gray_val -
    //   fh->mean_gray_val) > 20.f;
    // }
    // printf("disable_kf: %d, cur_mean_gray_val: %f, last_kf_mean_gray_val:
    // %f\n", disable_kf, fh->mean_gray_val, last_kf_mean_gray_val); for (int
    // cid = 0; cid < kCameraNumUsed; ++cid) {
    //   printf("each mean_gray_val: %f\n", fh->mean_gray_val_each[cid]);
    // }
    if (true && (disable_kf == 1)) {
      needToMakeKF = false;
      forceKF = false;
      forceNoKF = true;
    }
    printf(
        "++++ res_num_ratio 1: %f, time: %f, disable_kf_last: %d, "
        "lastSaturatedRatio: %f, needToMakeKF: %d, forceKF: %d, forceNoKF: "
        "%d, disable_kf: %d, points_wo_edge_ratio: %f\n",
        coarseTracker->lastResidualNum[0] / coarseTracker->firstCoarseResNum, image->timestamp_eval, disable_kf_last,
        coarseTracker->lastSaturatedRatio[0], needToMakeKF, forceKF, forceNoKF, disable_kf,
        coarseTracker->lastRS[0][9]);
    if (disable_kf_last == 1 && disable_kf != 1 && needToMakeKF == false && coarseTracker->lastRS[0][9] < 0.2) {
      needToMakeKF = true;
      forceKF = false;
      forceNoKF = false;
    }
    printf(
        "++++ res_num_ratio 2: %f, time: %f, disable_kf_last: %d, "
        "lastSaturatedRatio: %f, needToMakeKF: %d, forceKF: %d, forceNoKF: "
        "%d, disable_kf: %d, points_wo_edge_ratio: %f\n",
        coarseTracker->lastResidualNum[0] / coarseTracker->firstCoarseResNum, image->timestamp_eval, disable_kf_last,
        coarseTracker->lastSaturatedRatio[0], needToMakeKF, forceKF, forceNoKF, disable_kf,
        coarseTracker->lastRS[0][9]);
#endif
    printf(
        "++++ res_num_ratio 3: %f, time: %f, disable_kf_last: %d, "
        "lastSaturatedRatio: %f, needToMakeKF: %d, forceKF: %d, forceNoKF: "
        "%d, disable_kf: %d, points_wo_edge_ratio: %f\n",
        coarseTracker->lastResidualNum[0] / coarseTracker->firstCoarseResNum, image->timestamp_eval, disable_kf_last,
        coarseTracker->lastSaturatedRatio[0], needToMakeKF, forceKF, forceNoKF, disable_kf,
        coarseTracker->lastRS[0][9]);

#ifdef SHOW_ALIGN_FRAME
    char buf[100];
    snprintf(buf, 100,
             "/media/roger/Elements_SE/CI/dm_vio_results/"
             "align_frame_%015lu_%d_%d_%d.png",
             (uint64_t)(image->timestamp_eval * 1e9), pyrLevelsUsed - 1, pyrLevelsUsed - 1, 0);
    cv::Mat img = cv::imread(buf, -1);
    if (!img.empty()) {
      cv::putText(img, "tracker_kf: " + std::to_string((int)needToMakeKF_tracker), cv::Point(250, 30),
                  cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      cv::putText(img, "disable_kf_last: " + std::to_string((int)disable_kf_last), cv::Point(250, 130),
                  cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      cv::putText(img, "disable_kf: " + std::to_string((int)disable_kf), cv::Point(250, 230), cv::FONT_HERSHEY_COMPLEX,
                  1, cv::Scalar(0, 255, 0), 2);
      cv::putText(img, "final_kf: " + std::to_string((int)needToMakeKF), cv::Point(250, 330), cv::FONT_HERSHEY_COMPLEX,
                  1, cv::Scalar(0, 255, 0), 2);
      cv::putText(
          img,
          "thr: " +
              std::to_string(
                  (int)((setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                         setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
                         setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
                         setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0]))) *
                        100)) +
              " T: " +
              std::to_string(
                  (int)((setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0])) *
                        100)) +
              " RT: " +
              std::to_string(
                  (int)((setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0])) *
                        100)) +
              " AFFINE: " +
              std::to_string(
                  (int)((setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0]))) * 100)),
          cv::Point(10, 430), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      printf("time: %f, tres: [%f %f %f], refToFh: [%f %f]\n", image->timestamp_eval, tres[1], tres[2], tres[3],
             refToFh[0], refToFh[1]);
      cv::imwrite(buf, img);
    }
#endif

    if (needToMakeKF) {
      int prevKFId = fh->shell->trackingRef->id;
      // In non-RT mode this will always be accurate, but in RT mode the
      // printout in makeKeyframe is correct (because some of these KFs do not
      // end up getting created).
      int framesBetweenKFs = fh->shell->id - prevKFId - 1;

      // Enforce setting_minFramesBetweenKeyframes.
      printf(
          "res_num_ratio, framesBetweenKFs: %d, "
          "setting_minFramesBetweenKeyframes: %d\n",
          framesBetweenKFs, (int)setting_minFramesBetweenKeyframes);
      if (framesBetweenKFs < (int)setting_minFramesBetweenKeyframes)  // if integer value is smaller
                                                                      // we just skip.
      {
        std::cout << "1, Skipping KF because of minFramesBetweenKeyframes." << std::endl;
        needToMakeKF = false;
      } else if (framesBetweenKFs < setting_minFramesBetweenKeyframes)  // Enforce it for
                                                                        // non-integer values.
      {
        double fractionalPart = setting_minFramesBetweenKeyframes - (int)setting_minFramesBetweenKeyframes;
        framesBetweenKFsRest += fractionalPart;
        std::cout << "res_num_ratio, framesBetweenKFs: " << framesBetweenKFs << ", fractionalPart: " << fractionalPart
                  << ", framesBetweenKFsRest: " << framesBetweenKFsRest << std::endl;
        if (framesBetweenKFsRest >= 1.0) {
          std::cout << "2, Skipping KF because of minFramesBetweenKeyframes." << std::endl;
          needToMakeKF = false;
          framesBetweenKFsRest--;
        }
      }
    }

    if (setting_useIMU) {
      imuIntegration.finishCoarseTracking(*(fh->shell), needToMakeKF);
    }

    if (needToMakeKF && setting_useIMU && linearizeOperation) {
      imuIntegration.setGTData(gtData, fh->shell->id);
    }

    dmvio::TimeMeasurement timeLastStuff("afterCoarseTracking");

    for (IOWrap::Output3DWrapper* ow : outputWrapper) ow->publishCamPose(fh->shell, &Hcalib);

    lock.unlock();
    timeLastStuff.end();
    //[ ***step 7*** ] 把该帧发布出去
    coarseTrackingTime.end();
#if 0
    disable_kf = fh->mean_gray_val < 30.f || fh->mean_gray_val > 130.f;
    float last_kf_mean_gray_val = 0;
    if (allKeyFramesHistory.empty()) {
    } else {
      last_kf_mean_gray_val = allKeyFramesHistory.back()->mean_gray_val;
      // disable_kf = disable_kf || std::abs(last_kf_mean_gray_val - fh->mean_gray_val) > 20.f;
    }
    printf("disable_kf: %d, cur_mean_gray_val: %f, last_kf_mean_gray_val: %f\n", disable_kf, fh->mean_gray_val, last_kf_mean_gray_val);
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      printf("each mean_gray_val: %f\n", fh->mean_gray_val_each[cid]);
    }
    if (true && (disable_kf)) {
      needToMakeKF = false;
      forceKF = false;
      forceNoKF = true;
    }
#endif
    deliverTrackedFrame(fh, needToMakeKF, forceKF, forceNoKF);
    return;
  }
}

// TODO 把跟踪的帧, 给到建图线程, 设置成关键帧或非关键帧
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF, bool forceKF, bool forceNoKF) {
  disable_kf_last = disable_kf;
  printf(
      "res_num_ratio, deliverTrackedFrame, needKF: %d, forceKF: %d, "
      "forceNoKF: %d\n",
      needKF, forceKF, forceNoKF);
  dmvio::TimeMeasurement timeMeasurement("deliverTrackedFrame");
  // There seems to be exactly one instance where needKF is false but the mapper
  // creates a keyframe nevertheless: if it is the second tracked frame (so it
  // will become the third keyframe in total) There are also some cases where
  // needKF is true but the mapper does not create a keyframe.

  bool alreadyPreparedKF = setting_useIMU && imuIntegration.getPreparedKeyframe() != -1 && !linearizeOperation;

  if (!setting_debugout_runquiet) {
    std::cout << "Frame history size: " << allFrameHistory.size() << std::endl;
  }
  if ((needKF || (!secondKeyframeDone && !linearizeOperation)) && setting_useIMU && !alreadyPreparedKF) {
    // prepareKeyframe tells the IMU-Integration that this frame will become a
    // keyframe. -> don' marginalize it during addIMUData. Also resets the IMU
    // preintegration for the BA.
    if (!setting_debugout_runquiet) {
      std::cout << "Preparing keyframe: " << fh->shell->id << std::endl;
    }
    imuIntegration.prepareKeyframe(fh->shell->id);

    if (!needKF) {
      secondKeyframeDone = true;
    }
  } else {
    if (!setting_debugout_runquiet) {
      std::cout << "Creating a non-keyframe: " << fh->shell->id << std::endl;
    }
  }

  if (linearizeOperation) {
    if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
      MinimalImageF3 img(wG[0], hG[0], fh->dI);
      IOWrap::displayImage("frameToTrack", &img);
      while (true) {
        char k = IOWrap::waitKey(0);
        if (k == ' ') break;
        handleKey(k);
      }
      lastRefStopID = coarseTracker->refFrameID;
    } else
      handleKey(IOWrap::waitKey(1));

    // TODO important entrance, marginalization and sliding window optimization
    // are done in this function
    if (needKF) {
      if (setting_useIMU) {
        imuIntegration.keyframeCreated(fh->shell->id);
      }
      makeKeyFrame(fh, forceKF, forceNoKF);
    } else
      makeNonKeyFrame(fh);
  } else {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    unmappedTrackedFrames.push_back(fh);

    // If the prepared KF is still in the queue right now the current frame will
    // become a KF instead.
    if (alreadyPreparedKF && !imuIntegration.isPreparedKFCreated()) {
      imuIntegration.prepareKeyframe(fh->shell->id);
      needKF = true;
    }

    if (setting_useIMU) {
      if (needKF) needNewKFAfter = imuIntegration.getPreparedKeyframe();
    } else {
      if (needKF) needNewKFAfter = fh->shell->trackingRef->id;
    }
    trackedFrameSignal.notify_all();

    while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
      mappedFrameSignal.wait(lock);
    }

    lock.unlock();
  }
}

void FullSystem::mappingLoop() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

  while (runMapping) {
    while (unmappedTrackedFrames.size() == 0) {
      trackedFrameSignal.wait(lock);
      if (!runMapping) return;
    }

    FrameHessian* fh = unmappedTrackedFrames.front();
    unmappedTrackedFrames.pop_front();

    if (!setting_debugout_runquiet) {
      std::cout << "Current mapping id: " << fh->shell->id << " create KF after: " << needNewKFAfter << std::endl;
    }

    // guaranteed to make a KF for the very first two tracked frames.
    if (allKeyFramesHistory.size() <= 2) {
      if (setting_useIMU) {
        imuIntegration.keyframeCreated(fh->shell->id);
      }
      lock.unlock();
      makeKeyFrame(fh, false, false);
      lock.lock();
      mappedFrameSignal.notify_all();
      continue;
    }

    if (unmappedTrackedFrames.size() > 3) needToKetchupMapping = true;

    if (unmappedTrackedFrames.size() > 0)  // if there are other frames to track, do that first.
    {
      if (setting_useIMU && needNewKFAfter == fh->shell->id) {
        if (!dso::setting_debugout_runquiet) {
          std::cout << "WARNING: Prepared keyframe got skipped!" << std::endl;
        }
        imuIntegration.skipPreparedKeyframe();
        assert(false);
      }

      lock.unlock();
      makeNonKeyFrame(fh);
      lock.lock();

      if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
        FrameHessian* fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();
        {
          boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
          assert(fh->shell->trackingRef != 0);
          fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
          fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }
        delete fh;
      }

    } else {
      bool createKF =
          setting_useIMU ? needNewKFAfter == fh->shell->id : needNewKFAfter >= frameHessians.back()->shell->id;
      if (setting_realTimeMaxKF || createKF) {
        if (setting_useIMU) {
          imuIntegration.keyframeCreated(fh->shell->id);
        }
        lock.unlock();
        makeKeyFrame(fh, false, false);
        needToKetchupMapping = false;
        lock.lock();
      } else {
        lock.unlock();
        makeNonKeyFrame(fh);
        lock.lock();
      }
    }
    mappedFrameSignal.notify_all();
  }
  printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
  runMapping = false;
  trackedFrameSignal.notify_all();
  lock.unlock();

  mappingThread.join();
}

void FullSystem::makeNonKeyFrame(FrameHessian* fh) {
  dmvio::TimeMeasurement timeMeasurement("makeNonKeyframe");
  // needs to be set by mapping thread. no lock required since we are in mapping
  // thread.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);  // TODO 生命周期结束后自动解锁
    assert(fh->shell->trackingRef != 0);
    // mapping时将它当前位姿取出来得到camToWorld
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    /// 同时更新nullspace
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
    //    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),
    //                         fh->shell->cid_to_aff_g2l);
  }
  // TODO entrance, overload "traceNewCoarse"
  if (!disable_kf_real /*disable_kf != 1*/ || true) {
    traceNewCoarse(fh);
  }
  // TODO
  // 这个delete是为啥，把指针删掉了，那这块地址谁来指向？我以后要访问地址里的变量怎么办？
  // TODO 明白了，这是non-keyframe，旨在更新keyframe的idepth
  // interval，none-keyframe的数据之后不会再访问了，它的使命已经完成
  delete fh;
}

//@ 生成关键帧, 优化, 激活点, 提取点, 边缘化关键帧
// #define SHOW_NEWLY_PREDICTED_RESIDUALS
void FullSystem::makeKeyFrame(FrameHessian* fh, bool forceKF, bool forceNoKF) {
  dmvio::TimeMeasurement timeMeasurement("makeKeyframe");
  //[ ***step 1*** ] 设置当前估计的fh的位姿, 光度参数
  // needs to be set by mapping thread
  {
    // 同样取出位姿, 当前的作为最终值
    //? 为啥要从shell来设置 ???   答: 因为shell不删除, 而且参考帧还会被优化,
    // shell是桥梁
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
    int prevKFId = fh->shell->trackingRef->id;
    int framesBetweenKFs = fh->shell->id - prevKFId - 1;
    if (!setting_debugout_runquiet) {
      std::cout << "Frames between KFs: " << framesBetweenKFs << std::endl;
    }
  }
  //[ ***step 2*** ] 把这一帧来更新之前帧的未成熟点
  /// 不同于non-keyframe，这里是keyframe，所以不会delete fh
  // TODO depth filter
  if (disable_kf == 1 || disable_kf_real) {
    printf("disable_kf == 1 || disable_kf_real\n");
    std::exit(1);
  }
  traceNewCoarse(fh);  // 更新未成熟点(深度未收敛的点)

  boost::unique_lock<boost::mutex> lock(mapMutex);  // 建图锁
  //[ ***step 3*** ] 选择要边缘化掉的帧
  // =========================== Flag Frames to be Marginalized.
  // =========================
  // TODO Marginalization  entrance
  flagFramesForMarginalization(fh);  // TODO 这里没用最新帧，可以改进下

  //[ ***step 4*** ] 加入到关键帧序列
  // =========================== add New Frame to Hessian Struct.
  // =========================
  /// sliding window size, roughly around 7
  dmvio::TimeMeasurement timeMeasurementAddFrame("newFrameAndNewResidualsForOldPoints");
  fh->idx = frameHessians.size();
  frameHessians.push_back(fh);
  fh->frameID = allKeyFramesHistory.size();
  fh->shell->keyframeId = fh->frameID;
  allKeyFramesHistory.push_back(fh->shell);
  // TODO@ 向能量函数中增加一帧, 进行的操作:
  // 改变正规方程(绝对pose的雅可比转成相对pose的雅可比), 重新排ID, 共视关系
  ef->insertFrame(fh, &Hcalib);
  // TODO 每添加一个关键帧都会运行这个来设置位姿, 设置位姿线性化点, fix
  // linearization point
  // TODO
  // 设置当前估计相对于fej的增量,对pose来说还要把相对于fej的绝对增量转成相对增量
  setPrecalcValues();                        // 每添加一个关键帧都会运行这个来设置位姿,
                                             // 设置位姿线性化点
#if defined(SHOW_NEWLY_PREDICTED_RESIDUALS)  // && defined(USE_MULTI_CAM)
  MinimalImageB3* img_target;
  MinimalImageB3* edge_target;
  // MinimalImageB3 *edge_only_target;
  MinimalImageB3* label_only_target;
  MinimalImageB3* edge_dt_target;
  MinimalImageB3* dx_target;
  MinimalImageB3* dy_target;
  int lvl_check = 0;  // pyrLevelsUsed - 1;
  img_target = new MinimalImageB3(wG[0], hG[0]);
  edge_target = new MinimalImageB3(wG[0], hG[0]);
  // edge_only_target = new MinimalImageB3(wG[lvl_check], hG[lvl_check]);
  label_only_target = new MinimalImageB3(wG[lvl_check], hG[lvl_check]);
  edge_dt_target = new MinimalImageB3(wG[lvl_check], hG[lvl_check]);
  dx_target = new MinimalImageB3(wG[lvl_check], hG[lvl_check]);
  dy_target = new MinimalImageB3(wG[lvl_check], hG[lvl_check]);

  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec3f* colorRef = fh->dI + wG[0] * hG[0] * cam;
    Vec2i* edge_label_image_start = fh->edge_label_image[lvl_check] + wG[lvl_check] * hG[lvl_check] * cam;
    Vec3f* dt_dx_dy_start = fh->dt_dx_dy[lvl_check] + wG[lvl_check] * hG[lvl_check] * cam;
    float dt_len = fh->max_dt_dx_dy[lvl_check][cam][0] - fh->min_dt_dx_dy[lvl_check][cam][0];
    float dx_len = fh->max_dt_dx_dy[lvl_check][cam][1] - fh->min_dt_dx_dy[lvl_check][cam][1];
    float dy_len = fh->max_dt_dx_dy[lvl_check][cam][2] - fh->min_dt_dx_dy[lvl_check][cam][2];
    std::cerr << "cid: " << cam << ", lvl_check: " << lvl_check << ", max_label_num: " << fh->label_num[lvl_check][cam]
              << ", max dt_dx_dy: " << fh->max_dt_dx_dy[lvl_check][cam].transpose()
              << ", min dt_dx_dy: " << fh->min_dt_dx_dy[0][cam].transpose() << std::endl;
    for (int c = 1; c < wG[0] - 1; ++c) {
      for (int r = 1; r < hG[0] - 1; ++r) {
        int i = c + r * wG[0];
        // for (int i = 0; i < wG[0] * hG[0]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(colorRef + i))[0];
        if (colL < 0) colL = 0;
        if (colL > 255) colL = 255;
        img_target->at(i, cam) = Vec3b(colL, colL, colL);
        edge_target->at(i, cam) = Vec3b(colL, colL, colL);
        //}
      }
    }
    for (int c = 1; c < wG[lvl_check] - 1; ++c) {
      for (int r = 1; r < hG[lvl_check] - 1; ++r) {
        int i = c + r * wG[lvl_check];
        // for (int i = 0; i < wG[0] * hG[0]; i++) {
        // BRIGHTNESS TRANSFER
        // float colL = (*(colorRef + i))[0];
        float edge_val = (float)(*(edge_label_image_start + i))[0];
        float label_val = (float)(*(edge_label_image_start + i))[1];
        label_val =
            (fh->label_num[lvl_check][cam] > 0) ? 255.0f * (label_val) / (float)fh->label_num[lvl_check][cam] : 0;
        float dt = (*(dt_dx_dy_start + i))[0];
        float dx = (*(dt_dx_dy_start + i))[1];
        float dy = (*(dt_dx_dy_start + i))[2];
        if (dt > fh->max_dt_dx_dy[lvl_check][cam][0] || dt < fh->min_dt_dx_dy[lvl_check][cam][0]) {
          std::cerr << "dt > fh->max_dt_dx_dy[lvl_check][cam][0] || dt < "
                       "fh->min_dt_dx_dy[lvl_check][cam][0]"
                    << ", dt: " << dt << std::endl;
          std::exit(1);
        }
        if (dx > fh->max_dt_dx_dy[lvl_check][cam][1] || dx < fh->min_dt_dx_dy[lvl_check][cam][1]) {
          std::cerr << "dx > fh->max_dt_dx_dy[lvl_check][cam][1] || dx < "
                       "fh->min_dt_dx_dy[lvl_check][cam][1]"
                    << ", dx: " << dx << std::endl;
          std::exit(1);
        }
        if (dy > fh->max_dt_dx_dy[lvl_check][cam][2] || dy < fh->min_dt_dx_dy[lvl_check][cam][2]) {
          std::cerr << "dy > fh->max_dt_dx_dy[lvl_check][cam][2] || dy < "
                       "fh->min_dt_dx_dy[lvl_check][cam][2]"
                    << ", dy: " << dy << std::endl;
          std::exit(1);
        }
        dt = (dt_len > 0) ? 255.0f * (dt - fh->min_dt_dx_dy[lvl_check][cam][0]) / dt_len : 0;
        dx = (dx_len > 0) ? 255.0f * (dx - fh->min_dt_dx_dy[lvl_check][cam][1]) / dx_len : 0;
        dy = (dy_len > 0) ? 255.0f * (dy - fh->min_dt_dx_dy[lvl_check][cam][2]) / dy_len : 0;

        // if (colL < 0)
        //   colL = 0;
        // if (colL > 255)
        //   colL = 255;
        if (edge_val < 0) edge_val = 0;
        if (edge_val > 255) edge_val = 255;
        if (label_val < 0) label_val = 0;
        if (label_val > 255) label_val = 255;
        if (dt < 0) dt = 0;
        if (dt > 255) dt = 255;
        if (dx < 0) dx = 0;
        if (dx > 255) dx = 255;
        if (dy < 0) dy = 0;
        if (dy > 255) dy = 255;
        // img_target->at(i, cam) = Vec3b(colL, colL, colL);
        // edge_target->at(i, cam) = Vec3b(colL, colL, colL);
        // edge_only_target->at(i, cam) = Vec3b(edge_val, edge_val, edge_val);
        label_only_target->at(i, cam) = Vec3b(label_val, label_val, label_val);
        edge_dt_target->at(i, cam) = Vec3b(dt, edge_val, edge_val);
        dx_target->at(i, cam) = Vec3b(dx, dx, dx);
        dy_target->at(i, cam) = Vec3b(dy, dy, dy);
        //}
      }
    }
  }
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec2i* edge_pixel_start = fh->edge_pixels[0] + wG[0] * hG[0] * cam;
    for (int i = 0; i < fh->edge_pixel_num[0][cam]; ++i) {
      int epx = edge_pixel_start[i][0];
      int epy = edge_pixel_start[i][1];
      if (epx < 10 || epx >= wG[0] - 10 || epy < 10 || epy >= hG[0] - 10) continue;
      edge_target->setPixelCirc((float)epx + 0.5, (float)epy + 0.5, makeRainbow3B(0.1), cam);
    }
  }
#endif
  //[ ***step 5*** ] 构建之前关键帧与当前帧fh的残差(旧的), or before
  // optimization
  // =========================== add new residuals for old points
  // =========================
  int pid_count = 0;
  int pid_count_success = 0;
  int numFwdResAdde = 0;
  for (FrameHessian* fh1 : frameHessians)  // go through all active frames
  {
    if (fh1 == fh) continue;
    int point_checked_num = 0;
    for (PointHessian* ph : fh1->pointHessians) {
      //      for (int target_cid = 0; target_cid < kCameraNumUsed;
      //      ++target_cid) {
      //#ifdef DISABLE_CROSS_CID_ALIGN
      //        if (ph->host_cid != target_cid) {
      //          continue;
      //        }
      //#endif
      // TODO roger,
      // TODO 先无脑给最新帧的每一个cid都配上一个residual，最多再价格标志。
      PointFrameResidual *r = new PointFrameResidual(
            ph, fh1, fh, ph->host_cid/*,
            target_cid*/); // 新建当前帧fh和之前帧之间的残差
      /// 这时J只是开辟了空间，还没有赋值, 初值为0
      //  printf("r->J->resF[0]: %f \n",r->J->resF[0]);
      //  printf("r->J->resF(0): %f \n",r->J->resF(0));
      //        for (int target_cid = 0; target_cid < kCameraNumUsed;
      //        ++target_cid) {
      //            r->setState(ResState::IN, target_cid);
      //        }
      // TODO roger, 对于sw内的stable点,
      // 因为是最新帧，先无脑构建push进去，至于是不是inlier，优化时再判断
      ph->residuals.push_back(r);
      ef->insertResidual(r, Hcalib.p_multi_camera, true);
      ph->lastResiduals[1] = ph->lastResiduals[0];
      std::array<ResState, kCameraNumUsed> res_state{};
      //#ifdef USE_MULTI_CAM
      SE3 fhToNew_ = fh->PRE_worldToCam * ph->host->PRE_camToWorld;
      ph->idepth_before = ph->idepth;
      ph->idepth_backup = ph->idepth;
      ph->is_idp_optimized = false;
#if 1  // def USE_MULTI_CAM
      std::vector<ImmaturePoint*> toOptimize;
      ImmaturePoint* impt = new ImmaturePoint(ph->u, ph->v, ph->host, 0, &Hcalib, ph->host_cid, 0);
      assert(std::isfinite(impt->energyTH));
      impt->idepth_min = impt->idepth_max = ph->idepth;
      toOptimize.push_back(impt);
      std::vector<PointHessian*> optimized;
      // PointHessian *point_hessian;
      // optimized.push_back(point_hessian);
      optimized.resize(toOptimize.size());
      ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size() * kCameraNumUsed];
      optimized[0] = optimizeImmaturePoint(toOptimize[0], 5 /*10*/, tr, false, pid_count < 5);

      PointHessian* newpoint = optimized[0];
      // ph->idepth_before = ph->idepth;
      if (pid_count < 1) {
        printf("is_force_kf: %d, is_force_no_kf: %d\n", forceKF, forceNoKF);
      }
      pid_count++;
      if (newpoint != 0 && newpoint != (PointHessian*)((long)(-1))) {
        if (pid_count_success < 5) {
          printf("depth_diff: %f, pid_count_success: %d, forceKF: %d\n", 1 / newpoint->idepth - 1 / ph->idepth,
                 pid_count_success, forceKF);
        }
        pid_count_success++;
        if (!forceKF) {
          ph->setIdepthZero(newpoint->idepth);
          ph->setIdepth(newpoint->idepth);
          ph->idepth_backup = newpoint->idepth;
          if (newpoint->idepth != newpoint->idepth_backup) {
            printf("newpoint->idepth != newpoint->idepth_backup\n");
            std::exit(1);
          }
          ph->is_idp_optimized = true;
        }
        delete newpoint;
        optimized[0] = 0;
      } else if (newpoint == (PointHessian*)((long)(-1))) {
        // delete newpoint;
      } else {
        assert(newpoint == 0 /*|| newpoint == (PointHessian *)((long)(-1))*/);
      }
      delete impt;
      toOptimize[0] = 0;
      // delete point_hessian;
      delete[] tr;
      toOptimize.clear();
      optimized.clear();
#endif
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        r->setState(ResState::IN, target_cid);
        // 设置上上个残差
        res_state[target_cid] = ResState::IN;

        //                            r, ResState::IN);
        //                    std::pair<PointFrameResidual *, ResState>(
        //                            r, ResState::IN); // 当前的设置为上一个
#if defined(SHOW_NEWLY_PREDICTED_RESIDUALS)  // && defined(USE_MULTI_CAM)
        Vec2i* edge_label_image_start = fh->edge_label_image[0] + wG[0] * hG[0] * target_cid;
        Vec3f* dt_dx_dy_start = fh->dt_dx_dy[0] + wG[0] * hG[0] * target_cid;
        Vec2i* label2xy_start = fh->label2xy[0] + wG[0] * hG[0] * target_cid;
        SE3 fhToNew = fh->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() * fhToNew_ *
                      fh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
        Vec3f xyz_cur = fhToNew.rotationMatrix().cast<float>() *
                            (coarseDistanceMap->Ki[0] * Vec3f(ph->u, ph->v, 1) / (ph->idepth)) +
                        fhToNew.translation().cast<float>();
        Vec3f proj = coarseDistanceMap->K[0] * xyz_cur;
        proj /= proj[2];
        Vec3f xyz_cur0 = fhToNew.rotationMatrix().cast<float>() *
                             (coarseDistanceMap->Ki[0] * Vec3f(ph->u, ph->v, 1) / (ph->idepth_before)) +
                         fhToNew.translation().cast<float>();
        Vec3f proj0 = coarseDistanceMap->K[0] * xyz_cur0;
        proj0 /= proj0[2];
        if (xyz_cur[2] > 0.05 && xyz_cur0[2] > 0.05 &&
            (proj[0] > 10 && proj[1] > 10 && proj[0] < wG[0] - 10 && proj[1] < hG[0] - 10) &&
            (proj0[0] > 10 && proj0[1] > 10 && proj0[0] < wG[0] - 10 && proj0[1] < hG[0] - 10)) {
          img_target->setPixelCirc(proj0[0] + 0.5, proj0[1] + 0.5, makeRainbow3B(0.1), target_cid);
          if (ph->is_idp_optimized) {
            img_target->setPixel9(proj[0] + 0.5, proj[1] + 0.5, makeRainbow3B(1), target_cid);
          } else {
            img_target->setPixel9(proj[0] + 0.5, proj[1] + 0.5, makeRainbow3B(0.1), target_cid);
          }
          Vec2i proj_check = (proj.head(2) + Vec2f(0.5, 0.5)).cast<int>();
          int label = edge_label_image_start[proj_check[0] + proj_check[1] * wG[0]][1];
          float dt_val = dt_dx_dy_start[proj_check[0] + proj_check[1] * wG[0]][0];
          Vec2i nearestPt = label2xy_start[label];

          if (point_checked_num < 5) {
            Vec2i* edge_pixel_start = fh->edge_pixels[0] + wG[0] * hG[0] * target_cid;
            float min_dt_dist = 9999;
            Vec2i nearestPt_check = Vec2i::Zero();

            std::vector<Vec2i> nearest_pts;
            for (int i = 0; i < fh->edge_pixel_num[0][target_cid]; ++i) {
              float dist = (edge_pixel_start[i].cast<float>() - proj_check.cast<float>()).norm();
              int label_val = edge_label_image_start[i][1];
              if (label_val >= fh->edge_pixel_num[0][target_cid]) {
                printf("label over flow\n");
                std::exit(1);
              }
              if (dist < min_dt_dist) {
                min_dt_dist = dist;
                nearestPt_check = edge_pixel_start[i];
                // std::cout << "dist: " << dist << ", nearestPt_check: "
                //           << nearestPt_check.transpose()
                //           << ", proj: " << proj.transpose() << std::endl;
              }
            }
            for (int i = 0; i < fh->edge_pixel_num[0][target_cid]; ++i) {
              float dist = (edge_pixel_start[i].cast<float>() - proj_check.cast<float>()).norm();
              if (dist == min_dt_dist) {
                nearest_pts.emplace_back(edge_pixel_start[i]);
              }
            }
            bool has_same_pt = false;
            for (int i = 0; i < nearest_pts.size(); ++i) {
              if ((nearest_pts[i] - nearestPt).cast<float>().norm() < 0.0001) {
                has_same_pt = true;
              }
              // printf("cid: %d, i: %d, label: %d, nearest_pt: [%d %d]\n",
              //        target_cid, i, label, nearest_pts[i][0],
              //        nearest_pts[i][1]);
            }
            if (min_dt_dist < 2 && !has_same_pt /*(nearestPt_check - nearestPt).cast<float>().norm() > 0.0001*/) {
              std::vector<uint8_t> edge_data(wG[0] * hG[0]);
              std::vector<float> dt_data(wG[0] * hG[0]);
              std::vector<float> label_data(wG[0] * hG[0]);
              for (int i = 0; i < wG[0] * hG[0]; ++i) {
                int val = edge_label_image_start[i][0];
                if (val >= 255) {
                  val = 255;
                }
                if (val <= 0) {
                  val = 0;
                }
                edge_data[i] = (uint8_t)val;
                dt_data[i] = dt_dx_dy_start[i][0];
                label_data[i] = (float)edge_label_image_start[i][1];
              }
              cv::Mat edge = cv::Mat(hG[0], wG[0], CV_8UC1, edge_data.data()).clone();
              cv::Mat distanceTransformMap = cv::Mat(hG[0], wG[0], CV_32FC1, dt_data.data()).clone();
              cv::Mat labels = cv::Mat(hG[0], wG[0], CV_32FC1, label_data.data()).clone();

              cv::imwrite("edge.png", edge);
              cv::imwrite("dt.tiff", distanceTransformMap);
              cv::imwrite("labels.tiff", labels);
              printf(
                  "target_cid: %d, nearest_pts_size: %d, nearestPt check "
                  "failed, edge_pix_num: %d, point_checked_num: %d, dt_value: "
                  "%f, min_dt_dist: %f, nearest_pt_diff: %f, proj_check: [%d "
                  "%d], nearestPt: [%f %f], nearestPt_check: [%f %f]\n",
                  target_cid, nearest_pts.size(), fh->edge_pixel_num[0][target_cid], point_checked_num, dt_val,
                  min_dt_dist, (nearestPt_check - nearestPt).cast<float>().norm(), proj_check[0], proj_check[1],
                  (float)nearestPt[0], (float)nearestPt[1], (float)nearestPt_check[0], (float)nearestPt_check[1]);
              std::exit(1);
            }
            if (target_cid == 0) {
              point_checked_num++;
            }
          }
          edge_target->setPixelCirc((float)nearestPt[0], (float)nearestPt[1], Vec3b(0, 0, 255), target_cid);
          edge_target->setPixel9(proj[0] + 0.5, proj[1] + 0.5, Vec3b(0, 255, 0), target_cid);
        }
#endif
      }
      ph->lastResiduals[0] = std::pair<PointFrameResidual*, std::array<ResState, kCameraNumUsed>>(r, res_state);

      numFwdResAdde += 1;
      //}
    }
  }
#if defined(SHOW_NEWLY_PREDICTED_RESIDUALS)  // && defined(USE_MULTI_CAM)
#if 1
  IOWrap::displayImage("predicted vms in newest frame", img_target);
  IOWrap::displayImage("predicted edges in newest frame", edge_target);
  // IOWrap::displayImage("edge_only in newest frame", edge_only_target);
  IOWrap::displayImage("label_only in newest frame", label_only_target);
  IOWrap::displayImage("edge_dt in newest frame", edge_dt_target);
  IOWrap::displayImage("dx in newest frame", dx_target);
  IOWrap::displayImage("dy in newest frame", dy_target);
  IOWrap::waitKey(1);
#endif
  delete img_target;
  delete edge_target;
  // delete edge_only_target;
  delete label_only_target;
  delete edge_dt_target;
  delete dx_target;
  delete dy_target;
#endif
  if (false) {
    printf("frameHessians: %d\n", frameHessians.size());
    int fid = 0;
    for (FrameHessian* fh : frameHessians) {
      printf("fid: %d, points: %d\n", fid, fh->pointHessians.size());
      for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
        PointHessian* ph = fh->pointHessians[i];
        if (ph == 0) continue;
        std::cout << "before opt, ph->residuals: " << ph->residuals.size() << std::endl;
      }
      fid++;
    }
  }

  timeMeasurementAddFrame.end();

  ///[ ***step 6*** ] 激活所有关键帧上的部分未成熟点(构造新的残差)
  /// 开始往上面开辟的空间中填入残差，雅可比(no, not yet right?)
  // =========================== Activate Points (& flag for marginalization).
  // =========================
  // TODO triangulate immature points to active PointHessian
  activatePointsMT();
  ef->makeIDX();  // ? 为啥要重新设置ID呢, 是因为加新的帧了么

  if (setting_useGTSAMIntegration) {
    // Adds new keyframe to the BA graph, together with matching factors (e.g.
    // IMUFactors).
    baIntegration->addKeyframeToBA(fh->shell->id, fh->shell->camToWorld, ef->frames);
  }

  // =========================== OPTIMIZE ALL =========================
  //[ ***step 7*** ] 对滑窗内的关键帧进行优化(说的轻松, 里面好多问题)
  // TODO sliding window optimization entrance
  fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;  // 这两个不是一个值么???
  printf("before BA\n");
  float rmse = optimize(setting_maxOptIterations);
  printf("after BA, rmse: %f\n", rmse);
  // std::exit(1);
  if (false) {
    printf("frameHessians: %d\n", frameHessians.size());
    int fid = 0;
    for (FrameHessian* fh : frameHessians) {
      printf("fid: %d, points: %d\n", fid, fh->pointHessians.size());
      for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
        PointHessian* ph = fh->pointHessians[i];
        if (ph == 0) continue;
        std::cout << "after opt, ph->residuals: " << ph->residuals.size() << std::endl;
      }
      fid++;
    }
  }

  // =========================== Figure Out if INITIALIZATION FAILED
  // =========================
  //* 所有的关键帧数小于4，认为还是初始化，此时残差太大认为初始化失败
  printf("init rmse: %f\n", rmse);
#ifndef USE_MULTI_CAM
#ifndef USE_ZNCC
  std::vector<float> init_rmse_thr = {20, 13, 9};
#else
  std::vector<float> init_rmse_thr = {25, 20, 20};
#endif
#else
#ifndef USE_ZNCC
  std::vector<float> init_rmse_thr = {300, 300, 300};
#else
  std::vector<float> init_rmse_thr = {30, 30, 30};
#endif
#endif
  if (allKeyFramesHistory.size() <= 4) {
    if (allKeyFramesHistory.size() == 2 && rmse > init_rmse_thr[0] /*20*/ * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting. rmse: %f, sw_size: %d\n", rmse, allKeyFramesHistory.size());
      initFailed = true;  // 优化后的能量函数太大, 认为是跟丢了
      std::exit(1);
    }
    if (allKeyFramesHistory.size() == 3 && rmse > init_rmse_thr[1] /*13*/ * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting. rmse: %f, sw_size: %d\n", rmse, allKeyFramesHistory.size());
      initFailed = true;
      std::exit(1);
    }
    if (allKeyFramesHistory.size() == 4 && rmse > init_rmse_thr[2] /*9*/ * benchmark_initializerSlackFactor) {
      printf("I THINK INITIALIZATINO FAILED! Resetting. rmse: %f, sw_size: %d\n", rmse, allKeyFramesHistory.size());
      initFailed = true;
      std::exit(1);
    }
  }

  //[ ***step 8*** ] 去除外点, 把最新帧设置为参考帧
  // =========================== REMOVE OUTLIER =========================
  // TODO 是否可以更加严格一些
  removeOutliers();

  if (setting_useIMU) {
    imuIntegration.postOptimization(fh->shell->id);
  }

  bool imuReady = false;
  {
    dmvio::TimeMeasurement timeMeasurement("makeKeyframeChangeTrackingRef");
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);

    if (setting_useIMU) {
      imuReady = imuIntegration.finishKeyframeOptimization(fh->shell->id);
    }
    // TODO
    // 之前插入新关键帧，两个tracker指针内容交换了一下，这里给_forNewKF赋上新的内容
    coarseTracker_forNewKF->makeK(&Hcalib);
    // TODO roger, make reference depth map
    printf("end ==== res_num_ratio, setCoarseTrackingRef\n");
    coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

    coarseTracker_forNewKF->debugPlotIDepthMap(frameHessians, &minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
    coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
  }
  // for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  debugPlot("post Optimize");
  //}
  for (auto* ow : outputWrapper) {
    if (imuReady && !imuUsedBefore) {
      // Update state if this is the first time after IMU init.
      // VIO is now initialized the next published scale will be useful.
      ow->publishSystemStatus(dmvio::VISUAL_INERTIAL);
    }
    ow->publishTransformDSOToIMU(imuIntegration.getTransformDSOToIMU());
  }
  imuUsedBefore = imuReady;

  //[ ***step 9*** ] 标记删除和边缘化的点, 并删除&边缘化
  // =========================== (Activate-)Marginalize Points
  // =========================
  dmvio::TimeMeasurement timeMeasurementMarginalizePoints("marginalizeAndRemovePoints");
  flagPointsForRemoval();  // TODO
                           // 这里要把残差恢复至fej状态的（现在的残差是最新状态的，会小于fej状态下的残差，现在就是要让残差变大）
  ef->dropPointsF();       // TODO 扔掉drop的点
                           // 刚只扔了pointhessian，后端的能量函数还没扔掉，现在也把ef扔掉
  // TODO 每次设置线性化点都会更新零空间,
  // 因为刚删掉了一帧，所以要再更新一下零空间
  getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale, ef->lastNullspaces_affA, ef->lastNullspaces_affB);
  ef->marginalizePointsF();
  timeMeasurementMarginalizePoints.end();

  //[ ***step 10*** ] 生成新的点
  // =========================== add new Immature points & new residuals
  // =========================
  // TODO [detect new points]
  // TODO roger make new traces
  printf("make new traces\n");
  makeNewTraces(fh,
                0);  // TODO 刚去掉了一些点，现在肯定要再提一些点,
                     // 并且现在还是新插了关键帧，得保证又足够的immaturePoints去支持我的地图

  dmvio::TimeMeasurement timeMeasurementPublish("publishInMakeKeyframe");
  for (IOWrap::Output3DWrapper* ow : outputWrapper) {
    ow->publishGraph(ef->connectivityMap);
    ow->publishKeyframes(frameHessians, false, &Hcalib);
  }
  timeMeasurementPublish.end();

  // =========================== Marginalize Frames =========================
  //[ ***step 11*** ] 边缘化掉关键帧
  //* 边缘化一帧要删除or边缘化上面所有点
  // TODO 刚marg了点，现在要marg帧
  dmvio::TimeMeasurement timeMeasurementMargFrames("marginalizeFrames");
  for (unsigned int i = 0; i < frameHessians.size(); i++)
    if (frameHessians[i]->flaggedForMarginalization) {
      marginalizeFrame(frameHessians[i]);
      i = 0;
      if (setting_useGTSAMIntegration) {
        baIntegration->updateBAOrdering(ef->frames);
      }
    }
  timeMeasurementMargFrames.end();

  printLogLine();
  printEigenValLine();

  if (setting_useGTSAMIntegration) {
    baIntegration->updateBAValues(ef->frames);
  }

  if (setting_useIMU) {
    imuIntegration.finishKeyframeOperations(fh->shell->id);
  }
}
//#define SHOW_SEED_MASK_RET
void FullSystem::maskSeedsAcrossCids() {
#ifdef SHOW_SEED_MASK_RET
  MinimalImageB3* img_host;
#endif
  FrameHessian* firstFrame = coarseInitializer->firstFrame;
  SE3 firstToNew = coarseInitializer->thisToNext;
  for (int cid1 = 0; cid1 < kCameraNumUsed; ++cid1) {
    for (int i = 0; i < coarseInitializer->level_cid_to_numPoints[0][cid1]; i++) {
      Pnt* point1 = coarseInitializer->points[0] + i + coarseInitializer->level_cid_to_npts_success_offset[0][cid1];
      float idepth = point1->idepth;
      Vec3f uv_1 = Vec3f(point1->u, point1->v, 1);

      if (!point1->isGood) {
        continue;
      }
#ifdef SHOW_SEED_MASK_RET
      img_host = new MinimalImageB3(wG[0], hG[0]);

      for (int cam = 0; cam < kCameraNumUsed; ++cam) {
        Vec3f* colorRef = firstFrame->dI + wG[0] * hG[0] * cam;
        for (int i = 0; i < wG[0] * hG[0]; i++) {
          // BRIGHTNESS TRANSFER
          float colL = (*(colorRef + i))[0];
          if (colL < 0) colL = 0;
          if (colL > 255) colL = 255;
          img_host->at(i, cam) = Vec3b(colL, colL, colL);
        }
      }
      img_host->setPixelCirc(point1->u + 0.5, point1->v + 0.5, makeRainbow3B(1), cid1);
#endif
      for (int cid2 = 0; cid2 < kCameraNumUsed; ++cid2) {
        if (cid1 == cid2) {
          continue;
        }
        SE3 Tc2c1 = Hcalib.p_multi_camera->cid_to_T01_SE3_inv[cid2] * Hcalib.p_multi_camera->cid_to_T01_SE3[cid1];
        Vec3f xyz2 = (Tc2c1 * ((KiG[0] * uv_1) / idepth).cast<double>()).cast<float>();
        Vec3f uv_21 = KG[0] * xyz2;
        uv_21 /= uv_21[2];
        if (!(uv_21[0] > 10 && uv_21[1] > 10 && uv_21[0] < wG[0] - 10 && uv_21[1] < hG[0] - 10 && xyz2[2] > 0.1)) {
          continue;
        }
        // std::cout << "cid2; " << cid2 << ", uv_21: " << uv_21.transpose() <<
        // std::endl;
        for (int j = 0; j < coarseInitializer->level_cid_to_numPoints[0][cid2]; j++) {
          Pnt* point2 = coarseInitializer->points[0] + j + coarseInitializer->level_cid_to_npts_success_offset[0][cid2];
          if (!point2->isGood) {
            continue;
          }
          Vec3f uv_2 = Vec3f(point2->u, point2->v, 1);
#ifdef SHOW_SEED_MASK_RET
          img_host->setPixelCirc(point2->u + 0.5, point2->v + 0.5, makeRainbow3B(0.1), cid2);
#endif
        }
#ifdef SHOW_SEED_MASK_RET
        img_host->setPixelCirc(uv_21[0] + 0.5, uv_21[1] + 0.5, makeRainbow3B(1), cid2);
#endif
      }
#ifdef SHOW_SEED_MASK_RET
      IOWrap::displayImage("masked seeds", img_host);
      IOWrap::waitKey(0);
#endif
    }
  }
#ifdef SHOW_SEED_MASK_RET
  delete img_host;
#endif
}
//@ 从初始化中提取出信息, 用于跟踪.
void FullSystem::initializeFromInitializer(FrameHessian* newFrame) {
  boost::unique_lock<boost::mutex> lock(mapMutex);
  //[ ***step 1*** ] 把第一帧设置成关键帧, 加入队列, 加入EnergyFunctional
  // add firstframe.
  FrameHessian* firstFrame = coarseInitializer->firstFrame;  // 第一帧增加进地图
  firstFrame->idx = frameHessians.size();                    // 赋值给它id (0开始)
  frameHessians.push_back(firstFrame);                       // 地图内关键帧容器
  firstFrame->frameID = allKeyFramesHistory.size();          // 所有历史关键帧id
  allKeyFramesHistory.push_back(firstFrame->shell);          // 所有历史关键帧
  // TODO energy function means factor graph?
  // TODO 执行setAdjointsF()时,
  // 通过打印信息证明了这确实是fej的状态，只要frame被marg后，相同id的host target
  // pair的相对位姿就不可能变了
  ef->insertFrame(firstFrame, &Hcalib);
  ///* 计算frameHessian的预计算值, 和状态的delta值
  // TODO
  // 设置当前估计相对于fej的增量,对pose来说还要把相对于fej的绝对增量转成相对增量
  setPrecalcValues();  // 设置相对位姿预计算值
  // TODO 在gtsam里开辟状态变量的id，Hb的空间？
  baIntegration->addFirstBAFrame(firstFrame->shell->id);

  firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);              // 20%的点数目
  firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);  // 被边缘化
  firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);           // 丢掉的点

  // mask seeds across cids
  if (kCameraNumUsed > 1) {
    maskSeedsAcrossCids();
  }

  //[ ***step 2*** ] 求出平均尺度因子
  float sumID = 1e-5, numID = 1e-5;

  double sumFirst = 0.0;
  double sumSecond = 0.0;
  int num = 0;
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    for (int i = 0; i < coarseInitializer->level_cid_to_numPoints[0][cid]; i++) {
      //? iR的值到底是啥
      if (false) {
        sumID += coarseInitializer->points[0][i + coarseInitializer->level_cid_to_npts_success_offset[0][cid]]
                     .iR;  // 第0层点的中位值, 相当于
        numID++;
        num++;
      } else {
        Pnt point = coarseInitializer->points[0][i + coarseInitializer->level_cid_to_npts_success_offset[0][cid]];
        if (!point.isGood) {
          continue;
        }
        sumID += point.iR;  // 第0层点的中位值, 相当于
        numID++;
        num++;
      }
    }
  }
  //  sumFirst /= num;
  //  sumSecond /= num;

  float rescaleFactor = 1;
  if (kCameraNumUsed == 1) {
    rescaleFactor = 1 / (sumID / numID);
  }
  //[ ***step 4*** ] 设置第一帧和最新帧的待优化量, 参考帧
  SE3 firstToNew = coarseInitializer->thisToNext;
  std::cout << "Scaling with rescaleFactor: " << rescaleFactor << std::endl;
  firstToNew.translation() /= rescaleFactor;

  // randomly sub-select the points I need.
  // 目标点数 / 实际提取点数
  float keepPercentage = setting_desiredPointDensity / numID;  // coarseInitializer->numPoints[0];

  if (!setting_debugout_runquiet) {
    int point_count = 0;
    for (int id = 0; id < kCameraNumUsed; ++id) {
      if (false) {
        point_count += coarseInitializer->level_cid_to_numPoints[0][id];
      } else {
        for (int i = 0; i < coarseInitializer->level_cid_to_numPoints[0][id]; i++) {
          Pnt point = coarseInitializer->points[0][i + coarseInitializer->level_cid_to_npts_success_offset[0][id]];
          if (point.isGood) {
            point_count++;
          }
        }
      }
    }
    printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
           (int)(setting_desiredPointDensity), point_count
           /*coarseInitializer
                   ->level_cid_to_npts_success_offset[0][kCameraNumUsed - 1] +
           coarseInitializer->level_cid_to_numPoints[0][kCameraNumUsed - 1]*/
           /*numID */ /*coarseInitializer->numPoints[0]*/);
    if (kCameraNumUsed > 1) {
      assert(std::abs(static_cast<float>(point_count) - numID) < 0.0001);
    }
  }
  //[ ***step 3*** ] 创建PointHessian, 点加入关键帧, 加入EnergyFunctional
//#define CHECK_INIT
#ifdef CHECK_INIT
  std::cout << "Hcalib.intr:\n" << Hcalib.intr << std::endl;
  MinimalImageB3* img_host;
  MinimalImageB3* img_target;

  img_host = new MinimalImageB3(wG[0], hG[0]);
  img_target = new MinimalImageB3(wG[0], hG[0]);

  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec3f* colorRef = firstFrame->dI + wG[0] * hG[0] * cam;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorRef + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_host->at(i, cam) = Vec3b(colL, colL, colL);
    }
  }
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Eigen::Vector3f* colorCur = newFrame->dI + cam * wG[0] * hG[0];
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorCur + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_target->at(i, cam) = Vec3b(colL, colL, colL);
    }
  }
#endif
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int i = 0; i < coarseInitializer->level_cid_to_numPoints[0][host_cid]; i++) {
      if (rand() / (float)RAND_MAX > keepPercentage) {
        // continue; // 如果提取的点比较少, 不执行; 提取的多, 则随机干掉
      }
      Pnt* point = coarseInitializer->points[0] + i + coarseInitializer->level_cid_to_npts_success_offset[0][host_cid];
      // TODO roger, like SetFromImage in orca, 判断这个坐标纹理是否充分
      if (!point->isGood) {
        continue;
      }
      if (rand() / (float)RAND_MAX > keepPercentage) {
        continue;  // 如果提取的点比较少, 不执行; 提取的多, 则随机干掉
      }
      assert(point->host_cid == host_cid);
      ImmaturePoint* pt;
      if (kCameraNumUsed == 1 || true) {
        pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib, host_cid, 0);
      } else {
        pt = new ImmaturePoint(point->u, point->v, firstFrame, point->my_type, &Hcalib, host_cid, 0);
      }

      if (!std::isfinite(pt->energyTH)) {
        assert(!std::isfinite(pt->energyTH_converged));
        delete pt;
        continue;
      }  // 点值无穷大

      // 创建ImmaturePoint就为了创建PointHessian? 是为了接口统一吧
      if (kCameraNumUsed == 1) {
        pt->idepth_max = pt->idepth_min = 1;
      } else {
        pt->idepth_max = pt->idepth_min = point->idepth;
      }
      // std::cout << "idepth: " << point->idepth << std::endl;
      PointHessian* ph = new PointHessian(pt, &Hcalib, host_cid);
      assert(std::isfinite(pt->energyTH));
      assert(std::isfinite(pt->energyTH_converged));
      delete pt;
      // TODO roger, create patch, setFromImage, if fail, delete the point
      if (!std::isfinite(ph->energyTH)) {
        printf("energyTH MUST NOT be NAN, sth wrong\n");
        std::exit(1);
        delete ph;
        continue;
      }
      if (kCameraNumUsed == 1) {
        ph->setIdepthScaled(point->iR * rescaleFactor);  //? 为啥设置的是scaled之后的
      } else {
        ph->setIdepthScaled(point->idepth * rescaleFactor);  //? 为啥设置的是scaled之后的
      }
      ph->setIdepthZero(ph->idepth);  //! 设置初始先验值, 还有神奇的求零空间方法
#if 1                                 // ndef USE_MULTI_CAM
      ph->hasDepthPrior = true;
#endif
      ph->setPointStatus(PointHessian::ACTIVE);  // 激活点

      firstFrame->pointHessians.push_back(ph);
      ef->insertPoint(ph);
#ifdef CHECK_INIT
      img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(1), host_cid);
      img_host->setPixelCirc(point->u + 0.5, point->v + 0.5, makeRainbow3B(1), host_cid);

      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        SE3 refToNew = newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() * firstToNew *
                       newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
        Mat33f RKi = (refToNew.rotationMatrix() * Hcalib.intr_inv).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Vec3f pt = RKi * Vec3f(point->u, point->v, 1) + t * point->idepth * rescaleFactor;
        //        std::cout << "point->idepth * rescaleFactor: "
        //                  << point->idepth * rescaleFactor << std::endl;
        float u = pt[0] / pt[2];
        float v = pt[1] / pt[2];
        float new_idepth = point->idepth * rescaleFactor / pt[2];
        // 像素坐标pj
        float Ku = float(Hcalib.intr(0, 0)) * u + float(Hcalib.intr(0, 2));
        float Kv = float(Hcalib.intr(1, 1)) * v + float(Hcalib.intr(1, 2));
        //        std::cout << "init, "
        //                  << ", u: " << Ku << ", v: " << Kv
        //                  << ", idepth: " << new_idepth << std::endl;
        if (!(Ku > 10 && Kv > 10 && Ku < wG[0] - 20 && Kv < hG[0] - 20 && new_idepth > 0)) {
          //                isGood = false;
          //                break;
          continue;
        }
        //        img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(1),
        //        target_cid);
        img_target->setPixelCirc(Ku + 0.5, Kv + 0.5, makeRainbow3B(1), target_cid);
      }
      /////////////////////////////////////////////////////////////////////////////
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        SE3 refToNew = newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() * firstToNew *
                       newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
        Mat33f RKi = (refToNew.rotationMatrix() * Hcalib.intr_inv).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Vec3f pt = RKi * Vec3f(ph->u, ph->v, 1) + t * ph->idepth_zero_scaled;
        float u = pt[0] / pt[2];
        float v = pt[1] / pt[2];
        float new_idepth = ph->idepth_zero_scaled / pt[2];
        // 像素坐标pj
        float Ku = float(Hcalib.intr(0, 0)) * u + float(Hcalib.intr(0, 2));
        float Kv = float(Hcalib.intr(1, 1)) * v + float(Hcalib.intr(1, 2));

        //        std::cout << "LBA, "
        //                  << ", u: " << Ku << ", v: " << Kv
        //                  << ", idepth: " << new_idepth << std::endl;
        if (!(Ku > 10 && Kv > 10 && Ku < wG[0] - 20 && Kv < hG[0] - 20 && new_idepth > 0)) {
          //                isGood = false;
          //                break;
          continue;
        }
        img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(0.1), target_cid);
        //        img_target->setPixelCirc(Ku + 0.5, Kv + 0.5,
        //        makeRainbow3B(0.1),
        //                                 target_cid);
      }

#endif
    }
  }
#ifdef CHECK_INIT
  IOWrap::displayImage("check init host", img_host);
  IOWrap::displayImage("check init target", img_target);
  IOWrap::waitKey(0);
  delete img_host;
  delete img_target;
#endif
  // really no lock required, as we are initializing.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    firstFrame->shell->camToWorld = firstPose;
    firstFrame->shell->aff_g2l = AffLight(0, 0);
    firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);
    firstFrame->shell->trackingRef = 0;
    firstFrame->shell->camToTrackingRef = SE3();
    firstFrame->shell->keyframeId = 0;

    newFrame->shell->camToWorld = firstPose * firstToNew.inverse();
    newFrame->shell->aff_g2l = AffLight(0, 0);
    newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
    newFrame->shell->trackingRef = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();
  }
  // TODO update states
  imuIntegration.finishCoarseTracking(*(newFrame->shell), true);

  initialized = true;
  printf("### ### INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}
#define SHOW_DETECTION_MASK
void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth) {
  dmvio::TimeMeasurement timeMeasurement("makeNewTraces");
  pixelSelector->allowFast = true;
  // int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0],
  // hG[0], setting_desiredDensity);
#ifdef SHOW_DETECTION_MASK
  MinimalImageB3* img = new MinimalImageB3(wG[0], hG[0]);
  // img->setBlack();
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec3f* colorRef = newFrame->dI + wG[0] * hG[0] * cam;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(colorRef + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img->at(i, cam) = Vec3b(colL, colL, colL);
    }
  }
#endif
  int numPointsTotal = 0;
  // TODO roger, in LBA, we only detect new points at level 0
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    numPointsTotal += pixelSelector->makeMaps(newFrame, selectionMap + wG[0] * hG[0] * cid,
                                              setting_desiredImmatureDensity, 1, false, 1, cid);
  }
  newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
  // fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);
  int new_pt_num_before = (int)newFrame->immaturePoints.size();
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    int pt_num_not_found_in_edge = 0;
    int new_pt = 0;
    for (int y = patternPaddingSeed + 1; y < hG[0] - patternPaddingSeed - 2; y++)
      for (int x = patternPaddingSeed + 1; x < wG[0] - patternPaddingSeed - 2; x++) {
        int i = x + y * wG[0];
        if (selectionMap[i + wG[0] * hG[0] * cid] == 0) continue;

        ImmaturePoint* impt = new ImmaturePoint(x, y, newFrame, selectionMap[i + wG[0] * hG[0] * cid], &Hcalib, cid, 0);
        // float dist = coarseDistanceMap
        //                       ->fwdWarpedIDDistFinal[x /2 + wG[1] * y/2 +
        //                                              wG[1] * hG[1] * cid] +
        //                                              (ptp[0] -
        //                                              floorf((float)(ptp[0])));
#ifdef SHOW_DETECTION_MASK
        img->setPixelCirc(impt->u, impt->v, Vec3b(0, 0, 255), cid);
#endif
#ifdef USE_EDGE_ALIGN
        bool found = false;
        Vec2i* edge_pixel_start = newFrame->edge_pixels[0] + wG[0] * hG[0] * cid;
        for (int i = 0; i < newFrame->edge_pixel_num[0][cid]; ++i) {
          if ((edge_pixel_start[i] - Vec2i(x, y)).cast<float>().norm() < 1.5f) {
            found = true;
            break;
          }
        }
        if (!found) {
          pt_num_not_found_in_edge++;
          // float nan_before = pt->energyTH;
          impt->energyTH = NAN;
          // printf("nan: [%f %f]\n", nan_before, pt->energyTH);
        } else {
#ifdef SHOW_DETECTION_MASK
          img->setPixelCirc(impt->u, impt->v, Vec3b(255, 0, 0), cid);
#endif
        }
#endif
        if (!std::isfinite(impt->energyTH)) {
          delete impt;  // 投影得到的不是有穷数
        } else {
          newFrame->immaturePoints.push_back(impt);
          new_pt++;
        }
#ifdef SHOW_DETECTION_MASK
        img->setPixel9(impt->u + 0.5, impt->v + 0.5, Vec3b(0, 255, 0), cid);
#endif
      }
    printf("cid: %d, new_pt: %d, pt_num_not_found_in_edge: %d\n", cid, new_pt, pt_num_not_found_in_edge);
  }
#ifdef SHOW_DETECTION_MASK
  IOWrap::displayImage("new immature point detection mask cur frame", img);
  IOWrap::waitKey(1);
  delete img;
#endif
  printf("NEW IMMATURE POINTS: [before after]: [%d %d]!\n", new_pt_num_before, (int)newFrame->immaturePoints.size());
}

//* 计算frameHessian的预计算值, 和状态的delta值
//@ 设置关键帧之间的关系
void FullSystem::setPrecalcValues() {
  for (FrameHessian* fh : frameHessians) {
    fh->targetPrecalc.resize(frameHessians.size());            // 每个目标帧预运算容器, 大小是关键帧数
    for (unsigned int i = 0; i < frameHessians.size(); i++) {  //? 还有自己和自己的???
      ///@ 计算优化前和优化后的相对位姿, 相对光度变化, 及中间变量
      // TODO 自己当自己的host
      fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }
  }
  /// 这是为了后面使用固定线性化点吧（FEJ）
  // TODO
  // 设置当前估计相对于fej的增量,对pose来说还要把相对于fej的绝对增量转成相对增量
  ef->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine() {
  dmvio::TimeMeasurement timeMeasurementMargFrames("printLogLine");
  if (frameHessians.size() == 0) return;

  if (!setting_debugout_runquiet)
    printf(
        "LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. "
        "a=%f, b=%f. Window %d (%d)\n",
        allKeyFramesHistory.back()->id, statistics_lastFineTrackRMSE, ef->resInA, ef->resInL, ef->resInM,
        (int)statistics_numForceDroppedResFwd, (int)statistics_numForceDroppedResBwd,
        allKeyFramesHistory.back()->aff_g2l.a, allKeyFramesHistory.back()->aff_g2l.b,
        frameHessians.back()->shell->id - frameHessians.front()->shell->id, (int)frameHessians.size());

  if (!setting_logStuff) return;

  if (numsLog != 0) {
    (*numsLog) << allKeyFramesHistory.back()->id << " " << statistics_lastFineTrackRMSE << " "
               << (int)statistics_numCreatedPoints << " " << (int)statistics_numActivatedPoints << " "
               << (int)statistics_numDroppedPoints << " " << (int)statistics_lastNumOptIts << " " << ef->resInA << " "
               << ef->resInL << " " << ef->resInM << " " << statistics_numMargResFwd << " " << statistics_numMargResBwd
               << " " << statistics_numForceDroppedResFwd << " " << statistics_numForceDroppedResBwd << " "
               << frameHessians.back()->aff_g2l().a << " " << frameHessians.back()->aff_g2l().b << " "
               << frameHessians.back()->shell->id - frameHessians.front()->shell->id << " " << (int)frameHessians.size()
               << " "
               << "\n";
    numsLog->flush();
  }
}

void FullSystem::printEigenValLine() {
  dmvio::TimeMeasurement timeMeasurementMargFrames("printEigenValLine");
  if (!setting_logStuff) return;
  if (ef->lastHS.rows() < 12) return;

  MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
  MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
  int n = Hp.cols() / STATE_DIM;
  assert(Hp.cols() % STATE_DIM == 0);

  // sub-select
  for (int i = 0; i < n; i++) {
    MatXX tmp6 = Hp.block(i * STATE_DIM, 0, 6, n * STATE_DIM);
    Hp.block(i * 6, 0, 6, n * STATE_DIM) = tmp6;

    MatXX tmp2 = Ha.block(i * STATE_DIM + 6, 0, 2, n * STATE_DIM);
    Ha.block(i * 2, 0, 2, n * 8) = tmp2;
  }
  for (int i = 0; i < n; i++) {
    MatXX tmp6 = Hp.block(0, i * STATE_DIM, n * STATE_DIM, 6);
    Hp.block(0, i * 6, n * STATE_DIM, 6) = tmp6;

    MatXX tmp2 = Ha.block(0, i * STATE_DIM + 6, n * STATE_DIM, 2);
    Ha.block(0, i * 2, n * STATE_DIM, 2) = tmp2;
  }

  VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
  VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
  VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
  VecX diagonal = ef->lastHS.diagonal();

  std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
  std::sort(eigenP.data(), eigenP.data() + eigenP.size());
  std::sort(eigenA.data(), eigenA.data() + eigenA.size());

  int nz = std::max(100, setting_maxFrames * 10);

  if (eigenAllLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
    (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
    eigenAllLog->flush();
  }
  if (eigenALog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(eigenA.size()) = eigenA;
    (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
    eigenALog->flush();
  }
  if (eigenPLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(eigenP.size()) = eigenP;
    (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
    eigenPLog->flush();
  }

  if (DiagonalLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(diagonal.size()) = diagonal;
    (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
    DiagonalLog->flush();
  }

  if (variancesLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
    (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
    variancesLog->flush();
  }

  std::vector<VecX>& nsp = ef->lastNullspaces_forLogging;
  (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
  for (unsigned int i = 0; i < nsp.size(); i++)
    (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
  (*nullspacesLog) << "\n";
  nullspacesLog->flush();
}

void FullSystem::printFrameLifetimes() {
  if (!setting_logStuff) return;

  boost::unique_lock<boost::mutex> lock(trackMutex);

  std::ofstream* lg = new std::ofstream();
  lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
  lg->precision(15);

  for (FrameShell* s : allFrameHistory) {
    (*lg) << s->id << " " << s->marginalizedAt << " " << s->statistics_goodResOnThis << " "
          << s->statistics_outlierResOnThis << " " << s->movedByOpt;

    (*lg) << "\n";
  }

  lg->close();
  delete lg;
}

void FullSystem::printEvalLine() { return; }

}  // namespace dso
