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

#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"
#include "util/color_map.h"
#include "util/settings.h"
#include "vector"
#include <math.h>

#include "IMU/IMUIntegration.hpp"

namespace dso {
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;
struct MultiCamera;

class CoarseTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CoarseTracker(int w, int h, dmvio::IMUIntegration& imuIntegration);

  ~CoarseTracker();

  bool trackNewestCoarse(bool& disable_kf_bak, const std::vector<FrameHessian*>& frameHessians, int all_keyframe_size,
                         FrameHessian* lastRef, FrameHessian* newFrameHessian, SE3& lastToNew_out,
                         AffLight& aff_g2l_out, int coarsestLvl, Vec5 minResForAbort,
                         IOWrap::Output3DWrapper* wrap = 0);

  void setCoarseTrackingRef(std::vector<FrameHessian*> frameHessians);

  void makeK(CalibHessian* HCalib);

  bool debugPrint, debugPlot;
  // ColorMap color_map = ColorMap(GetColorMap("jet"));

  Mat33f K[PYR_LEVELS];   // * kCameraNumUsed];
  Mat33f Ki[PYR_LEVELS];  // * kCameraNumUsed];
  float fx[PYR_LEVELS];   // * kCameraNumUsed];
  float fy[PYR_LEVELS];   // * kCameraNumUsed];
  float fxi[PYR_LEVELS];  // * kCameraNumUsed];
  float fyi[PYR_LEVELS];  // * kCameraNumUsed];
  float cx[PYR_LEVELS];   // * kCameraNumUsed];
  float cy[PYR_LEVELS];   // * kCameraNumUsed];
  float cxi[PYR_LEVELS];  // * kCameraNumUsed];
  float cyi[PYR_LEVELS];  // * kCameraNumUsed];
  int w[PYR_LEVELS];      // * kCameraNumUsed];
  int h[PYR_LEVELS];      // * kCameraNumUsed];

  void debugPlotIDepthMap(std::vector<FrameHessian*> frameHessians, float* minID, float* maxID,
                          std::vector<IOWrap::Output3DWrapper*>& wraps) const;

  void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*>& wraps);

  bool NeedKF();

  FrameHessian* lastRef;  //!< 参考帧
  AffLight lastRef_aff_g2l;
  // std::array<AffLight, kCameraNumUsed> a_lastRef_aff_g2l;
  FrameHessian* newFrame;  //!< 新来的一帧
  int refFrameID;          //!< 参考帧id

  // act as pure ouptut
  Vec10 lastResiduals;
  Vec10 lastResidualNum;
  Vec10 lastSaturatedRatio;
  std::array<VecTrack, PYR_LEVELS> lastRS;
  Vec3 lastFlowIndicators;  //!< 光流指示用, 只有平移和, 旋转+平移的像素移动
  double firstCoarseRMSE;
  float firstCoarseResNum;
  float firstSaturatedRatio;
  SE3 thisToNext;
  Mat33 dRwb;

 private:
  void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians);

  float* idepth[PYR_LEVELS];
  float* weightSums[PYR_LEVELS];
  float* weightSums_bak[PYR_LEVELS];

  Vec6 calcResAndGS(int lvl, MatState& H_out, VecState& b_out, const SE3& refToNew, AffLight aff_g2l, float cutoffTH);

  VecTrack calcRes(const bool& disable_kf, const int& iter, const std::vector<FrameHessian*>& frameHessians,
                   int all_keyframe_size, bool is_imu_ready, int lvl_target_, FrameHessian* lastRef, int lvl,
                   const SE3& refToNew_, AffLight aff_g2l, float cutoffTH, bool show_image = false);

  void calcGSSSE(bool fix_ab_, bool is_imu_ready, int lvl_target_, int lvl, MatState& H_out, VecState& b_out,
                 const SE3& refToNew, AffLight aff_g2l, int& N, MultiCamera* p_multi_camera);

  void calcGS(int lvl, MatState& H_out, VecState& b_out, const SE3& refToNew, AffLight aff_g2l);

  // pc buffers
  float* pc_u[PYR_LEVELS];               //!< 每层上的有逆深度点的坐标x
  float* pc_v[PYR_LEVELS];               //!< 每层上的有逆深度点的坐标y
  float* pc_idepth[PYR_LEVELS];          //!< 每层上点的逆深度
  float* pc_color[PYR_LEVELS];           //!< 每层上点的颜色值
  int pc_n[PYR_LEVELS][kCameraNumUsed];  //!< 每层上点的个数
  // int image_info_offset[PYR_LEVELS][kCameraNumUsed];

  // warped buffers
  float* buf_warped_idepth[kCameraNumUsed * kCameraNumUsed];    //[kCameraNumUsed * kCameraNumUsed];
                                                                ////!< 投影得到的点的逆深度
  float* buf_warped_u[kCameraNumUsed * kCameraNumUsed];         //[kCameraNumUsed * kCameraNumUsed]; //!<
                                                                //投影得到的归一化坐标
  float* buf_warped_v[kCameraNumUsed * kCameraNumUsed];         //[kCameraNumUsed * kCameraNumUsed]; //!<
                                                                //同上
  float* buf_warped_dx[kCameraNumUsed * kCameraNumUsed];        //[kCameraNumUsed * kCameraNumUsed];
                                                                ////!< 投影点的图像梯度
  float* buf_warped_dy[kCameraNumUsed * kCameraNumUsed];        //[kCameraNumUsed * kCameraNumUsed];
                                                                ////!< 投影点的图像梯度
  float* buf_warped_residual[kCameraNumUsed * kCameraNumUsed];  //[kCameraNumUsed * kCameraNumUsed];
                                                                ////!< 投影得到的残差
  float* buf_warped_weight[kCameraNumUsed * kCameraNumUsed];    //[kCameraNumUsed * kCameraNumUsed];
                                                                ////!< 投影的huber函数权重
  float* buf_warped_refColor[kCameraNumUsed * kCameraNumUsed];  //[kCameraNumUsed * kCameraNumUsed];
                                                                ////!< 投影点参考帧上的灰度值
  int buf_warped_n[kCameraNumUsed * kCameraNumUsed];            //!< 投影点的个数

  std::vector<float*> ptrToDelete;  //!< 所有的申请的内存指针, 用于析构删除
  Accumulator9 acc;

  dmvio::IMUIntegration& imuIntegration;
};

class CoarseDistanceMap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CoarseDistanceMap(int w, int h);

  ~CoarseDistanceMap();

  void makeDistanceMap(std::vector<FrameHessian*> frameHessians, FrameHessian* frame, const int& target_cid);

  void makeInlierVotes(std::vector<FrameHessian*> frameHessians);

  void makeK(CalibHessian* HCalib);

  float* fwdWarpedIDDistFinal;  //!< 距离场的数值

  Mat33f K[PYR_LEVELS];   // * kCameraNumUsed];
  Mat33f Ki[PYR_LEVELS];  // * kCameraNumUsed];
  float fx[PYR_LEVELS];   // * kCameraNumUsed];
  float fy[PYR_LEVELS];   // * kCameraNumUsed];
  float fxi[PYR_LEVELS];  // * kCameraNumUsed];
  float fyi[PYR_LEVELS];  // * kCameraNumUsed];
  float cx[PYR_LEVELS];   // * kCameraNumUsed];
  float cy[PYR_LEVELS];   // * kCameraNumUsed];
  float cxi[PYR_LEVELS];  // * kCameraNumUsed];
  float cyi[PYR_LEVELS];  // * kCameraNumUsed];
  int w[PYR_LEVELS];      // * kCameraNumUsed];
  int h[PYR_LEVELS];      // * kCameraNumUsed];

  void addIntoDistFinal(int u, int v, const int& target_cid);

 private:
  PointFrameResidual** coarseProjectionGrid;
  int* coarseProjectionGridNum;
  Eigen::Vector2i* bfsList1[kCameraNumUsed];  //!< 投影到frame的坐标
  Eigen::Vector2i* bfsList2[kCameraNumUsed];  //!< 和1轮换使用

  void growDistBFS(int bfsNum, const int& target_cid);
};

}  // namespace dso
