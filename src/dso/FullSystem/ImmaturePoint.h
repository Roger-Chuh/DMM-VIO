/**
 * This file is part of DSO.
 *
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

#include "util/NumType.h"

#include "FullSystem/HessianBlocks.h"

namespace dso {

// ImmaturePoint的状态, 残差和目标帧
struct ImmaturePointTemporaryResidual {
 public:
  ResState state_state;     //!< 逆深度残差的状态
  double state_energy;      //!< 残差值
  ResState state_NewState;  //!< 新计算的逆深度残差的状态
  double state_NewEnergy;   //!< 新计算的残差值
  FrameHessian* target;
  float hw_use;
};

enum ImmaturePointStatus {
  IPS_GOOD = 0,  // traced well and good
  // 搜索区间超出图像, 尺度变化太大, 两次残差都大于阈值, 不再搜索
  IPS_OOB,  // OOB: end tracking & marginalize!
  // 第一次残差大于阈值, 外点
  IPS_OUTLIER,  // energy too high: if happens again: outlier!
  // 搜索区间太短，但是没激活
  IPS_SKIPPED,  // traced well and good (but not actually traced).
  // 梯度和极线夹角太大
  IPS_BADCONDITION,  // not traced because of bad condition.
  IPS_UNINITIALIZED
};  // not even traced once.

class ImmaturePoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  float setting_outlierTH_trace_on = setting_outlierTH_epi_trace_on;    // 9; // setting_huberTH_loose;
  float setting_outlierTH_linearize = setting_outlierTH_epi_linearize;  // 9;    // setting_huberTH_loose;

  float setting_huberTH_trace_on = setting_huberTH_epi_trace_on;    // 9; // setting_huberTH_loose;
  float setting_huberTH_linearize = setting_huberTH_epi_linearize;  // 9;    // setting_huberTH_loose;

  float setting_energyTH_trace_on = setting_energyTH_epi_trace_on;    // 9; // setting_huberTH_loose;
  float setting_energyTH_linearize = setting_energyTH_epi_linearize;  // 9;    // setting_huberTH_loose;

  // static values
  float color[MAX_RES_PER_POINT_SEED];               //!< 原图上pattern上对应的像素值
  float distance_transform[MAX_RES_PER_POINT_SEED];  //!< 原图上pattern上对应的像素值
  float weights[MAX_RES_PER_POINT_SEED];             //!< 原图上pattern对应的权重(与梯度成反比)
  float weights_gray[MAX_RES_PER_POINT_SEED];        //!< 原图上pattern对应的权重(与梯度成反比)
  float color_converged[MAX_RES_PER_POINT];          //!< 原图上pattern上对应的像素值
  float weights_converged[MAX_RES_PER_POINT];        //!< 原图上pattern对应的权重(与梯度成反比)
  float weights_converged_gray[MAX_RES_PER_POINT];
  Mat22f gradH, gradH_converged;  //!< 图像梯度hessian矩阵
  Vec2f gradH_ev;
  Mat22f gradH_eig;
  float energyTH;
  float energyTH_converged;
  float u, v;  //!< host里的像素坐标
  FrameHessian* host;
  int idxInImmaturePoints;

  std::array<float, kCameraNumUsed> quality;  //!< 第二误差/第一误差 作为搜索质量, 越大越好

  float my_type;

  float idepth_min;  //!< 逆深度范围
  float idepth_max;

  float idp = -1.f;  //!< 逆深度范围

  int host_cid;
  int host_level;
  float zncc_opt = 0;
  //  ImmaturePoint(int u_, int v_, FrameHessian *host_, float type,
  //                CalibHessian *HCalib);
  ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib, const int& host_cid_,
                const int& host_level_);

  ~ImmaturePoint();

  ImmaturePointStatus traceOn(const int& target_cid, FrameHessian* frame, const Mat33f& hostToFrame_KRKi,
                              const Vec3f& hostToFrame_Kt, const Vec2f& hostToFrame_affine, CalibHessian* HCalib,
                              bool debugPrint = false, int lvl = 0, bool is_first_frame = false,
                              bool show_image = false);

  std::array<ImmaturePointStatus, kCameraNumUsed> lastTraceStatus;  //!< 上一次跟踪状态
  std::array<Vec2f, kCameraNumUsed> lastTraceUV;                    //!< 上一次搜索得到的位置
  std::array<float, kCameraNumUsed> lastTracePixelInterval;         //!< 上一次的搜索范围长度
  std::array<float, kCameraNumUsed> hw_use;
  float idepth_GT;

  double linearizeResidual(const int& target_cid, CalibHessian* HCalib, const float outlierTHSlack,
                           ImmaturePointTemporaryResidual* tmpRes, float& Hdd, float& bd, float idepth,
                           int lvl_target = 0, bool print_info = false);

  float getdPixdd(CalibHessian* HCalib, ImmaturePointTemporaryResidual* tmpRes, float idepth);

  float calcResidual(CalibHessian* HCalib, const float outlierTHSlack, ImmaturePointTemporaryResidual* tmpRes,
                     float idepth);
  float CalcZncc(const Eigen::MatrixXf& host_, const Eigen::MatrixXf& target_);

 private:
};

}  // namespace dso
