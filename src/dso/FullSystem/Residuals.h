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

#include "util/globalCalib.h"
#include "vector"

#include "OptimizationBackend/RawResidualJacobian.h"
#include "util/NumType.h"
#include "util/globalFuncs.h"
#include <fstream>
#include <iostream>

namespace dso {
class PointHessian;

class FrameHessian;

class CalibHessian;

class EFResidual;

enum ResLocation { ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE };
enum ResState { IN = 0, OOB, OUTLIER };

struct FullJacRowT {
  Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

class PointFrameResidual {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EFResidual *efResidual;

  int host_cid;

  // int target_cid;

  static int instanceCounter;

  std::array<ResState, kCameraNumUsed> state_state; //!< 上一次的残差状态
  std::array<double, kCameraNumUsed> state_energy;  //!< 上一次的能量值
  std::array<ResState, kCameraNumUsed> state_NewState; //!< 新的一次计算的状态
  std::array<double, kCameraNumUsed>
      state_NewEnergy; //!< 新的能量, 如果大于阈值则把等于阈值
  std::array<double, kCameraNumUsed>
      state_NewEnergyWithOutlier; //!< 可能具有外点的能量, 可能大于阈值
  std::array<double, kCameraNumUsed> state_zncc_angle;            //!<
  std::array<double, kCameraNumUsed> state_hw;                    //!<
  std::array<Vec2f, kCameraNumUsed> state_residual_residual_gray; //!<
  void setState(ResState s, int cid) {
    //      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    state_state[cid] = s;
    //      }
  }

  PointHessian *point;                    //!< 点
  FrameHessian *host;                     //!< 主帧
  FrameHessian *target;                   //!< 目标帧
  RawResidualJacobian *J[kCameraNumUsed]; //!< 残差对变量的各种雅克比

  std::array<bool, kCameraNumUsed> isNew;

  //    Eigen::Vector2f projectedTo[MAX_RES_PER_POINT * kCameraNumUsed]; //!<
  //    各个patch的投影坐标 std::array<Vec3f, kCameraNumUsed> centerProjectedTo;
  //    //!< patch的中心点投影 [像素x, 像素y, 新帧逆深度]
  std::array<Eigen::Vector2f, kCameraNumUsed>
      projectedTo[MAX_RES_PER_POINT]; //!< 各个patch的投影坐标
  std::array<Vec3f, kCameraNumUsed>
      centerProjectedTo; //!< patch的中心点投影 [像素x, 像素y, 新帧逆深度]
                         //!< 用来初始化新点的逆深度

  ~PointFrameResidual();

  PointFrameResidual();

  //  PointFrameResidual(PointHessian *point_, FrameHessian *host_,
  //                     FrameHessian *target_);

  PointFrameResidual(PointHessian *point_, FrameHessian *host_,
                     FrameHessian *target_, const int &host_cid_/*,
                     const int &target_cid_*/);

  Vec6f linearize(CalibHessian *HCalib, int target_cid_now,
                  double *p_other_residual = nullptr);

  void resetOOB(int cid) {
    //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    state_NewEnergy[cid] = state_energy[cid] = 0;
    state_NewState[cid] = ResState::OUTLIER;
    setState(ResState::IN, cid);
    //    }

    // setState(ResState::IN);
  };

  void applyRes(bool copyJacobians, int cid);

  void debugPlot(int cid);

  void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M,
                 int res);
};
} // namespace dso
