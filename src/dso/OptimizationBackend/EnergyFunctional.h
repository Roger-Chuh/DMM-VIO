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

#include "map"
#include "util/IndexThreadReduce.h"
#include "util/NumType.h"
#include "vector"
#include <math.h>

namespace dmvio {
class BAGTSAMIntegration;
}

namespace dso {

class PointFrameResidual;

class CalibHessian;

class FrameHessian;

class PointHessian;

class EFResidual;

class EFPoint;

class EFFrame;

class EnergyFunctional;

class AccumulatedTopHessian;

class AccumulatedTopHessianSSE;

class AccumulatedSCHessian;

class AccumulatedSCHessianSSE;

extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;

class EnergyFunctional {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  friend class EFFrame;

  friend class EFPoint;

  friend class EFResidual;

  friend class AccumulatedTopHessian;

  friend class AccumulatedTopHessianSSE;

  friend class AccumulatedSCHessian;

  friend class AccumulatedSCHessianSSE;

  EnergyFunctional(dmvio::BAGTSAMIntegration &gtsamIntegration);

  ~EnergyFunctional();

  EFResidual *insertResidual(PointFrameResidual *r, MultiCamera *p_multi_camera,
                             bool add_connection = true);

  EFFrame *insertFrame(FrameHessian *fh, CalibHessian *Hcalib);

  EFPoint *insertPoint(PointHessian *ph);

  void dropResidual(EFResidual *r, bool delete_connection = true);

  void marginalizeFrame(EFFrame *fh);

  void removePoint(EFPoint *ph);

  void marginalizePointsF();

  void dropPointsF();

  void solveSystemF(int iteration, double lambda, CalibHessian *HCalib);

  double calcMEnergyF(bool useNewValues);

  double calcLEnergyF_MT();

  void makeIDX();

  void setDeltaF(CalibHessian *HCalib);

  void setAdjointsF(CalibHessian *Hcalib);

  std::vector<EFFrame *> frames; //!< 能量函数中的帧
  int nPoints, nFrames, nResiduals; //!< EFPoint的数目, EFframe关键帧数, 残差数

  // HMForGTSAM, bMForGTSAM only contain marginalized points until the next time
  // a keyframe is marginalized. With each keyframe marginalization the
  // information in them is transferred to the GTSAMIntegration.
  MatXX HM, HMForGTSAM; //!< 优化的Hessian矩阵, 边缘化掉逆深度
  VecX bM, bMForGTSAM;  //!< 优化的Jr项, 边缘化掉逆深度

  int resInA, resInL, resInM; //!< 分别是在计算A, L, 边缘化H和b中残差的数量
  MatXX lastHS;
  VecX lastbS;
  VecX lastX;
  std::vector<VecX> lastNullspaces_forLogging;
  std::vector<VecX> lastNullspaces_pose;
  std::vector<VecX> lastNullspaces_scale;
  std::vector<VecX> lastNullspaces_affA;
  std::vector<VecX> lastNullspaces_affB;

  IndexThreadReduce<Vec10> *red;

  std::map<uint64_t, // 历史ID
           Eigen::Vector2i, std::less<uint64_t>,
           Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>>
      connectivityMap; //!< 关键帧之间的连接关系, first: 前32表示host ID,
                       //!< 后32位表示target ID; second:数目 [0] 普通的, [1]
                       //!< 边缘化的

private:
  VecX getStitchedDeltaF() const;

  void resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT);

  void resubstituteFPt(const VecCf &xc, Mat1Statef *xAd, int min, int max,
                       Vec10 *stats, int tid);

  void accumulateAF_MT(MatXX &H, VecX &b, bool MT);

  void accumulateLF_MT(MatXX &H, VecX &b, bool MT);

  void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

  void calcLEnergyPt(int min, int max, Vec10 *stats, int tid);

  void orthogonalize(VecX *b, MatXX *H);

  Mat1Statef *adHTdeltaF; //!< host和target之间位姿的增量, 一共[帧数×帧数]个

  MatState *adHost; //!< 伴随矩阵, double
  MatState *adTarget;

  MatStatef *adHostF; //!< 伴随矩阵, float
  MatStatef *adTargetF;

  VecC cPrior;   //!< setting_initialCalibHessian 信息矩阵
  VecCf cDeltaF; //!< 相机内参增量
  VecCf cPriorF;

  AccumulatedTopHessianSSE *accSSE_top_L; //!<
  AccumulatedTopHessianSSE *accSSE_top_A; //!<

  AccumulatedSCHessianSSE *accSSE_bot;

  std::vector<EFPoint *> allPoints;
  std::vector<EFPoint *> allPointsToMarg;

  float currentLambda;

  dmvio::BAGTSAMIntegration &gtsamIntegration;
};
} // namespace dso
