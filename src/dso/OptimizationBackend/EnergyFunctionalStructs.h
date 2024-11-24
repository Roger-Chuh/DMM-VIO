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

#include "OptimizationBackend/RawResidualJacobian.h"
#include "util/NumType.h"
#include "vector"
#include <math.h>

namespace dso {

class PointFrameResidual;

class CalibHessian;

class FrameHessian;

class PointHessian;

class EFResidual;

class EFPoint;

class EFFrame;

class EnergyFunctional;

class EFResidual {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  //  inline EFResidual(PointFrameResidual *org, EFPoint *point_, EFFrame
  //  *host_,
  //                    EFFrame *target_)
  //      : data(org), point(point_), host(host_), target(target_) {
  //    isLinearized = false;
  //    isActiveAndIsGoodNEW = false;
  //    J = new RawResidualJacobian();
  //    assert(((long)this) % 16 == 0);
  //    assert(((long)J) % 16 == 0);
  //  }
  inline EFResidual(PointFrameResidual *org, EFPoint *point_, EFFrame *host_,
                    EFFrame *target_, int host_cid_, int target_cid_,
                    MultiCamera *p_multi_camera_)
      : data(org), point(point_), host(host_), target(target_),
        host_cid(host_cid_), target_cid(target_cid_),
        p_multi_camera(p_multi_camera_) {
    isLinearized = false;
    isActiveAndIsGoodNEW = false;
    J = new RawResidualJacobian(host_cid, target_cid);
    assert(((long)this) % 16 == 0);
    assert(((long)J) % 16 == 0);
  }

  inline ~EFResidual() { delete J; }

  void takeDataF();

  void fixLinearizationF(EnergyFunctional *ef);

  MultiCamera *p_multi_camera;
  // structural pointers
  PointFrameResidual *data;
  int hostIDX, targetIDX; //!< 残差对应的 host 和 Target ID号
  EFPoint *point;         //!< 残差点
  EFFrame *host;          //!< 主
  EFFrame *target;        //!< 目标
  int idxInAll;           //!< 所有残差中的id

  RawResidualJacobian *J; //!< 用来计算jacob, res值

  VecNRf res_toZeroF; //!< 更新delta后的线性残差
  VecStatef JpJdF;    //!< 逆深度Jaco和位姿+光度Jaco的Hessian

  int host_cid, target_cid;
  // status.
  bool isLinearized; //!< 计算完成res_toZeroF

  // if residual is not OOB & not OUTLIER & should be used during accumulations
  bool isActiveAndIsGoodNEW; //!< 激活的还可以参与优化
  inline const bool &isActive() const {
    return isActiveAndIsGoodNEW;
  } //!< 是不是激活的取决于残差状态
};

enum EFPointStatus { PS_GOOD = 0, PS_MARGINALIZE, PS_DROP };

class EFPoint {
  // todo roger,
  // 存放属于同一个pid的所有残差，同一个fid的不同cid都放在residualsAll里
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EFPoint(PointHessian *d, EFFrame *host_, const int &host_cid_,
          const int &target_cid_ = -1)
      : data(d), host(host_), host_cid(host_cid_), target_cid(target_cid_) {
    takeData();
    stateFlag = EFPointStatus::PS_GOOD;
  }

  void takeData();

  PointHessian *data; //!< PointHessian数据

  float priorF; //!< 逆深度先验信息矩阵, 初始化之后的有
  float deltaF; //!< 当前逆深度和线性化处的差, 没有使用FEJ, 就是0

  // constant info (never changes in-between).
  int idxInPoints; //!< 当前点在EFFrame中id
  EFFrame *host;

  // contains all residuals.
  // todo roger,
  // 存放属于同一个pid的所有残差，同一个fid的不同cid都放在residualsAll里
  std::vector<EFResidual *> residualsAll; //!< 该点的所有残差

  int host_cid;
  int target_cid;

  float bdSumF;    //!< 当前残差 + 边缘化先验残差
  float HdiF;      //!< 逆深度hessian的逆, 协方差
  float Hdd_accLF; //!< 边缘化, 逆深度的hessian
  VecCf Hcd_accLF; //!< 边缘化, 逆深度和内参的hessian
  float bd_accLF;  //!< 边缘化, J逆深度*残差
  float Hdd_accAF; //!< 正常逆深度的hessian
  VecCf Hcd_accAF; //!< 正常逆深度和内参的hessian
  float bd_accAF;  //!< 正常 J逆深度*残差

  EFPointStatus stateFlag; //!< 点的状态
};

class EFFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EFFrame(FrameHessian *d) : data(d) { takeData(); }
  //  EFFrame(std::array<FrameHessian *, kCameraNumUsed> a_d) : a_data(a_d) {
  //    takeData();
  //  }

  void takeData();
  // TODO 存的跟先验有关的东西，没什么有用的干货
  //! 位姿 0-5, 光度ab 6-7

  VecState prior; //!< 位姿只有第一帧有先验 // prior hessian (diagonal)
  VecState delta_prior; //!< 相对于先验的增量	// = state-state_prior (E_prior
                        //!< = (delta_prior)' * diag(prior) * (delta_prior)
  VecState delta; //!< 相对于线性化点位姿, 光度的增量	// state - state_zero.
  // todo roger host在同一个fid下的所有fidde点都存在points里
  std::vector<EFPoint *> points; //!< 帧上所有点
  FrameHessian *data;            //!< 对应FrameHessian数据
  std::array<FrameHessian *, kCameraNumUsed> a_data; //!< 对应FrameHessian数据
  //? 和FrameHessian中的idx有啥不同
  int idx; //!< 在能量函数中帧id // idx in frames.

  int cid;

  int frameID; //!< 所有历史帧ID
};

} // namespace dso
