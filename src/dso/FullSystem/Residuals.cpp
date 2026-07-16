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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

//#define USE_INVERSE_COMPOSITIONAL

namespace dso {
int PointFrameResidual::instanceCounter = 0;

long runningResID = 0;

PointFrameResidual::PointFrameResidual() {
  assert(false);
  instanceCounter++;
}

PointFrameResidual::~PointFrameResidual() {
  assert(efResidual == 0);
  instanceCounter--;
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    delete[] J[cid];
  }
  // delete[] J;
}

// PointFrameResidual::PointFrameResidual(PointHessian *point_,
//                                       FrameHessian *host_,
//                                       FrameHessian *target_)
//    : point(point_), host(host_), target(target_) {
//  efResidual = 0;
//  instanceCounter++;
//  resetOOB();
//  // TODO 这时J只是开辟了空间，还没有赋值
//  J = new RawResidualJacobian(); // 各种雅克比
//  assert(((long)J) % 16 == 0);   // 16位对齐
//
//  isNew = true;
//}
PointFrameResidual::PointFrameResidual(PointHessian *point_,
                                       FrameHessian *host_,
                                       FrameHessian *target_,
                                       const int &host_cid_/*,
                                       const int &target_cid_*/)
    : point(point_), host(host_), target(target_), host_cid(host_cid_)/*,
      target_cid(target_cid_)*/ {
  efResidual = 0;
  instanceCounter++;
  // resetOOB();
  // TODO 这时J只是开辟了空间，还没有赋值
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    resetOOB(cid);
    J[cid] =
        new RawResidualJacobian(host_cid, cid /*target_cid*/); // 各种雅克比
    isNew[cid] = true;
    assert(((long)(J[cid])) % 16 == 0); // 16位对齐
  }
  // assert(((long)J) % 16 == 0); // 16位对齐
}

//@ 求对各个参数的导数, 和能量值
//#define SHOW_IMAGE
Vec6f PointFrameResidual::linearize(CalibHessian *HCalib, int target_cid_now,
                                    double *p_other_residual) {
  Vec6f ret = Vec6f::Zero();
  // printf("fx fy cx cy: [%f %f %f %f]\n", HCalib->fxl(), HCalib->fyl(),
  // HCalib->cxl(), HCalib->cyl());
  J[target_cid_now]->ResetValues();
  state_NewEnergyWithOutlier[target_cid_now] = -1;
  state_zncc_angle[target_cid_now] = NAN;
  state_hw[target_cid_now] = NAN;
  state_residual_residual_gray[target_cid_now] = Vec2f::Constant(NAN);

  if (state_state[target_cid_now] == ResState::OOB) {
    // printf("oob\n");
    state_NewState[target_cid_now] = ResState::OOB;
    ret[0] = state_energy[target_cid_now];
    return ret;
  }
  // TODO 同一个host有多个target，合理
  FrameFramePrecalc *precalc = &(
      host->targetPrecalc[target
                              ->idx]); // 得到这个目标帧在主帧上的一些预计算参数
                                       //                              host_cid;
  //                              target_cid_now;
  //  std::cout << "T_th:\n" << precalc->PRE_RTll << std::endl;
  //  std::cout << "t_th:\n" << precalc->PRE_tTll << std::endl;
  float energyLeft = 0;
  float energyLeft_gray = 0;
  const Eigen::Vector3f *dIl_gray = target->dI + wG[0] * hG[0] * target_cid_now;
  const Eigen::Vector3f *host_dIl_gray = host->dI + wG[0] * hG[0] * host_cid;
#ifndef USE_EDGE_ALIGN
  const Eigen::Vector3f *dIl = target->dI + wG[0] * hG[0] * target_cid_now;
  const Eigen::Vector3f *host_dIl = host->dI + wG[0] * hG[0] * host_cid;
#else
  const Eigen::Vector3f *dIl =
      target->dt_dx_dy_0 + wG[0] * hG[0] * target_cid_now;
  const Eigen::Vector3f *host_dIl = host->dt_dx_dy_0 + wG[0] * hG[0] * host_cid;
#endif
  bool show_image = host_cid == 2 && target_cid_now == 3;
#ifdef SHOW_IMAGE
  MinimalImageB3 *img_host;
  MinimalImageB3 *img_target;
  if (show_image) {
    img_host = new MinimalImageB3(wG[0], hG[0]);
    img_target = new MinimalImageB3(wG[0], hG[0]);
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = host_dIl_gray[i][0];
      if (colL < 0)
        colL = 0;
      if (colL > 255)
        colL = 255;
      img_host->at(i, host_cid) = Vec3b(colL, colL, colL);
      colL = dIl_gray[i][0];
      if (colL < 0)
        colL = 0;
      if (colL > 255)
        colL = 255;
      img_target->at(i, target_cid_now) = Vec3b(colL, colL, colL);
    }
    img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                        host_cid);
  }
#endif
  // const float* const Il = target->I;
  // const Mat33f &PRE_KRKiTll_orig = precalc->PRE_KRKiTll; // todo relative
  // pose after optimize
#if 0
  Mat33f KK = Mat33f::Zero();
  KK(0, 0) = HCalib->fxl();
  KK(1, 1) = HCalib->fyl();
  KK(0, 2) = HCalib->cxl();
  KK(1, 2) = HCalib->cyl();
  KK(2, 2) = 1;
  Mat44f T10 = Mat44f::Identity();
  T10.topLeftCorner<3, 3>() = KK.inverse() * precalc->PRE_KRKiTll * KK;
  T10.topRightCorner<3, 1>() = KK.inverse() * precalc->PRE_KtTll;

  Mat44f T10_ = (host->p_multi_camera->cid_to_T01[target_cid_now].inverse() * T10.cast<double>() * host->p_multi_camera->cid_to_T01[host_cid]).cast<float>();

  const Mat33f &PRE_KRKiTll = KK * T10_.topLeftCorner<3, 3>() * KK.inverse();
  // const Vec3f &PRE_KtTll = precalc->PRE_KtTll; //
  const Vec3f &PRE_KtTll = KK * T10_.topRightCorner<3, 1>();
  //TODO roger, adj jac
#else
  const Mat33f &PRE_KRKiTll =
      precalc
          ->a_PRE_KRKiTll[host_cid * kCameraNumUsed +
                          target_cid_now]; // todo relative pose after optimize
  const Vec3f &PRE_KtTll =
      precalc->a_PRE_KtTll[host_cid * kCameraNumUsed + target_cid_now]; //
  // TODO roger, adj jac
  const Mat66 &extra_pose_jac =
      host->p_multi_camera->cid_to_T01_inv_Adj[target_cid_now];
#endif
#if 0
  const Mat33f &PRE_RTll_0 =
      precalc->PRE_RTll_0; // todo relative pose before optimize
  const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
#else
  const Mat33f &PRE_RTll_0 =
      precalc
          ->a_PRE_RTll_0[host_cid * kCameraNumUsed +
                         target_cid_now]; // todo relative pose before optimize
  const Vec3f &PRE_tTll_0 =
      precalc->a_PRE_tTll_0[host_cid * kCameraNumUsed + target_cid_now];
#endif
  const float *const color = point->color; // host帧上颜色
  const float *const weights = point->weights;
  const float *const weights_gray = point->weights_gray;

  Vec2f affLL = precalc->PRE_aff_mode; // 待优化的a和b, 就是host和target合的
  float b0 = precalc->PRE_b0_mode; // 主帧的单独 b

  //! x=0时候求几何的导数, 使用FEJ!! ,逆深度没有使用FEJ
  Vec6f d_xi_x, d_xi_y;
  Vec4f d_C_x, d_C_y;
  float d_d_x, d_d_y;
  Eigen::Matrix<float, 2, 6> d_uv_d_pose, d_uv_d_pose_inverse_comp,
      d_uv_d_pose_fwd_jac;
  Eigen::Matrix<float, 2, 3> d_uv_d_pt3d, d_uv_host_d_n_host,
      d_uv_target_d_x_target_scaled;
  Eigen::Matrix<float, 3, 6> d_pt3d_d_pose, d_pt3d_d_pose_inverse_comp;
  {
    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3f KliP;
    /// PRE_RTll_0 means FEJ
    if (!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,
                      HCalib, PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv,
                      KliP, new_idepth)) {
      state_NewState[target_cid_now] = ResState::OOB;
      // printf("oob\n");
#ifdef SHOW_IMAGE
      if (show_image) {
        delete img_host;
        delete img_target;
      }
#endif
      ret[0] = state_energy[target_cid_now];
      return ret;
    } // 投影不在图像里, 则返回OOB

    centerProjectedTo[target_cid_now] = Vec3f(Ku, Kv, new_idepth);

    Vec3f n_host = Vec3f((point->u - HCalib->cxl()) * HCalib->fxli(),
                         (point->v - HCalib->cyl()) * HCalib->fyli(), 1);

    Vec3f X_target_scaled =
        PRE_RTll_0 * n_host + PRE_tTll_0 * point->idepth_zero_scaled;

    Mat33f X_target_scaled_skew;
    X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
        X_target_scaled(2), (0), -X_target_scaled(0), -X_target_scaled(1),
        X_target_scaled(0), (0);

    d_uv_d_pt3d << HCalib->fxl() / X_target_scaled(2), 0,
        -HCalib->fxl() * X_target_scaled(0) / X_target_scaled(2) /
            X_target_scaled(2),
        0, HCalib->fyl() / X_target_scaled(2),
        -HCalib->fyl() * X_target_scaled(1) / X_target_scaled(2) /
            X_target_scaled(2);
    d_pt3d_d_pose.leftCols(3) = point->idepth_zero_scaled * Mat33f::Identity();
    d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;

    d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;

    d_uv_host_d_n_host << HCalib->fxl(), 0, -HCalib->fxl() * n_host(0), 0,
        HCalib->fyl(), -HCalib->fyl() * n_host(1);

    // TODO inverse comp d_uv_d_pose
    d_uv_d_pose_inverse_comp =
        d_uv_host_d_n_host * PRE_RTll_0.transpose() * d_pt3d_d_pose;

    d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
    d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;

    // diff d_idepth
    // TODO 这些在初始化都写过了又写一遍 !!! 放到一起好不好, ai
    // TODO same as CoarseInitializer.cpp -> CoarseInitializer::calcResAndGS
    //* 像素点对host上逆深度求导(由于乘了SCALE_IDEPTH倍, 因此乘上)
    /// d_idepth / d_x,  d_idepth / d_y
    // TODO 带0的都是使用FEJ的？ PRE_tTll_0,
    // 并不是，带0只是表示这是优化之前的state estimation
    // drescale = 1/X_target_scaled(2);
    // drescale = rou2 / rou1
    // u v是target帧的归一化坐标
    Vec2f d_uv_d_d;
    d_uv_d_d = d_uv_host_d_n_host * PRE_RTll_0.transpose() * PRE_tTll_0;

    Vec2f d_uv_d_d_fwd_jac;
    d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled * PRE_tTll_0;

#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
    d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH *
            HCalib->fxl();
    d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH *
            HCalib->fyl();
#else
    d_d_x = d_uv_d_d_fwd_jac(0);
    d_d_y = d_uv_d_d_fwd_jac(1);
#endif
#else
    d_d_x = d_uv_d_d(0);
    d_d_y = d_uv_d_d(1);
#endif

    //* 像素点对相机内参fx fy cx cy的导数第一部分
    // diff calib
    //! [0]: 1/Pz'*Px*(R20*Px'/Pz' - R00)
    //! [1]: 1/Pz'*Py*fx/fy*(R21*Px'/Pz' - R01)
    //! [2]: 1/Pz'*(R20*Px'/Pz' - R00)
    //! [3]: 1/Pz'*fx/fy*(R21*Px'/Pz' - R01)
    d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
    d_C_x[3] = HCalib->fxl() * drescale *
               (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
    // TODO KliP: host帧归一化坐标
    d_C_x[0] = KliP[0] * d_C_x[2];
    d_C_x[1] = KliP[1] * d_C_x[3];

    //! [0]: 1/Pz'*Px*fy/fy*(R20*Py'/Pz' - R10)
    //! [1]: 1/Pz'*Py*(R21*Py'/Pz' - R11)
    //! [2]: 1/Pz'*fy/fy*(R20*Py'/Pz' - R10)
    //! [3]: 1/Pz'*(R21*Py'/Pz' - R11)
    d_C_y[2] = HCalib->fyl() * drescale *
               (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
    d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
    d_C_y[0] = KliP[0] * d_C_y[2];
    d_C_y[1] = KliP[1] * d_C_y[3];

    Eigen::Matrix<float, 2, 4> d_uv_d_C, d_uv_d_C_inverse_comp;

    //* 第二部分 同样project时候一样使用了scaled的内参
    //! [Px'/Pz'  0  1  0;
    //!  0  Py'/Pz'  0  1]
#if 1
    //   #ifndef USE_INVERSE_COMPOSITIONAL
    d_C_x[0] = (d_C_x[0] + u) * SCALE_F; // TODO d_u2_d_fx
    d_C_x[1] *= SCALE_F;
    d_C_x[2] = (d_C_x[2] + 1) * SCALE_C; // TODO d_u2_d_cx
    d_C_x[3] *= SCALE_C;

    d_C_y[0] *= SCALE_F;
    d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
    d_C_y[2] *= SCALE_C;
    d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;
#else
    d_C_x[0] = n_host(0) * SCALE_F; // TODO d_u2_d_fx
    d_C_x[1] = 0 * SCALE_F;
    d_C_x[2] = 1 * SCALE_C; // TODO d_u2_d_cx
    d_C_x[3] = 0 * SCALE_C;

    d_C_y[0] = 0 * SCALE_F;
    d_C_y[1] = n_host(1) * SCALE_F;
    d_C_y[2] = 0 * SCALE_C;
    d_C_y[3] = 1 * SCALE_C;
#endif

#ifndef USE_INVERSE_COMPOSITIONAL
    //* 像素点对位姿的导数, 位移在前!
    //! 公式见初始化那儿
    // TODO same as CoarseInitializer.cpp -> CoarseInitializer::calcResAndGS
#ifndef USE_ZNCC
    d_xi_x[0] = new_idepth * HCalib->fxl();
    d_xi_x[1] = 0;
    d_xi_x[2] = -new_idepth * u * HCalib->fxl();
    d_xi_x[3] = -u * v * HCalib->fxl();
    d_xi_x[4] = (1 + u * u) * HCalib->fxl();
    d_xi_x[5] = -v * HCalib->fxl();

    d_xi_y[0] = 0;
    d_xi_y[1] = new_idepth * HCalib->fyl();
    d_xi_y[2] = -new_idepth * v * HCalib->fyl();
    d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
    d_xi_y[4] = u * v * HCalib->fyl();
    d_xi_y[5] = u * HCalib->fyl();
#else
    d_xi_x = d_uv_d_pose_fwd_jac.row(0).transpose();
    d_xi_y = d_uv_d_pose_fwd_jac.row(1).transpose();
#endif
#else
    d_xi_x = d_uv_d_pose_inverse_comp.row(0).transpose();
    d_xi_y = d_uv_d_pose_inverse_comp.row(1).transpose();
#endif
#if 0
            Eigen::Matrix<float,2,12> show;
            show.block<1,6>(0,0) = d_xi_x.transpose();
            show.block<1,6>(1,0) = d_xi_y.transpose();
            show.rightCols(6) = show.leftCols(6) - d_uv_d_pose;
            std::cout<<"pose jac diff:\n"<<show<<std::endl;
#endif
    Mat26f d_uv_d_pose;
    d_uv_d_pose.row(0) = d_xi_x.transpose();
    d_uv_d_pose.row(1) = d_xi_y.transpose();
    d_uv_d_pose = (d_uv_d_pose.cast<double>() * extra_pose_jac).cast<float>();
    d_xi_x = d_uv_d_pose.row(0).transpose();
    d_xi_y = d_uv_d_pose.row(1).transpose();
  }

  {
    // TODO 终于找到给J赋值的地方了
    J[target_cid_now]->Jpdxi[0] = d_xi_x;
    J[target_cid_now]->Jpdxi[1] = d_xi_y;

    J[target_cid_now]->Jpdc[0] = d_C_x;
    J[target_cid_now]->Jpdc[1] = d_C_y;

    J[target_cid_now]->Jpdd[0] = d_d_x;
    J[target_cid_now]->Jpdd[1] = d_d_y;
#if 0
            Eigen::Matrix<float, 2,2> d_uv_d_c_show;
            d_uv_d_c_show.col(0) = J->Jpdd;
            d_uv_d_c_show.col(1) = d_uv_d_c_show.col(0) - d_uv_d_pt3d*PRE_tTll_0;
            std::cout<<"d_uv_d_c_show:\n"<<d_uv_d_c_show<<std::endl;
#endif
  }

  float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
  float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
  float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

  float wJI2_sum = 0;

  Eigen::MatrixXf Mat_ZNSSD_I;
  Eigen::MatrixXf J_ZNSSD_mean;
  Eigen::MatrixXf J_ZNSSD_J_I_host;
  Eigen::MatrixXf J_ZNSSD_J_I_target;
  Eigen::MatrixXf grad_new_host;
  Eigen::MatrixXf grad_new_target;
  float host_val_mean;
  float target_val_mean;
  Eigen::MatrixXf ones;
  float host_sigma, target_sigma;

  Eigen::MatrixXf host_info, target_info;
  size_t count = 0;
#ifdef USE_EDGE_ALIGN
  Vec2i *edge_label_image_start =
      target->edge_label_image[0] + wG[0] * hG[0] * target_cid_now;
  Vec2i *label2xy_start = target->label2xy[0] + wG[0] * hG[0] * target_cid_now;
  Vec2i proj_check =
      (centerProjectedTo[target_cid_now].head(2) + Vec2f(0.5, 0.5)).cast<int>();
  int label = edge_label_image_start[proj_check[0] + proj_check[1] * wG[0]][1];
  Vec2f nearestPt = label2xy_start[label].cast<float>();
  Vec3f nearestHitColor =
      (getInterpolatedElement33(dIl, centerProjectedTo[target_cid_now][0],
                                centerProjectedTo[target_cid_now][1], wG[0]));
#endif
  for (int idx = 0; idx < patternNum; idx++) {
    float Ku, Kv;
    //? 为啥这里使用idepth_scaled, 上面使用的是zero； 答：
    //其实和上面一样的....同时调用了setIdepth() setIdepthZero()
    //! 答: 这里是求图像导数, 由于线性误差大, 就不使用FEJ, 所以使用当前的状态
    // TODO  这里求残差用的是最新状态重投影，而不是fej状态重投影
    if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1],
                      point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv)) {
      continue;
    }

#ifdef SHOW_IMAGE
    if (show_image) {
      img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(1),
                            target_cid_now);
    }
#endif

    Vec3f hitColor = (getInterpolatedElement33(dIl_gray, Ku, Kv, wG[0]));
    // float residual = hitColor[0] - (float) (affLL[0] * color[idx] +
    // affLL[1]);
    Vec3f hostColor =
        (getInterpolatedElement33(host_dIl_gray, point->u + patternP[idx][0],
                                  point->v + patternP[idx][1], wG[0]));
    float host_value_corrected = (float)(affLL[0] * color[idx] + affLL[1]);
#ifdef USE_EDGE_ALIGN
    Vec3f hitColor_edge = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
    if (true && ((std::abs(hitColor_edge[1]) < 0.001f &&
    std::abs(hitColor_edge[2]) < 0.001f) ||(std::abs(nearestHitColor[1]) < 0.001f &&
    std::abs(nearestHitColor[2]) < 0.001f))) {
      hitColor[0] = NAN;
    }
#endif
    if (!std::isfinite((float)hitColor[0])) {
      continue;
    }
#if 0
    hostColor[0] = host_value_corrected;
#else
    hostColor[0] = color[idx];
#endif
    host_info.conservativeResize(count + 1, 3);
    target_info.conservativeResize(count + 1, 3);
    host_info.row(count) = hostColor.transpose();
    target_info.row(count) = hitColor.transpose();
    count++;
  }

  int patch_num = host_info.rows();
  float ws2 = 1;
  float zncc = 0;
  float hw = 1;
  float hw_gray = 1;
  float angle = 0;
  if (patch_num != 0) {

    host_val_mean = host_info.col(0).sum() / patch_num;
    target_val_mean = target_info.col(0).sum() / patch_num;

    ones.conservativeResize(patch_num, 1);
    ones.setOnes();
    host_info.col(0) = host_info.col(0) - host_val_mean * ones;
    target_info.col(0) = target_info.col(0) - target_val_mean * ones;
    host_sigma = host_info.col(0).norm();
    target_sigma = target_info.col(0).norm();
    host_info.col(0) /= host_sigma;
    target_info.col(0) /= target_sigma;

    Mat_ZNSSD_I.conservativeResize(patch_num, patch_num);
    Mat_ZNSSD_I.setIdentity();

    J_ZNSSD_mean =
        Mat_ZNSSD_I - (ones / static_cast<float>(patch_num)) * ones.transpose();

    J_ZNSSD_J_I_host =
        setting_variableScale *
        ((Mat_ZNSSD_I - (host_info.col(0) * host_info.col(0).transpose())) /
         host_sigma * J_ZNSSD_mean);
    J_ZNSSD_J_I_target =
        setting_variableScale *
        ((Mat_ZNSSD_I - (target_info.col(0) * target_info.col(0).transpose())) /
         target_sigma * J_ZNSSD_mean);

    grad_new_host =
        J_ZNSSD_J_I_host * host_info.rightCols(2); // "new" gradient: 8x2
    grad_new_target =
        J_ZNSSD_J_I_target * target_info.rightCols(2); // "new" gradient: 8x2

    zncc = target_info.col(0).dot(host_info.col(0));
    if (host_sigma < 3.f || target_sigma < 3.f ||
        patch_num != MAX_RES_PER_POINT) {
      state_NewState[target_cid_now] = ResState::OOB;
      // printf("oob\n");
#ifdef SHOW_IMAGE
      if (show_image) {
        delete img_host;
        delete img_target;
      }
#endif
      ret[0] = state_energy[target_cid_now];
      return ret;
    }
    // printf("zncc: %f, target_sigma: %f\n", zncc, target_sigma);
    assert(std::abs(zncc) < 1.00001);
    float r2 = 2 - 2 * zncc;
    ws2 = 2.0 / (r2 + 2.0);
    if (true) {
      assert(std::abs(zncc) < 1.00001);
      angle = (kOur_PI - std::acos(zncc)) / kOur_PI;
      if (false) {
        std::cout << "host_info.col(0): " << host_info.col(0).transpose()
                  << ", target_info.col(0): " << target_info.col(0).transpose()
                  << std::endl;
        printf("angle: %f, zncc: %f, std::acos(zncc): %f, host_sigma: %f, "
               "target_sigma: %f\n",
               angle, zncc, std::acos(zncc), host_sigma, target_sigma);
        std::cout << "host_fid: " << host->idx
                  << ", target_fid: " << target->idx
                  << ", host_cid: " << host_cid
                  << ", target_cid: " << target_cid_now << std::endl;
      }
      angle = std::isnan(angle) ? 1 : angle;
      // assert(angle >= 0);
      assert(angle >= 0.0001);
      ws2 = angle <= 1 ? angle : 1;
      ws2 *= ws2;
    }

#ifdef USE_ZNCC_WEIGHT
    if (zncc < -111110.5 /*0.8*/) {
      state_NewState[target_cid_now] = ResState::OOB;
#ifdef SHOW_IMAGE
      if (show_image) {
        delete img_host;
        delete img_target;
      }
#endif
      ret[0] = state_energy[target_cid_now];
      return ret;
    }
#endif
    host_info.col(0) *= setting_variableScale;
    target_info.col(0) *= setting_variableScale;
  }
  //    std::cout << "lba, grad_new_host: \n" << grad_new_host << std::endl;
  //    std::cout << "lba, grad_new_target: \n" << grad_new_target << std::endl;

  int cnt = 0;
  float residual;
  float residual_gray;
  bool has_nan_res = false;
  int continued_count = 0;
#ifdef SHOW_IMAGE
  if (show_image) {
    printf("======= show image ======\n");
  }
#endif
    Vec2f res_sum = Vec2f::Zero();
    int res_count = 0;
  for (int idx = 0; idx <
#ifndef USE_CENTER_PIXEL_ONLY
  patternNum
#else
  1
#endif
  ; idx++) {
    float Ku, Kv;
    //? 为啥这里使用idepth_scaled, 上面使用的是zero； 答：
    //其实和上面一样的....同时调用了setIdepth() setIdepthZero()
    //! 答: 这里是求图像导数, 由于线性误差大, 就不使用FEJ, 所以使用当前的状态
    // TODO  这里求残差用的是最新状态重投影，而不是fej状态重投影
    if (!projectPoint(point->u + patternP[idx][0] * pattern_scale_extra_edge, point->v + patternP[idx][1] * pattern_scale_extra_edge,
                      point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv)) {
      state_NewState[target_cid_now] = ResState::OOB;
#ifdef SHOW_IMAGE
      if (show_image) {
        delete img_host;
        delete img_target;
      }
#endif
      ret[0] = state_energy[target_cid_now];
      return ret;
    }

    // 像素坐标
    projectedTo[target_cid_now][idx /*+ MAX_RES_PER_POINT * target_cid*/][0] =
        Ku;
    projectedTo[target_cid_now][idx /*+ MAX_RES_PER_POINT * target_cid*/][1] =
        Kv;
    if (idx == 0) {
        if ((Vec2f(Ku, Kv) - centerProjectedTo[target_cid_now].head(2)).norm() > 0.001) {
            // printf("center_projection differs too much, diff = %f, [%f %f], [%f %f]\n",(Vec2f(Ku, Kv) - centerProjectedTo[target_cid_now].head(2)).norm(), Ku, Kv, centerProjectedTo[target_cid_now][0], centerProjectedTo[target_cid_now][1]);
        }
    }
    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
    Vec3f hitColor_gray = (getInterpolatedElement33(dIl_gray, Ku, Kv, wG[0]));
    //* 残差对光度仿射a求导
    //! 光度参数使用固定线性化点了
    float drdA = (color[idx] - b0);
#ifdef USE_EDGE_ALIGN
    if (true &&
        ((std::abs(hitColor[1]) < 0.001f && std::abs(hitColor[2]) < 0.001f) || (std::abs(nearestHitColor[1]) < 0.001f && std::abs(nearestHitColor[2]) < 0.001f))) {
      hitColor[0] = NAN;
    } else {
    }
#endif
    if (!std::isfinite((float)hitColor[0])) {
      state_NewState[target_cid_now] = ResState::OOB;
      // printf("oob, (float)hitColor[0]: %f\n", (float)hitColor[0]);
#ifdef SHOW_IMAGE
      if (show_image) {
        delete img_host;
        delete img_target;
      }
#endif
      ret[0] = state_energy[target_cid_now];
      return ret;
    }
    residual_gray =
        hitColor_gray[0] - (float)(affLL[0] * color[idx] + affLL[1]);
    if (p_other_residual) {
      *p_other_residual = residual_gray;
    }

#ifndef USE_ZNCC
#ifndef USE_EDGE_ALIGN
    residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);
#else
    Vec3f hit_color_bak = setting_variableScale_edge * hitColor;
#if 1
    //hitColor[0] = setting_variableScale_edge * (nearestPt - Vec2f(Ku, Kv)).norm();
    //hitColor.tail(2) = nearestHitColor.tail(2);
#endif
    residual = hitColor[0];
    // printf("idx: %d, target_uv: [%f %f], res: %f\n", idx, Ku, Kv, residual);
#endif

#else
    float residual_bak =
        hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);
    if (residual_bak != residual_gray) {
      printf("residual_bak != residual_gray\n");
      std::exit(1);
    }
    /*float*/ residual = 1 * (target_info(cnt, 0) - host_info(cnt, 0));
    // printf("idx: %d, zncc_residual: %f, zncc: %f, angle: %f, is_inlier:
    // %d\n", idx, residual, zncc, angle, angle > 0.7);
    if (std::isnan(residual)) {
      continued_count++;
      //            isGood = false;
      //            break;
      if (!has_nan_res) {
        has_nan_res = true;
      }
      cnt++;
      continue;
      state_NewState[target_cid_now] = ResState::OOB;
#ifdef SHOW_IMAGE
      if (show_image) {
        delete img_host;
        delete img_target;
      }
#endif
      ret[0] = state_energy[target_cid_now];
      return ret;
    }
#endif
    Vec3f hostColor =
        (getInterpolatedElement33(host_dIl, point->u + patternP[idx][0],
                                  point->v + patternP[idx][1], wG[0]));
    Vec3f hostColor_gray =
        (getInterpolatedElement33(host_dIl_gray, point->u + patternP[idx][0],
                                  point->v + patternP[idx][1], wG[0]));

    if (pattern_scale_extra_edge != 1) {
        if (idx == 0 && hostColor_gray[0] != color[idx]) {
            printf("idx == 0 && hostColor_gray[0] != color[idx], [%f %f], pix: [%f "
                   "%f]\n", hostColor_gray[0], color[idx], point->u, point->v);
            std::exit(1);
        }
    } else {
        if (hostColor_gray[0] != color[idx]) {
            printf("idx == 0 && hostColor_gray[0] != color[idx], [%f %f], pix: [%f "
                   "%f]\n", hostColor_gray[0], color[idx], point->u, point->v);
            std::exit(1);
        }
    }
    if (std::abs(hostColor_gray[0] - color[idx]) > 0.001f) {
      printf("hostColor_gray[0] != color[idx], [%f %f], pix: [%f %f]\n",
             hostColor_gray[0], color[idx], point->u, point->v);
      std::exit(1);
    }
      res_sum += Vec2f(residual, residual_gray);
      res_count++;
    // printf("value1: %f, value check: %f\n", hostColor[0], color[idx]);
    // assert(hostColor[0] == color[idx]);
    //        //* 残差对光度仿射a求导
    //        //! 光度参数使用固定线性化点了
    //		float drdA = (color[idx]-b0);
    //		if(!std::isfinite((float)hitColor[0]))
    //		{ state_NewState = ResState::OOB; return state_energy; }
#ifdef SHOW_IMAGE
    if (show_image) {
      std::cout << "idx: " << idx << ", host_pix: " << point->u << ", "
                << point->v << ", target_pix: " << Ku << ", " << Kv
                << ", hostColor_gray(raw): " << hostColor_gray.transpose()
                << ", hitColor_gray: " << hitColor_gray.transpose()
                << ", hitColor(or dt, maybe over wrote): "
                << hitColor.transpose()
#ifdef USE_EDGE_ALIGN
                << ", hit_color_bak: " << hit_color_bak.transpose()
                << ", nearestHitColor: " << nearestHitColor.transpose()
                << ", dist_to_nearest_pt: "
                << (nearestPt - Vec2f(Ku, Kv)).norm()
#endif
                << ", affLL: " << affLL.transpose()
                << ", color_raw[idx]: " << color[idx]
                << ", gray_diff: " << hostColor_gray[0] - hitColor_gray[0]
                << std::endl;
      // IOWrap::displayImage("host", img_host);
      // IOWrap::displayImage("target", img_target);
      // IOWrap::waitKey(0);
    }
#endif
    float w_gray = sqrtf(setting_outlierTHSumComponent /
                         (setting_outlierTHSumComponent +
                          hitColor_gray.tail<2>().squaredNorm()));
#ifndef USE_ZNCC
#ifndef USE_EDGE_ALIGN
    float w = w_gray;
#else
    float w = sqrtf(
        setting_outlierTHSumComponent /
        (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
#endif

#else
    // float w = sqrtf(setting_outlierTHSumComponent /
    // (setting_outlierTHSumComponent +
    // grad_new_target.row(cnt).squaredNorm()));
    float w = sqrtf(
        setting_outlierTHSumComponent /
        (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
#endif
#ifndef USE_ZNCC_WEIGHT
    w = 0.5f * (w + weights[idx]);
    w_gray = 0.5f * (w_gray + weights_gray[idx]);
#else
    w = 0.5f * (w + weights[idx]); // std::sqrt(ws2);
    w_gray = 0.5f * (w_gray + weights_gray[idx]);
#endif

#ifndef USE_ZNCC
#if 1
    hw_gray =
        fabsf(residual_gray) < (setting_huberTH_LBA /*+ std::abs(affLL[1])*/)
            ? 1
            : (setting_huberTH_LBA /*+ std::abs(affLL[1])*/) / fabsf(residual_gray);
    hw =
        fabsf(residual) < (setting_huberTH_LBA /*+ std::abs(affLL[1])*/)
            ? 1
            : (setting_huberTH_LBA /*+ std::abs(affLL[1])*/) / fabsf(residual);
#else
#if 0 // eachErrDim == 2
        fabsf(residual_gray) < (setting_huberTH /*+ std::abs(affLL[1])*/)
            ? 1
            : (setting_huberTH /*+ std::abs(affLL[1])*/) / fabsf(residual_gray);
#else
    fabsf(residual) < (setting_huberTH /*+ std::abs(affLL[1])*/)
        ? 1
        : (setting_huberTH /*+ std::abs(affLL[1])*/) / fabsf(residual);
#endif
#ifndef USE_EDGE_ALIGN
    if (true) {
      // ws2 *= ws2;
      hw = ws2 > setting_huberTH_zncc_LBA ? 1 : ws2 / setting_huberTH_zncc_LBA;
    }
#endif
#endif
    energyLeft += w * w * hw * residual * residual * (2 - hw);

#if defined(USE_EDGE_ALIGN) && eachErrDim == 2
    energyLeft_gray += w_gray * w_gray * hw_gray * residual_gray *
                       residual_gray * (2 - hw_gray);
#endif
#else
    hw_gray =
        fabsf(residual_gray) < (setting_huberTH /*+ std::abs(affLL[1])*/)
            ? 1
            : (setting_huberTH /*+ std::abs(affLL[1])*/) / fabsf(residual_gray);
    hw =
        fabsf(residual) < (setting_huberTH_LBA + std::abs(affLL[1]))
            ? 1
            : (setting_huberTH_LBA + std::abs(affLL[1])) / fabsf(residual);
    if (true) {
      hw = ws2;
      hw = ws2 > setting_huberTH_zncc_LBA ? 1 : ws2 / setting_huberTH_zncc_LBA;
    }
    // energyLeft += w * w * hw * residual * residual * (2 - hw);
    energyLeft += hw * residual * residual * (2 - hw);
#if eachErrDim == 2
    energyLeft_gray += w_gray * w_gray * hw_gray * residual_gray *
                       residual_gray * (2 - hw_gray);
#endif
#endif

    {
      // printf("weights: %f, w: %f, hw: %f, residual: %f\n", weights[idx], w,
      // hw, residual); printf("hw: %f\n", hw);
      if (hw < 1)
        hw = sqrtf(hw);
      hw = hw * w;
      if (hw_gray < 1)
        hw_gray = sqrtf(hw_gray);
      hw_gray = hw_gray * w_gray;

      hitColor[1] *= hw;
      hitColor[2] *= hw;

      hostColor[1] *= hw;
      hostColor[2] *= hw;

      hitColor_gray[1] *= hw_gray;
      hitColor_gray[2] *= hw_gray;

      hostColor_gray[1] *= hw_gray;
      hostColor_gray[2] *= hw_gray;

      grad_new_target.row(cnt) *= hw;
      grad_new_host.row(cnt) *= hw;
      // printf("residual_gray: %f, residual: %f, host_gray: %f, target_gray: %f, [a b]: [%f %f]\n", residual_gray, residual, color[idx], hitColor_gray[0], affLL[0], affLL[1]);
      //! 残差 res*w*sqrt(hw)
      J[target_cid_now]->resF[idx] = residual * hw;
#if /*defined(USE_EDGE_ALIGN) &&*/ eachErrDim == 2
      J[target_cid_now]->resF[idx + patternNum] = residual_gray * hw_gray;
#endif

      //! 图像导数 dx dy
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
      J[target_cid_now]->JIdx[0][idx] = hitColor[1];
      J[target_cid_now]->JIdx[1][idx] = hitColor[2];
// #if defined(USE_EDGE_ALIGN) && eachErrDim == 2
//       J[target_cid_now]->JIdx[0][idx + patternNum] =
//           0; // default value is zero anyway
//       J[target_cid_now]->JIdx[1][idx + patternNum] = 0;
// #endif
#else
      J[target_cid_now]->JIdx[0][idx] = grad_new_target(cnt, 0);
      J[target_cid_now]->JIdx[1][idx] = grad_new_target(cnt, 1);
// #if eachErrDim == 2
//         J[target_cid_now]->JIdx[0][idx + patternNum] =
//           0; // default value is zero anyway
//         J[target_cid_now]->JIdx[1][idx + patternNum] = 0;
// #endif
#endif
#else
#ifndef USE_ZNCC
      J[target_cid_now]->JIdx[0][idx] = affLL[0] * hostColor[1];
      J[target_cid_now]->JIdx[1][idx] = affLL[0] * hostColor[2];
#else
      J[target_cid_now]->JIdx[0][idx] = affLL[0] * grad_new_host(cnt, 0);
      J[target_cid_now]->JIdx[1][idx] = affLL[0] * grad_new_host(cnt, 1);
#endif
#endif
#if eachErrDim == 2
      // default value is zero anyway
      J[target_cid_now]->JIdx[0][idx + patternNum] = 0;
      J[target_cid_now]->JIdx[1][idx + patternNum] = 0;
#endif
      //! 对光度合成后a b的导数 [Ii-b0  1]
      //! Ij - a*Ii - b  (a = tj*e^aj / ti*e^ai,   b = bj - a*bi) //TODO true
      //! dat Ij - [a*(Ii-b0) + b]
      // TODO bug 正负号有影响 ??? ab部分好确实差了一个负号
#if 0
#ifndef USE_INVERSE_COMPOSITIONAL
#if 0 // ndef USE_EDGE_ALIGN
      J[target_cid_now]->JabF[0][idx] = drdA * hw;
      J[target_cid_now]->JabF[1][idx] = hw;
#else
      J[target_cid_now]->JabF[0][idx + (eachErrDim - 1) * patternNum] =
          drdA * hw_gray;
      J[target_cid_now]->JabF[1][idx + (eachErrDim - 1) * patternNum] = hw_gray;
#endif
#else
      J[target_cid_now]->JabF[0][idx] = drdA * hw;
      J[target_cid_now]->JabF[1][idx] = 1 * hw;
#endif
#else
      J[target_cid_now]->JabF[0][idx + (eachErrDim - 1) * patternNum] =
          drdA * hw_gray;
      J[target_cid_now]->JabF[1][idx + (eachErrDim - 1) * patternNum] = hw_gray;
#endif
#ifndef USE_INVERSE_COMPOSITIONAL
      //! dIdx&dIdx hessian block
      // Jt * W * J = [gx; gy] * [gx gy] = [gxgx gxgy; gxgy gygy]
#ifndef USE_ZNCC
      JIdxJIdx_00 += hitColor[1] * hitColor[1];
      JIdxJIdx_11 += hitColor[2] * hitColor[2];
      JIdxJIdx_10 += hitColor[1] * hitColor[2];
      //! dIdx&dIdab hessian block
#if !defined(USE_EDGE_ALIGN)
      JabJIdx_00 += drdA * hw * hitColor[1];
      JabJIdx_01 += drdA * hw * hitColor[2];
      JabJIdx_10 += hw * hitColor[1];
      JabJIdx_11 += hw * hitColor[2];
#endif
#else
      JIdxJIdx_00 += grad_new_target(cnt, 0) * grad_new_target(cnt, 0);
      JIdxJIdx_11 += grad_new_target(cnt, 1) * grad_new_target(cnt, 1);
      JIdxJIdx_10 += grad_new_target(cnt, 0) * grad_new_target(cnt, 1);
      //! dIdx&dIdab hessian block
      // TODO 即使用了zncc，但关于ab的雅可比任然需要用梯度
#if 0 // 使用zncc时。梯度和ab没有交叉项，因为zncc残差需要用到梯度，但不需要用到ab
#if 1
      JabJIdx_00 += drdA * hw * grad_new_target(cnt, 0);
      JabJIdx_01 += drdA * hw * grad_new_target(cnt, 1);
      JabJIdx_10 += hw * grad_new_target(cnt, 0);
      JabJIdx_11 += hw * grad_new_target(cnt, 1);
#else
      JabJIdx_00 += drdA * hw * hitColor(cnt, 0);
      JabJIdx_01 += drdA * hw * hitColor(cnt, 1);
      JabJIdx_10 += hw * hitColor(cnt, 0);
      JabJIdx_11 += hw * hitColor(cnt, 1);
#endif
#endif
#endif
      //! dIdab&dIdab hessian block
      JabJab_00 += drdA * drdA * hw_gray * hw_gray;
      JabJab_01 += drdA * hw_gray * hw_gray;
      JabJab_11 += hw_gray * hw_gray;
#else
      //! dIdx&dIdx hessian block
      // Jt * W * J = [gx; gy] * [gx gy] = [gxgx gxgy; gxgy gygy]
#ifndef USE_ZNCC
      JIdxJIdx_00 += affLL[0] * affLL[0] * hostColor[1] * hostColor[1];
      JIdxJIdx_11 += affLL[0] * affLL[0] * hostColor[2] * hostColor[2];
      JIdxJIdx_10 += affLL[0] * affLL[0] * hostColor[1] * hostColor[2];
      //! dIdx&dIdab hessian block
      JabJIdx_00 += drdA * hw * affLL[0] * hostColor[1];
      JabJIdx_01 += drdA * hw * affLL[0] * hostColor[2];
      JabJIdx_10 += hw * affLL[0] * hostColor[1];
      JabJIdx_11 += hw * affLL[0] * hostColor[2];
#else
      JIdxJIdx_00 +=
          affLL[0] * affLL[0] * grad_new_host(cnt, 0) * grad_new_host(cnt, 0);
      JIdxJIdx_11 +=
          affLL[0] * affLL[0] * grad_new_host(cnt, 1) * grad_new_host(cnt, 1);
      JIdxJIdx_10 +=
          affLL[0] * affLL[0] * grad_new_host(cnt, 0) * grad_new_host(cnt, 1);
      //! dIdx&dIdab hessian block
#if 0 // 使用zncc时。梯度和ab没有交叉项，因为zncc残差需要用到梯度，但不需要用到ab
      JabJIdx_00 += drdA * hw * affLL[0] * grad_new_host(cnt, 0);
      JabJIdx_01 += drdA * hw * affLL[0] * grad_new_host(cnt, 1);
      JabJIdx_10 += hw * affLL[0] * grad_new_host(cnt, 0);
      JabJIdx_11 += hw * affLL[0] * grad_new_host(cnt, 1);
#endif
#endif
      //! dIdab&dIdab hessian block
      JabJab_00 += drdA * drdA * hw_gray * hw_gray;
      JabJab_01 += drdA * hw_gray * hw_gray;
      JabJab_11 += hw_gray * hw_gray;
#endif
#ifndef USE_ZNCC
      wJI2_sum += hw * hw *
                  (hitColor_gray[1] * hitColor_gray[1] +
                   hitColor_gray[2] * hitColor_gray[2]);
#else
      wJI2_sum += hw * hw * (grad_new_target.row(cnt).squaredNorm());
#endif
      if (setting_affineOptModeA < 0) {
#if 0 // ndef USE_EDGE_ALIGN
        J[target_cid_now]->JabF[0][idx] = 0;
#else
        J[target_cid_now]->JabF[0][idx + (eachErrDim - 1) * patternNum] = 0;
#endif
      }
      if (setting_affineOptModeB < 0) {
#if 0 // ndef USE_EDGE_ALIGN
        J[target_cid_now]->JabF[1][idx] = 0;
#else
        J[target_cid_now]->JabF[1][idx + (eachErrDim - 1) * patternNum] = 0;
#endif
      }
#if eachErrDim == 1 && (defined(USE_ZNCC) || defined(USE_EDGE_ALIGN))
      J[target_cid_now]->JabF[0][idx] = 0;
      J[target_cid_now]->JabF[1][idx] = 0;
#endif
    }
    cnt++;
  }
    if (res_count != cnt) {
        printf("res_count != cnt\n");
        std::exit(3);
    }
    res_sum /= (float)res_count;
#ifndef USE_CENTER_PIXEL_ONLY
  assert(cnt == count);
#else
  assert(cnt == 1);
#endif

  J[target_cid_now]->JIdx2(0, 0) =
      JIdxJIdx_00; // TODO gradient related 2x2, top left
  J[target_cid_now]->JIdx2(0, 1) = JIdxJIdx_10; // TODO 梯度x梯度部分的小hessian
  J[target_cid_now]->JIdx2(1, 0) = JIdxJIdx_10;
  J[target_cid_now]->JIdx2(1, 1) = JIdxJIdx_11;
#ifndef USE_ZNCC
  J[target_cid_now]->JabJIdx(0, 0) = JabJIdx_00; // TODO buttom left
  J[target_cid_now]->JabJIdx(0, 1) =
      JabJIdx_01; // TODO 光度x梯度部分的小hessian
  J[target_cid_now]->JabJIdx(1, 0) = JabJIdx_10;
  J[target_cid_now]->JabJIdx(1, 1) = JabJIdx_11;
  J[target_cid_now]->Jab2(0, 0) = JabJab_00; // TODO buttom right
  J[target_cid_now]->Jab2(0, 1) = JabJab_01; // TODO 光度x光度部分的小hessian
  J[target_cid_now]->Jab2(1, 0) = JabJab_01;
  J[target_cid_now]->Jab2(1, 1) = JabJab_11;
#if defined(USE_EDGE_ALIGN) && eachErrDim == 1
  J[target_cid_now]->Jab2.setZero();
#endif
#else
  J[target_cid_now]->JabJIdx.setZero();
#if eachErrDim == 1
  J[target_cid_now]->Jab2.setZero();
#endif
#endif

  state_NewEnergyWithOutlier[target_cid_now] = energyLeft;
  if (has_nan_res && false) {
    std::cout << "continued_count: " << continued_count << std::endl;
    std::cout << "J[target_cid_now]->JIdx2:\n"
              << J[target_cid_now]->JIdx2 << std::endl;
    std::cout << "J[target_cid_now]->JabJIdx:\n"
              << J[target_cid_now]->JabJIdx << std::endl;
    std::cout << "J[target_cid_now]->Jab2:\n"
              << J[target_cid_now]->Jab2 << std::endl;
    std::cout << "J[target_cid_now]->JabF: "
              << J[target_cid_now]->JabF[0].transpose() << std::endl;
    std::cout << "J[target_cid_now]->resF: "
              << J[target_cid_now]->resF.transpose() << std::endl;
    std::cout << "J[target_cid_now]->Jpdxi: "
              << J[target_cid_now]->Jpdxi[0].transpose() << std::endl;
    std::cout << "J[target_cid_now]->Jpdc: "
              << J[target_cid_now]->Jpdc[0].transpose() << std::endl;
    std::cout << "J[target_cid_now]->Jpdd: "
              << J[target_cid_now]->Jpdd.transpose() << std::endl;
    std::cout << "J[target_cid_now]->JIdx: "
              << J[target_cid_now]->JIdx[0].transpose() << std::endl;
    std::cout << "J[target_cid_now]->JIdy: "
              << J[target_cid_now]->JIdx[1].transpose() << std::endl;

    std::cout << "energyLeft: " << energyLeft
              << ", energyLeft_gray: " << energyLeft_gray << std::endl;
    // std::exit(31);
  }
#if eachErrDim == 1
#if defined(USE_EDGE_ALIGN) || defined(USE_ZNCC)
  if (J[target_cid_now]->JabF[0].norm() > 0 ||
      J[target_cid_now]->JabF[1].norm() > 0 ||
      J[target_cid_now]->Jab2.norm() > 0 ||
      J[target_cid_now]->JabJIdx.norm() > 0) {
    printf("J[target_cid_now]->JabF.norm() > 0||J[target_cid_now]->Jab2.norm() "
           "> 0 || J[target_cid_now]->JabJIdx.norm() > 0\n");
    std::exit(1);
  }
#endif
#else
#if defined(USE_EDGE_ALIGN) || defined(USE_ZNCC)
  if (/*J[target_cid_now]->resF.head(patternNum).norm() < 0.0001 ||*/
#ifndef USE_CENTER_PIXEL_ONLY
  J[target_cid_now]->resF.tail(patternNum).norm() < 0.0001 ||
#endif
      (J[target_cid_now]->JIdx[0].head(patternNum).norm() < 0.0001 &&
      J[target_cid_now]->JIdx[1].head(patternNum).norm() < 0.0001) ||
      J[target_cid_now]->JIdx[0].tail(patternNum).norm() > 0 ||
      J[target_cid_now]->JIdx[1].tail(patternNum).norm() > 0 ||
      J[target_cid_now]->JabF[0].tail(patternNum).norm() < 0.0001 ||
      J[target_cid_now]->JabF[1].tail(patternNum).norm() < 0.0001 ||
      J[target_cid_now]->JabF[0].head(patternNum).norm() > 0 ||
      J[target_cid_now]->JabF[1].head(patternNum).norm() > 0 ||
      J[target_cid_now]->Jab2.norm() < 0.0001 ||
      J[target_cid_now]->JabJIdx.norm() > 0) {
    printf("J[target_cid_now]->resF.head(patternNum).norm(): %f, "
           "J[target_cid_now]->resF.tail(patternNum).norm(): %f, "
           "J[target_cid_now]->JIdx[0].head(patternNum).norm(): %f, "
           "J[target_cid_now]->JIdx[1].head(patternNum).norm(): %f, "
           "J[target_cid_now]->JIdx[0].tail(patternNum).norm(): %f, "
           "J[target_cid_now]->JIdx[1].tail(patternNum).norm(): %f, "
           "J[target_cid_now]->JabF[0].tail(patternNum).norm(): %f, "
           "J[target_cid_now]->JabF[1].tail(patternNum).norm(): %f, "
           "J[target_cid_now]->JabF[0].head(patternNum).norm(): %f, "
           "J[target_cid_now]->JabF[1].head(patternNum).norm(): %f, "
           "J[target_cid_now]->Jab2.norm(): %f, "
           "J[target_cid_now]->JabJIdx.norm(): %f\n",
           J[target_cid_now]->resF.head(patternNum).norm(),
           J[target_cid_now]->resF.tail(patternNum).norm(),
           J[target_cid_now]->JIdx[0].head(patternNum).norm(),
           J[target_cid_now]->JIdx[1].head(patternNum).norm(),
           J[target_cid_now]->JIdx[0].tail(patternNum).norm(),
           J[target_cid_now]->JIdx[1].tail(patternNum).norm(),
           J[target_cid_now]->JabF[0].tail(patternNum).norm(),
           J[target_cid_now]->JabF[1].tail(patternNum).norm(),
           J[target_cid_now]->JabF[0].head(patternNum).norm(),
           J[target_cid_now]->JabF[1].head(patternNum).norm(),
           J[target_cid_now]->Jab2.norm(), J[target_cid_now]->JabJIdx.norm());
    std::exit(1);
  }
#endif
#endif
#ifdef SHOW_IMAGE
  if (show_image) {

    IOWrap::displayImage("host", img_host);
    IOWrap::displayImage("target", img_target);
    IOWrap::waitKey(0);

    delete img_host;
    delete img_target;
  }
#endif
  //* 大于阈值则视为有外点
#ifndef USE_ZNCC
  assert(continued_count == 0);
#endif
  if ((zncc < setting_outlierTH_zncc_LBA && angle < setting_outlierTH_zncc_angle_LBA) ||
#ifndef USE_ZNCC
      energyLeft > std::max<float>(host->frameEnergyTH,
                                   target->frameEnergyTH) /*|| wJI2_sum < 2*/
#else
      zncc < 0.8
#endif
      || continued_count == MAX_RES_PER_POINT) {
    // printf("residual: %f, energyLeft: %f, host->frameEnergyTH: %f,
    // target->frameEnergyTH: %f\n", residual, energyLeft, host->frameEnergyTH,
    // target->frameEnergyTH);
    energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
    state_NewState[target_cid_now] = ResState::OUTLIER;
  } else {
    state_NewState[target_cid_now] = ResState::IN;
  }
  state_zncc_angle[target_cid_now] = angle;
  state_hw[target_cid_now] = hw;
  state_residual_residual_gray[target_cid_now] = res_sum;
  state_NewEnergy[target_cid_now] = energyLeft;
  ret << energyLeft, energyLeft_gray,angle,hw, hw_gray, 1.0f;
  return ret;
}

void PointFrameResidual::debugPlot(int cid) {
  if (state_state[cid] == ResState::OOB)
    return;
  Vec3b cT = Vec3b(0, 0, 0);

  if (freeDebugParam5 == 0) {
    float rT = 20 * sqrt(state_energy[cid] / 9);
    if (rT < 0)
      rT = 0;
    if (rT > 255)
      rT = 255;
    cT = Vec3b(0, 255 - rT, rT);
  } else {
    if (state_state[cid] == ResState::IN)
      cT = Vec3b(255, 0, 0);
    else if (state_state[cid] == ResState::OOB)
      cT = Vec3b(255, 255, 0);
    else if (state_state[cid] == ResState::OUTLIER)
      cT = Vec3b(0, 0, 255);
    else
      cT = Vec3b(255, 255, 255);
  }

  for (int i = 0; i < patternNum; i++) {
    if ((projectedTo[cid][i][0] > 2 && projectedTo[cid][i][1] > 2 &&
         projectedTo[cid][i][0] < wG[0] - 3 &&
         projectedTo[cid][i][1] < hG[0] - 3))
      target->debugImage->setPixel1((float)projectedTo[cid][i][0],
                                    (float)projectedTo[cid][i][1], cT, cid);
  }
}

//@ 把计算的残差,雅克比值给EFResidual, 更新残差的状态(好坏)
void PointFrameResidual::applyRes(bool copyJacobians, int cid) {
  if (copyJacobians) {
    if (state_state[cid] == ResState::OOB) {
      assert(!efResidual->isActiveAndIsGoodNEW[cid]);
      return; // can never go back from OOB
    }
    if (state_NewState[cid] == ResState::IN) // && )
    {
      // printf("good res\n");
      efResidual->isActiveAndIsGoodNEW[cid] = true;
      //? 指针好恶心, 计算好了调用这个函数
      efResidual->takeDataF(cid); // 从当前取jacobian数据
    } else {
      // printf("bad res, state_NewState: %d\n", state_NewState);
      efResidual->isActiveAndIsGoodNEW[cid] = false;
    }
  }

  setState(state_NewState[cid], cid);
  state_energy[cid] = state_NewEnergy[cid];
}
} // namespace dso
