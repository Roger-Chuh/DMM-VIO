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

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "IOWrapper/ImageRW.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/TimeMeasurement.h"
#include <algorithm>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

//! 生成2^b个字节对齐
template <int b, typename T>
T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
  const int padT = 1 + ((1 << b) / sizeof(T));
  T *ptr = new T[size + padT];
  rawPtrVec.push_back(ptr);
  T *alignedPtr =
      (T *)((((uintptr_t)(ptr + padT)) >> b)
            << b); //! 左移右移之后就会按照2的b次幂字节对齐, 丢掉不对齐的
  return alignedPtr;
}

//@ 构造函数, 申请内存, 初始化
CoarseTracker::CoarseTracker(int ww, int hh,
                             dmvio::IMUIntegration &imuIntegration)
    : lastRef_aff_g2l(0, 0), imuIntegration(imuIntegration) {
  // make coarse tracking templates.
  int offset = 0;
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    int wl = ww >> lvl;
    int hl = hh >> lvl;

    idepth[lvl] = allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);
    weightSums[lvl] =
        allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);
    weightSums_bak[lvl] =
        allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);

    pc_u[lvl] = allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);
    pc_v[lvl] = allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);
    pc_idepth[lvl] =
        allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);
    pc_color[lvl] =
        allocAligned<4, float>(wl * hl * kCameraNumUsed, ptrToDelete);
    //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    //      image_info_offset[lvl][cid] = offset;
    //      offset += wl * hl;
    //    }
  }

  // warped buffers
  for (int cid = 0; cid < kCameraNumUsed * kCameraNumUsed; ++cid) {
    buf_warped_idepth[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_u[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_v[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_dx[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_dy[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_residual[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_weight[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
    buf_warped_refColor[cid] = allocAligned<4, float>(ww * hh, ptrToDelete);
  }
  newFrame = 0;
  lastRef = 0;
  debugPlot = debugPrint = true;
  w[0] = h[0] = 0;
  refFrameID = -1;
}

CoarseTracker::~CoarseTracker() {
  for (float *ptr : ptrToDelete)
    delete[] ptr;
  ptrToDelete.clear();
}

//@ 构造内参矩阵, 以及一些中间量,
// TODO  每个类都有这个, 直接用一个多好
void CoarseTracker::makeK(CalibHessian *HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < pyrLevelsUsed; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0, 0);
    fyi[level] = Ki[level](1, 1);
    cxi[level] = Ki[level](0, 2);
    cyi[level] = Ki[level](1, 2);
  }
}

//@ 使用在当前帧上投影的点的逆深度, 来生成每个金字塔层上点的逆深度值
// TODO roger, 在换tracking ref时，把上一个ref的深度warp到新ref上
void CoarseTracker::makeCoarseDepthL0(
    std::vector<FrameHessian *> frameHessians) {
  // make coarse tracking templates for latstRef.
  memset(idepth[0], 0, sizeof(float) * w[0] * h[0] * kCameraNumUsed); // 第0层
  memset(weightSums[0], 0, sizeof(float) * w[0] * h[0] * kCameraNumUsed);

  //[ ***step 1*** ] 计算其它点在最新帧投影第0层上的各个像素的逆深度权重,
  //和加权逆深度
  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    for (FrameHessian *fh : frameHessians) {
      for (PointHessian *ph : fh->pointHessians) {
        // 点的上一次残差正常
        //* 优化之后上一次不好的置为0，用来指示，而点是没有删除的，残差删除了
        //      for (int target_cid = 0; target_cid < kCameraNumUsed;
        //      ++target_cid) {
        // printf("enter:\n");
        if (ph->lastResiduals[0].first != 0 &&
            ph->lastResiduals[0].second[target_cid] ==
                ResState::IN /*&& ph->host_cid == target_cid*/) {
          // printf("hit\n");
          PointFrameResidual *r = ph->lastResiduals[0].first;
          // assert(r->target_cid == target_cid);
          // if (r->target_cid != target_cid) {
          // continue;
          //}
          assert(r->efResidual->isActive(target_cid) &&
                 r->target ==
                     lastRef); // 点的残差是好的, 上一次优化的target是这次的ref
          // TODO roger,
          // 我其实是知道这个r是往哪个相机投影得到的，所以centerProjectedTo不需要用array，都已经具体到残差r了，肯定不需要用array了
          int u = r->centerProjectedTo[target_cid][0] + 0.5f; // 四舍五入
          int v = r->centerProjectedTo[target_cid][1] + 0.5f;
          float new_idepth = r->centerProjectedTo[target_cid][2];
          float weight =
              sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12)); // 协方差逆做权重

          idepth[0][u + w[0] * v + target_cid * w[0] * h[0]] +=
              new_idepth * weight; // 加权后的
          weightSums[0][u + w[0] * v + target_cid * w[0] * h[0]] += weight;
        }
      }
    }
    //  }
    //  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    //[ ***step 2*** ] 从下层向上层生成逆深度和权重
    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
      int lvlm1 = lvl - 1;
      int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1], hlm1 = h[lvlm1];

      float *idepth_l = idepth[lvl] + target_cid * hl * wl;
      float *weightSums_l = weightSums[lvl] + target_cid * hl * wl;

      float *idepth_lm = idepth[lvlm1] + target_cid * hlm1 * wlm1;
      float *weightSums_lm = weightSums[lvlm1] + target_cid * hlm1 * wlm1;

      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++) {
          int bidx = 2 * x + 2 * y * wlm1;
          //? 为什么不除以4   答: 后面除以权重的和了 nice!
          idepth_l[x + y * wl] = idepth_lm[bidx] + idepth_lm[bidx + 1] +
                                 idepth_lm[bidx + wlm1] +
                                 idepth_lm[bidx + wlm1 + 1];

          weightSums_l[x + y * wl] =
              weightSums_lm[bidx] + weightSums_lm[bidx + 1] +
              weightSums_lm[bidx + wlm1] + weightSums_lm[bidx + wlm1 + 1];
        }
    }

    //[ ***step 3*** ] 0和1层 对于没有深度的像素点, 使用周围斜45度的四个点来填充
    // dilate idepth by 1.
    for (int lvl = 0; lvl < 2; lvl++) {
      int numIts = 1;
      int wl = w[lvl], hl = h[lvl];
      for (int it = 0; it < numIts; it++) {
        int wh = w[lvl] * h[lvl] - w[lvl]; // 空出一行
        float *weightSumsl = weightSums[lvl] + target_cid * wl * hl;
        float *weightSumsl_bak = weightSums_bak[lvl] + target_cid * wl * hl;
        memcpy(weightSumsl_bak, weightSumsl,
               w[lvl] * h[lvl] * sizeof(float)); // 备份
        float *idepthl = idepth[lvl] +
                         target_cid * wl * hl; // dotnt need to make a temp copy
        // of depth, since I only
        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for (int i = w[lvl] + 1; i < wh - 1; i++) // 上下各空一行
        {
          if (weightSumsl_bak[i] <= 0) {
            // 使用四个角上的点来填充没有深度的
            // bug: 对于竖直边缘上的点不太好把, 使用上两行的来计算
            float sum = 0, num = 0, numn = 0;
            if (weightSumsl_bak[i + 1 + wl] > 0) {
              sum += idepthl[i + 1 + wl];
              num += weightSumsl_bak[i + 1 + wl];
              numn++;
            }
            if (weightSumsl_bak[i - 1 - wl] > 0) {
              sum += idepthl[i - 1 - wl];
              num += weightSumsl_bak[i - 1 - wl];
              numn++;
            }
            if (weightSumsl_bak[i + wl - 1] > 0) {
              sum += idepthl[i + wl - 1];
              num += weightSumsl_bak[i + wl - 1];
              numn++;
            }
            if (weightSumsl_bak[i - wl + 1] > 0) {
              sum += idepthl[i - wl + 1];
              num += weightSumsl_bak[i - wl + 1];
              numn++;
            }
            if (numn > 0) {
              idepthl[i] = sum / numn;
              weightSumsl[i] = num / numn;
            }
          }
        }
      }
    }

    //[ ***step 4*** ] 2层向上, 对于没有深度的像素点, 使用上下左右的四个点来填充
    // dilate idepth by 1 (2 on lower levels).
    for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
      int wh = w[lvl] * h[lvl] - w[lvl];
      int wl = w[lvl];
      int hl = h[lvl];
      float *weightSumsl = weightSums[lvl] + target_cid * wl * hl;
      float *weightSumsl_bak = weightSums_bak[lvl] + target_cid * wl * hl;
      memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
      float *idepthl =
          idepth[lvl] +
          target_cid * wl * hl; // dotnt need to make a temp copy of
      // depth, since I only
      // read values with weightSumsl>0, and write ones with weightSumsl<=0.
      for (int i = w[lvl] + 1; i < wh - 1; i++) {
        if (weightSumsl_bak[i] <= 0) {
          float sum = 0, num = 0, numn = 0;
          if (weightSumsl_bak[i + 1] > 0) {
            sum += idepthl[i + 1];
            num += weightSumsl_bak[i + 1];
            numn++;
          }
          if (weightSumsl_bak[i - 1] > 0) {
            sum += idepthl[i - 1];
            num += weightSumsl_bak[i - 1];
            numn++;
          }
          if (weightSumsl_bak[i + wl] > 0) {
            sum += idepthl[i + wl];
            num += weightSumsl_bak[i + wl];
            numn++;
          }
          if (weightSumsl_bak[i - wl] > 0) {
            sum += idepthl[i - wl];
            num += weightSumsl_bak[i - wl];
            numn++;
          }
          if (numn > 0) {
            idepthl[i] = sum / numn;
            weightSumsl[i] = num / numn;
          }
        }
      }
    }

    //[ ***step 5*** ] 归一化点的逆深度并赋值给成员变量pc_*
    // normalize idepths and weights.
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
      int wl = w[lvl];
      int hl = h[lvl];
      float *weightSumsl = weightSums[lvl] + target_cid * wl * hl;
      float *idepthl = idepth[lvl] + target_cid * wl * hl;
      Eigen::Vector3f *dIRefl = lastRef->dIp[lvl] + target_cid * wl * hl;

      // int wl = w[lvl], hl = h[lvl];

      int lpc_n = 0;
      //!!!! 指针, 只是把指针传过去, 怎么总想有没有赋值, 智障

      float *lpc_u = pc_u[lvl] + wl * hl * target_cid;
      float *lpc_v = pc_v[lvl] + wl * hl * target_cid;
      float *lpc_idepth = pc_idepth[lvl] + wl * hl * target_cid;
      float *lpc_color = pc_color[lvl] + wl * hl * target_cid;

      for (int y = 2; y < hl - 2; y++)
        for (int x = 2; x < wl - 2; x++) {
          int i = x + y * wl;

          if (weightSumsl[i] > 0) // 有值的
          {
            idepthl[i] /= weightSumsl[i];
            lpc_u[lpc_n] = x;
            lpc_v[lpc_n] = y;
            lpc_idepth[lpc_n] = idepthl[i];
            lpc_color[lpc_n] = dIRefl[i][0];

            if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0)) {
              idepthl[i] = -1;
              continue; // just skip if something is wrong.
            }
            lpc_n++;
          } else
            idepthl[i] = -1;

          weightSumsl[i] = 1; // 求完就变成1了
        }

      pc_n[lvl][target_cid] = lpc_n;
    }
  }
}

//@ 对跟踪的最新帧和参考帧之间的残差, 求 Hessian 和 b
void CoarseTracker::calcGSSSE(bool fix_ab_, bool is_imu_ready, int lvl_target_,
                              int lvl, MatState &H_out, VecState &b_out,
                              const SE3 &refToNew, AffLight aff_g2l, int &N,
                              MultiCamera *p_multi_camera) {
  // acc.initialize();
  int lvl_target = lvl_target_ >= 0 ? lvl_target_ : lvl;
  bool fix_ab = fix_ab_; // lvl_target >= 2 || lvl >= 2;
  __m128 fxl = _mm_set1_ps(fx[lvl /* + host_cid * PYR_LEVELS*/]);
  __m128 fyl = _mm_set1_ps(fy[lvl /* + host_cid * PYR_LEVELS*/]);
  __m128 fxl_target = _mm_set1_ps(fx[lvl_target /* + host_cid * PYR_LEVELS*/]);
  __m128 fyl_target = _mm_set1_ps(fy[lvl_target /* + host_cid * PYR_LEVELS*/]);
  __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
  __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(
      lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
      aff_g2l)[0]));

  __m128 one = _mm_set1_ps(1);
  __m128 minusOne = _mm_set1_ps(-1);
  __m128 zero = _mm_set1_ps(0);
  H_out.setZero();
  b_out.setZero();
  MatState H_temp;
  VecState b_temp;
  N = 0;
  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    acc.initialize();
    const Mat66 &extra_pose_jac =
        newFrame->p_multi_camera->cid_to_T01_inv_Adj[target_cid];
    for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
      int n = buf_warped_n[host_cid * kCameraNumUsed + target_cid];
      N += n;
      assert(n % 4 == 0);
      for (int i = 0; i < n; i += 4) {
        //        printf("i: %d, n: %d, index: %lld, value: %f, target_cid: %d,
        //        "
        //               "host_cid: %d\n",
        //               i, n, host_cid * kCameraNumUsed + target_cid + i,
        //               buf_warped_dx[host_cid * kCameraNumUsed +
        //               target_cid][i], target_cid, host_cid);
        __m128 dx = _mm_mul_ps(
            _mm_load_ps(buf_warped_dx[host_cid * kCameraNumUsed + target_cid] +
                        i),
            fxl_target); //! dx*fx
        __m128 dy = _mm_mul_ps(
            _mm_load_ps(buf_warped_dy[host_cid * kCameraNumUsed + target_cid] +
                        i),
            fyl_target); //! dy*fy
        __m128 u = _mm_load_ps(
            buf_warped_u[host_cid * kCameraNumUsed + target_cid] + i);
        __m128 v = _mm_load_ps(
            buf_warped_v[host_cid * kCameraNumUsed + target_cid] + i);
        __m128 id = _mm_load_ps(
            buf_warped_idepth[host_cid * kCameraNumUsed + target_cid] + i);

        acc.updateSSE_eighted(
            _mm_mul_ps(id, dx), // 对位移x导数
            _mm_mul_ps(id, dy), // 对位移y导数
            _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx),
                                                       _mm_mul_ps(v, dy)))),
            _mm_sub_ps(
                zero,
                _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(u, v), dx),
                    _mm_mul_ps(
                        dy,
                        _mm_add_ps(one, _mm_mul_ps(v, v))))), // 对旋转xi_1求导
            _mm_add_ps(
                _mm_mul_ps(_mm_mul_ps(u, v), dy),
                _mm_mul_ps(
                    dx, _mm_add_ps(one, _mm_mul_ps(u, u)))), // 对旋转xi_2求导
            _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)), // 对旋转xi_3求导
            (fix_ab || setting_affineOptModeA < 0)
                ? zero
                : _mm_mul_ps(
                      a, _mm_sub_ps(
                             b0,
                             _mm_load_ps(
                                 buf_warped_refColor[host_cid * kCameraNumUsed +
                                                     target_cid] +
                                 i))), // 对目标帧a求导 Jac = a * (I0 + b0)
                                       // 经过ab校正后的host帧的灰度值？
            (fix_ab || setting_affineOptModeB < 0)
                ? zero
                : minusOne, // 对目标帧b求导 Jac = -1
            _mm_load_ps(
                buf_warped_residual[host_cid * kCameraNumUsed + target_cid] +
                i), // 残差
            _mm_load_ps(
                buf_warped_weight[host_cid * kCameraNumUsed + target_cid] +
                i)); // huber权重
      }
    }
    acc.finish();
    H_temp = acc.H.topLeftCorner<STATE_DIM, STATE_DIM>()
                 .cast<double>(); // * (1.0f / N);
    b_temp =
        acc.H.topRightCorner<STATE_DIM, 1>().cast<double>(); // * (1.0f / N);
    H_out.topLeftCorner<6, 6>() += extra_pose_jac.transpose() *
                                   H_temp.topLeftCorner<6, 6>() *
                                   extra_pose_jac;
    H_out.block<2, 2>(6, 6) += H_temp.block<2, 2>(6, 6);
    H_out.block<2, 6>(6, 0) += H_temp.block<2, 6>(6, 0) * extra_pose_jac;
    H_out.block<6, 2>(0, 6) +=
        extra_pose_jac.transpose() * H_temp.block<6, 2>(0, 6);
    b_out.head<6>() += extra_pose_jac.transpose() * b_temp.head<6>();
    b_out.segment<2>(6) += b_temp.segment<2>(6);
  }
  // loop all the buffer and cumulate into H and b
  // acc will collect all values in the memory slots into one: H and b.
  // remember the arrow shape on the paper:
  /*      X is H, - is b.
   *      R R R T T T a b -
   *      x x x x x x x x -
   *      x x x x x x x x -
   *      x x x x x x x x -
   *      x x x x x x x x -
   *      x x x x x x x x -
   *      x x x x x x x x -
   *      x x x x x x x x -
   *      - - - - - - - - 1
   *
   *      H is a matrix contains 8 directional column vectors
   *      each column is 0-6 residual's jacobian of SE3,
   * */
  // todo roger, [H b] = 9 * 9
  // acc.finish();
#if 0
  H_out.topLeftCorner<6, 6>() = acc.H.topLeftCorner<6, 6>().cast<double>();// * (1.0f / n);
  b_out.head<6>() = acc.H.topRightCorner<6, 1>().cast<double>();// * (1.0f / n);
  H_out.block<2, 2>(6 + host_cid * 2, 6 + target_cid * 2) = acc.H.block<2, 2>(6,6).cast<double>();// * (1.0f / n);
  if (host_cid == target_cid) {
      b_out.segment<2>(6 + host_cid * 2) = acc.H.block<2, 1>(6, 8).cast<double>();// * (1.0f / n);
  }
#else
  // N += n;
  /// TODO roger, 这只是tracking， 不优化idp，所以H
  /// b的维度只有8维，不需要schur补，并且4目共享同一个ab
  //  H_out =
  //      acc.H.topLeftCorner<STATE_DIM, STATE_DIM>().cast<double>() * (1.0f /
  //      N);
  //  b_out = acc.H.topRightCorner<STATE_DIM, 1>().cast<double>() * (1.0f / N);
  H_out *= (1.0f / N);
  b_out *= (1.0f / N);
#endif
  // scale H and b.
  /// H is a 8*8 matrix, the 8 rows are for 8 different directions (scratch that
  /// , it's obvious that H_out is an arrow shaped matrix)
#if 1
  H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
  H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
  H_out.block<8, 1>(0, 6) *= SCALE_A;
  H_out.block<8, 1>(0, 7) *= SCALE_B;
  H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
  H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
  H_out.block<1, 8>(6, 0) *= SCALE_A;
  H_out.block<1, 8>(7, 0) *= SCALE_B;
  b_out.segment<3>(0) *= SCALE_XI_ROT;
  b_out.segment<3>(3) *= SCALE_XI_TRANS;
  b_out.segment<1>(6) *= SCALE_A;
  b_out.segment<1>(7) *= SCALE_B;
#endif
}
  static float FindMedian(const std::vector<float> &numbers) {
  if (numbers.empty()) {
    return 0.0f;
  }
  std::vector<float> sortedNumbers = numbers;
  std::sort(sortedNumbers.begin(), sortedNumbers.end());

  size_t size = sortedNumbers.size();
  if (size % 2 == 0) {
    // 偶数个数，取中间两个数的平均值
    return (float)(sortedNumbers[size / 2 - 1] + sortedNumbers[size / 2]) / 2.0;
  } else {
    // 奇数个数，取中间那个数
    return (float)sortedNumbers[size / 2];
  }
}
//@ 计算当前位姿投影得到的残差(能量值), 并进行一些统计
//! 构造尽量多的点, 有助于跟踪
//#define SHOW_TRACK_RES
#define USE_MULTI_KEYFRAME_DISTANCE_MAP
//#define SHOW_KF_PROJ
////////////////////////// #define SHOW_ALIGN_FRAME
//#define SHOW_ERROR_DISTRIBUTION
VecTrack CoarseTracker::calcRes(const bool &disable_kf, const int &iter,
                            const std::vector<FrameHessian *> &frameHessians,
                            int all_keyframe_size, bool is_imu_ready,
                            int lvl_target_, FrameHessian *lastRef, int lvl,
                            const SE3 &refToNew_, AffLight aff_g2l,
                            float cutoffTH, bool show_image) {

  bool force_host_target_same_cid = false; // false;
  float count_thr = 10.0;//200.0;
  int count_step = 1;

  if (!force_host_target_same_cid) {
    count_thr = 10.0;//200.0;
    count_step = 1;//4;
  }
  float setting_huberTH_use;
  float dt_cutoffTH_use;
  int lvl_target = lvl_target_ >= 0 ? lvl_target_ : lvl;
#if 1
  if (lvl >= setting_pyrLvlWithAffineFixed &&
      all_keyframe_size > setting_kfNumWithAffineFixed) {
    setting_huberTH_use = setting_huberTH_loose_tracker;
    dt_cutoffTH_use = setting_dtCutoffTH_loose;
  } else {
    setting_huberTH_use = setting_huberTH_tracker;
    dt_cutoffTH_use = setting_dtCutoffTH;
  }
#else
  if (lvl >= setting_pyrLvlWithAffineFixed ||
      all_keyframe_size <= setting_kfNumWithAffineFixed) {
    setting_huberTH_use = setting_huberTH_loose_tracker;
    dt_cutoffTH_use = setting_dtCutoffTH_loose;
  } else {
    setting_huberTH_use = setting_huberTH_tracker;
    dt_cutoffTH_use = setting_dtCutoffTH;
  }
#endif
#ifdef USE_EDGE_ALIGN
  float thr_draw = dt_cutoffTH_use * setting_variableScale;
#else
  float thr_draw = cutoffTH;
#endif
  bool show_kf_disp = lvl_target == 0 && lvl == 0 && iter == 0;
  float hw_sum = 0;
  Vec2f res_sum = Vec2f::Zero();
  float res_count = 0;
  float E = 0;
  int numTermsInE = 0;
  int point_num_without_edges = 0;
  int depth_map_point_num = 0;
  // int numTermsInWarped = 0;
  int numSaturated = 0;
  float sumSquaredShiftT = 0;
  float sumSquaredShiftRT = 0;
  float sumSquaredShiftNum = 0;
  int numTermsInWarpedSum = 0;
#if 0 //def SAVE_IMAGES
  bool show_align_res = true;
#else
  bool show_align_res = lvl == 0 && lvl == lvl_target && iter == -1;
#endif
  std::array<float, kCameraNumUsed * kCameraNumUsed> a_sumSquaredShiftT,
      a_sumSquaredShiftRT, a_sumSquaredShiftNum;
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
      a_sumSquaredShiftT[host_cid * kCameraNumUsed + target_cid] = 0;
      a_sumSquaredShiftRT[host_cid * kCameraNumUsed + target_cid] = 0;
      a_sumSquaredShiftNum[host_cid * kCameraNumUsed + target_cid] = 0;
    }
  }
#ifdef SHOW_ALIGN_FRAME
  MinimalImageB3 *img_target_align;
  if (show_align_res) {
    img_target_align = new MinimalImageB3(wG[lvl_target], hG[lvl_target]);
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      const Eigen::Vector3f *dIl_gray = newFrame->dIp[lvl_target] + wG[lvl_target] * hG[lvl_target] * cid;
      for (int i = 0; i < wG[lvl_target] * hG[lvl_target]; i++) {
        float colL = dIl_gray[i][0];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        img_target_align->at(i, cid) = Vec3b(colL, colL, colL);
      }
    }
  }
#endif
  std::vector<std::array<float, kCameraNumUsed * kCameraNumUsed>>
      v_a_sumSquaredShiftT, v_a_sumSquaredShiftRT, v_a_sumSquaredShiftNum;

  std::vector<float> Ts, RTs;
  int show_point_count = 0, valid_point_count = 0;
#ifdef USE_MULTI_KEYFRAME_DISTANCE_MAP
  SE3 Twb_cur = lastRef->shell->camToWorld * refToNew_.inverse();
  if (lvl_target == 0 && lvl == 0) {
#ifdef SHOW_KF_PROJ
    std::vector<MinimalImageB3 *> v_img_host;
    std::vector<MinimalImageB3 *> v_img_target;
#endif
    float fxl_target = fx[lvl_target];
    float fyl_target = fy[lvl_target];
    int wl = w[lvl];
    int hl = h[lvl];
    float cxl_target = cx[lvl_target];
    float cyl_target = cy[lvl_target];
    int wl_target = w[lvl_target];
    int hl_target = h[lvl_target];
    int kf_count = 0;
    for (FrameHessian *fh : frameHessians) {
      kf_count++;

      printf("aa KFs: [%d / %d], pointHessians: %d, is_tracking_ref: %d\n",
             kf_count, frameHessians.size(), fh->pointHessians.size(),
             fh == lastRef);

      if (fh == newFrame) {
        // this should never happen, since fh hasn't been pushed in
        // frameHessians yet, and we don't know whether it is a kf
        printf("bb fh[%llu] == newFrame[%llu]\n", fh, newFrame);
        std::exit(3);
        continue;
      }
      fh->shell->aff_g2l;
      Vec2f affLL =
          AffLight::fromToVecExposure(fh->ab_exposure, newFrame->ab_exposure,
                                      fh->aff_g2l(), aff_g2l)
              .cast<float>();
      SE3 T10 = Twb_cur.inverse() * fh->shell->camToWorld;
      std::array<float, kCameraNumUsed * kCameraNumUsed>
          temp_a_sumSquaredShiftT, temp_a_sumSquaredShiftRT,
          temp_a_sumSquaredShiftNum;
      for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
        for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
          temp_a_sumSquaredShiftT[host_cid * kCameraNumUsed + target_cid] = 0;
          temp_a_sumSquaredShiftRT[host_cid * kCameraNumUsed + target_cid] = 0;
          temp_a_sumSquaredShiftNum[host_cid * kCameraNumUsed + target_cid] = 0;
        }
      }

#ifdef SHOW_KF_PROJ
      MinimalImageB3 *img_host;
      MinimalImageB3 *img_target;
      int extra_scale = 0;//1;
      float extra_scale_coord = std::pow(2, -extra_scale);
      if (show_kf_disp && !fh->pointHessians.empty()) {
        img_host =
            new MinimalImageB3(w[lvl + extra_scale], h[lvl + extra_scale]);
        img_target = new MinimalImageB3(w[lvl_target + extra_scale],
                                        h[lvl_target + extra_scale]);
        for (int cam = 0; cam < kCameraNumUsed; ++cam) {
          Vec3f *colorRef = fh->dIp[lvl + extra_scale] +
                            wG[lvl + extra_scale] * hG[lvl + extra_scale] * cam;
          for (int i = 0; i < wG[lvl + extra_scale] * hG[lvl + extra_scale];
               i++) {
            // BRIGHTNESS TRANSFER
            float colL = (*(colorRef + i))[0];
            if (colL < 0)
              colL = 0;
            if (colL > 255)
              colL = 255;
            img_host->at(i, cam) = Vec3b(colL, colL, colL);
          }
        }
        for (int cam = 0; cam < kCameraNumUsed; ++cam) {
          Vec3f *colorRef =
              newFrame->dIp[lvl_target + extra_scale] +
              wG[lvl_target + extra_scale] * hG[lvl_target + extra_scale] * cam;
          for (int i = 0;
               i < wG[lvl_target + extra_scale] * hG[lvl_target + extra_scale];
               i++) {
            // BRIGHTNESS TRANSFER
            float colL = (*(colorRef + i))[0];
            if (colL < 0)
              colL = 0;
            if (colL > 255)
              colL = 255;
            img_target->at(i, cam) = Vec3b(colL, colL, colL);
          }
        }
      }
#endif

      for (PointHessian *ph : fh->pointHessians) {
        int good_res_count = 0;

        for (int id = 0; id < ph->residuals.size(); ++id) {
          for (int cid = 0; cid < kCameraNumUsed; ++cid) {
            if (ph->residuals[id]->state_state[cid] == ResState::IN) {
              good_res_count++;
            }
          }
        }

        if (good_res_count <= 1) {
          // till this point, we don't have any residuals on the newest frame
          // yet, no wonder before_CoarseTracker is zero
          if (fh == lastRef) {
            printf("cc good_res_count: %d\n", good_res_count);
          }
          // continue;
        }
        Eigen::Vector3f *dIHostl_gray = fh->dIp[lvl] + wl * hl * ph->host_cid;
        Eigen::Vector3f *dIHostl = fh->dIp[lvl] + wl * hl * ph->host_cid;
        for (int target_cam = 0; target_cam < kCameraNumUsed; ++target_cam) {
          if (ph->host_cid != target_cam && force_host_target_same_cid) {
            continue;
          }
          Eigen::Vector3f *dINewl_gray =
              newFrame->dIp[lvl_target] + wl_target * hl_target * target_cam;
          Eigen::Vector3f *dINewl =
              newFrame->dt_dx_dy[lvl_target] + wl_target * hl_target * target_cam;
          const float *const color = ph->color; // host帧上颜色
          SE3 fhToNew_ = newFrame->PRE_worldToCam * fh->PRE_camToWorld;
          SE3 fhToNew =
              newFrame->p_multi_camera->cid_to_T01_SE3[target_cam].inverse() *
              T10 * newFrame->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
          Mat33f RKi = (fhToNew.rotationMatrix().cast<float>() * Ki[lvl]);
          Vec3f t = (fhToNew.translation()).cast<float>();
          Vec3f pt = RKi * Vec3f(ph->u, ph->v, 1) + t * ph->idepth;
          float u = pt[0] / pt[2]; // 归一化坐标
          float v = pt[1] / pt[2];
          float Ku = fxl_target * u + cxl_target; // 像素坐标
          float Kv = fyl_target * v + cyl_target;

          float new_idepth = ph->idepth / pt[2]; // 当前帧上的深度
          Vec3f hitColor, hostColor, hitColor_gray, hostColor_gray;
          hostColor = getInterpolatedElement33(dIHostl, ph->u, ph->v, wl);
          hostColor_gray = getInterpolatedElement33(dIHostl_gray, ph->u, ph->v, wl);
          bool is_in_frame = true, is_valid_projection = true;
          //* 图像边沿, 深度为负 则跳过
          if (!(Ku > 2 && Kv > 2 && Ku < wl_target - 3 && Kv < hl_target - 3 &&
                new_idepth > 0)) {
            is_in_frame = false;
            hitColor = Vec3f::Constant(std::nan(""));
            hitColor_gray = Vec3f::Constant(std::nan(""));
          } else {
            hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl_target);
            hitColor_gray = getInterpolatedElement33(dINewl_gray, Ku, Kv, wl_target);
          }
#ifndef USE_EDGE_ALIGN
          float residual =
              hitColor_gray[0] - (float)(affLL[0] * hostColor_gray[0] + affLL[1]);
#else
          float residual = hitColor[0];
#endif
          if (!std::isfinite((float)hitColor_gray[0])
#ifdef USE_EDGE_ALIGN
          || !std::isfinite((float)hitColor[0])
#endif
          ) {
            is_valid_projection = false;
          }
          // printf("setting_huberTH_tracker: %f, residual_residual: %f, is_in_frame: %d,is_valid_projection: %d, [host target]: [%d %d], [fh == lastRef]: %d\n", setting_huberTH_tracker, residual, is_in_frame, is_valid_projection,
          //        ph->host_cid, target_cam, fh == lastRef);
          valid_point_count++;
          // printf("is_in_frame: %d, is_valid_projection: %d, new_idepth: %f, target_cam: %d, lvl_target: %d, [wl_target hl_target]: [%d %d], target_uv: [%f %f]\n", is_in_frame, is_valid_projection, new_idepth, target_cam, lvl_target, wl_target, hl_target, Ku, Kv);
          if (is_in_frame && is_valid_projection &&
              (ph->host_cid == target_cam || !force_host_target_same_cid) &&
              std::abs(residual) < setting_huberTH_tracker &&
              (valid_point_count % count_step == 0)) {
            //* 只正的平移 // translation only (positive)
            Vec3f ptT = Ki[lvl] * Vec3f(ph->u, ph->v, 1) + t * ph->idepth;
            float uT = ptT[0] / ptT[2];
            float vT = ptT[1] / ptT[2];
            float KuT = fxl_target * uT + cxl_target;
            float KvT = fyl_target * vT + cyl_target;

            //* 只负的平移// translation only (negative)
            /// warpping
            Vec3f ptT2 = Ki[lvl] * Vec3f(ph->u, ph->v, 1) - t * ph->idepth;
            float uT2 = ptT2[0] / ptT2[2];
            float vT2 = ptT2[1] / ptT2[2];
            float KuT2 = fxl_target * uT2 + cxl_target;
            float KvT2 = fyl_target * vT2 + cyl_target;

            //* 旋转+负的平移//translation and rotation (negative)
            Vec3f pt3 = RKi * Vec3f(ph->u, ph->v, 1) - t * ph->idepth;
            float u3 = pt3[0] / pt3[2];
            float v3 = pt3[1] / pt3[2];
            float Ku3 = fxl_target * u3 + cxl_target;
            float Kv3 = fyl_target * v3 + cyl_target;
            // printf("fh[%llu] == lastRef[%llu]\n", fh, lastRef);
            if (fh == lastRef && force_host_target_same_cid && false) {
              sumSquaredShiftT +=
                  (KuT - ph->u) * (KuT - ph->u) + (KvT - ph->v) * (KvT - ph->v);
              sumSquaredShiftT += (KuT2 - ph->u) * (KuT2 - ph->u) +
                                  (KvT2 - ph->v) * (KvT2 - ph->v);
              sumSquaredShiftRT +=
                  (Ku - ph->u) * (Ku - ph->u) + (Kv - ph->v) * (Kv - ph->v);
              sumSquaredShiftRT +=
                  (Ku3 - ph->u) * (Ku3 - ph->u) + (Kv3 - ph->v) * (Kv3 - ph->v);
              sumSquaredShiftNum += 2;
            }
            temp_a_sumSquaredShiftT[ph->host_cid * kCameraNumUsed +
                                    target_cam] +=
                (KuT - ph->u) * (KuT - ph->u) + (KvT - ph->v) * (KvT - ph->v);
            temp_a_sumSquaredShiftT[ph->host_cid * kCameraNumUsed +
                                    target_cam] +=
                (KuT2 - ph->u) * (KuT2 - ph->u) +
                (KvT2 - ph->v) * (KvT2 - ph->v);
            temp_a_sumSquaredShiftRT[ph->host_cid * kCameraNumUsed +
                                     target_cam] +=
                (Ku - ph->u) * (Ku - ph->u) + (Kv - ph->v) * (Kv - ph->v);
            temp_a_sumSquaredShiftRT[ph->host_cid * kCameraNumUsed +
                                     target_cam] +=
                (Ku3 - ph->u) * (Ku3 - ph->u) + (Kv3 - ph->v) * (Kv3 - ph->v);
            temp_a_sumSquaredShiftNum[ph->host_cid * kCameraNumUsed +
                                      target_cam] += 2.0;
#ifdef SHOW_KF_PROJ
            show_point_count++;
            if (show_kf_disp && (show_point_count % 32 == 0 || true)) {
              //              printf("residual_residual: %f, is_in_frame: %d, "
              //                     "is_valid_projection: %d, [host target]:
              //                     [%d %d], [fh == " "lastRef]: %d\n",
              //                     residual, is_in_frame, is_valid_projection,
              //                     ph->host_cid, target_cam, fh == lastRef);

              if ((extra_scale_coord * ph->u > 10 &&
                   extra_scale_coord * ph->v > 10 &&
                   extra_scale_coord * ph->u < w[lvl + extra_scale] - 10 &&
                   extra_scale_coord * ph->v < h[lvl + extra_scale] - 10)) {
                img_host->setPixelCirc(extra_scale_coord * ph->u + 0.5,
                                       extra_scale_coord * ph->v + 0.5,
                                       makeRainbow3B(1), ph->host_cid);
              }
              if ((extra_scale_coord * Ku > 10 && extra_scale_coord * Kv > 10 &&
                   extra_scale_coord * Ku < w[lvl_target + extra_scale] - 10 &&
                   extra_scale_coord * Kv < h[lvl_target + extra_scale] - 10)) {
                img_target->setPixelCirc(extra_scale_coord * Ku + 0.5,
                                         extra_scale_coord * Kv + 0.5,
                                         makeRainbow3B(1), target_cam);
              }
              // IOWrap::displayImage("host frame 111", img_host);
              // IOWrap::displayImage("target frame 111", img_target);
              // printf("host_cid: %d, target_cid: %d, good_res_count: %d,all_res_count: %d\n",
              //        ph->host_cid, target_cam, good_res_count, ph->residuals.size());
              // IOWrap::waitKey(0);
              //              delete img_host;
              //              delete img_target;
            }
#endif
          }
        }
      }
      // printf("123, iter: %d\n", iter);
#ifdef SHOW_KF_PROJ
      if (show_kf_disp && !fh->pointHessians.empty()) {
        // printf("emplace1\n");
        v_img_host.emplace_back(img_host);
        v_img_target.emplace_back(img_target);
        // printf("emplace11\n");
      }
#endif
      v_a_sumSquaredShiftT.emplace_back(temp_a_sumSquaredShiftT);
      v_a_sumSquaredShiftRT.emplace_back(temp_a_sumSquaredShiftRT);
      v_a_sumSquaredShiftNum.emplace_back(temp_a_sumSquaredShiftNum);
    }
#ifdef SHOW_KF_PROJ

    for (int id = 0; id < v_img_host.size(); ++id) {
      // printf("show1\n");
      IOWrap::displayImage(("kf_host_" + std::to_string(id)).c_str(),
                           v_img_host[id]);
      IOWrap::waitKey(1);
      // printf("show11\n");
    }
    for (int id = 0; id < v_img_target.size(); ++id) {
      // printf("show2\n");
      IOWrap::displayImage(("kf_target_" + std::to_string(id)).c_str(),
                           v_img_target[id]);
      IOWrap::waitKey(1);
      // printf("show22\n");
    }
    for (int id = 0; id < v_img_host.size(); ++id) {
      // printf("delete1\n");
      delete v_img_host[id];
      // printf("delete11\n");
    }
    for (int id = 0; id < v_img_target.size(); ++id) {
      // printf("delete2\n");
      delete v_img_target[id];
      // printf("delete22\n");
    }
    // printf("clear1\n");
    v_img_host.clear();
    v_img_target.clear();
    // printf("clear2\n");
#endif
  }
#endif
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
#ifdef USE_EDGE_ALIGN
      float dt_len = newFrame->max_dt_dx_dy[lvl_target][target_cid][0] -
                     newFrame->min_dt_dx_dy[lvl_target][target_cid][0];
#endif
      int numTermsInWarped = 0;
      // int host_info_offset = image_info_offset[lvl][host_cid]; // kImageWidth
      // * kImageHeight * host_cid
      // * PYR_LEVELS;
      int wl = w[lvl];
      int hl = h[lvl];
      int wl_target = w[lvl_target];
      int hl_target = h[lvl_target];
      Eigen::Vector3f *dINewl_gray = newFrame->dIp[lvl_target] + wl_target * hl_target * target_cid;
#ifndef USE_EDGE_ALIGN
      Eigen::Vector3f *dINewl = dINewl_gray;
#else
      Eigen::Vector3f *dINewl = newFrame->dt_dx_dy[lvl_target] + wl_target * hl_target * target_cid;
      float gray_val_target = (*(newFrame->dIp[lvl_target] + wl_target * hl_target * target_cid))[0];
#endif
      float fxl = fx[lvl];
      float fyl = fy[lvl];
      float cxl = cx[lvl];
      float cyl = cy[lvl];
      float fxl_target = fx[lvl_target];
      float fyl_target = fy[lvl_target];
      float cxl_target = cx[lvl_target];
      float cyl_target = cy[lvl_target];

      SE3 refToNew =
          newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
          refToNew_ * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
      Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
      Vec3f t = (refToNew.translation()).cast<float>();
      // 这个函数会把前后两帧的光度参数变成两个值
      Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure,
                                                newFrame->ab_exposure,
                                                lastRef_aff_g2l, aff_g2l)
                        .cast<float>();

      //          float sumSquaredShiftT = 0;
      //          float sumSquaredShiftRT = 0;
      //          float sumSquaredShiftNum = 0;
      // 经过huber函数后的能量阈值
      float maxEnergy =
          2 * setting_huberTH_use * cutoffTH -
          setting_huberTH_use *
              setting_huberTH_use; // energy for r=setting_coarseCutoffTH.

      MinimalImageB3 *resImage = 0; // 自己定义的图像 nb
      if (debugPlot) {
        resImage = new MinimalImageB3(wl, hl);
        resImage->setConst(Vec3b(255, 255, 255));
      }
      //* 投影在ref帧上的点
      int nl = pc_n[lvl][host_cid];
      // printf("nl: %d\n", nl);
      float *lpc_u = pc_u[lvl] + wl * hl * host_cid;
      float *lpc_v = pc_v[lvl] + wl * hl * host_cid;
      float *lpc_idepth = pc_idepth[lvl] + wl * hl * host_cid;
      float *lpc_color = pc_color[lvl] + wl * hl * host_cid;

      int address_offset =
          w[0] * h[0] * (host_cid * kCameraNumUsed + target_cid);

      for (int i = 0; i < nl; i++) {
        float id = lpc_idepth[i];
        float x = lpc_u[i];
        float y = lpc_v[i];
        //! 投影点
        Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
        float u = pt[0] / pt[2]; // 归一化坐标
        float v = pt[1] / pt[2];
        float Ku = fxl_target * u + cxl_target; // 像素坐标
        float Kv = fyl_target * v + cyl_target;

        float new_idepth = id / pt[2]; // 当前帧上的深度

        //#ifndef USE_MULTI_KEYFRAME_DISTANCE_MAP
        if (lvl_target == 0 && lvl == 0 && i % 1/*32*/ == 0 &&
            (host_cid == target_cid ||
             !force_host_target_same_cid)) //* 第0层 每隔32个点
        {
          //* 只正的平移 // translation only (positive)
          Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
          float uT = ptT[0] / ptT[2];
          float vT = ptT[1] / ptT[2];
          float KuT = fxl_target * uT + cxl_target;
          float KvT = fyl_target * vT + cyl_target;

          //* 只负的平移// translation only (negative)
          /// warpping
          Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
          float uT2 = ptT2[0] / ptT2[2];
          float vT2 = ptT2[1] / ptT2[2];
          float KuT2 = fxl_target * uT2 + cxl_target;
          float KvT2 = fyl_target * vT2 + cyl_target;

          //* 旋转+负的平移//translation and rotation (negative)
          Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
          float u3 = pt3[0] / pt3[2];
          float v3 = pt3[1] / pt3[2];
          float Ku3 = fxl_target * u3 + cxl_target;
          float Kv3 = fyl_target * v3 + cyl_target;

          // translation and rotation (positive)
          // already have it.
          //* 统计像素的移动大小
#if 0
          sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
          sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
          sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
          sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
          sumSquaredShiftNum += 2;
#endif
#if 1
          float refColor = lpc_color[i];
          Vec3f hitColor, hitColor_gray;
          bool is_in_frame = true, is_valid_projection = true;
          //* 图像边沿, 深度为负 则跳过
          if (!(Ku > 2 && Kv > 2 && Ku < wl_target - 3 && Kv < hl_target - 3 &&
                new_idepth > 0)) {
            is_in_frame = false;
            hitColor = Vec3f::Constant(std::nan(""));
            hitColor_gray = Vec3f::Constant(std::nan(""));
          } else {
            hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl_target);
            hitColor_gray = getInterpolatedElement33(dINewl_gray, Ku, Kv, wl_target);
          }

          if (!std::isfinite((float)hitColor_gray[0])
#ifdef USE_EDGE_ALIGN
|| !std::isfinite((float)hitColor[0])
#endif
          ) {
            is_valid_projection = false;
          }
          /// 只算host点的残差，不算8个邻域内的残差了?
          /// 计算残差
          float residual_gray =
              hitColor_gray[0] - (float)(affLL[0] * refColor + affLL[1]);
#ifndef USE_EDGE_ALIGN
          float residual = residual_gray;
#else
          float residual = hitColor[0];
#endif
          float hw =
              fabs(residual) < (setting_huberTH_use /*+ std::abs(affLL[1])*/)
                  ? 1
                  : (setting_huberTH_use /*+ std::abs(affLL[1])*/) /
                        fabs(residual);
          // printf("residual_residual: %f, is_in_frame: %d,
          // is_valid_projection: %d\n", residual, is_in_frame,
          // is_valid_projection);
          float res_draw = fabsf(residual) > thr_draw ? thr_draw : fabsf(residual);
          if (!std::isfinite(res_draw)) {
            // printf("res_draw1: %f, residual: %f\n", res_draw, residual);
            // std::exit(2);
          }
          double color_draw = (res_draw - 0) / (thr_draw - 0);
          if (!std::isfinite(color_draw)) color_draw = 0.0;
          if (color_draw < 0.0) color_draw = 0.0;
          if (color_draw > 1.0) color_draw = 1.0;
          Vec3 bgr_map = color_map.GetBgr(color_draw) * 255;
#ifdef SHOW_ALIGN_FRAME
          if (show_align_res && is_in_frame && is_valid_projection && std::isfinite(residual) && std::isfinite(residual_gray)) {
            // img_target_align->setPixelCirc(Ku, Kv, Vec3b(0, 0, 255), target_cid);
            img_target_align->setPixelCirc(Ku, Kv, Vec3b(bgr_map[0], bgr_map[1], bgr_map[2]), target_cid);
          }
#endif
          if (is_in_frame && is_valid_projection &&
              std::abs(residual) < /*setting_huberTH_use */ setting_huberTH_tracker) {
            if (!std::isfinite(residual)) {
              printf("residual has nan\n");
              std::exit(3);
            }
#ifdef SHOW_ALIGN_FRAME
            if (show_align_res) {
              // img_target_align->setPixelCirc(Ku, Kv, Vec3b(255, 0, 0), target_cid);
              img_target_align->setPixelCirc(Ku, Kv, Vec3b(bgr_map[0], bgr_map[1], bgr_map[2]), target_cid);
            }
#endif
            //            // translation and rotation (positive)
            //            // already have it.
            //            //* 统计像素的移动大小
            //            sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y)
            //            * (KvT - y); sumSquaredShiftT +=
            //                (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
            //            sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) *
            //            (Kv - y); sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) +
            //            (Kv3 - y) * (Kv3 - y); sumSquaredShiftNum += 2;
#if 1
            sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
            sumSquaredShiftT +=
                (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
            sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
            sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
            sumSquaredShiftNum += 2;
#endif
#if 1 // ndef USE_MULTI_KEYFRAME_DISTANCE_MAP
            a_sumSquaredShiftT[host_cid * kCameraNumUsed + target_cid] +=
                (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
            a_sumSquaredShiftT[host_cid * kCameraNumUsed + target_cid] +=
                (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
            a_sumSquaredShiftRT[host_cid * kCameraNumUsed + target_cid] +=
                (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
            a_sumSquaredShiftRT[host_cid * kCameraNumUsed + target_cid] +=
                (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
            a_sumSquaredShiftNum[host_cid * kCameraNumUsed + target_cid] += 2.0;
#endif
          }
#endif
        }
        //* 图像边沿, 深度为负 则跳过
        if (!(Ku > 2 && Kv > 2 && Ku < wl_target - 3 && Kv < hl_target - 3 &&
              new_idepth > 0))
          continue;

        // 计算残差
        float refColor = lpc_color[i];
        Vec3f hitColor_gray = getInterpolatedElement33(dINewl_gray, Ku, Kv, wl_target);
        Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl_target);
        if (!std::isfinite((float)hitColor_gray[0])
#ifdef USE_EDGE_ALIGN
        || !std::isfinite((float)hitColor[0])
#endif
        ) {
          continue;
        }


#ifndef USE_EDGE_ALIGN
        hitColor = hitColor_gray;
        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
#else
        float residual_gray = hitColor_gray[0] - (float)(affLL[0] * refColor + affLL[1]);
        float residual = hitColor[0];
#endif

#ifdef SHOW_ALIGN_FRAME
        if (show_align_res) {
          // img_target_align->setPixel9(Ku, Kv, Vec3b(0, 255, 0), target_cid);
          float res_draw = fabsf(residual) > thr_draw ? thr_draw : fabsf(residual);
          if (!std::isfinite(res_draw)) {
            printf("res_draw2: %f, residual: %f\n", res_draw, residual);
            std::exit(2);
          }
          double color_draw = (res_draw - 0) / (thr_draw - 0);
          if (!std::isfinite(color_draw)) color_draw = 0.0;
          if (color_draw < 0.0) color_draw = 0.0;
          if (color_draw > 1.0) color_draw = 1.0;
          Vec3 bgr_map = color_map.GetBgr(color_draw) * 255;
          img_target_align->setPixelCirc(Ku, Kv, Vec3b(bgr_map[0], bgr_map[1], bgr_map[2]), target_cid);
        }
#endif


        /// 只算host点的残差，不算8个邻域内的残差了?
        float hw =
            fabs(residual) < (setting_huberTH_use /*+ std::abs(affLL[1])*/)
                ? 1
                : (setting_huberTH_use /*+ std::abs(affLL[1])*/) /
                      fabs(residual);
        depth_map_point_num++;
#ifdef USE_EDGE_ALIGN
        if (fabs(residual) > dt_cutoffTH_use * setting_variableScale) {
          point_num_without_edges++;
          }
#endif
        if (fabs(residual) > cutoffTH) {
          if (debugPlot) {
            resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255), host_cid);
          }
          if (
#ifdef USE_EDGE_ALIGN
        fabs(residual) < dt_cutoffTH_use * setting_variableScale  &&
#endif
           true) {
            E += maxEnergy; // 能量值
            numTermsInE++;  // E 中数目
            numSaturated++; // 大于阈值数目
            hw_sum += hw;
          } else {
            // point_num_without_edges++;
          }
        } else {
          if (debugPlot)
            resImage->setPixel4(
                lpc_u[i], lpc_v[i],
                Vec3b(residual + 128, residual + 128, residual + 128),
                host_cid);
//           if (
// #ifdef USE_EDGE_ALIGN
//         fabs(residual) > dt_cutoffTH_use * setting_variableScale  ||
// #endif
//            false) {
//             point_num_without_edges++;
//           }
          E += hw * residual * residual * (2 - hw);
          numTermsInE++;
          hw_sum += hw;
          res_sum += Vec2f(residual, fabsf(residual_gray));
          res_count +=1;
          // TODO 为凑雅可比buffer一些中间变量，这些变量不一定有明确物理含义
          buf_warped_idepth[host_cid * kCameraNumUsed + target_cid]
                           [numTermsInWarped /* + address_offset*/] =
                               new_idepth;
          buf_warped_u[host_cid * kCameraNumUsed + target_cid]
                      [numTermsInWarped /* + address_offset*/] = u;
          buf_warped_v[host_cid * kCameraNumUsed + target_cid]
                      [numTermsInWarped /* + address_offset*/] = v;
          buf_warped_dx[host_cid * kCameraNumUsed + target_cid]
                       [numTermsInWarped /* + address_offset*/] = hitColor[1];
          buf_warped_dy[host_cid * kCameraNumUsed + target_cid]
                       [numTermsInWarped /* + address_offset*/] = hitColor[2];
          buf_warped_residual[host_cid * kCameraNumUsed + target_cid]
                             [numTermsInWarped /* + address_offset*/] =
                                 residual;
          buf_warped_weight[host_cid * kCameraNumUsed + target_cid]
                           [numTermsInWarped /* + address_offset*/] = hw;
          buf_warped_refColor[host_cid * kCameraNumUsed + target_cid]
                             [numTermsInWarped /* + address_offset*/] =
#if 1 // ndef USE_EDGE_ALIGN
                                 lpc_color[i];
#else
                                 0;
#endif
#ifdef SHOW_TRACK_RES
          if (show_image) {
            printf("residual: %f\n", residual);
          }
          // show_image = true; // i % 300 == 0;
          MinimalImageB3 *img_host;
          MinimalImageB3 *img_target;
          MinimalImageB3 *edge_dt_target;
          if (show_image && (Ku > 15 && Kv > 15 && Ku < wl_target - 15 &&
                             Kv < hl_target - 15 && new_idepth > 0)) {
            img_host = new MinimalImageB3(wG[lvl], hG[lvl]);
            img_target = new MinimalImageB3(wG[lvl_target], hG[lvl_target]);
#ifdef USE_EDGE_ALIGN
            edge_dt_target = new MinimalImageB3(wG[lvl_target], hG[lvl_target]);
#endif
            refFrameID;

            for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
              // BRIGHTNESS TRANSFER
              float colL =
                  (*(lastRef->dIp[lvl] + wG[lvl] * hG[lvl] * host_cid + i))[0];
              if (colL < 0)
                colL = 0;
              if (colL > 255)
                colL = 255;
              img_host->at(i, host_cid) = Vec3b(colL, colL, colL);
              //              colL = (*(newFrame->dIp[lvl] + wG[lvl] * hG[lvl] *
              //              target_cid +
              //                        i))[0];
              //              if (colL < 0)
              //                colL = 0;
              //              if (colL > 255)
              //                colL = 255;
              //              img_target->at(i, target_cid) = Vec3b(colL, colL,
              //              colL);
            }
            for (int i = 0; i < wG[lvl_target] * hG[lvl_target]; i++) {
              // BRIGHTNESS TRANSFER
              float colL =
                  (*(newFrame->dIp[lvl_target] +
                     wG[lvl_target] * hG[lvl_target] * target_cid + i))[0];
              if (colL < 0)
                colL = 0;
              if (colL > 255)
                colL = 255;
              img_target->at(i, target_cid) = Vec3b(colL, colL, colL);
#ifdef USE_EDGE_ALIGN
              float edge_val = (float)(*(
                  newFrame->edge_label_image[lvl_target] +
                  wG[lvl_target] * hG[lvl_target] * target_cid + i))[0];
              float dt_val = (float)(*(
                  newFrame->dt_dx_dy[lvl_target] +
                  wG[lvl_target] * hG[lvl_target] * target_cid + i))[0];
              dt_val =
                  (dt_len > 0)
                      ? 255.0f *
                            (dt_val - newFrame->min_dt_dx_dy[lvl_target]
                                                            [target_cid][0]) /
                            dt_len
                      : 0;
              edge_dt_target->at(i, target_cid) =
                  Vec3b(dt_val, edge_val, edge_val);
#endif
            }

            img_host->setPixel9(x + 0.5, y + 0.5, makeRainbow3B(1), host_cid);
            img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(1),
                                  target_cid);
            IOWrap::displayImage("host", img_host);
            IOWrap::displayImage("target", img_target);
#ifdef USE_EDGE_ALIGN
            edge_dt_target->setPixelCirc(Ku + 0.5, Kv + 0.5, Vec3b(0, 0, 255),
                                         target_cid);
            IOWrap::displayImage("dt_target", edge_dt_target);
#endif
            IOWrap::waitKey(0);

            delete img_host;
            delete img_target;
#ifdef USE_EDGE_ALIGN
            delete edge_dt_target;
#endif
          }
#endif
          numTermsInWarped++;
        }
      }
      numTermsInWarpedSum += numTermsInWarped;
      //* 16字节对齐, 填充上
      while (numTermsInWarped % 4 != 0) {
        buf_warped_idepth[host_cid * kCameraNumUsed + target_cid]
                         [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_u[host_cid * kCameraNumUsed + target_cid]
                    [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_v[host_cid * kCameraNumUsed + target_cid]
                    [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_dx[host_cid * kCameraNumUsed + target_cid]
                     [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_dy[host_cid * kCameraNumUsed + target_cid]
                     [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_residual[host_cid * kCameraNumUsed + target_cid]
                           [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_weight[host_cid * kCameraNumUsed + target_cid]
                         [numTermsInWarped /* + address_offset*/] = 0;
        buf_warped_refColor[host_cid * kCameraNumUsed + target_cid]
                           [numTermsInWarped /* + address_offset*/] = 0;
        numTermsInWarped++;
      }
      buf_warped_n[host_cid * kCameraNumUsed + target_cid] = numTermsInWarped;
      if (debugPlot) {
        IOWrap::displayImage("RES", resImage, false);
        IOWrap::waitKey(0);
        delete resImage;
      }
    }
  }
  //  if (debugPlot) {
  //    IOWrap::displayImage("RES", resImage, false);
  //    IOWrap::waitKey(0);
  //    delete resImage;
  //  }

  VecTrack rs = VecTrack::Zero();
  rs[0] = E;           // 投影的能量值
  rs[1] = numTermsInE; // 投影的点的数目
  rs[2] = sumSquaredShiftT /
          (sumSquaredShiftNum + 0.1); // 纯平移时 平均像素移动的大小
  rs[3] = 0;
  rs[4] = sumSquaredShiftRT /
          (sumSquaredShiftNum + 0.1); // 平移+旋转 平均像素移动大小
  rs[5] = numSaturated / (float)numTermsInE; // 大于cutoff阈值的百分比
  rs[6] = hw_sum;
  rs.segment<2>(7) = res_sum.cast<double>() / res_count;
  rs[9] = float(point_num_without_edges) / float(depth_map_point_num);
  float magic_num_all = 987654.0;
  float magic_num = 987654.0;
  for (int i = 0; i < rs.size(); ++i) {
   if (std::isfinite(rs[i]) && rs[i] > magic_num_all) {
     magic_num_all = rs[i];
   }
    if (std::isfinite(rs[i]) && rs[i] > magic_num) {
      magic_num = rs[i];
    }
  }
  magic_num_all += 10;
  magic_num += 10;

  float min_T_all = magic_num_all;  // 123456;
  float min_RT_all = magic_num_all; // 123456;
  float min_T = magic_num;  // 123456;
  float min_RT = magic_num; // 123456;

  float count_sum = 0;
  if (lvl == 0 && lvl_target == 0) {
    std::cout << "disable_kf: "<< disable_kf<< ", before_CoarseTracker, rs: " << rs.transpose()
              << ", sumSquaredShiftNum: " << sumSquaredShiftNum << ", count_sum: " << count_sum << std::endl;
  }
  if (true) {
#ifndef USE_MULTI_KEYFRAME_DISTANCE_MAP
    // float min_T = magic_num;  // 123456;
    // float min_RT = magic_num; // 123456;
    if (lvl_target == 0 && lvl == 0) {
    for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        float count =
            a_sumSquaredShiftNum[host_cid * kCameraNumUsed + target_cid] + 0.1;
        if (lvl == 0 && lvl_target == 0 &&
            (host_cid == target_cid || !force_host_target_same_cid)) {
          printf(
              "host_cid: %d, target_cid: %d, count: %f, [T RT]value: [%f %f], med[T RT]value_sw: [%f %f]\n",
              host_cid, target_cid, count,
              a_sumSquaredShiftT[host_cid * kCameraNumUsed + target_cid] /
                  (count),
              a_sumSquaredShiftRT[host_cid * kCameraNumUsed + target_cid] /
                  (count), FindMedian(Ts), FindMedian(RTs));
        }
        if (count > 1.0) {
          count_sum += count;
        }
        if (count < count_thr /*100.0*/) {
          continue;
        }
        float T = a_sumSquaredShiftT[host_cid * kCameraNumUsed + target_cid] /
                  (count);
        float RT = a_sumSquaredShiftRT[host_cid * kCameraNumUsed + target_cid] /
                   (count);
        Ts.emplace_back(T);
        RTs.emplace_back(RT);
        if (min_T > T) {
          min_T = T;
        }
        if (min_RT > RT) {
          min_RT = RT;
        }
      }
    }
    VecTrack rs_before1 = rs;
    if (std::abs(min_T - magic_num /*123456*/) > 1) {
      rs_before1[2] = rs[2] = min_T;
      if (Ts.empty()) {
        printf("Ts.empty()\n");
        std::exit(1);
      }
      rs[2] = FindMedian(Ts);
    }
    if (std::abs(min_RT - magic_num /*123456*/) > 1) {
      rs_before1[4] = rs[4] = min_RT;
      if (RTs.empty()) {
        printf("RTs.empty()\n");
        std::exit(1);
      }
      rs[4] = FindMedian(RTs);
    }
    std::cout << "rs_before1: " << rs_before1.transpose()<< "\nrs_ minT: " << min_T << ", minRT: " << min_RT<<"\nrs_after: " << rs.transpose() << std::endl;
    if (!disable_kf && (std::abs(min_T - magic_num /*123456*/) < 1 || std::abs(min_RT - magic_num /*123456*/) < 1)) {
      printf("tracking lost, too few tracked vms, std::abs(min_T - magic_num /*123456*/) < 1 || std::abs(min_RT - magic_num /*123456*/) < 1\n");
      std::exit(1);
    }
    } else {
      if (sumSquaredShiftT > 0.0001 || sumSquaredShiftRT > 0.0001) {
        printf("sumSquaredShiftT > 0.0001 || sumSquaredShiftRT > 0.0001\n");
        std::exit(1);
      }
    }
#else
    if (lvl_target == 0 && lvl == 0) {
      v_a_sumSquaredShiftT.emplace_back(a_sumSquaredShiftT);
      v_a_sumSquaredShiftRT.emplace_back(a_sumSquaredShiftRT);
      v_a_sumSquaredShiftNum.emplace_back(a_sumSquaredShiftNum);
      for (int id = 0; id < v_a_sumSquaredShiftT.size(); ++id) {
        for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
          for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
            float count = v_a_sumSquaredShiftNum[id][host_cid * kCameraNumUsed +
                                                     target_cid] +
                          0.1;
            if (lvl == 0 && lvl_target == 0 &&
                (host_cid == target_cid || !force_host_target_same_cid)) {
#if 1//ndef USE_EDGE_ALIGN
              printf("[kf_id / kf]: [%d / %d], host_cid: %d, target_cid: %d, "
                     "count: %f, mean[T RT]value: [%f %f], med[T RT]value_sw: [%f %f]\n",
                     id, v_a_sumSquaredShiftT.size(), host_cid, target_cid,
                     count,
                     v_a_sumSquaredShiftT[id][host_cid * kCameraNumUsed +
                                              target_cid] /
                         (count),
                     v_a_sumSquaredShiftRT[id][host_cid * kCameraNumUsed +
                                               target_cid] /
                         (count), FindMedian(Ts), FindMedian(RTs));
#endif
            }
            if (count > 1.0) {
              count_sum += count;
            }
            if (count < count_thr /*100.0*/) {
              continue;
            }
            float T = v_a_sumSquaredShiftT[id][host_cid * kCameraNumUsed +
                                               target_cid] /
                      (count);
            float RT = v_a_sumSquaredShiftRT[id][host_cid * kCameraNumUsed +
                                                 target_cid] /
                       (count);
            Ts.emplace_back(T);
            RTs.emplace_back(RT);
            if (min_T_all > T) {
              min_T_all = T;
            }
            if (min_RT_all > RT) {
              min_RT_all = RT;
            }
          }
        }
      }
      VecTrack rs_before1 = rs;
      float min_T_all_bak = min_T_all,min_RT_all_bak = min_RT_all;
      if ((min_T_all > sumSquaredShiftT / (sumSquaredShiftNum + 0.1)) && (sumSquaredShiftT / (sumSquaredShiftNum + 0.1) > 0.1)) {
        min_T_all = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
      }
      if ((min_RT_all > sumSquaredShiftRT / (sumSquaredShiftNum + 0.1)) && (sumSquaredShiftRT / (sumSquaredShiftNum + 0.1) > 0.1)) {
        min_RT_all = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
      }
      VecTrack rs_before2 = rs;
      if (std::abs(min_T_all - magic_num_all /*123456*/) > 1) {
        rs_before2[2] = rs[2] = min_T_all;
        if (Ts.empty()) {
          printf("Ts.empty()\n");
          if (std::abs(min_T_all_bak - magic_num_all /*123456*/) > 1) {
            printf("Ts.empty() 2\n");
            std::exit(1);
          }
        } else {
          rs[2] = FindMedian(Ts);
        }

      }
      if (std::abs(min_RT_all - magic_num_all /*123456*/) > 1) {
        rs_before2[4] = rs[4] = min_RT_all;
        if (RTs.empty()) {
          printf("RTs.empty()\n");
          if (std::abs(min_RT_all_bak - magic_num_all /*123456*/) > 1) {
            printf("RTs.empty() 2\n");
            std::exit(1);
          }
        } else {
          rs[4] = FindMedian(RTs);
        }

      }
      std::cout << "rs_before1: " << rs_before1.transpose()<< "\nrs_ minT_all_bak: " << min_T_all_bak << ", minRT_all_bak: " << min_RT_all_bak<< "\nrs_ minT_all_use: " << min_T_all << ", minRT_all_use: " << min_RT_all<< "\nrs_before2: " << rs_before2.transpose()<< "\nrs_after: " << rs.transpose() << std::endl;

    if (!disable_kf && (std::abs(min_T_all_bak - magic_num_all /*123456*/) < 1 || std::abs(min_RT_all_bak - magic_num_all /*123456*/) < 1)) {
      printf("tracking lost, too few tracked vms, std::abs(min_T_all_bak - magic_num_all /*123456*/) < 1 || std::abs(min_RT_all_bak - magic_num_all /*123456*/) < 1\n");
      std::exit(1);
    }
    }else {
      if (sumSquaredShiftT > 0.0001 || sumSquaredShiftRT > 0.0001) {
        printf("sumSquaredShiftT > 0.0001 || sumSquaredShiftRT > 0.0001\n");
        std::exit(1);
      }
    }
#endif
  }
  std::cout << "after_CoarseTracker, rs: " << rs.transpose()
              << ", sumSquaredShiftNum: " << sumSquaredShiftNum << ", count_sum: " << count_sum << std::endl;
  if (lvl == 0 && lvl_target == 0) {
    // printf("sumSquaredShiftT: %f\n", sumSquaredShiftT);
    // std::cout << "rs: " << rs.transpose() << std::endl;
    if (!disable_kf && (rs[2] < 0.00001 || rs[4] < 0.00001)) {
      printf("tracking lost, rs[2] < 0.1 || rs[4] < 0.1, sumSquaredShiftT: %f, sumSquaredShiftRT: %f, sumSquaredShiftNum: %f\n", sumSquaredShiftT, sumSquaredShiftRT, sumSquaredShiftNum);
      std::exit(1);
    }
  }
#ifdef SHOW_ALIGN_FRAME
  if (show_align_res) {
    for (int cam = 0; cam < kCameraNumUsed; ++cam) {
      Vec2i *edge_pixel_start = newFrame->edge_pixels[lvl_target] + wG[lvl_target] * hG[lvl_target] * cam;
      for (int i = 0; i < newFrame->edge_pixel_num[lvl_target][cam]; ++i) {
        int epx = edge_pixel_start[i][0];
        int epy = edge_pixel_start[i][1];
        if (epx < 10 || epx >= wG[lvl_target] - 10 || epy < 10 || epy >= hG[lvl_target] - 10)
          continue;
        img_target_align->setPixel1((float)epx + 0.5, (float)epy + 0.5, Vec3b(255,0, 255), cam);
      }
    }
    img_target_align->putText(100, 20, std::to_string((int)(rs[9] * 100)).c_str(), Vec3b(255, 255,0),0);
    // img_target_align->putText(20, 100, std::to_string((int)disable_kf).c_str(), Vec3b(0, 255,0),0);
    if (lvl_target == 0) {
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        img_target_align->putText(20, 20, std::to_string((int)(newFrame->mean_gray_val_each[cid])).c_str(), Vec3b(0, 255,255),cid);
      }
      if (newFrame && newFrame->shell) {
        char ts_buf[32];
        snprintf(ts_buf, sizeof(ts_buf), "%.3f", newFrame->shell->timestamp_eval);
        img_target_align->putText(20, 200, ts_buf, Vec3b(255, 255,0),0);
      }
    }
    IOWrap::displayImage("align frame res", img_target_align);
#ifdef SAVE_IMAGES
    if (newFrame && newFrame->shell) {
      char buf[100];
      snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/align_frame_%015lu_%d_%d_%d.png", (uint64_t)(newFrame->shell->timestamp_eval * 1e9), pyrLevelsUsed - 1 - lvl, pyrLevelsUsed - 1 - lvl_target, iter + 1);
      IOWrap::writeImage(buf, img_target_align);
    }
#endif
    IOWrap::waitKey(1);

    delete img_target_align;
  }
#endif
  return rs;
}

//@ 把优化完的最新帧设为参考帧
void CoarseTracker::setCoarseTrackingRef(
    std::vector<FrameHessian *> frameHessians) {
  assert(frameHessians.size() > 0);
  lastRef = frameHessians.back();
  printf("change tracking ref\n");
  makeCoarseDepthL0(frameHessians); // 生成逆深度估值

  refFrameID = lastRef->shell->id;
  lastRef_aff_g2l = lastRef->aff_g2l();

  firstCoarseRMSE = -1;
}

//@ 对新来的帧进行跟踪, 优化得到位姿, 光度参数
bool CoarseTracker::trackNewestCoarse(bool& disable_kf_bak,
    const std::vector<FrameHessian *> &frameHessians, int all_keyframe_size,
    FrameHessian *lastRef, FrameHessian *newFrameHessian, SE3 &lastToNew_out,
    AffLight &aff_g2l_out, int coarsestLvl, Vec5 minResForAbort,
    IOWrap::Output3DWrapper *wrap) {
  bool disable_kf_orig = disable_kf_bak;
  bool &disable_kf = disable_kf_bak;
  debugPlot = setting_render_displayCoarseTrackingFull;
  debugPrint = !setting_debugout_runquiet;

  assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);
  printf("coarsestLvl: %d\n", coarsestLvl);

  lastResiduals.setConstant(NAN);
  lastFlowIndicators.setConstant(1000);

  newFrame = newFrameHessian;
#ifndef USE_MULTI_CAM
  int maxIterations[] = {10, 20, 50, 50, 50, 50, 50, 50}; // 不同层迭代的次数
#else
  // int maxIterations[] = {10, 20, 20, 20, 20, 20, 20, 20}; // 不同层迭代的次数
  int maxIterations[] = {5, 5, 10, 10, 20, 20, 20, 20}; // 不同层迭代的次数
#endif
  float lambdaExtrapolationLimit = 0.001;

  SE3 refToNew_current = lastToNew_out; // 优化的初始值
  AffLight aff_g2l_current = aff_g2l_out;
  //  std::array<AffLight, kCameraNumUsed> a_aff_g2l_current;
  //  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //      a_aff_g2l_current[cid] = aff_g2l_current;
  //  }

  bool haveRepeated = false; // 是否重复计算了

  MatState H;
  VecState b;
  int lastLvl = -1, lastLvl_target = -1;
#if 1 // def USE_MULTI_CAM
  bool use_inner_loop = true;
#else
  bool use_inner_loop = false;
#endif
#ifdef USE_EDGE_ALIGN
  use_inner_loop = false;
#endif
  int inner_loop_start_lvl = use_inner_loop ? pyrLevelsUsed - 1 : 0;
  ;
  int lvl_target = 0;
  int max_iter = -1;
  bool is_imu_ready =
      dso::setting_useIMU && imuIntegration.isCoarseInitialized();
  for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
    if (use_inner_loop) {
      haveRepeated = false;
      disable_kf = disable_kf_orig;
    }
    for (int lvl_target_ = inner_loop_start_lvl /*pyrLevelsUsed - 1*/;
         lvl_target_ >= 0; lvl_target_--) {
      if (use_inner_loop) {
        lvl_target = lvl_target_;
        if (lvl_target_ < lvl) {
          // continue;
        }
        max_iter = maxIterations[lvl] > maxIterations[lvl_target]
                       ? maxIterations[lvl]
                       : maxIterations[lvl_target];
      } else {
        lvl_target = -1; // lvl;
        max_iter = maxIterations[lvl];
      }
      bool fix_ab = all_keyframe_size <= setting_kfNumWithAffineFixed ||
                    (lvl >= 20 || lvl_target >= 20); // lvl != lvl_target;
#ifdef USE_EDGE_ALIGN
      fix_ab = true;
#endif
      float levelCutoffRepeat = 1;
      float setting_coarseCutoffTH_use;
#if 1
      if (lvl >= setting_pyrLvlWithAffineFixed &&
          all_keyframe_size > setting_kfNumWithAffineFixed) {
        setting_coarseCutoffTH_use = setting_coarseCutoffTH_loose;
      } else {
        setting_coarseCutoffTH_use = setting_coarseCutoffTH;
      }
#else
      if (lvl >= setting_pyrLvlWithAffineFixed ||
          all_keyframe_size <= setting_kfNumWithAffineFixed) {
        setting_coarseCutoffTH_use = setting_coarseCutoffTH_loose;
          } else {
            setting_coarseCutoffTH_use = setting_coarseCutoffTH;
          }
#endif
      //[ ***step 1*** ] 计算残差, 保证最多60%残差大于阈值, 计算正规方程
      // TODO preCalculate some values w.r.t. current state estimate
      ///         buf_warped_idepth
      ///			buf_warped_u
      ///			buf_warped_v
      ///			buf_warped_dx
      ///			buf_warped_dy
      ///			buf_warped_residual
      ///			buf_warped_weight
      ///			buf_warped_refColor
      VecTrack resOld = VecTrack::Zero();
      //    for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
      //      for (int target_cid = 0; target_cid < kCameraNumUsed;
      //      ++target_cid)
      //      {
      printf("aa\n");
      resOld =
          calcRes(disable_kf, -1, frameHessians, all_keyframe_size, is_imu_ready,
                  lvl_target, lastRef, lvl, refToNew_current, aff_g2l_current,
                  setting_coarseCutoffTH_use * levelCutoffRepeat,
                  lvl == 0 && lvl_target == 0);
      NAN_CHECK_EIGEN(resOld, "calcRes resOld");
      std::cout << "resOld111: " << resOld.transpose() << std::endl;
      printf("bb, disable_kf: %d, disable_kf_orig: %d\n", disable_kf, disable_kf_orig);
      //      }
      //    }
      //* 保证大于阈值的点小于60%
      //TODO 视觉点太少，求解时也会不稳定, 还是想尝试edge_scale_extra = 0
      const int min_tracked_num = 100;
      if (!std::isfinite(resOld[5]) || ((int)(resOld[1] * (1-resOld[5])) < min_tracked_num)) {
        disable_kf = true;
      }
      printf("cc, disable_kf: %d, disable_kf_orig: %d\n", disable_kf, disable_kf_orig);
      int increase_cutoff_count = 0;
      int increase_cutoff_count_thr = 100;//5;//3;
      while (std::isfinite(resOld[5]) && increase_cutoff_count < increase_cutoff_count_thr && resOld[5] >
#ifndef USE_EDGE_ALIGN
      0.6
#else
      0.8//0.97
#endif
      && (levelCutoffRepeat < 50 || resOld[5] > 0.99)) {
#ifndef USE_EDGE_ALIGN
        levelCutoffRepeat *= 2; // 超过阈值的多, 则放大阈值重新计算
#else
        levelCutoffRepeat *= 2;//1.2; // 超过阈值的多, 则放大阈值重新计算
#endif
        resOld.setZero();
        //      for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
        //        for (int target_cid = 0; target_cid < kCameraNumUsed;
        //        ++target_cid) {
        resOld =
            calcRes(disable_kf,-1, frameHessians, all_keyframe_size, is_imu_ready,
                    lvl_target, lastRef, lvl, refToNew_current, aff_g2l_current,
                    setting_coarseCutoffTH_use * levelCutoffRepeat,
                    lvl == 0 && lvl_target == 0);
        //        }
        //      }
        if (!std::isfinite(resOld[5]) || ((int)(resOld[1] * (1-resOld[5])) < min_tracked_num)) {
          disable_kf = true;
        }
        std::cout << "disable_kf: " << disable_kf <<  ", resOld: " << resOld.transpose() << std::endl;
        increase_cutoff_count++;
        if ((int)(resOld[1] * (1-resOld[5])) < min_tracked_num) {
          increase_cutoff_count = 10000;
        }
        if (!setting_debugout_runquiet)
          printf("all_keyframe_size: %d, setting_coarseCutoffTH_use: %f, [lvl lvl_target]: [%d %d],INCREASING cutoff to %f, increase_cutoff_count: %d, levelCutoffRepeat: %f, haveRepeated: %d, [lvl lvl_target]: [%d %d], (ratio is %f), calced_num: %d, inlier_num: %d!\n",all_keyframe_size, setting_coarseCutoffTH_use, lvl, lvl_target,
                 setting_coarseCutoffTH_use * levelCutoffRepeat, increase_cutoff_count, levelCutoffRepeat, haveRepeated, lvl, lvl_target, resOld[5], (int)resOld[1], (int)(resOld[1] * (1-resOld[5])));
      }
      if (increase_cutoff_count >= increase_cutoff_count_thr) {
        disable_kf = true;
      }
      // refToNew_current is the camera pose
      // aff_g2l_current is the photometric
      // refToNew_current is not used in this function
      // this function only updates H and b and the aff_g2l_current
      // calculate GradientS use intel SSE.
      float lambda = 0.01;
      int fails = 0;
      {
        H.setZero();
        b.setZero();
        int res_count = 0;
        //      for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
        //        for (int target_cid = 0; target_cid < kCameraNumUsed;
        //        ++target_cid) {
        calcGSSSE(fix_ab, is_imu_ready, lvl_target, lvl, H, b, refToNew_current,
                  aff_g2l_current, res_count, newFrame->p_multi_camera);
        NAN_CHECK_EIGEN(H, "calcGSSSE H(iter start)");
        NAN_CHECK_EIGEN(b, "calcGSSSE b(iter start)");
        if (disable_kf) {
          b.setZero();
        }
        //                  if (debugPrint) {
        //                      Vec2f relAff = AffLight::fromToVecExposure(
        //                              lastRef->ab_exposure,
        //                              newFrame->ab_exposure,
        //                              lastRef_aff_g2l, aff_g2l_current)
        //                              .cast<float>();
        //                      printf(
        //                              "[host target]: [%d %d], lvl%d, it %d
        //                              (l=%f / %f) %s: %.3f->%.3f (%d -> %d)
        //                              (|inc| = %f)! \t", host_cid,
        //                              target_cid, lvl, -1, lambda, 1.0f,
        //                              "INITIA", 0.0f, resOld[0] / resOld[1],
        //                              0, (int)resOld[1], 0.0f);
        //                      std::cout <<
        //                      refToNew_current.log().transpose() << " AFF "
        //                                << aff_g2l_current.vec().transpose()
        //                                << " (rel "
        //                                << relAff.transpose() << ")\n";
        //                  }
//        }
//      }
#if 0
          H *= (1.0f / res_count);
          b *= (1.0f / res_count);
          H.block<STATE_DIM, 3>(0, 0) *= SCALE_XI_ROT;
          H.block<STATE_DIM, 3>(0, 3) *= SCALE_XI_TRANS;
          H.block<STATE_DIM, 1>(0, 6) *= SCALE_A;
          H.block<STATE_DIM, 1>(0, 7) *= SCALE_B;
          H.block<3, STATE_DIM>(0, 0) *= SCALE_XI_ROT;
          H.block<3, STATE_DIM>(3, 0) *= SCALE_XI_TRANS;
          H.block<1, STATE_DIM>(6, 0) *= SCALE_A;
          H.block<1, STATE_DIM>(7, 0) *= SCALE_B;
          b.segment<3>(0) *= SCALE_XI_ROT;
          b.segment<3>(3) *= SCALE_XI_TRANS;
          b.segment<1>(6) *= SCALE_A;
          b.segment<1>(7) *= SCALE_B;
#endif
      }
      //    float lambda = 0.01;

      if (debugPrint) {
        Vec2f relAff = AffLight::fromToVecExposure(
                           lastRef->ab_exposure, newFrame->ab_exposure,
                           lastRef_aff_g2l, aff_g2l_current)
                           .cast<float>();
        printf(
            "lvl %d, lvl_target: %d, it %d (l=%.3f / %.3f) %s: %.3f->%.3f (inlier: %d -> %d) (|inc| = "
            "%.3f)! \t",
            lvl, lvl_target, -1, lambda, 1.0f, "INITIAL_ERR", 0.0f, /*mean_err: */resOld[0] / resOld[1], 0,
            /*inlier_num: */(int)resOld[1], /*inc norm: */0.0f);
        std::cout << refToNew_current.log().transpose() << " AFF "
                  << aff_g2l_current.vec().transpose() << " (rel "
                  << relAff.transpose() << ")\n";
      }

      //[ ***step 2*** ] 迭代优化
      for (int iteration = 0; iteration < max_iter /*maxIterations[lvl]*/;
           iteration++) {
        dmvio::TimeMeasurement timeMeasurement("coarseTrackingIteration");
        //[ ***step 2.1*** ] 计算增量
        Mat88 Hl = H;
        for (int i = 0; i < 8; i++)
          Hl(i, i) *= (1 + lambda);

        //? lambda太小的化, 就给增量一个因子, 啥原理????
        float extrapFac = 1;
        if (lambda < lambdaExtrapolationLimit)
          extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));

        SE3 refToNew_new;
        AffLight aff_g2l_new = aff_g2l_current;
        std::cout << "!!!!!!!!!!!!!!! is_imu_ready: " << is_imu_ready
                  << ", !!!!!!!!!!!!!!! fix_ab: " << fix_ab
                  << ", !!!!!!!!!!!!!!!!!!!!!! iter: " << iteration
                  << ", h_lvl: " << lvl << ", t_lvl: " << lvl_target
                  << ", aff_g2l_new: " << aff_g2l_new.vec().transpose()
                  << std::endl;
        double incNorm;
        NAN_CHECK_EIGEN(H, "coarse H");
        NAN_CHECK_EIGEN(b, "coarse b");
        if (dso::setting_useIMU && imuIntegration.isCoarseInitialized()) {
          // The idea of the integration of the IMU (and GTSAM) into the coarse
          // tracking is to replace the line Vec8 inc = Hl.ldlt().solve(-b);
          // with a call to computeCoarseUpdate, which will add GTSAM factors
          // before calculating the update.

          double incA, incB;
          // Note that we pass H instead of Hl as the lambda multiplication is
          // done inside...
          // TODO rog, like align frame in orca next, but with imu factors
          Vec8 inc_gtsam = Vec8::Zero();
          refToNew_new = imuIntegration.computeCoarseUpdate(inc_gtsam,
              H, b, extrapFac, lambda, incA, incB, incNorm, disable_kf);
          NAN_CHECK_SCALAR(incA, "computeCoarseUpdate incA");
          NAN_CHECK_SCALAR(incB, "computeCoarseUpdate incB");
          NAN_CHECK_SCALAR(incNorm, "computeCoarseUpdate incNorm");

          if (fix_ab) {
            if (!(std::abs(incA) == 0 && std::abs(incB) == 0)) {
              printf("!(std::abs(incA) == 0 && std::abs(incB) == 0)\n");
              std::exit(1);
            }
          }

          SE3 oldVal = refToNew_current;
          SE3 newVal = refToNew_new;
          dso::Vec6 increment = (newVal * oldVal.inverse()).log();

          dso::Vec8 totalIncrement;
          totalIncrement.segment(0, 6) = increment;

          totalIncrement(6) = incA;
          totalIncrement(7) = incB;
          printf("tracking_disable_kf: %d, totalIncrement.norm(): %f, incNorm: %f\n", disable_kf, totalIncrement.norm(), incNorm);
          if (disable_kf) {
            if (totalIncrement.norm() > 1e-8 || incNorm > 0.0 || inc_gtsam.norm() > 0.0){
              printf("disable_kf && totalIncrement.norm() > 0.0, totalIncrement.norm(): %g, incNorm: %g, b:[%g %g %g %g %g %g %g %g]\n", totalIncrement.norm(), incNorm, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
              std::exit(2);
            }
          }

          incA *= SCALE_A;
          incB *= SCALE_B;
          // TODO rog, affine的更新没用gtsam去批量update
          // values，而是在流程外面手动更新
          aff_g2l_new.a += incA;
          aff_g2l_new.b += incB;
        } else {
          // TODO rog, align frame without imu factors
          Vec8 inc = Vec8::Zero();
          if (!disable_kf) {
            inc = Hl.ldlt().solve(-b);

            NAN_CHECK_EIGEN(inc, "coarse LDLT inc");

            if (fix_ab || (setting_affineOptModeA < 0 &&
                           setting_affineOptModeB < 0)) // fix a, b
            {
              inc = Hl.ldlt().solve(-b);
              if (fix_ab) {
                assert(inc.tail<2>().norm() == 0);
              }
              inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
              NAN_CHECK_EIGEN(inc.head<6>(), "coarse LDLT sub inc");
              inc.tail<2>().setZero();
            }
            if (!(setting_affineOptModeA < 0) &&
                setting_affineOptModeB < 0) // fix b
            {
              inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
              NAN_CHECK_EIGEN(inc.head<7>(), "coarse LDLT fix-b inc");
              inc.tail<1>().setZero();
            }
            if (setting_affineOptModeA < 0 &&
                !(setting_affineOptModeB < 0)) // fix a
            {
              //? 怎么又换了个方法求....
              MatState HlStitch = Hl;
              VecState bStitch = b;
              HlStitch.col(6) = HlStitch.col(7);
              HlStitch.row(6) = HlStitch.row(7);
              bStitch[6] = bStitch[7];
              Vec7 incStitch =
                  HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
              NAN_CHECK_EIGEN(incStitch, "coarse LDLT fix-a incStitch");
              inc.setZero();
              inc.head<6>() = incStitch.head<6>();
              inc[6] = 0;
              inc[7] = incStitch[6];
            }
          }
          inc *= extrapFac;
          NAN_CHECK_EIGEN(inc, "coarse inc after extrapFac");
          printf("tracking_disable_kf: %d, inc.norm(): %f\n", disable_kf, inc.norm());
          if (disable_kf) {
            if (inc.norm() > 0.0){
              printf("disable_kf && inc.norm() > 0.0, dx: %f\n", inc.norm());
              std::exit(2);
            }
          }
          VecState incScaled = inc;
          incScaled.segment<3>(0) *= SCALE_XI_ROT;
          incScaled.segment<3>(3) *= SCALE_XI_TRANS;
          incScaled.segment<1>(6) *= SCALE_A;
          incScaled.segment<1>(7) *= SCALE_B;

          if (!std::isfinite(incScaled.sum()))
            incScaled.setZero();
          //[ ***step 2.2*** ] 使用增量更新后, 重新计算能量值
          // exp: first three: translational part, last three: rotational part.
          // Note: gtsam::Pose3 contains first rotational and then translational
          // part!
          refToNew_new =
              SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
          aff_g2l_new = aff_g2l_current;
          aff_g2l_new.a += incScaled[6];
          aff_g2l_new.b += incScaled[7];

          incNorm = inc.head(6).norm();
        }
        // std::array<AffLight, kCameraNumUsed> a_aff_g2l_new;
        //      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        //          a_aff_g2l_new[cid] = aff_g2l_new;
        //      }

        VecTrack resNew = VecTrack::Zero();
        //      for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
        //        for (int target_cid = 0; target_cid < kCameraNumUsed;
        //        ++target_cid) {
        resNew =
            calcRes(disable_kf, iteration, frameHessians, all_keyframe_size, is_imu_ready,
                    lvl_target, lastRef, lvl, refToNew_new, aff_g2l_new,
                    setting_coarseCutoffTH_use * levelCutoffRepeat,
                    lvl == 0 && lvl_target == 0);
        NAN_CHECK_EIGEN(resNew, "calcRes resNew");
        //        }
        //      }

        bool accept = (resNew[0] / resNew[1]) <
                      (resOld[0] / resOld[1]); // 平均能量值小则接受

        if (debugPrint) {
          Vec2f relAff = AffLight::fromToVecExposure(
                             lastRef->ab_exposure, newFrame->ab_exposure,
                             lastRef_aff_g2l, aff_g2l_new)
                             .cast<float>();
          printf("lvl %d, lvl_target: %d, it %d (l=%f / %f) %s: mean_energy: [%.3f->%.3f], mean_res: [%.3f->%.3f || %.3f->%.3f], num: [(%d -> %d)], hw_tracker: [(%f -> %f)], Saturated_ratio: [(%f -> %f)] (|inc| = "
                 "%f)! \t",
                 lvl, lvl_target, iteration, lambda, extrapFac,
                 (accept ? "ACCEPT" : "REJECT"), resOld[0] / resOld[1],
                 resNew[0] / resNew[1],resOld[7],resNew[7],resOld[8],resNew[8], (int)resOld[1], (int)resNew[1],resOld[6] / resOld[1],
                 resNew[6] / resNew[1],resOld[5], resNew[5],
                 incNorm);
          std::cout << refToNew_new.log().transpose() << " AFF "
                    << aff_g2l_new.vec().transpose() << " (rel "
                    << relAff.transpose() << ")\n";
        }
        if (accept) {
          {
            H.setZero();
            b.setZero();
            int res_count2 = 0;
            //          for (int host_cid = 0; host_cid < kCameraNumUsed;
            //          ++host_cid) {
            //            for (int target_cid = 0; target_cid < kCameraNumUsed;
            //                 ++target_cid) {
            calcGSSSE(fix_ab, is_imu_ready, lvl_target, lvl, H, b, refToNew_new,
                      aff_g2l_new, res_count2, newFrame->p_multi_camera);
            NAN_CHECK_EIGEN(H, "calcGSSSE H(accept)");
            NAN_CHECK_EIGEN(b, "calcGSSSE b(accept)");
            if (disable_kf) {
              b.setZero();
            }
//            }
//          }
#if 0
                  H *= (1.0f / res_count2);
                  b *= (1.0f / res_count2);
                  H.block<STATE_DIM, 3>(0, 0) *= SCALE_XI_ROT;
                  H.block<STATE_DIM, 3>(0, 3) *= SCALE_XI_TRANS;
                  H.block<STATE_DIM, 1>(0, 6) *= SCALE_A;
                  H.block<STATE_DIM, 1>(0, 7) *= SCALE_B;
                  H.block<3, STATE_DIM>(0, 0) *= SCALE_XI_ROT;
                  H.block<3, STATE_DIM>(3, 0) *= SCALE_XI_TRANS;
                  H.block<1, STATE_DIM>(6, 0) *= SCALE_A;
                  H.block<1, STATE_DIM>(7, 0) *= SCALE_B;
                  b.segment<3>(0) *= SCALE_XI_ROT;
                  b.segment<3>(3) *= SCALE_XI_TRANS;
                  b.segment<1>(6) *= SCALE_A;
                  b.segment<1>(7) *= SCALE_B;
#endif
          }
          resOld = resNew;
          // TODO update state estimate
          // TODO 这里用了fej吗? i guess not, it's just coaseTracking, far from
          // fixed lag smoothing
          aff_g2l_current = aff_g2l_new;
          refToNew_current = refToNew_new;
          if (dso::setting_useIMU)
            imuIntegration.acceptCoarseUpdate();
          lambda *= 0.5;
          fails = 0;
          if (lambda < lambdaExtrapolationLimit) {
            lambda = lambdaExtrapolationLimit;
          }
        } else {
          fails++;
          if (fails < 2) {
            lambda *= 4;
          } else {
            lambda *= 4;
          }
          if (lambda < lambdaExtrapolationLimit) {
            // lambda = lambdaExtrapolationLimit;
          }
          if (lambda > 10000) {
            lambda = 10000;
          }
        }

        lastLvl = lvl;
        lastLvl_target = use_inner_loop ? lvl_target : lvl;
        if (!(incNorm > 1e-3) || fails >= 3 /*200*/) {
          if (debugPrint)
            printf("inc too small, break! fails: %d\n", fails);
          break;
        }
      }
      //[ ***step 3*** ] 记录上一次残差, 光流指示,
      //如果调整过阈值则重新计算这一层
      // set last residual for that level, as well as flow indicators.
      lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
      lastResidualNum[lvl] = (float)(resOld[1] * (1-resOld[5]));
      lastSaturatedRatio[lvl] = resOld[5];
      lastRS[lvl] = resOld;
      // TODO average optical flow
      lastFlowIndicators = resOld.segment<3>(2);
      if (std::isnan(lastResiduals[lvl])) {
        printf("lastResiduals has nan\n");
        return false;
      }
      if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl]) {
        return false; //! 如果算出来大于最好的直接放弃
      }
      if (levelCutoffRepeat > 1 && !haveRepeated && !disable_kf) {
        if (use_inner_loop) {
          lvl_target_++;
        } else {
          lvl++; // 这一层重新算一遍
        }
        haveRepeated = true;
        printf("REPEAT LEVEL because levelCutoffRepeat is enlarged!, lvl: %d, lvl_target_: %d\n", lvl, lvl_target_);
      }
    }
  }

  // set!
  lastToNew_out = refToNew_current;
  aff_g2l_out = aff_g2l_current;

  bool trackingGood = true;
  //[ ***step 4*** ] 判断优化失败情况
  if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2
#ifndef USE_ZNCC
#ifdef USE_MULTI_CAM
                                                                  * 15.0
#endif
#else
#ifdef USE_MULTI_CAM
                                                                  * 150.0
#endif
#endif
                                       )) ||
      (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200))) {
    trackingGood = false;
  }
  Vec2f relAff =
      AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                                  lastRef_aff_g2l, aff_g2l_out)
          .cast<float>();

  printf("aff_g2l_out: [%f %f], relAff: [%f %f]\n", fabsf(aff_g2l_out.a), fabsf(aff_g2l_out.b), fabsf(logf((float)relAff[0])), fabsf((float)relAff[1]));
  if ((setting_affineOptModeA == 0 &&
       (fabsf(logf((float)relAff[0])) > 1.5
#ifndef USE_ZNCC
#ifdef USE_MULTI_CAM
                                            * 15.0
#endif
#else
#ifdef USE_MULTI_CAM
                                            * 150.0
#endif
#endif
        )) ||
      (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200))) {
    trackingGood = false;
  }
  // 固定情况
  for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
    if (setting_affineOptModeA < 0) {
      aff_g2l_out.a = 0;
    }
    if (setting_affineOptModeB < 0) {
      aff_g2l_out.b = 0;
    }
  }
  if (lastLvl == 0 && lastLvl_target == 0) {
    if (dso::setting_useIMU) {
      imuIntegration.addVisualToCoarseGraph(H, b, trackingGood
#ifdef USE_EDGE_ALIGN
      && lastResidualNum[0] > 1000/*3000*/ && !disable_kf/*_orig*/
#endif
      );
    }
  }
  if (!(trackingGood && lastResidualNum[0] > 3000)) {
    // disable_kf = true;
  }
  return trackingGood;
}

void CoarseTracker::debugPlotIDepthMap(std::vector<FrameHessian *> frameHessians,
    float *minID_pt, float *maxID_pt,
    std::vector<IOWrap::Output3DWrapper *> &wraps) const {
  dmvio::TimeMeasurement timeMeasurement("debugPlotIDepthMap");
  if (wraps.empty() && !debugSaveImages) {
    return;
  }
  if (w[1] == 0)
    return;

  // int lvl = 0;
  for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {
    std::vector<float> allID; // TODO idepth numbers, sorted
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      for (int i = 0; i < h[lvl] * w[lvl]; i++) {
        if (idepth[lvl][i + cid * w[lvl] * h[lvl]] > 0) {
          assert((idepth[lvl] + cid * w[lvl] * h[lvl])[i] ==
                 idepth[lvl][i + cid * w[lvl] * h[lvl]]);
          allID.push_back((idepth[lvl] + cid * w[lvl] * h[lvl])[i]);
        }
      }
    }
    std::sort(allID.begin(), allID.end());
    int n = allID.size() - 1;
    if (n <= 0) {
      return;
    }

    float minID_new = allID[(int)(n * 0.05)];
    float maxID_new = allID[(int)(n * 0.95)];
    // float minID_new = allID[(int)(n * 0.2)];
    // float maxID_new = allID[(int)(n * 0.8)];

    float minID, maxID;
    minID = minID_new;
    maxID = maxID_new;
    if (minID_pt != 0 && maxID_pt != 0) {
      if (*minID_pt < 0 || *maxID_pt < 0) {
        *maxID_pt = maxID;
        *minID_pt = minID;
      } else {

        // slowly adapt: change by maximum 10% of old span.
        float maxChange = 0.3 * (*maxID_pt - *minID_pt);

        if (minID < *minID_pt - maxChange)
          minID = *minID_pt - maxChange;
        if (minID > *minID_pt + maxChange)
          minID = *minID_pt + maxChange;

        if (maxID < *maxID_pt - maxChange)
          maxID = *maxID_pt - maxChange;
        if (maxID > *maxID_pt + maxChange)
          maxID = *maxID_pt + maxChange;

        *maxID_pt = maxID;
        *minID_pt = minID;
      }
    }

    MinimalImageB3 mf(w[lvl], h[lvl]);
    mf.setBlack();
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      Eigen::Vector3f *colorRef = lastRef->dIp[lvl] + h[lvl] * w[lvl] * cid;
      for (int i = 0; i < h[lvl] * w[lvl]; i++) {
        int c = colorRef[i][0];// * 0.9f;
        if (c > 255)
          c = 255;
        mf.at(i, cid) = Vec3b(c, c, c); // TODO one channel to three channel
      }
    }
    int wl = w[lvl];
    int hl = h[lvl];
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      for (int y = 3; y < h[lvl] - 3; y++) {
        for (int x = 3; x < wl - 3; x++) {
          int idx = x + y * wl;
          float sid = 0, nid = 0;
          float *bp = idepth[lvl] + idx + cid * wl * hl;

          if (bp[0] > 0) {
            sid += bp[0];
            nid++;
          }
          if (bp[1] > 0) {
            sid += bp[1];
            nid++;
          }
          if (bp[-1] > 0) {
            sid += bp[-1];
            nid++;
          }
          if (bp[wl] > 0) {
            sid += bp[wl];
            nid++;
          }
          if (bp[-wl] > 0) {
            sid += bp[-wl];
            nid++;
          }

          if (bp[0] > 0 || nid >= 3) {
            float id = ((sid / nid) - minID) / ((maxID - minID));
            mf.setPixelCirc(x, y, makeJet3B(id), cid);
            // mf.at(idx) = makeJet3B(id);
          }
        }
      }
    }
#ifdef USE_EDGE_ALIGN
    if (newFrame != frameHessians.back()) {
      printf("newFrame != frameHessians.back(), [%p, %p], frameHessians.size(): %d\n", (void*)newFrame, (void*)frameHessians.back(), frameHessians.size());
      // std::exit(1);
    }
    if (lastRef != frameHessians.back()) {
      printf("lastRef != frameHessians.back(), [%p, %p], frameHessians.size(): %d\n", (void*)lastRef, (void*)frameHessians.back(), frameHessians.size());
      std::exit(1);
    }
    FrameHessian* new_frame;
    new_frame = frameHessians.back();
    for (int cam = 0; cam < kCameraNumUsed; ++cam) {
      Vec2i *edge_pixel_start = new_frame->edge_pixels[lvl] + wG[lvl] * hG[lvl] * cam;
      for (int i = 0; i < new_frame->edge_pixel_num[lvl][cam]; ++i) {
        int epx = edge_pixel_start[i][0];
        int epy = edge_pixel_start[i][1];
        if (epx < 10 || epx >= wG[lvl] - 10 || epy < 10 || epy >= hG[lvl] - 10)
          continue;
        mf.setPixel4((float)epx + 0.5, (float)epy + 0.5, Vec3b(255,0, 255), cam);
      }
    }
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      mf.putText(20, 20, std::to_string(int(lastRef->mean_gray_val_each[cid])).c_str(), Vec3b(0, 255,255),cid);
    }
    if (lastRef && lastRef->shell) {
      char ts_buf[32];
      snprintf(ts_buf, sizeof(ts_buf), "%.3f", lastRef->shell->timestamp_eval);
      mf.putText(20, 100, ts_buf, Vec3b(255, 255,0),0);
    }
#endif
    IOWrap::displayImage(("coarseDepth LVL: " + std::to_string(lvl)).c_str(),
                         &mf, false);
    IOWrap::waitKey(1);
    if (lvl == 0) {
#ifdef SAVE_IMAGES
      if (lastRef && lastRef->shell) {
        char buf[100];
        snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/kf_depth_%015lu.png", (uint64_t)(lastRef->shell->timestamp_eval * 1e9));
        IOWrap::writeImage(buf, &mf);
      }
#endif
      printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
      for (IOWrap::Output3DWrapper *ow : wraps) {
        ow->pushDepthImage(&mf, lastRef->mean_gray_val_each);
      }

      if (debugSaveImages) {
        char buf[1000];
        snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png",
                 lastRef->shell->id, refFrameID);
        IOWrap::writeImage(buf, &mf);
      }
    }
  }
}

void CoarseTracker::debugPlotIDepthMapFloat(
    std::vector<IOWrap::Output3DWrapper *> &wraps) {
  dmvio::TimeMeasurement timeMeasurement("debugPlotIDepthMapFloat");
  if (w[1] == 0)
    return;
  int lvl = 0;
  MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
  for (IOWrap::Output3DWrapper *ow : wraps)
    ow->pushDepthImageFloat(&mim, lastRef);
}

bool CoarseTracker::NeedKF() {
  bool needToMakeKF = true;
  return needToMakeKF;
}

CoarseDistanceMap::CoarseDistanceMap(int ww,
                                     int hh) { //* 在第一层上算的, 所以除4
  fwdWarpedIDDistFinal = new float[ww * hh / 4 * kCameraNumUsed];

  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    bfsList1[cid] = new Eigen::Vector2i[ww * hh / 4 * kCameraNumUsed];
    bfsList2[cid] = new Eigen::Vector2i[ww * hh / 4 * kCameraNumUsed];
  }
  int fac = 1 << (pyrLevelsUsed - 1);

  coarseProjectionGrid =
      new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
  coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

  w[0] = h[0] = 0;
}

CoarseDistanceMap::~CoarseDistanceMap() {
  delete[] fwdWarpedIDDistFinal;
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    delete[] bfsList1[cid];
    delete[] bfsList2[cid];
  }
  delete[] coarseProjectionGrid;
  delete[] coarseProjectionGridNum;
}

///@ 对于目前所有的地图点投影, 生成距离场图
void CoarseDistanceMap::makeDistanceMap(
    std::vector<FrameHessian *> frameHessians, FrameHessian *frame,
    const int &target_cid) {
  int w1 = w[1]; //? 为啥使用第一层的
  int h1 = h[1];
  int wh1 = w1 * h1;
  for (int i = 0; i < wh1; i++)
    fwdWarpedIDDistFinal[i + wh1 * target_cid] = 1000;

  // make coarse tracking templates for latstRef.
  int numItems = 0;

  for (FrameHessian *fh : frameHessians) {
    if (frame == fh)
      continue;

    //    SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
    //    /// old keyframe的0层投影到newest keyframe的1层？
    //    Mat33f KRKi =
    //        (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]); //
    //        0层到1层变换
    //    Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

    for (PointHessian *ph : fh->pointHessians) {
      assert(ph->status == PointHessian::ACTIVE);

      SE3 fhToNew = fh->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
                    frame->PRE_worldToCam * fh->PRE_camToWorld *
                    fh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
      /// old keyframe的0层投影到newest keyframe的1层？
      Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() *
                     Ki[0]); // 0层到1层变换
      Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

      // TODO old keyframe的0层投影到newest keyframe的1层？
      // TODO in pixel coordinate
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) +
                  Kt * ph->idepth_scaled; // 投影到frame帧
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;
      if (!(u > 0 && v > 0 && u < w[1] && v < h[1]))
        continue;
      fwdWarpedIDDistFinal[u + w1 * v + wh1 * target_cid] = 0;
      bfsList1[target_cid][numItems] = Eigen::Vector2i(u, v);
      numItems++;
    }
  }
  /// now in this function, use dfs to grow distance in the bfslist1 list.
  growDistBFS(numItems,
              target_cid); /// numItems is the total number of point hessians in
                           /// all frames in the sliding window.
}

void CoarseDistanceMap::makeInlierVotes(
    std::vector<FrameHessian *> frameHessians) {}

//@ 生成每一层的距离, 第一层为1, 第二层为2....
/// record all the neighbourhood of point hessians in the sliding window frames
/// in a BFS fashion they store all the neighbour points in bfsList1 and
/// bfsList2 the pattern is interlacing four directions and eight directions.
void CoarseDistanceMap::growDistBFS(int bfsNum, const int &target_cid) {
  int cid = target_cid;
  assert(w[0] != 0);
  int w1 = w[1], h1 = h[1];
  // TODO loop all points for 40 times
  for (int k = 1; k < 40; k++) {
    int bfsNum2 = bfsNum;
    //* 每一次都是在上一次的点周围找
    /// reprojections from older keyframes[0] to newest keyframe[1]
    std::swap<Eigen::Vector2i *>(bfsList1[cid],
                                 bfsList2[cid]); // 每次迭代一遍就交换
    bfsNum = 0;

    if (k % 2 == 0) // 偶数
    {
      for (int i = 0; i < bfsNum2; i++) {
        int x = bfsList2[cid][i][0];
        int y = bfsList2[cid][i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
          continue;
        int idx = x + y * w1; /// this makes up the distance index to find in
                              /// forward warped idepth distance final
        /// remember in coarseTracker make Depth map, fwdWarpedIDDistFinal was
        /// initialized as 1000 for each value. fwd's value will be quickly
        /// marked as k or k-1 or k-2 or ... any value between 0..k this way
        /// they will keep search the other direction which was not retrieved
        /// before. the following four if statements are searching right left up
        /// and down four directions. the reason they swap the bfsList is that
        /// they want to interlace the directions, just merge the direction
        /// patterns into List.

        //* 右边
        /// fwdWarpedIDDistFinal = 1000 or 0

        if (fwdWarpedIDDistFinal[idx + 1 + w1 * h1 * cid] > k) // 没有赋值的位置
        {
          fwdWarpedIDDistFinal[idx + 1 + w1 * h1 * cid] =
              k; // 赋值为2, 4, 6 ....
          /// k should be recording the depth of search.
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x + 1, y);
          bfsNum++;
        }
        //* 左边
        if (fwdWarpedIDDistFinal[idx - 1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx - 1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x - 1, y);
          bfsNum++;
        }
        //* 下边
        if (fwdWarpedIDDistFinal[idx + w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx + w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x, y + 1);
          bfsNum++;
        }
        //* 上边
        if (fwdWarpedIDDistFinal[idx - w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx - w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x, y - 1);
          bfsNum++;
        }
      }
    } else {
      for (int i = 0; i < bfsNum2; i++) {
        int x = bfsList2[cid][i][0];
        int y = bfsList2[cid][i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
          continue;
        int idx = x + y * w1;
        //* 上下左右
        if (fwdWarpedIDDistFinal[idx + 1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx + 1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x + 1, y);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx - 1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x - 1, y);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx + w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx + w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx - w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x, y - 1);
          bfsNum++;
        }

        /// everything above is same when k is even number.
        /// what make this different is it search the points
        /// that is left upper corner of the pixel
        /// and bottom right corner, top right, bottom left
        /// four corners.
        //* 四个角
        if (fwdWarpedIDDistFinal[idx + 1 + w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx + 1 + w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x + 1, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1 + w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx - 1 + w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x - 1, y + 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx - 1 - w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx - 1 - w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x - 1, y - 1);
          bfsNum++;
        }
        if (fwdWarpedIDDistFinal[idx + 1 - w1 + w1 * h1 * cid] > k) {
          fwdWarpedIDDistFinal[idx + 1 - w1 + w1 * h1 * cid] = k;
          bfsList1[cid][bfsNum] = Eigen::Vector2i(x + 1, y - 1);
          bfsNum++;
        }
      }
    }
  }
}

//@ 在点(u, v)附近生成距离场
void CoarseDistanceMap::addIntoDistFinal(int u, int v, const int &target_cid) {
  if (w[0] == 0)
    return;
  bfsList1[target_cid][0] = Eigen::Vector2i(u, v);
  fwdWarpedIDDistFinal[u + w[1] * v + w[1] * h[1] * target_cid] = 0;
  growDistBFS(1, target_cid);
}

void CoarseDistanceMap::makeK(CalibHessian *HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < pyrLevelsUsed; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0, 0);
    fyi[level] = Ki[level](1, 1);
    cxi[level] = Ki[level](0, 2);
    cyi[level] = Ki[level](1, 2);
  }
}

} // namespace dso
