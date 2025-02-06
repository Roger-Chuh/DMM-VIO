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

#include "FullSystem/CoarseInitializer.h"
#include "../camera_model/camera_base.h"
#include "../camera_model/pinhole_camera.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "algs_tools_images_buffer.h"
#include "depth_filter_DSM.h"
#include "util/nanoflann.h"
#include <opencv2/highgui/highgui.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh, MultiCamera *p_cam)
    : thisToNext_aff(0, 0), thisToNext(SE3()) {
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    points[lvl] = 0;
    // numPoints[lvl] = 0;
  }
  Rwb = Mat33::Identity();
  JbBuffer =
      new Vec10f[ww * hh * kCameraNumUsed /* * kCameraNumUsed * kCameraNumUsed*/
  ]; // todo roger, align with host uv, rounding pixel
  JbBuffer_new =
      new Vec10f[ww * hh *
                 kCameraNumUsed /* * kCameraNumUsed * kCameraNumUsed*/];

  frameID = -1;
#if !defined(USE_MULTI_CAM) || defined(USE_ZNCC)
  fixAffine = true;
#else
  fixAffine = true;                                       // false;
#endif
  printDebug = false;
  //! 这是
  wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
  wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
  wM.diagonal()[6] = SCALE_A;
  wM.diagonal()[7] = SCALE_B;
  p_depth_filter_DSM_ = new DepthFilterDSM(p_cam, &estimator_config_);
}

CoarseInitializer::~CoarseInitializer() {
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    if (points[lvl] != 0)
      delete[] points[lvl];
  }

  delete[] JbBuffer;
  delete[] JbBuffer_new;
  delete p_depth_filter_DSM_;
}
#define ALWAYS_USE_IDP_PRIOR
bool CoarseInitializer::trackFrame(
    FrameHessian *newFrameHessian,
    std::vector<IOWrap::Output3DWrapper *> &wraps, const Mat33 &Rwb) {
  if (!Rwb.hasNaN() && false) {
    std::cout << "init Twb in init before:\n"
              << thisToNext.matrix3x4() << std::endl;
    thisToNext.setRotationMatrix(Rwb);
    std::cout << "Rwb:\n" << Rwb << std::endl;
    std::cout << "init Twb in init Rwb:\n"
              << thisToNext.matrix3x4() << std::endl;
  }
  newFrame = newFrameHessian;
  //[ ***step 1*** ] 先显示新来的帧
  // 新的一帧, 在跟踪之前显示的
  for (IOWrap::Output3DWrapper *ow : wraps)
    ow->pushLiveFrame(newFrameHessian);
#ifndef USE_MULTI_CAM
  // int maxIterations[] = {5, 5, 10, 30, 50, 50, 50, 50};
  int maxIterations[] = {10, 20, 50, 50, 50, 50, 50, 50}; // 不同层迭代的次数
  // int maxIterations[] = {50, 50, 50, 50, 50, 50, 50, 50}; // 不同层迭代的次数
#else
  int maxIterations[] = {10, 10, 10, 10, 10, 20, 20, 20}; // 不同层迭代的次数
#endif
//? 调参
#ifndef USE_ZNCC
  alphaK = 2.5 * 2.5; //*freeDebugParam1*freeDebugParam1;
  alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
#else
  alphaK =
      2.5 *
      2.5; // 2.5*2.5;//0.0150*0.0150;//0.005*0.005;//0.010*0.010;//0.0150*0.0150;//*freeDebugParam1*freeDebugParam1;
  alphaW = 150 * 150;       //*freeDebugParam2*freeDebugParam2;
#endif
  regWeight = 0.8;    //*freeDebugParam4;
  couplingWeight = 1; //*freeDebugParam5;
  //[ ***step 2*** ] 初始化每个点逆深度为1, 初始化光度参数, 位姿SE3
  /// if it's not snapped? what snapped mean? stored? successfully tracked?
  /// these initialization steps shows that snapped means established a stable
  /// tacking in that frame.
  /// ###########################
  /// Now I know, snapped is a flag returned by the tracker, if tracker
  /// successfully locked this frame,
  // TODO that means it was snapped, which is tracked. that's why if it's
  // tracked, all idepth and hessian stuff would be already available.
  // TODO if not tracked or no tracking successful, initialize all selected
  // point in this frame
  if (!snapped) //! snapped应该指的是位移足够大了，不够大就重新优化
  {
    // 初始化
    if (kCameraNumUsed == 1
#ifdef FIX_ZERO_TRANS_IN_INIT
        || true
#endif
    ) {
      thisToNext.translation().setZero();
    }
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        int npts = level_cid_to_numPoints[lvl][cid];
        Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
        for (int i = 0; i < npts; i++) {
          ptsl[i].v_energy_vec.clear();
          ptsl[i].energy_size = 0;
          ptsl[i].min_energy = ptsl[i].median_energy = 999999;
          ptsl[i].max_zncc = -999999;
          if (kCameraNumUsed == 1) {
            ptsl[i].iR = 1; // TODO 每个点的深度初值都赋1，hessian赋0
            ptsl[i].idepth_new = 1;
          } else {
            if (
#ifdef ALWAYS_USE_IDP_PRIOR
                true ||
#endif
                false) {
              ptsl[i].iR =
                  ptsl[i].iR_triangle; // TODO 每个点的深度初值都赋1，hessian赋0
              ptsl[i].idepth_new = ptsl[i].idepth_new_triangle;
              //            ptsl[i].idepth_new = ptsl[i].idepth_new_triangle =
              //                ptsl[i].iR_triangle;
            }
          }
          ptsl[i].lastHessian = 0;
          assert(ptsl[i].host_cid == cid);
          ptsl[i].host_cid = cid;
        }
      }
    }
  }

  SE3 refToNew_current = thisToNext;

  AffLight refToNew_aff_current = thisToNext_aff;
  // 如果都有仿射系数, 则估计一个初值
  if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
    refToNew_aff_current =
        AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure),
                 0); // coarse approximation.

  Vec3f latestRes = Vec3f::Zero();
  // 从顶层开始估计
  /// start from lowest resolution
  bool use_inner_loop = true;
  int inner_loop_start_lvl = use_inner_loop ? pyrLevelsUsed - 1 : 0;
  if (use_inner_loop) {
  }
  int lvl_target = 0;
  int max_iter = 10;
  for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {
    printf("---------------------------------------------\n");
    //[ ***step 3*** ] 使用计算过的上一层来初始化下一层
    // 顶层未初始化到, reset来完成
    if (lvl < pyrLevelsUsed - 1 /*&& kCameraNumUsed == 1*/) {
      /// from coarse image to fine image, hence "down"
      // TODO roger,
      // 用上一轮优化过的模糊的金字塔给当前较高分辨率的金字塔提供初值
      // 要在内循环外调用，因为一定要优化最充分的小图去预测大图
      propagateDown(lvl + 1);
    }

    Mat88f H, Hsc;
    Vec8f b, bsc;
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
      // TODO roger,
      // 要在内循环内调用，要及时更新当前host图上的iR作为优化的prior，以及重置isGood，energy，idepth_new这些标志位
      resetPoints(lvl); // 这里对顶层进行初始化!
      //[ ***step 4*** ] 迭代之前计算能量, Hessian等
      /// resOld = [energy snapped ptsnum]
      /// resOld = [能量值, ? , 使用的点的个数]

      Vec3f resOld = Vec3f::Zero();
      int N = 0;
      for (int l = 0; l < PYR_LEVELS; ++l) {
        // JbBuffer_new[l].setZero();
      }
      H.setZero();
      Hsc.setZero();
      b.setZero();
      bsc.setZero();
      //    for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
      //      for (int target_cid = 0; target_cid < kCameraNumUsed;
      //      ++target_cid)
      //      {
      resOld = calcResAndGS(-1, max_iter /*maxIterations[lvl]*/, lvl, H, b, Hsc,
                            bsc, refToNew_current, refToNew_aff_current, false,
                            N, lvl <= 0, lvl_target);
      //      }
      //    }
      applyStep(lvl); // 新的能量付给旧的

      float lambda = 0.1;
      float eps = 1e-4;
      int fails = 0;
      // 初始信息
      if (printDebug) {
        printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) "
               "(|inc| = %f)! \t",
               lvl, 0, lambda, "INITIA", sqrtf((float)(resOld[0] / resOld[2])),
               sqrtf((float)(resOld[1] / resOld[2])),
               sqrtf((float)(resOld[0] / resOld[2])),
               sqrtf((float)(resOld[1] / resOld[2])),
               (resOld[0] + resOld[1]) / resOld[2],
               (resOld[0] + resOld[1]) / resOld[2], 0.0f);
        std::cout << refToNew_current.log().transpose() << " AFF "
                  << refToNew_aff_current.vec().transpose() << "\n";
      }
      //[ ***step 5*** ] 迭代求解
      int iteration = 0;
      while (true) {
        //[ ***step 5.1*** ] 计算边缘化后的Hessian矩阵, 以及一些骚操作
        /// 吧idepth边缘化掉，剩下8维
        Mat88f Hl = H;
        /// lambda: dampping factor in L-M
        for (int i = 0; i < 8; i++)
          Hl(i, i) *= (1 + lambda); // 这不是LM么,论文说没用, 嘴硬
        // 舒尔补, 边缘化掉逆深度状态
        Hl -= Hsc * (1 / (1 + lambda));
        Vec8f bl =
            b -
            bsc * (1 / (1 + lambda)); // 因为dd必定是对角线上的, 所以也乘倒数
        //? wM为什么这么乘, 它对应着状态的SCALE
        //? (0.01f/(w[lvl]*h[lvl]))是为了减小数值, 更稳定?
        Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
        bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

        //[ ***step 5.2*** ] 求解增量
        Vec8f inc;
        SE3 refToNew_new;
        if (fixAffine) // 固定光度参数
        {
          // Note as we set the weights of rotation and translation to 1 the wM
          // is just the identity in this case.
          inc.head<6>() =
              -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
          inc.tail<2>().setZero();
        } else
          inc = -(wM * (Hl.ldlt().solve(bl))); //=-H^-1 * b.

        double incNorm = inc.head(6).norm();

        //[ ***step 5.3*** ] 更新状态, doStep中更新逆深度
        /// lifting
        refToNew_new =
            SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;

        AffLight refToNew_aff_new = refToNew_aff_current;
        refToNew_aff_new.a += inc[6];
        refToNew_aff_new.b += inc[7];
        std::cout << "inc: " << inc.transpose() << std::endl;
        doStep(lvl, lambda, inc);
        // std::cout << "inc: " << inc.transpose() << std::endl;
        // std::exit(1);
        //[ ***step 5.4*** ] 计算更新后的能量并且与旧的对比判断是否accept
        Mat88f H_new, Hsc_new;
        Vec8f b_new, bsc_new;
        Vec3f resNew = Vec3f::Zero();
        int N2 = 0;
        H_new.setZero();
        Hsc_new.setZero();
        b_new.setZero();
        bsc_new.setZero();
        //      for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
        //        for (int target_cid = 0; target_cid < kCameraNumUsed;
        //        ++target_cid) {
        resNew =
            calcResAndGS(iteration, max_iter /*maxIterations[lvl]*/, lvl, H_new,
                         b_new, Hsc_new, bsc_new, refToNew_new,
                         refToNew_aff_new, false, N2, lvl <= 0, lvl_target);
        //        }
        //      }
        Vec3f regEnergy = calcEC(lvl);

        //      std::cout << "resNew: " << resNew.transpose()
        //                << ", regEnergy: " << regEnergy.transpose() <<
        //                std::endl;

        float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
        float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

        bool accept = eTotalOld > eTotalNew;

        printf(
            "accept: %d, level: %d, [eTotalOld / eTotalNew]: [%f / %f], diff: "
            "%f, diff_ratio: %f, iter: %d, level: %d, incNorm: %f, lambda: "
            "%f. [lvl_h / lvl_t]: [%d %d]\n",
            accept, lvl, eTotalOld, eTotalNew, eTotalNew - eTotalOld,
            (eTotalNew - eTotalOld) / eTotalOld, iteration, lvl, incNorm,
            lambda, lvl, lvl_target);
        if (printDebug) {
          printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + "
                 "%.5f (%.2f->%.2f) (|inc| = %f)! \t",
                 lvl, iteration, lambda, (accept ? "ACCEPT" : "REJECT"),
                 sqrtf((float)(resOld[0] / resOld[2])),
                 sqrtf((float)(regEnergy[0] / regEnergy[2])),
                 sqrtf((float)(resOld[1] / resOld[2])),
                 sqrtf((float)(resNew[0] / resNew[2])),
                 sqrtf((float)(regEnergy[1] / regEnergy[2])),
                 sqrtf((float)(resNew[1] / resNew[2])), eTotalOld / resNew[2],
                 eTotalNew / resNew[2], incNorm);
          std::cout << refToNew_new.log().transpose() << " AFF "
                    << refToNew_aff_new.vec().transpose() << "\n";
        }
        //[ ***step 5.5*** ] 接受的话, 更新状态,; 不接受则增大lambda
        bool quitOpt = false;
        if (accept) {
          int point_count = 0;
          for (int id = 0; id < kCameraNumUsed; ++id) {
            point_count += level_cid_to_numPoints[lvl][id];
          }
          //        printf("alphaK: %f, level_cid_to_numPoints[lvl]: %d,
          //        resNew[1]: %f\n",
          //               alphaK, point_count, resNew[1]);

          //? 这是啥   答：应该是位移足够大，才开始优化IR
          if (
#if 1 // ndef ALWAYS_USE_IDP_PRIOR
                  kCameraNumUsed > 1 ||
#endif
                  resNew[1] ==
                                            alphaK * static_cast<float>(point_count)
                      /*(level_cid_to_numPoints[lvl][kCameraNumUsed - 1] +
                       level_cid_to_npts_success_offset[lvl][kCameraNumUsed -
                                                             1])*/) { // 当 alphaEnergy
            // > alphaK*npts
            printf("##################################################### "
                   "SNAPPED!!! ###########################\n");
            snapped = true;
          } else {
          }
          H = H_new;
          b = b_new;
          Hsc = Hsc_new;
          bsc = bsc_new;
          resOld = resNew;
          refToNew_aff_current = refToNew_aff_new;
          refToNew_current = refToNew_new;
          applyStep(lvl);
          if (kCameraNumUsed == 1 || true) {
            optReg(lvl); // 更新iR
          }
          lambda *= 0.5;
          fails = 0;
          if (lambda < 0.000001) {
            lambda = 0.000001;
          }
        } else {
          fails++;
          if (fails < 2) {
            lambda *= 4;
          } else {
            lambda *= 4;
          }
          if (lambda > 10000) {
            lambda = 10000;
            quitOpt = true;
          }
        }

        debugPlot(lvl, wraps, refToNew_current, true);
        // bool quitOpt = false;
        // 迭代停止条件, 收敛/大于最大次数/失败2次以上
        if (!(incNorm > eps) || iteration >= max_iter /*maxIterations[lvl]*/ ||
            fails >= 3 /*200*/) {
          Mat88f H, Hsc;
          Vec8f b, bsc;

          quitOpt = true;
        }

        if (quitOpt)
          break;
        iteration++;
      }
      latestRes = resOld;
    }
  }

  //[ ***step 6*** ] 优化后赋值位姿, 从底层计算上层点的深度
  std::cout << "refToNew_current: \n"
            << refToNew_current.matrix3x4() << std::endl;
  thisToNext = refToNew_current;
  thisToNext_aff = refToNew_aff_current;
#if 1 // ndef USE_MULTI_CAM
  for (int i = 0; i < pyrLevelsUsed - 1; i++) {
    propagateUp(i);
  }
#endif
  frameID++;
  if (!snapped)
    snappedAt = 0;

  if (snapped && snappedAt == 0)
    snappedAt = frameID; // 位移足够的帧数

  debugPlot(0, wraps, thisToNext, false);

  // 位移足够大, 再优化5帧才行
  if (kCameraNumUsed == 1) {
    return snapped && frameID > snappedAt + 5;
  } else {
    // snapped = true;
    return snapped && frameID >= snappedAt + 1; // 0;
  }
}

void CoarseInitializer::debugPlot(int lvl,
                                  std::vector<IOWrap::Output3DWrapper *> &wraps,
                                  SE3 T_th, bool show_details) {
  bool needCall = false;
  if (!show_details) {
    for (IOWrap::Output3DWrapper *ow : wraps)
      needCall = needCall || ow->needPushDepthImage();
    if (!needCall)
      return;
  }
  int wl = w[lvl], hl = h[lvl];
  if (!show_details) {
    MinimalImageB3 iRImg(wl, hl);

    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      Eigen::Vector3f *colorRef =
          firstFrame->dIp[lvl] + hG[lvl] * wG[lvl] * cid;
      for (int i = 0; i < wl * hl; i++)
        iRImg.at(i, cid) =
            Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);
    }
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      int npts = level_cid_to_numPoints[lvl][cid];

      float nid = 0, sid = 0;
      for (int i = 0; i < npts; i++) {
        Pnt *point =
            points[lvl] + i + level_cid_to_npts_success_offset[lvl][cid];
        if (point->isGood) {
          nid++;
#ifndef USE_MULTI_CAM
          sid += point->iR;
#else
          sid += point->iR; // point->idepth;
#endif
        }
      }
      float fac = nid / sid;

      for (int i = 0; i < npts; i++) {
        Pnt *point =
            points[lvl] + level_cid_to_npts_success_offset[lvl][cid] + i;

        if (!point->isGood) {
          iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0),
                          point->host_cid);
        } else {
          // printf("good point\n");
          iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f,
#ifndef USE_MULTI_CAM
                          makeRainbow3B(point->iR * fac),
#else
                          makeRainbow3B(point->iR /*point->idepth*/ * fac),
#endif
                          point->host_cid);
        }
      }
    }
    // IOWrap::displayImage("idepth-R", &iRImg, false);
    for (IOWrap::Output3DWrapper *ow : wraps) {
      ow->pushDepthImage(&iRImg);
    }
  }

  if (true) {
    Mat3 intr = Mat3::Identity();
    intr(0, 0) = fx[lvl];
    intr(1, 1) = fy[lvl];
    intr(0, 2) = cx[lvl];
    intr(1, 2) = cy[lvl];
    Mat3 intr_inv = intr.inverse();

    MinimalImageB3 *img_host;
    MinimalImageB3 *img_target;

    img_host = new MinimalImageB3(wG[lvl], hG[lvl]);
    img_target = new MinimalImageB3(wG[lvl], hG[lvl]);

    for (int cam = 0; cam < kCameraNumUsed; ++cam) {
      Vec3f *colorRef = firstFrame->dIp[lvl] + wG[lvl] * hG[lvl] * cam;
      for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
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
      Eigen::Vector3f *colorCur = newFrame->dIp[lvl] + cam * wG[lvl] * hG[lvl];
      for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(colorCur + i))[0];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        img_target->at(i, cam) = Vec3b(colL, colL, colL);
      }
    }
    // IOWrap::displayImage("host", img_host);
    // IOWrap::displayImage("target", img_target);
    // IOWrap::waitKey(0);
    /////////////////////////////////////////////////////////////////////
    for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
      float nid = 0, sid = 0;
      std::array<float, kCameraNumUsed> target_cid_to_nid;
      std::array<float, kCameraNumUsed> target_cid_to_sid;
      for (int id = 0; id < kCameraNumUsed; ++id) {
        //          target_cid_to_nid[id] = 0;
        //          target_cid_to_sid[id] = 0;
      }
      for (int i = 0; i < level_cid_to_numPoints[lvl][host_cid]; i++) {
        Pnt *point =
            points[lvl] + i + level_cid_to_npts_success_offset[lvl][host_cid];
        if (point->isGood) {
          nid += 1;
#ifndef USE_MULTI_CAM
          sid += point->iR;
#else
          sid += point->iR; // point->idepth;
                            // sid += point->idepth;
#endif
        }
      }
      float fac = nid / sid;
      std::array<float, kCameraNumUsed> target_cid_to_fac;
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        target_cid_to_nid[target_cid] = 0;
        target_cid_to_sid[target_cid] = 0;
        SE3 Tth;
        if (!show_details) {
          Tth = thisToNext;
        } else {
          Tth = T_th;
        }
        SE3 refToNew =
            newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
            Tth * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
        Mat33f RKi = (refToNew.rotationMatrix() * intr_inv).cast<float>();
        for (int i = 0; i < level_cid_to_numPoints[lvl][host_cid]; i++) {
          Pnt *point_temp =
              points[lvl] + i + level_cid_to_npts_success_offset[lvl][host_cid];
          // TODO roger, like SetFromImage in orca, 判断这个坐标纹理是否充分
          if (!point_temp->isGood) {
            continue;
          }

          Vec3f t = refToNew.translation().cast<float>();
          Vec3f pt = RKi * Vec3f(point_temp->u, point_temp->v, 1) +
                     t * point_temp->idepth;
          // std::cout << "point->idepth: " << point->idepth << std::endl;
          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          float new_idepth = point_temp->idepth / pt[2];
          // 像素坐标pj
          float Ku = float(intr(0, 0)) * u + float(intr(0, 2));
          float Kv = float(intr(1, 1)) * v + float(intr(1, 2));
          //                std::cout << "init, "
          //                          << ", u: " << Ku << ", v: " << Kv
          //                          << ", idepth: " << new_idepth <<
          //                          std::endl;
          if (!(Ku > 15 && Kv > 15 && Ku < wG[lvl] - 15 && Kv < hG[lvl] - 15 &&
                new_idepth > 0)) {
            //                isGood = false;
            //                break;
            continue;
          }
          target_cid_to_nid[target_cid] += 1;
          target_cid_to_sid[target_cid] += new_idepth;
        }
        target_cid_to_fac[target_cid] =
            target_cid_to_nid[target_cid] / target_cid_to_sid[target_cid];
      }
      for (int i = 0; i < level_cid_to_numPoints[lvl][host_cid]; i++) {
        Pnt *point =
            points[lvl] + i + level_cid_to_npts_success_offset[lvl][host_cid];
        // TODO roger, like SetFromImage in orca, 判断这个坐标纹理是否充分
        if (!point->isGood) {
          continue;
        }
        if (point->isGood) {
          img_host->setPixel9(point->u + 0.5, point->v + 0.5,
                              makeRainbow3B(point->iR /*point->idepth*/ * fac),
                              host_cid);
          img_host->setPixelCirc(
              point->u + 0.5, point->v + 0.5,
              makeRainbow3B(point->iR /*point->idepth*/ * fac), host_cid);
        } else {
          img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(10),
                              host_cid);
          img_host->setPixelCirc(point->u + 0.5, point->v + 0.5,
                                 makeRainbow3B(10), host_cid);
        }

        for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
          SE3 Tth;
          if (!show_details) {
            Tth = thisToNext;
          } else {
            Tth = T_th;
          }
          SE3 refToNew =
              newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
              Tth * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
          Mat33f RKi = (refToNew.rotationMatrix() * intr_inv).cast<float>();
          Vec3f t = refToNew.translation().cast<float>();
          Vec3f pt = RKi * Vec3f(point->u, point->v, 1) + t * point->idepth;
          // std::cout << "point->idepth: " << point->idepth << std::endl;
          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          float new_idepth = point->idepth / pt[2];
          // 像素坐标pj
          float Ku = float(intr(0, 0)) * u + float(intr(0, 2));
          float Kv = float(intr(1, 1)) * v + float(intr(1, 2));
          //                std::cout << "init, "
          //                          << ", u: " << Ku << ", v: " << Kv
          //                          << ", idepth: " << new_idepth <<
          //                          std::endl;
          if (!(Ku > 15 && Kv > 15 && Ku < wG[lvl] - 15 && Kv < hG[lvl] - 15 &&
                new_idepth > 0)) {
            //                isGood = false;
            //                break;
            continue;
          }
          if (point->isGood && point->is_valid_project[target_cid]) {
            img_target->setPixel9(
                Ku + 0.5, Kv + 0.5,
                makeRainbow3B(new_idepth * target_cid_to_fac[target_cid]),
                target_cid);
            img_target->setPixelCirc(
                Ku + 0.5, Kv + 0.5,
                makeRainbow3B(new_idepth * target_cid_to_fac[target_cid]),
                target_cid);
          } else {
            if (false) {
              img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(10),
                                    target_cid);
            }
            //            img_target->setPixelCirc(Ku + 0.5, Kv + 0.5,
            //            makeRainbow3B(10),
            //                                     target_cid);
          }
        }
      }
    }
    IOWrap::displayImage("host", img_host, true);
    IOWrap::displayImage("target", img_target, true);
    if (show_details) {
      IOWrap::waitKey(lvl == 0 ? 1000 : 1);
    } else {
      IOWrap::waitKey(1);
    }
    delete img_host;
    delete img_target;
  }
}

//* 计算能量函数和Hessian矩阵, 以及舒尔补, sc代表Schur
// calculates residual, Hessian and Hessian-block neede for re-substituting
// depth.
#if 0
Vec3f CoarseInitializer::calcResAndGS_bak(int lvl, MatStatef &H_out,
                                          VecStatef &b_out, MatStatef &H_out_sc,
                                          VecStatef &b_out_sc,
                                          const SE3 &refToNew_,
                                          AffLight refToNew_aff, bool plot,
                                          int &N, bool show_image) {
  int wl = w[lvl], hl = h[lvl];
  // 当前层图像及梯度
  Accumulator11 E;   // 1*1 的累加器
  acc9.initialize(); // 初始值, 分配空间
  E.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
      const Mat66 &extra_pose_jac =
          newFrame->p_multi_camera->cid_to_T01_inv_Adj[target_cid];
      Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl * hl;
      Eigen::Vector3f *colorNew = newFrame->dIp[lvl] + target_cid * wl * hl;
      //! 旋转矩阵R * 内参矩阵K_inv
      SE3 refToNew =
          newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
          refToNew_ * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
      Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
      Vec3f t = refToNew.translation().cast<float>();
      Eigen::Vector2f r2new_aff =
          Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);
      // 该层的相机参数
      float fxl = fx[lvl];
      float fyl = fy[lvl];
      float fxli = 1 / fx[lvl];
      float fyli = 1 / fy[lvl];
      float cxl = cx[lvl];
      float cyl = cy[lvl];

      //  Accumulator11 E;   // 1*1 的累加器
      //  acc9.initialize(); // 初始值, 分配空间
      //  E.initialize();

      int npts = level_cid_to_numPoints[lvl][host_cid];
      Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][host_cid];
      for (int i = 0; i < npts; i++) {
        show_image = host_cid == 2 && lvl == 0 && target_cid == 3 && i > 200;
        /// ptsl + i same as ptsl[i];
        Pnt *point = ptsl + i;
        assert(point->host_cid == host_cid);
        if (point->host_cid != host_cid) {
          //  continue;
        }
        point->maxstep = 1e10;
        if (!point->isGood) // 点不好
        {
          E.updateSingle((float)(point->energy[0])); // 累加
          point->energy_new = point->energy;
          point->isGood_new = false;
          continue;
        }

        /// VecNeighbourResidualFloat
        /// dp here 0-5 is d_residual / d_SE3, 6-7 is d_residual / d_a and
        /// d_residual / d_b 6dof pose residual
        // TODO !< 用来计算Schur的 0-7: sum(dd * dp). 8: sum(res*dd). 9:
        // 1/(1+sum(dd*dd))=inverse hessian entry
        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        /// affine residual x 2
        VecNRf dp6;
        VecNRf dp7;

        VecNRf dd;
        VecNRf r;
        // todo roger,
        // 不同level共用这个JbBuffer，所以开辟空间肯定要按最大的level0去开辟
        JbBuffer_new[i + h[0] * w[0] * host_cid].setZero(); // 10*1 向量

        // sum over all residuals.
        bool isGood = true;
        float energy = 0;
        Eigen::Matrix<float, 2, 6> d_uv_d_pose, d_uv_d_pose_inverse_comp,
            d_uv_d_pose_fwd_jac;
        Eigen::Matrix<float, 2, 3> d_uv_d_pt3d, d_uv_host_d_n_host,
            d_uv_target_d_x_target_scaled;
        Eigen::Matrix<float, 3, 6> d_pt3d_d_pose, d_pt3d_d_pose_inverse_comp;
        Eigen::Matrix<float, 3, 3> Rot;
        Eigen::Matrix<float, 3, 1> trans;
        Vec4f d_C_x, d_C_y;
        float drescale;
        int show_cnt = 0;

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

#ifdef SHOW_INIT_IMAGE

        MinimalImageB3 *img_host;
        MinimalImageB3 *img_target;
        if (show_image) {
          img_host = new MinimalImageB3(wG[lvl], hG[lvl]);
          img_target = new MinimalImageB3(wG[lvl], hG[lvl]);

          for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
            // BRIGHTNESS TRANSFER
            float colL = (*(colorRef + i))[0];
            if (colL < 0)
              colL = 0;
            if (colL > 255)
              colL = 255;
            img_host->at(i, host_cid) = Vec3b(colL, colL, colL);
            colL = (*(colorNew + i))[0];
            if (colL < 0)
              colL = 0;
            if (colL > 255)
              colL = 255;
            img_target->at(i, target_cid) = Vec3b(colL, colL, colL);
          }

          img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                              host_cid);
        }

#endif

        int count = 0;
        for (int idx = 0; idx < patternNum; idx++) {
          // pattern的坐标偏移
          int dx = patternP[idx][0];
          int dy = patternP[idx][1];

          //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
          /// Pj = [x y z]
          /// Pj' * Z = Pj
          /// Pj' = [x/z y/z 1]
          Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) +
                     t * point->idepth_new;

          // Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() *
          // Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

          Vec3f n_host = Vec3f((point->u + dx - cxl) * fxli,
                               (point->v + dy - cyl) * fyli, 1);
          Rot = refToNew.rotationMatrix().cast<float>();
          trans = refToNew.translation().cast<float>();
          Vec3f X_target_scaled =
              refToNew.rotationMatrix().cast<float>() * n_host +
              refToNew.translation().cast<float>() * point->idepth_new;
          Mat33f X_target_scaled_skew;
          X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
              X_target_scaled(2), (0), -X_target_scaled(0), -X_target_scaled(1),
              X_target_scaled(0), (0);
          d_uv_d_pt3d << fxl / X_target_scaled(2), 0,
              -fxl * X_target_scaled(0) / X_target_scaled(2) /
                  X_target_scaled(2),
              0, fyl / X_target_scaled(2),
              -fyl * X_target_scaled(1) / X_target_scaled(2) /
                  X_target_scaled(2);
          d_pt3d_d_pose.leftCols(3) = point->idepth_new * Mat33f::Identity();
          d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
          d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
          d_uv_host_d_n_host << fxl, 0, -fxl * n_host(0), 0, fyl,
              -fyl * n_host(1);
          d_uv_d_pose_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              d_pt3d_d_pose;

          Vec2f d_uv_d_d_inverse_comp;
          d_uv_d_d_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              refToNew.translation().cast<float>();

          d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
          d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;

          Vec2f d_uv_d_d_fwd_jac;
          d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled *
                             refToNew.translation().cast<float>();
          // printf("xyz: %f\n",pt[2]);
          // 归一化坐标 Pj

          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          // 像素坐标pj
          float Ku = fxl * u + cxl;
          float Kv = fyl * v + cyl;

#ifdef SHOW_INIT_IMAGE
          if (show_image) {
            img_target->setPixel9(Ku, Kv, makeRainbow3B(1), target_cid);
          }
#endif

          // dpi/pz'
          /// 这2个相除应该没什么几何含义，相当于rou1/rou2吧，为了计算雅可比的
          /// idepth_new is the estimated z in host frame, and pt[2] is
          /// projected z in new frame.
          float new_idepth = point->idepth_new / pt[2];
          // 落在边缘附近，深度小于0, 则不好
          if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 &&
                new_idepth > 0)) {
            //                isGood = false;
            //                break;
            continue;
          }
          // 插值得到新图像中的 patch 像素值，(输入3维，输出3维像素值 +
          // x方向梯度 + y方向梯度)
          Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
          Vec3f hostColor = getInterpolatedElement33(colorRef, point->u + dx,
                                                     point->v + dy, wl);
          // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv,
          // wl);
          float host_value_corrected =
              (float)(r2new_aff[0] * hostColor[0] + r2new_aff[1]);

          // 参考帧上的 patch 上的像素值, 输出一维像素值
          // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
          float rlR = getInterpolatedElement31(colorRef, point->u + dx,
                                               point->v + dy, wl);
          // 像素值有穷, good
          if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
            //                isGood = false;
            //                break;
            continue;
          }
          hostColor[0] = host_value_corrected;
          host_info.conservativeResize(count + 1, 3);
          target_info.conservativeResize(count + 1, 3);
          host_info.row(count) = hostColor.transpose();
          target_info.row(count) = hitColor.transpose();
          count++;
        }

#ifdef SHOW_INIT_IMAGE
        //    std::cout << "idx: " << idx << ", hostColor: " <<
        //    hostColor.transpose()
        //              << ", hitColor: " << hitColor.transpose()
        //              << ", affLL: " << affLL.transpose()
        //              << ", color[idx]: " << color[idx] << std::endl;
        if (show_image) {
          IOWrap::displayImage("host", img_host);
          IOWrap::displayImage("target", img_target);
          IOWrap::waitKey(0);

          delete img_host;
          delete img_target;
        }
#endif

        int patch_num = host_info.rows();
        if (patch_num != 0) {

          host_val_mean = host_info.col(0).sum() / patch_num;
          target_val_mean = target_info.col(0).sum() / patch_num;

          ones.conservativeResize(patch_num, 1);
          ones.setOnes();
#ifdef USE_ZNCC
          host_info.col(0) = host_info.col(0) - host_val_mean * ones;
          target_info.col(0) = target_info.col(0) - target_val_mean * ones;
          host_sigma = host_info.col(0).norm();
          target_sigma = target_info.col(0).norm();
          host_info.col(0) /= host_sigma;
          target_info.col(0) /= target_sigma;

          Mat_ZNSSD_I.conservativeResize(patch_num, patch_num);
          Mat_ZNSSD_I.setIdentity();

          J_ZNSSD_mean = Mat_ZNSSD_I - (ones / static_cast<float>(patch_num)) *
                                           ones.transpose();

          J_ZNSSD_J_I_host = setting_variableScale *
                             ((Mat_ZNSSD_I - (host_info.col(0) *
                                              host_info.col(0).transpose())) /
                              host_sigma * J_ZNSSD_mean);
          J_ZNSSD_J_I_target =
              setting_variableScale *
              ((Mat_ZNSSD_I -
                (target_info.col(0) * target_info.col(0).transpose())) /
               target_sigma * J_ZNSSD_mean);

          grad_new_host =
              J_ZNSSD_J_I_host * host_info.rightCols(2); // "new" gradient: 8x2
          grad_new_target = J_ZNSSD_J_I_target *
                            target_info.rightCols(2); // "new" gradient: 8x2
          host_info.col(0) *= setting_variableScale;
          target_info.col(0) *= setting_variableScale;
#else
          grad_new_host = host_info.rightCols(2);     // "new" gradient: 8x2
          grad_new_target = target_info.rightCols(2); // "new" gradient: 8x2
#endif
          //            std::cout << "init, grad_new_host: \n" << grad_new_host
          //            << std::endl; std::cout << "init, grad_new_target: \n"
          //            << grad_new_target << std::endl;
          //
          //            std::cout << "init, grad_old_host: \n" <<
          //            host_info.rightCols(2) << std::endl; std::cout << "init,
          //            grad_old_target: \n" << target_info.rightCols(2) <<
          //            std::endl;
        }

        int cnt = 0;
        for (int idx = 0; idx < patternNum; idx++) {
          int dx = patternP[idx][0];
          int dy = patternP[idx][1];

          //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
          /// Pj = [x y z]
          /// Pj' * Z = Pj
          /// Pj' = [x/z y/z 1]
          Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) +
                     t * point->idepth_new;

          // Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() *
          // Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

          Vec3f n_host = Vec3f((point->u + dx - cxl) * fxli,
                               (point->v + dy - cyl) * fyli, 1);
          Rot = refToNew.rotationMatrix().cast<float>();
          trans = refToNew.translation().cast<float>();
          Vec3f X_target_scaled =
              refToNew.rotationMatrix().cast<float>() * n_host +
              refToNew.translation().cast<float>() * point->idepth_new;
          Mat33f X_target_scaled_skew;
          X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
              X_target_scaled(2), (0), -X_target_scaled(0), -X_target_scaled(1),
              X_target_scaled(0), (0);
          d_uv_d_pt3d << fxl / X_target_scaled(2), 0,
              -fxl * X_target_scaled(0) / X_target_scaled(2) /
                  X_target_scaled(2),
              0, fyl / X_target_scaled(2),
              -fyl * X_target_scaled(1) / X_target_scaled(2) /
                  X_target_scaled(2);
          d_pt3d_d_pose.leftCols(3) =
              point->idepth_new * Mat33f::Identity(); /// [t R]
          d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
          d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
          d_uv_host_d_n_host << fxl, 0, -fxl * n_host(0), 0, fyl,
              -fyl * n_host(1);
          d_uv_d_pose_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              d_pt3d_d_pose;

          Vec2f d_uv_d_d_inverse_comp;
          d_uv_d_d_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              refToNew.translation().cast<float>();

          d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
          d_uv_d_pose_fwd_jac = (d_uv_target_d_x_target_scaled.cast<double>() *
                                 d_pt3d_d_pose.cast<double>())
                                    .cast<float>();

          Mat26f d_uv_d_pose_fwd_jac_use =
              (d_uv_target_d_x_target_scaled.cast<double>() *
               d_pt3d_d_pose.cast<double>() * extra_pose_jac)
                  .cast<float>();

          Vec2f d_uv_d_d_fwd_jac;
          d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled *
                             refToNew.translation().cast<float>();

          Mat26f d_uv_d_pose_inverse_comp_use =
              (d_uv_host_d_n_host.cast<double>() *
               refToNew.rotationMatrix().transpose() *
               d_pt3d_d_pose.cast<double>() * extra_pose_jac)
                  .cast<float>();
          // printf("xyz: %f\n",pt[2]);
          // 归一化坐标 Pj

          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          float Ku = fxl * u + cxl;
          float Kv = fyl * v + cyl;
          // dpi/pz'
          /// 这2个相除应该没什么几何含义，相当于rou1/rou2吧，为了计算雅可比的
          /// idepth_new is the estimated z in host frame, and pt[2] is
          /// projected z in new frame.
          float new_idepth = point->idepth_new / pt[2];
#if 0
                        drescale = 1/X_target_scaled(2);
                d_C_x[2] = drescale*(Rot(2,0)*u-Rot(0,0));
                d_C_x[3] = fxl * drescale*(Rot(2,1)*u-Rot(0,1)) * fyli;
                //TODO KliP: host帧归一化坐标
                d_C_x[0] = n_host[0]*d_C_x[2];
                d_C_x[1] = n_host[1]*d_C_x[3];

                d_C_y[2] = fyl * drescale*(Rot(2,0)*v-Rot(1,0)) * fxli;
                d_C_y[3] = drescale*(Rot(2,1)*v-Rot(1,1));
                d_C_y[0] = n_host[0]*d_C_y[2];
                d_C_y[1] = n_host[1]*d_C_y[3];

                d_C_x[0] = (d_C_x[0]+u);//TODO d_u2_d_fx
                d_C_x[1] *= 1;
                d_C_x[2] = (d_C_x[2]+1);//TODO d_u2_d_cx
                d_C_x[3] *= 1;

                d_C_y[0] *= 1;
                d_C_y[1] = (d_C_y[1]+v)*1;
                d_C_y[2] *= 1;
                d_C_y[3] = (d_C_y[3]+1)*1;
#endif
          // 落在边缘附近，深度小于0, 则不好
          if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 &&
                new_idepth > 0)) {
            isGood = false;
            break;
          }
          // 插值得到新图像中的 patch 像素值，(输入3维，输出3维像素值 +
          // x方向梯度 + y方向梯度)
          Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
          Vec3f hostColor = getInterpolatedElement33(colorRef, point->u + dx,
                                                     point->v + dy, wl);
          // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv,
          // wl);

          // 参考帧上的 patch 上的像素值, 输出一维像素值
          // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
          float rlR = getInterpolatedElement31(colorRef, point->u + dx,
                                               point->v + dy, wl);

          // 像素值有穷, good
          if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
            isGood = false;
            break;
          }

#ifndef USE_ZNCC
          // 残差
          float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
          // Huber权重
          float hw = fabs(residual) < setting_huberTH
                         ? 1
                         : setting_huberTH / fabs(residual);
          // huberweight * (2-huberweight) = Objective Function
          // robust 权重和函数之间的关系
          energy += hw * residual * residual * (2 - hw);
#else
          float residual_bak = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
          float residual = 1 * (target_info(cnt, 0) - host_info(cnt, 0));
          // printf("residual: %f\n", residual);

          if (std::isnan(residual)) {
            isGood = false;
            break;
          }

          // printf("residual: %f\n", residual);
          float hw = fabs(residual) < setting_huberTH
                         ? 1
                         : setting_huberTH / fabs(residual);
          energy += hw * residual * residual * (2 - hw);
#endif

          // Pj 对 逆深度 di 求导
          //! 1/Pz * (tx - u*tz), u = px/pz
          float dxdd = (t[0] - t[2] * u) / pt[2];
          //! 1/Pz * (ty - v*tz), u = py/pz
          float dydd = (t[1] - t[2] * v) / pt[2];

          if (hw < 1)
            hw = sqrtf(hw); //?? 为啥开根号, 答: 鲁棒核函数等价于加权最小二乘
#ifndef USE_ZNCC
          //! dxfx, dyfy
          float dxInterp = hw * hitColor[1] * fxl;
          float dyInterp = hw * hitColor[2] * fyl;
#else
          float dxInterp = hw * grad_new_target(cnt, 0) * fxl;
          float dyInterp = hw * grad_new_target(cnt, 1) * fyl;
#endif
          // TODO* 残差对 j(新状态) 位姿求导, 6

          Eigen::Matrix<float, 3, 6> show;
          show(0, 0) = new_idepth * dxInterp;
          show(0, 1) = new_idepth * dyInterp;
          show(0, 2) = -new_idepth * (u * dxInterp + v * dyInterp);
          show(0, 3) = -u * v * dxInterp - (1 + v * v) * dyInterp;
          show(0, 4) = (1 + u * u) * dxInterp + u * v * dyInterp;
          show(0, 5) = -v * dxInterp + u * dyInterp;

#ifndef USE_ZNCC
          show.row(1) =
              show.row(0) -
              hw * Vec2f(hitColor[1], hitColor[2]).transpose() * d_uv_d_pose;
          Vec6f d_res_d_pose_inverse_comp =
              hw * Vec2f(hostColor[1], hostColor[2]).transpose() *
              d_uv_d_pose_inverse_comp;
          Vec6f d_res_d_pose_fwd_jac =
              hw *
              Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1))
                  .transpose() *
              d_uv_d_pose_fwd_jac;
          show.row(2) = show.row(0) - d_res_d_pose_fwd_jac.transpose();
          Vec6f d_res_d_pose_fwd_jac_use =
              (static_cast<double>(hw) *
               Vec2(hitColor[1], hitColor[2]).transpose() *
               d_uv_d_pose_fwd_jac_use.cast<double>())
                  .cast<float>();
          Vec6f d_res_d_pose_inverse_comp_use =
              (static_cast<double>(hw) *
               Vec2(hostColor[1], hostColor[2]).transpose() *
               d_uv_d_pose_inverse_comp_use.cast<double>())
                  .cast<float>();
          assert(std::abs(hostColor[1] - grad_new_host(cnt, 0)) == 0);
          assert(std::abs(hostColor[2] - grad_new_host(cnt, 1)) == 0);
#else
          show.row(1) = show.row(0) - hw *
                                          Vec2f(grad_new_target(cnt, 0),
                                                grad_new_target(cnt, 1))
                                              .transpose() *
                                          d_uv_d_pose;
          Vec6f d_res_d_pose_inverse_comp =
              hw *
              Vec2f(grad_new_host(cnt, 0), grad_new_host(cnt, 1)).transpose() *
              d_uv_d_pose_inverse_comp;
          Vec6f d_res_d_pose_fwd_jac =
              hw *
              Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1))
                  .transpose() *
              d_uv_d_pose_fwd_jac;
          Vec6f d_res_d_pose_fwd_jac_use =
              (static_cast<double>(hw) *
               Vec2(grad_new_target(cnt, 0), grad_new_target(cnt, 1))
                   .transpose() *
               d_uv_d_pose_fwd_jac_use.cast<double>())
                  .cast<float>();
          Vec6f d_res_d_pose_inverse_comp_use =
              (static_cast<double>(hw) *
               Vec2(grad_new_host(cnt, 0), grad_new_host(cnt, 1)).transpose() *
               d_uv_d_pose_inverse_comp_use.cast<double>())
                  .cast<float>();
#endif

          Vec3f d_uv_d_idp_show;
          d_uv_d_idp_show(0) = dxInterp * dxdd + dyInterp * dydd;
#ifndef USE_ZNCC
          d_uv_d_idp_show(1) = d_uv_d_idp_show(0) -
                               hw *
                                   Vec2f(hitColor[1], hitColor[2]).transpose() *
                                   d_uv_d_pt3d * trans;
          // std::cout<<"d_uv_d_c_show:\n"<<d_uv_d_c_show<<std::endl;
          float d_res_d_idp_inverse_comp =
              hw * Vec2f(hostColor[1], hostColor[2]).transpose() *
              d_uv_d_d_inverse_comp;
          float d_res_d_idp_fwd_jac =
              hw *
              Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1))
                  .transpose() *
              d_uv_d_d_fwd_jac;
          d_uv_d_idp_show(2) = d_uv_d_idp_show(0) - d_res_d_idp_fwd_jac;
#else
          d_uv_d_idp_show(1) =
              d_uv_d_idp_show(0) -
              hw *
                  Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1))
                      .transpose() *
                  d_uv_d_pt3d * trans;
          float d_res_d_idp_inverse_comp =
              hw *
              Vec2f(grad_new_host(cnt, 0), grad_new_host(cnt, 1)).transpose() *
              d_uv_d_d_inverse_comp;
          float d_res_d_idp_fwd_jac =
              hw *
              Vec2f(grad_new_target(cnt, 0), grad_new_target(cnt, 1))
                  .transpose() *
              d_uv_d_d_fwd_jac;
#endif

          if (show_cnt < -10) {
            std::cout << "[du dv hw]: [" << dx << ", " << dy << ", " << hw
                      << "], tracking pose jac diff: \n"
                      << show << std::endl;
            std::cout << "d_uv_d_idp_show: \n"
                      << d_uv_d_idp_show.transpose() << std::endl
                      << std::endl
                      << std::endl;
            show_cnt++;
          }

#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC

          dp0[idx] = new_idepth * dxInterp; //! dpi/pz' * dxfx
          dp1[idx] = new_idepth * dyInterp; //! dpi/pz' * dyfy
          dp2[idx] =
              -new_idepth *
              (u * dxInterp +
               v * dyInterp); //! -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
          dp3[idx] =
              -u * v * dxInterp -
              (1 + v * v) *
                  dyInterp; //! - px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy
          dp4[idx] =
              (1 + u * u) * dxInterp +
              u * v * dyInterp; //! (1+px'^2/pz'^2)*dxfx + px'py'/pz'^2*dxfy
          dp5[idx] =
              -v * dxInterp + u * dyInterp; //! -py'/pz'*dxfx + px'/pz'*dyfy

          dp0[idx] = d_res_d_pose_fwd_jac_use(0);
          dp1[idx] = d_res_d_pose_fwd_jac_use(1);
          dp2[idx] = d_res_d_pose_fwd_jac_use(2);
          dp3[idx] = d_res_d_pose_fwd_jac_use(3);
          dp4[idx] = d_res_d_pose_fwd_jac_use(4);
          dp5[idx] = d_res_d_pose_fwd_jac_use(5);
#else
          dp0[idx] = d_res_d_pose_fwd_jac_use(0);
          dp1[idx] = d_res_d_pose_fwd_jac_use(1);
          dp2[idx] = d_res_d_pose_fwd_jac_use(2);
          dp3[idx] = d_res_d_pose_fwd_jac_use(3);
          dp4[idx] = d_res_d_pose_fwd_jac_use(4);
          dp5[idx] = d_res_d_pose_fwd_jac_use(5);
#endif
#else
          dp0[idx] = d_res_d_pose_inverse_comp_use(0);
          dp1[idx] = d_res_d_pose_inverse_comp_use(1);
          dp2[idx] = d_res_d_pose_inverse_comp_use(2);
          dp3[idx] = d_res_d_pose_inverse_comp_use(3);
          dp4[idx] = d_res_d_pose_inverse_comp_use(4);
          dp5[idx] = d_res_d_pose_inverse_comp_use(5);
#endif
          // TODO* 残差对光度参数求导, 2
          dp6[idx] = -hw * r2new_aff[0] * rlR; //! exp(aj-ai)*I(pi)
          dp7[idx] = -hw * 1;                  //! 对 b 导
          // TODO* 残差对 i(旧状态) 逆深度求导, 1
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
          dd[idx] = dxInterp * dxdd +
                    dyInterp * dydd; //! dxfx * 1/Pz * (tx - u*tz) +　dyfy *
                                     //! 1/Pz * (tx - u*tz)
#else
          dd[idx] = d_res_d_idp_fwd_jac;
#endif
#else
          dd[idx] = d_res_d_idp_inverse_comp;
#endif
          // TODO* 残差 res, 1
          r[idx] = hw * residual; //! 残差 res

          //#else
          //            // Pj 对 逆深度 di 求导
          //			//! 1/Pz * (tx - u*tz), u = px/pz
          //			float dxdd = (t[0]-t[2]*u)/pt[2];
          //			//! 1/Pz * (ty - v*tz), u = py/pz
          //			float dydd = (t[1]-t[2]*v)/pt[2];
          //
          //			if(hw < 1) hw = sqrtf(hw); //?? 为啥开根号, 答:
          //鲁棒核函数等价于加权最小二乘
          //			//! dxfx, dyfy
          //			float dxInterp = hw*hitColor[1]*fxl;
          //			float dyInterp = hw*hitColor[2]*fyl;
          //			//TODO* 残差对 j(新状态) 位姿求导, 6
          //			dp0[idx] = new_idepth*dxInterp; //! dpi/pz' *
          // dxfx 			dp1[idx] = new_idepth*dyInterp; //!
          // dpi/pz'
          // * dyfy 			dp2[idx] = -new_idepth*(u*dxInterp +
          // v*dyInterp); //! -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
          // dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp; //! -
          // px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy dp4[idx] =
          // (1+u*u)*dxInterp + u*v*dyInterp; //! (1+px'^2/pz'^2)*dxfx +
          // px'py'/pz'^2*dxfy dp5[idx] = -v*dxInterp + u*dyInterp; //!
          // -py'/pz'*dxfx + px'/pz'*dyfy
          //			//TODO* 残差对光度参数求导, 2
          //			dp6[idx] = - hw*r2new_aff[0] * rlR; //!
          // exp(aj-ai)*I(pi) 			dp7[idx] = - hw*1;	//! 对 b
          // 导
          //			//TODO* 残差对 i(旧状态) 逆深度求导, 1
          //			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
          ////! dxfx * 1/Pz * (tx - u*tz) +　dyfy * 1/Pz * (tx - u*tz)
          //            //TODO* 残差 res, 1
          //			r[idx] = hw*residual; //! 残差 res
          //
          //#endif

          //* 像素误差对逆深度的导数，取模倒数
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
          float maxstep =
              1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm(); //? 为什么这么设置
#else
          float maxstep = 1.0f / d_uv_d_d_fwd_jac.norm();
#endif
#else
          float maxstep =
              1.0f / d_uv_d_d_inverse_comp.norm(); //? 为什么这么设置
#endif
          if (maxstep < point->maxstep)
            point->maxstep = maxstep;

          // immediately compute dp*dd' and dd*dd' in JbBuffer1.
          //* 计算Hessian的第一行(列), 及Jr 关于逆深度那一行
          // 用来计算舒尔补
          // |H11  H12| * |x1| = |b1|
          // |H21  H22|   |x2|   |b2|
          //
          //
          //
          //
          //
          //
          //
          //
          //
          JbBuffer_new[i + h[0] * w[0] * host_cid][0] += dp0[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][1] += dp1[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][2] += dp2[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][3] += dp3[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][4] += dp4[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][5] += dp5[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][6] += dp6[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][7] += dp7[idx] * dd[idx];
          JbBuffer_new[i + h[0] * w[0] * host_cid][8] += r[idx] * dd[idx];
          // TODO 10 = 6 + 2 + 1 + 1 = pose + affine + idepth + res
          // TODO H22 hessian约等于JTJ，当变量为1维时，hessian = J^2
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] +=
              dd[idx] *
              dd[idx]; /// 1/(1+sum(dd*dd))=inverse depth hessian entry,
          /// while now is just sum(dd*dd), H_{\beta \beta}
          cnt++;
        }
        // 如果点的pattern(其中一个像素)超出图像,像素值无穷, 或者残差大于阈值
        if (!isGood || energy > point->outlierTH * 20) {
          E.updateSingle((float)(point->energy[0])); // 上一帧的加进来 //
          point->isGood_new = false;
          point->energy_new = point->energy; //上一次的给当前次的
          continue;
        }

        // 内点则加进能量函数
        // add into energy.
        /// energy = sum(weight * residual * residual * (2 - weight));
        E.updateSingle(energy);
        point->isGood_new = true;
        point->energy_new[0] = energy;

        //! 因为使用128位相当于每次加4个数, 因此i+=4, 妙啊!
        // update Hessian matrix.
        // update Hessian matrix.
        // update Hessian matrix.
        // acc += dp[0] + dp[4]
        // this trick unroll the loop into 4 blocks and speed up this for loop
        // in x86 SSE instruction set dp0 * dp0 + dp0*dp1 + .. + dp0*r
        //            dp1*dp1 + .. + dp1*r
        //                      ..
        //                           r * r
        // here ((float *) (&dp0)) + i we can see that &dp0 get the address of
        // dp0 convert this address into float * which occupy 4 size_of space.
        // and i is the offsets, which shift size_of 4 for each loop
        // this acc9 is aggregating inside each point, this is just summing up
        // the pattern, it will sum the points also
        for (int i = 0; i + 3 < patternNum;
             i += 4) // this for loop has 2 steps each step step 4 stride.
          // (align with SSE)
          acc9.updateSSE(_mm_load_ps(((float *)(&dp0)) +
                                     i), // _mm_load_ps load 4 float values
                                         // from pointer address at a time
                         _mm_load_ps(((float *)(&dp1)) + i),
                         _mm_load_ps(((float *)(&dp2)) + i),
                         _mm_load_ps(((float *)(&dp3)) + i),
                         _mm_load_ps(((float *)(&dp4)) + i),
                         _mm_load_ps(((float *)(&dp5)) + i),
                         _mm_load_ps(((float *)(&dp6)) + i),
                         _mm_load_ps(((float *)(&dp7)) + i),
                         _mm_load_ps(((float *)(&r)) + i));

        // 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
        // ((patternNum >> 2) << 2) this will align the patternNum to be n*4
        // which is required by SSE.
        // ((8 >> 2) << 2) = 8 so, this will jump this for loop directly.
        // this loop is prepared for the patternNum more than 8
        // for example if it's 10, then ((10 >> 2) << 2) = 8. i then loop start
        // from 8 to 10
        // TODO 这不是一路自加，pt+=4表示移位，其实算的就是Jt * J和-Jt*b
        // H += H_i
        // H = Jt * J,  b = -Jt * b
        // 老老实实对H和b进行累加
        for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
          acc9.updateSingle((float)dp0[i], (float)dp1[i], (float)dp2[i],
                            (float)dp3[i], (float)dp4[i], (float)dp5[i],
                            (float)dp6[i], (float)dp7[i], (float)r[i]);
      }
    }
  }
  E.finish();
  acc9.finish();

  //????? 这是在干吗???
  // calculate alpha energy, and decide if we cap it.
  Accumulator11 EAlpha;
  EAlpha.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < 1 /*kCameraNumUsed*/; ++target_cid) {
      int npts = level_cid_to_numPoints[lvl][host_cid];
      Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][host_cid];
      for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (!point->isGood_new) // 点不好用之前的
        {
          EAlpha.updateSingle((float)(point->energy[1]));
          //          E.updateSingle(
          //              (float)(point->energy[1])); //!
          //              又是故意这样写的，没用的代码,
          //! it should be EAlpha, not E,
          //! stop bullshitting me!!
        } else {
          // 最开始初始化都是成1
          /// res = 1 - idepth_new
#if 1 // ndef USE_MULTI_CAM
          if (kCameraNumUsed == 1) {
            point->energy_new[1] =
                (point->idepth_new - 1) * (point->idepth_new - 1); //? 什么原理?
          } else {
            point->energy_new[1] =
                (point->idepth_new - point->iR_triangle) *
                (point->idepth_new - point->iR_triangle); //? 什么原理?
          }
#else
          point->energy_new[1] = 0;
#endif
          EAlpha.updateSingle((float)(point->energy_new[1]));
          // E.updateSingle((float)(point->energy_new[1]));
        }
      }
    }
  }
  EAlpha.finish(); //! 只是计算位移是否足够大
  int point_count = 0;
  for (int id = 0; id < kCameraNumUsed; ++id) {
    point_count += level_cid_to_numPoints[lvl][id];
  }
  float alphaEnergy =
      alphaW * (EAlpha.A + refToNew_.translation().squaredNorm() *
                               (point_count)); // 平移越大, 越容易初始化成功?

  // printf("AE = %f * %f + %f\n", alphaW, EAlpha.A,
  // refToNew.translation().squaredNorm() * npts);

  // compute alpha opt.
  float alphaOpt;
  if (alphaEnergy > alphaK * (point_count)) // 平移大于一定值
  {
    alphaOpt = 0;
    alphaEnergy = alphaK * (point_count);
  } else {
    alphaOpt = alphaW;
  }

  acc9SC.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < 1 /*kCameraNumUsed*/; ++target_cid) {
      int npts = level_cid_to_numPoints[lvl][host_cid];
      Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][host_cid];
      for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (!point->isGood_new)
          continue;

        /// hessian约等于JTJ，当变量为1维时，hessian = J^2
        /// 1/(1+sum(dd*dd))=inverse depth hessian entry, while now is just
        /// sum(dd*dd), H_{\beta \beta}
        point->lastHessian_new = JbBuffer_new[i + h[0] * w[0] * host_cid][9];

        //? 这又是啥??? 对逆深度的值进行加权? 深度值归一化?
        // 前面Energe加上了（d-1)*(d-1), 所以dd = 1， r += (d-1)
        // TODO 因为初始化阶段idepth的目标值是1，所以idepth的残差就是 weight *
        // (1-idepth)^2了;
        // res = (point->idepth_new - 1); J = 1
        // b = Jt*res = 1 * res = res;
#if 1 // ndef USE_MULTI_CAM
        if (kCameraNumUsed == 1) {
          JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
              alphaOpt * (point->idepth_new - 1);
        } else {
          JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
              alphaOpt *
              (point->idepth_new -
               point->iR_triangle); // TODO Jt * w * res = 1 * w * res
        }
        JbBuffer_new[i + h[0] * w[0] * host_cid][9] +=
            alphaOpt; // TODO Jt * w * J = 1 * w *1
        if (alphaOpt == 0) {
          if (kCameraNumUsed == 1 || true) {
            JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
                couplingWeight * (point->idepth_new - point->iR);
          } else {
            JbBuffer_new[i + h[0] * w[0] * host_cid][8] += couplingWeight * 0;
          }
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] += couplingWeight;
        }
#endif
        /// refer to the equation (17) in DSO, here JbBuffer_new[i][9] is
        /// H^{-1}_{\beta \beta}
        if (kCameraNumUsed == 1) {
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] =
              1 / (1 + JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
        } else {
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] =
              1 / (
#ifdef FIX_ZERO_TRANS_IN_INIT
                      1 +
#endif
                      JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
        }
        //* 9做权重, 计算的是舒尔补项!
        //! dp*dd*(dd^2)^-1*dd*dp

        // H11_ = H11 - H12 * H22^-1 * H12.t ;
        // b1_ = b1 - H12 * H22^-1 * b2 ;
        acc9SC.updateSingleWeighted(
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][0],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][1],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][2],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][3],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][4],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][5],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][6],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][7],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][8],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
      }
    }
  }
  acc9SC.finish();

  // printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num,
  // (int)E.num, (int)acc9SC.num*9);
  H_out = acc9.H.topLeftCorner<8, 8>();       // / acc9.num;
  b_out = acc9.H.topRightCorner<8, 1>();      // / acc9.num;
  H_out_sc = acc9SC.H.topLeftCorner<8, 8>();  // / acc9.num;
  b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;

  //??? 啥意思
  // t*t*ntps
  // 给 t 对应的Hessian, 对角线加上一个数, b也加上
  // TODO 加权重
#if 1
  if (kCameraNumUsed == 1
#ifdef FIX_ZERO_TRANS_IN_INIT
      || true
#endif
  ) {
    H_out(0, 0) += alphaOpt * point_count;
    H_out(1, 1) += alphaOpt * point_count;
    H_out(2, 2) += alphaOpt * point_count;

    Vec3f tlog = refToNew_.log().head<3>().cast<float>();
    // TODO roger,
    // 要强制平移为0，因为默认平移初值是0，所以tlog其实是平移的残差，t_err =
    // t_new - t_orig, t_orig = zeros(3， 1)， J_d_terr_d_t = eye(3)，
    // H_d_err_d_t = eye(3) * eye(3) = eye(3)
    // TODO roger, b_new = H * delta_state + b_old, delta_state = tlog
    b_out[0] += tlog[0] * alphaOpt * point_count;
    b_out[1] += tlog[1] * alphaOpt * point_count;
    b_out[2] += tlog[2] * alphaOpt * point_count;
  }
  // Add zero prior to translation.
  // setting_weightZeroPriorDSOInitY is the squared weight of the prior
  // residual.
  if (kCameraNumUsed == 1
#ifdef FIX_ZERO_TRANS_IN_INIT
      || true
#endif
  ) {
    H_out(1, 1) += setting_weightZeroPriorDSOInitY;
    b_out(1) += setting_weightZeroPriorDSOInitY * refToNew_.translation().y();

    H_out(0, 0) += setting_weightZeroPriorDSOInitX;
    b_out(0) += setting_weightZeroPriorDSOInitX * refToNew_.translation().x();
  }
#endif
  /// E.A is actually equal to residual + (1-d)^2, this (1-d)^2 came from all
  /// good tracking points return vector is total error, alphaEnergy, and the
  /// good tracking number point E.num will be npts + num_good_points since the
  /// E += energy is for every point, but E += (d-1)^2 is only when the point is
  /// good so E.A = sum_npts(energy) + sum_isgood((d-1)^2)
  // 能量值, ? , 使用的点的个数
  return Vec3f(E.A, alphaEnergy, E.num);
}
#endif
//#define SHOW_INIT_IMAGE
#define USE_8_RES_MULTI_CAM
Vec3f CoarseInitializer::calcResAndGS(int iter, int max_iter, int lvl,
                                      MatStatef &H_out, VecStatef &b_out,
                                      MatStatef &H_out_sc, VecStatef &b_out_sc,
                                      const SE3 &refToNew_,
                                      AffLight refToNew_aff, bool plot, int &N,
                                      bool show_image, int lvl_target_) {
  int lvl_target = 0;
  //  int lvl_target = lvl < pyrLevelsUsed - 1 ? lvl + 1 : lvl;
  //  if (lvl == pyrLevelsUsed - 1) {
  //    lvl_target = lvl;
  //  } else {
  //    if (iter % 2 == 0) {
  //      lvl_target = lvl + 1;
  //    } else {
  //      lvl_target = lvl;
  //    }
  //  }

  if (lvl_target_ < 0) {
    lvl_target = lvl < pyrLevelsUsed - 1 ? lvl + 1 : lvl;
    if (lvl == pyrLevelsUsed - 1) {
      lvl_target = lvl;
    } else {
      if (iter % 2 == 0) {
        lvl_target = lvl + 1;
      } else {
        lvl_target = lvl;
      }
    }
  } else {
    lvl_target = lvl_target_;
  }
  // bool use_cross_camera = lvl >= 2 ? true : false;
  bool use_cross_camera = true; // lvl % 2 == 1;
  // bool use_cross_camera = iter % 2 == 0;
  if (lvl_target_ < 0) {
    use_cross_camera = lvl % 2 == 1;
  } else {
    use_cross_camera = true;
  }
  float ratio = static_cast<float>(iter) / static_cast<float>(max_iter);
  // bool calc_stats_only = false;//true;
#ifndef USE_ZNCC
  float ratio_thr1 = 10.5;
  bool calc_stats_only = false; // true;
  float ratio_thr2 = 0.5;
  ratio_thr1 = -10.5;
  calc_stats_only = true;
  ratio_thr2 = -10.5;
#else
  float ratio_thr1 = -10.5;
  bool calc_stats_only = true;
  float ratio_thr2 = -10.5;
#endif
  if (calc_stats_only) {
    // ratio_thr1 = -10.0;
  }
  // float ratio_thr2 = 10.5;
  if (lvl == 0) {
    if (!calc_stats_only) {
      // ratio_thr1 = 10.5;
    }
    // ratio_thr2 = 10.5;
  }
  calc_stats_only = calc_stats_only && ratio > ratio_thr1;
  int wl = w[lvl], hl = h[lvl];
  int wl_target = w[lvl_target], hl_target = h[lvl_target];
  // 当前层图像及梯度
  Accumulator11 E;   // 1*1 的累加器
  acc9.initialize(); // 初始值, 分配空间
  E.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    int bad_pid_count = 0;
    //    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    //      const Mat66 &extra_pose_jac =
    //          newFrame->p_multi_camera->cid_to_T01_inv_Adj[target_cid];
    //      Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl *
    //      hl; Eigen::Vector3f *colorNew = newFrame->dIp[lvl] + target_cid * wl
    //      * hl;
    //      //! 旋转矩阵R * 内参矩阵K_inv
    //      SE3 refToNew =
    //          newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
    //          refToNew_ * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
    //      Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
    //      Vec3f t = refToNew.translation().cast<float>();
    //      Eigen::Vector2f r2new_aff =
    //          Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);
    // 该层的相机参数
    float fxl = fx[lvl];
    float fyl = fy[lvl];
    float fxli = 1 / fx[lvl];
    float fyli = 1 / fy[lvl];
    float cxl = cx[lvl];
    float cyl = cy[lvl];
    float fxl_target = fx[lvl_target];
    float fyl_target = fy[lvl_target];
    float fxli_target = 1 / fx[lvl_target];
    float fyli_target = 1 / fy[lvl_target];
    float cxl_target = cx[lvl_target];
    float cyl_target = cy[lvl_target];

    //  Accumulator11 E;   // 1*1 的累加器
    //  acc9.initialize(); // 初始值, 分配空间
    //  E.initialize();

    int npts = level_cid_to_numPoints[lvl][host_cid];
    printf("host_cid: %d, npts: %d, iter: %d\n", host_cid, npts, iter);
    Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][host_cid];
    for (int i = 0; i < npts; i++) {
      VecBigf dp0 = VecBigf::Zero();
      VecBigf dp1 = VecBigf::Zero();
      VecBigf dp2 = VecBigf::Zero();
      VecBigf dp3 = VecBigf::Zero();
      VecBigf dp4 = VecBigf::Zero();
      VecBigf dp5 = VecBigf::Zero();
      /// affine residual x 2
      VecBigf dp6 = VecBigf::Zero();
      VecBigf dp7 = VecBigf::Zero();

      VecBigf dd = VecBigf::Zero();
      VecBigf r = VecBigf::Zero();
      int inlier_cid_count = 0;
      int is_bad_res_count = 0;
      std::array<int, kCameraNumUsed> cam_info;
      for (int id = 0; id < kCameraNumUsed; ++id) {
        cam_info[id] = -1;
      }
      int good_cam_num = 0;
      int bad_cam_num = 0;
      float energy = 0;
      Pnt *point = ptsl + i;
      point->valid_cid_num = -1;
      bool break_inner_loop = false;
      // int cnt = 0;

#if 1
      Eigen::Matrix<float, 2, 6> d_uv_d_pose, d_uv_d_pose_inverse_comp,
          d_uv_d_pose_fwd_jac;
      Eigen::Matrix<float, 2, 3> d_uv_d_pt3d, d_uv_host_d_n_host,
          d_uv_target_d_x_target_scaled;
      Eigen::Matrix<float, 3, 6> d_pt3d_d_pose, d_pt3d_d_pose_inverse_comp;
      Eigen::Matrix<float, 3, 3> Rot;
      Eigen::Matrix<float, 3, 1> trans;
      Vec4f d_C_x, d_C_y;
      float drescale;

      int show_cnt = 0;

      Eigen::MatrixXf Mat_ZNSSD_I;
      Eigen::MatrixXf J_ZNSSD_mean;
      Eigen::MatrixXf J_ZNSSD_J_I_host;
      Eigen::MatrixXf J_ZNSSD_J_I_target;
      Eigen::MatrixXf grad_new_host;
      Eigen::MatrixXf grad_new_target;
      float host_val_mean;
      float target_val_mean;
      Eigen::MatrixXf ones, ones_temp;
      float host_sigma, target_sigma;

      Eigen::MatrixXf host_info, target_info, host_info_temp, target_info_temp,
          host_info_big, target_info_big;
      //        host_info.resize(MAX_RES_PER_POINT * kCameraNumUsed, 3);
      //        target_info.resize(MAX_RES_PER_POINT * kCameraNumUsed, 3);
      //        host_info.setZero();
      //        target_info.setZero();

      int count = 0;
      std::array<int, kCameraNumUsed> a_count;
      std::map<int, Vec3f> index_to_host_value;
      std::map<int, Vec3f> index_to_target_value;
      std::map<int, int> index_to_count, index_to_count_big;
      std::array<Eigen::MatrixXf, kCameraNumUsed> a_host_info, a_target_info,
          a_grad_new_host, a_grad_new_target;

      Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl * hl;
#ifdef SHOW_INIT_IMAGE
      int show_step = 300;
      MinimalImageB3 *img_host;
      MinimalImageB3 *img_target;
      if (show_image && i % show_step == 0) {
        img_host = new MinimalImageB3(wG[lvl], hG[lvl]);
        img_target = new MinimalImageB3(wG[lvl_target], hG[lvl_target]);

        for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
          // BRIGHTNESS TRANSFER
          float colL = (*(colorRef + i))[0];
          if (colL < 0)
            colL = 0;
          if (colL > 255)
            colL = 255;
          img_host->at(i, host_cid) = Vec3b(colL, colL, colL);
        }
        for (int cam = 0; cam < kCameraNumUsed; ++cam) {
          Eigen::Vector3f *colorCur =
              newFrame->dIp[lvl_target] + cam * wl_target * hl_target;
          for (int i = 0; i < wG[lvl_target] * hG[lvl_target]; i++) {
            // BRIGHTNESS TRANSFER
            float colL = (*(colorCur + i))[0];
            if (colL < 0)
              colL = 0;
            if (colL > 255)
              colL = 255;
            img_target->at(i, cam) = Vec3b(colL, colL, colL);
          }
        }

        img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                            host_cid);
        img_host->setPixelCirc(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                               host_cid);
      }

#endif
      int host_target_info_size = 0;
      float host_sigma_temp = -1;
      float target_sigma_temp = -1;
      MatXXf a_target_sigma_temp, a_err_mean, a_err_max, a_err_min, a_grad_mean,
          a_grad_max, a_grad_min, a_zncc_mean;
      a_target_sigma_temp.resize(kCameraNumUsed, 1);
      a_target_sigma_temp.setOnes();
      a_target_sigma_temp *= -20;
      a_grad_mean = a_grad_max = a_grad_min = a_err_mean = a_err_max =
          a_err_min = a_zncc_mean = a_target_sigma_temp;
      for (int target_cam_id = 0; target_cam_id < kCameraNumUsed;
           ++target_cam_id) {
        // Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl *
        // hl;
        if (!use_cross_camera && host_cid != target_cam_id) {
          continue;
        }
        Eigen::Vector3f *colorNew =
            newFrame->dIp[lvl_target] + target_cam_id * wl_target * hl_target;
        //! 旋转矩阵R * 内参矩阵K_inv
        SE3 refToNew =
            newFrame->p_multi_camera->cid_to_T01_SE3[target_cam_id].inverse() *
            refToNew_ * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
        Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Eigen::Vector2f r2new_aff =
            Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);
#ifdef SHOW_INIT_IMAGE

        //        MinimalImageB3 *img_target;
        //        if (show_image) {
        //
        //          img_target = new MinimalImageB3(wG[lvl], hG[lvl]);
        //
        //          for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
        //            // BRIGHTNESS TRANSFER
        //
        //            float colL = (*(colorNew + i))[0];
        //            if (colL < 0)
        //              colL = 0;
        //            if (colL > 255)
        //              colL = 255;
        //            img_target->at(i, target_cam_id) = Vec3b(colL, colL,
        //            colL);
        //          }
        //
        //        }

#endif
        int count_each_cam = 0;
        std::vector<Vec2f> uv_draw;
        float err_sum = 0, grad_sum = 0;
        float err_max = -10, grad_max = -10;
        float err_min = 99910, grad_min = 99910;
        float zncc_patch = 0;
        // float host_sigma_temp;
        // float target_sigma_temp;
        for (int idx = 0; idx < patternNum; idx++) {
          // pattern的坐标偏移
          int dx = patternP[idx][0];
          int dy = patternP[idx][1];

          //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
          /// Pj = [x y z]
          /// Pj' * Z = Pj
          /// Pj' = [x/z y/z 1]
          Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) +
                     t * point->idepth_new;

          // Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() *
          // Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

          Vec3f n_host = Vec3f((point->u + dx - cxl) * fxli,
                               (point->v + dy - cyl) * fyli, 1);
          Rot = refToNew.rotationMatrix().cast<float>();
          trans = refToNew.translation().cast<float>();
          Vec3f X_target_scaled =
              refToNew.rotationMatrix().cast<float>() * n_host +
              refToNew.translation().cast<float>() * point->idepth_new;
          Mat33f X_target_scaled_skew;
          X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
              X_target_scaled(2), (0), -X_target_scaled(0), -X_target_scaled(1),
              X_target_scaled(0), (0);
          d_uv_d_pt3d << fxl_target / X_target_scaled(2), 0,
              -fxl_target * X_target_scaled(0) / X_target_scaled(2) /
                  X_target_scaled(2),
              0, fyl_target / X_target_scaled(2),
              -fyl_target * X_target_scaled(1) / X_target_scaled(2) /
                  X_target_scaled(2);
          d_pt3d_d_pose.leftCols(3) = point->idepth_new * Mat33f::Identity();
          d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
          d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
          d_uv_host_d_n_host << fxl, 0, -fxl * n_host(0), 0, fyl,
              -fyl * n_host(1);
          d_uv_d_pose_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              d_pt3d_d_pose;

          Vec2f d_uv_d_d_inverse_comp;
          d_uv_d_d_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              refToNew.translation().cast<float>();

          d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
          d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;

          Vec2f d_uv_d_d_fwd_jac;
          d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled *
                             refToNew.translation().cast<float>();
          // printf("xyz: %f\n",pt[2]);
          // 归一化坐标 Pj

          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          // 像素坐标pj
          float Ku = fxl_target * u + cxl_target;
          float Kv = fyl_target * v + cyl_target;

          uv_draw.emplace_back(Vec2f(Ku, Kv));

          // dpi/pz'
          /// 这2个相除应该没什么几何含义，相当于rou1/rou2吧，为了计算雅可比的
          /// idepth_new is the estimated z in host frame, and pt[2] is
          /// projected z in new frame.
          float new_idepth = point->idepth_new / pt[2];
          // 落在边缘附近，深度小于0, 则不好
          if (!(Ku > 1 && Kv > 1 && Ku < wl_target - 2 && Kv < hl_target - 2 &&
                new_idepth > 0)) {
            //                isGood = false;
            //                break;
            continue;
          }
          // 插值得到新图像中的 patch 像素值，(输入3维，输出3维像素值 +
          // x方向梯度 + y方向梯度)
          Vec3f hitColor =
              getInterpolatedElement33(colorNew, Ku, Kv, wl_target);
          Vec3f hostColor = getInterpolatedElement33(colorRef, point->u + dx,
                                                     point->v + dy, wl);
          // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv,
          // wl);
          float host_value_corrected =
              (float)(r2new_aff[0] * hostColor[0] + r2new_aff[1]);

          // 参考帧上的 patch 上的像素值, 输出一维像素值
          // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
          float rlR = getInterpolatedElement31(colorRef, point->u + dx,
                                               point->v + dy, wl);
          assert(std::abs(rlR - hostColor[0]) < 0.00001);
          assert(std::abs(r2new_aff[0] * rlR + r2new_aff[1] -
                          host_value_corrected) < 0.00001);
          // 像素值有穷, good
          if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
            //                isGood = false;
            //                break;
            continue;
          }
          float residual_temp = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
          // Huber权重
          float hw_temp = fabs(residual_temp) < setting_huberTH
                              ? 1
                              : setting_huberTH / fabs(residual_temp);
          // huberweight * (2-huberweight) = Objective Function
          // robust 权重和函数之间的关系
          float energy_temp =
              hw_temp * residual_temp * residual_temp * (2 - hw_temp);
          if (std::abs(residual_temp) /*energy_temp*/ >
              30 /*static_cast<float>(20 * (lvl + 1))*/) {
            // continue;
          }
          err_sum += std::abs(residual_temp);
          if (err_max < std::abs(residual_temp)) {
            err_max = std::abs(residual_temp);
          }
          if (err_min > std::abs(residual_temp)) {
            err_min = std::abs(residual_temp);
          }
          float grad_norm = hitColor.segment<2>(1).norm();
          grad_sum += grad_norm;
          if (grad_max < grad_norm) {
            grad_max = grad_norm;
          }
          if (grad_min > grad_norm) {
            grad_min = grad_norm;
          }
          hostColor[0] = host_value_corrected;
          host_info_big.conservativeResize(count + 1, 3);
          target_info_big.conservativeResize(count + 1, 3);
          host_info_big.row(count) = hostColor.transpose();
          target_info_big.row(count) = hitColor.transpose();
          {
            host_info_temp.conservativeResize(count_each_cam + 1, 3);
            target_info_temp.conservativeResize(count_each_cam + 1, 3);
            host_info_temp.row(count_each_cam) = hostColor.transpose();
            target_info_temp.row(count_each_cam) = hitColor.transpose();
          }

          index_to_host_value.emplace(std::make_pair(
              idx + MAX_RES_PER_POINT * target_cam_id, hostColor));
          index_to_target_value.emplace(std::make_pair(
              idx + MAX_RES_PER_POINT * target_cam_id, hitColor));
          index_to_count_big.emplace(
              std::make_pair(idx + MAX_RES_PER_POINT * target_cam_id, count));

          count_each_cam++;
          count++;
        }
        if (ratio > ratio_thr1) {
          if (count_each_cam >= MAX_RES_PER_POINT) {
            assert(count_each_cam == MAX_RES_PER_POINT);
            float host_val_mean_temp =
                host_info_temp.col(0).sum() /
                static_cast<float>(host_info_temp.rows());
            float target_val_mean_temp =
                target_info_temp.col(0).sum() /
                static_cast<float>(target_info_temp.rows());
            ones_temp.conservativeResize(host_info_temp.rows(), 1);
            ones_temp.setOnes();

            host_info_temp.col(0) =
                host_info_temp.col(0) - host_val_mean_temp * ones_temp;
            target_info_temp.col(0) =
                target_info_temp.col(0) - target_val_mean_temp * ones_temp;
            host_sigma_temp = host_info_temp.col(0).norm();
            target_sigma_temp = target_info_temp.col(0).norm();
            a_target_sigma_temp(target_cam_id, 0) = target_sigma_temp;
            host_info_temp.col(0) /= host_sigma_temp;
            target_info_temp.col(0) /= target_sigma_temp;
            float zncc = host_info_temp.col(0).dot(target_info_temp.col(0));
            if (true) {
              if (std::isfinite(zncc)) {
                //                  printf("zncc: %f, host_sigma_temp: %f,
                //                  target_sigma_temp: %f\n", zncc,
                //                  host_sigma_temp,
                //                         target_sigma_temp);
                assert(std::abs(zncc) < 1.00001);
                //                  std::cout << "host_info_temp: " <<
                //                  host_info_temp.col(0).transpose() << ",
                //                  target_info_temp: "
                //                            <<
                //                            target_info_temp.col(0).transpose()
                //                            << std::endl;
                float angle = (kOur_PI - std::acos(zncc)) / kOur_PI;
                angle = std::isnan(angle) ? 1 : angle;
                // printf("angle: %f\n", angle);
                assert(angle >= 0.0001);
                zncc_patch = angle <= 1 ? angle : 1;
              }
            }
          }
          if (target_sigma_temp < 0.0001) {
            // printf("i: %d, host: %d, target: %d, target_sigma_temp: %f\n", i,
            // point->host_cid, target_cam_id, target_sigma_temp);
          }
          if (count_each_cam > 0 && count_each_cam == MAX_RES_PER_POINT) {
            // printf("i: %d, host: %d, target: %d, target_sigma_temp: %f\n", i,
            // point->host_cid, target_cam_id, target_sigma_temp);
            if ((host_sigma_temp > 3.0f && target_sigma_temp > 2.0f) /*||
                (calc_stats_only && host_sigma_temp > 0.003f &&
                 target_sigma_temp > 0.003f)*/) {
              err_sum /= static_cast<float>(count_each_cam);
              a_err_mean(target_cam_id, 0) = err_sum;
              a_err_max(target_cam_id, 0) = err_max;
              a_err_min(target_cam_id, 0) = err_min;

              grad_sum /= static_cast<float>(count_each_cam);
              a_grad_mean(target_cam_id, 0) = grad_sum;
              a_grad_max(target_cam_id, 0) = grad_max;
              a_grad_min(target_cam_id, 0) = grad_min;
              a_zncc_mean(target_cam_id, 0) = zncc_patch;
              //                 printf("err_sum: %f, err_max: %f, zncc: %f,
              //                 host_sigma_temp: %f, target_sigma_temp: %f\n",
              //                 err_sum, err_max, zncc_patch, host_sigma_temp,
              //                 target_sigma_temp);
              if (!calc_stats_only &&
                  ((err_min > 30000.f ||
#ifndef USE_ZNCC
                    err_sum > 20.f
#else
                    err_sum > 140.f
#endif
                    || err_max > 1000.f) ||
                   grad_sum < 3.f ||
#ifndef USE_ZNCC
                   (lvl >= 0 ? zncc_patch < zncc_thr[lvl] - 0.5f : false
#else
                   (lvl == 0 ? zncc_patch < zncc_thr[lvl] - 0.5f : false
#endif
                    ))) {
                count_each_cam = 0;
              } else {
                // printf("err_sum: %f, err_max: %f, zncc: %f, host_sigma_temp:
                // %f, target_sigma_temp: %f\n", err_sum, err_max, zncc_patch,
                // host_sigma_temp, target_sigma_temp);
              }
            } else {
              //              printf("i: %d, host_sigma_temp: %f,
              //              target_sigma_temp: %f\n", i,
              //                     host_sigma_temp, target_sigma_temp);
              count_each_cam = 0;
            }
          }
        }
        if (count_each_cam >= MAX_RES_PER_POINT) {
          cam_info[target_cam_id] = 1;
          good_cam_num++;
          assert(host_info_temp.rows() == MAX_RES_PER_POINT);
#ifdef SHOW_INIT_IMAGE
          if (show_image && i % show_step == 0) {
            for (const Vec2f &uv : uv_draw) {
              img_target->setPixel9(
                  uv[0], uv[1],
                  makeRainbow3B(1.f / static_cast<float>(target_cam_id + 1)),
                  target_cam_id);
            }
          }
#endif
          a_host_info[target_cam_id] = host_info_temp;
          a_target_info[target_cam_id] = target_info_temp;
          for (int id = 0; id < host_info_temp.rows(); ++id) {
            host_info.conservativeResize(host_target_info_size + 1, 3);
            target_info.conservativeResize(host_target_info_size + 1, 3);
            host_info.row(host_target_info_size) = host_info_temp.row(id);
            target_info.row(host_target_info_size) = target_info_temp.row(id);
            index_to_count.emplace(std::make_pair(
                id + MAX_RES_PER_POINT * target_cam_id, host_target_info_size));
            host_target_info_size++;
          }
        }
        //#ifdef SHOW_INIT_IMAGE
        //        //    std::cout << "idx: " << idx << ", hostColor: " <<
        //        //    hostColor.transpose()
        //        //              << ", hitColor: " << hitColor.transpose()
        //        //              << ", affLL: " << affLL.transpose()
        //        //              << ", color[idx]: " << color[idx] <<
        //        std::endl; if (show_image) {
        //          IOWrap::displayImage(("host_" +
        //          std::to_string(host_cid)).data(), img_host);
        //          IOWrap::displayImage(("target_" +
        //          std::to_string(target_cam_id)).data(), img_target);
        //          IOWrap::waitKey(0);
        //
        //          delete img_host;
        //          delete img_target;
        //        }
        //#endif
        a_count[target_cam_id] = count_each_cam;
      }
#ifdef SHOW_INIT_IMAGE
      //    std::cout << "idx: " << idx << ", hostColor: " <<
      //    hostColor.transpose()
      //              << ", hitColor: " << hitColor.transpose()
      //              << ", affLL: " << affLL.transpose()
      //              << ", color[idx]: " << color[idx] << std::endl;
      if (show_image && i % show_step == 0) {
        printf("pid: %d\n", i);
        IOWrap::displayImage("host_", img_host);
        IOWrap::displayImage("target_", img_target);
        IOWrap::waitKey(0);
        delete img_host;
        delete img_target;
      }
#endif
      // printf("cc\n");
      assert(host_target_info_size == host_info.rows());
      int patch_num = host_info.rows();
      // assert(patch_num == MAX_RES_PER_POINT * kCameraNumUsed);
      if (patch_num != 0) {

        for (int id = 0; id < kCameraNumUsed; ++id) {
          if (cam_info[id] > 0) {
            // printf("id: %d\n", id);
            host_val_mean = a_host_info[id].col(0).sum() /
                            static_cast<float>(a_target_info[id].rows());
            target_val_mean = a_target_info[id].col(0).sum() /
                              static_cast<float>(a_target_info[id].rows());
            // printf("host_val_mean: %f, target_val_mean: %f\n", host_val_mean,
            // target_val_mean);

            ones.conservativeResize(a_target_info[id].rows(), 1);
            ones.setOnes();
#ifdef USE_ZNCC
            a_host_info[id].col(0) =
                a_host_info[id].col(0) - host_val_mean * ones;
            a_target_info[id].col(0) =
                a_target_info[id].col(0) - target_val_mean * ones;
            host_sigma = a_host_info[id].col(0).norm();
            target_sigma = a_target_info[id].col(0).norm();
            a_host_info[id].col(0) /= host_sigma;
            a_target_info[id].col(0) /= target_sigma;

            Mat_ZNSSD_I.conservativeResize(a_target_info[id].rows(),
                                           a_target_info[id].rows());
            Mat_ZNSSD_I.setIdentity();

            J_ZNSSD_mean =
                Mat_ZNSSD_I -
                (ones / static_cast<float>(a_target_info[id].rows())) *
                    ones.transpose();

            J_ZNSSD_J_I_host =
                setting_variableScale *
                ((Mat_ZNSSD_I - (a_host_info[id].col(0) *
                                 a_host_info[id].col(0).transpose())) /
                 host_sigma * J_ZNSSD_mean);
            J_ZNSSD_J_I_target =
                setting_variableScale *
                ((Mat_ZNSSD_I - (a_target_info[id].col(0) *
                                 a_target_info[id].col(0).transpose())) /
                 target_sigma * J_ZNSSD_mean);

            a_grad_new_host[id] =
                J_ZNSSD_J_I_host *
                a_host_info[id].rightCols(2); // "new" gradient: 8x2
            a_grad_new_target[id] =
                J_ZNSSD_J_I_target *
                a_target_info[id].rightCols(2); // "new" gradient: 8x2
            a_host_info[id].col(0) *= setting_variableScale;
            a_target_info[id].col(0) *= setting_variableScale;
#else
            a_grad_new_host[id] =
                a_host_info[id].rightCols(2); // "new" gradient: 8x2
            a_grad_new_target[id] =
                a_target_info[id].rightCols(2);     // "new" gradient: 8x2
#endif
          }
        }
        // printf("dddd\n");
        host_val_mean = host_info.col(0).sum() / patch_num;
        // printf("ee\n");
        target_val_mean = target_info.col(0).sum() / patch_num;

        ones.conservativeResize(patch_num, 1);
        ones.setOnes();
#ifdef USE_ZNCC
        host_info.col(0) = host_info.col(0) - host_val_mean * ones;
        target_info.col(0) = target_info.col(0) - target_val_mean * ones;
        host_sigma = host_info.col(0).norm();
        target_sigma = target_info.col(0).norm();
        host_info.col(0) /= host_sigma;
        target_info.col(0) /= target_sigma;

        Mat_ZNSSD_I.conservativeResize(patch_num, patch_num);
        Mat_ZNSSD_I.setIdentity();

        J_ZNSSD_mean = Mat_ZNSSD_I - (ones / static_cast<float>(patch_num)) *
                                         ones.transpose();

        J_ZNSSD_J_I_host =
            setting_variableScale *
            ((Mat_ZNSSD_I - (host_info.col(0) * host_info.col(0).transpose())) /
             host_sigma * J_ZNSSD_mean);
        J_ZNSSD_J_I_target = setting_variableScale *
                             ((Mat_ZNSSD_I - (target_info.col(0) *
                                              target_info.col(0).transpose())) /
                              target_sigma * J_ZNSSD_mean);

        grad_new_host =
            J_ZNSSD_J_I_host * host_info.rightCols(2); // "new" gradient: 8x2
        grad_new_target = J_ZNSSD_J_I_target *
                          target_info.rightCols(2); // "new" gradient: 8x2
        host_info.col(0) *= setting_variableScale;
        target_info.col(0) *= setting_variableScale;
        // printf("ff\n");
#else
        grad_new_host = host_info.rightCols(2);     // "new" gradient: 8x2
        grad_new_target = target_info.rightCols(2); // "new" gradient: 8x2
#endif
        //            std::cout << "init, grad_new_host: \n" << grad_new_host
        //            << std::endl; std::cout << "init, grad_new_target: \n"
        //            << grad_new_target << std::endl;
        //
        //            std::cout << "init, grad_old_host: \n" <<
        //            host_info.rightCols(2) << std::endl; std::cout << "init,
        //            grad_old_target: \n" << target_info.rightCols(2) <<
        //            std::endl;
      }

#endif

      JbBuffer_new[i + h[0] * w[0] * host_cid].setZero(); // 10*1 向量
      int cnt = 0;
      std::array<int, kCameraNumUsed> a_cnt;
      point->maxstep = 1e10;
      if (!point->isGood) // 点不好
      {
        E.updateSingle((float)(point->energy[0])); // 累加
        point->energy_new = point->energy;
        point->isGood_new = false;
        continue;
      }
      is_bad_res_count = 0;
      energy = 0;
      float target_grad_sum = 0;
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        a_cnt[target_cid] = 0;
        if (!use_cross_camera && host_cid != target_cid) {
          continue;
        }
        if (cam_info[target_cid] < 0) {
          continue;
        }
        int target_cid_use = 0; // target_cid;
        const Mat66 &extra_pose_jac =
            newFrame->p_multi_camera->cid_to_T01_inv_Adj[target_cid];
        // Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl *
        // hl;
        Eigen::Vector3f *colorNew =
            newFrame->dIp[lvl_target] + target_cid * wl_target * hl_target;
        //! 旋转矩阵R * 内参矩阵K_inv
        SE3 refToNew =
            newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
            refToNew_ * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
        Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Eigen::Vector2f r2new_aff =
            Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);
        //        show_image =
        //            host_cid == 2 && lvl == 0 && target_cid == 3 && i > 200 &&
        //            false;
        /// ptsl + i same as ptsl[i];
        // Pnt *point = ptsl + i;
        assert(point->host_cid == host_cid);
        if (point->host_cid != host_cid) {
          //  continue;
        }
        // point->maxstep = 1e10;
        //        if (!point->isGood) // 点不好
        //        {
        //          E.updateSingle((float)(point->energy[0])); // 累加
        //          point->energy_new = point->energy;
        //          point->isGood_new = false;
        //          // break_inner_loop = true;
        //          is_bad_res_count = 1000;
        //          break;
        //          continue;
        //        }

        /// VecNeighbourResidualFloat
        /// dp here 0-5 is d_residual / d_SE3, 6-7 is d_residual / d_a and
        /// d_residual / d_b 6dof pose residual
        // TODO !< 用来计算Schur的 0-7: sum(dd * dp). 8: sum(res*dd). 9:
        // 1/(1+sum(dd*dd))=inverse hessian entry
        //        VecBigf dp0;
        //          VecBigf dp1;
        //          VecBigf dp2;
        //          VecBigf dp3;
        //          VecBigf dp4;
        //          VecBigf dp5;
        //        /// affine residual x 2
        //          VecBigf dp6;
        //          VecBigf dp7;
        //
        //          VecBigf dd;
        //          VecBigf r;
        // todo roger,
        // 不同level共用这个JbBuffer，所以开辟空间肯定要按最大的level0去开辟
        // JbBuffer_new[i + h[0] * w[0] * host_cid].setZero(); // 10*1 向量

        // sum over all residuals.
//        bool isGood = true;
//        float energy = 0;
#if 0
        Eigen::Matrix<float, 2, 6> d_uv_d_pose, d_uv_d_pose_inverse_comp,
            d_uv_d_pose_fwd_jac;
        Eigen::Matrix<float, 2, 3> d_uv_d_pt3d, d_uv_host_d_n_host,
            d_uv_target_d_x_target_scaled;
        Eigen::Matrix<float, 3, 6> d_pt3d_d_pose, d_pt3d_d_pose_inverse_comp;
        Eigen::Matrix<float, 3, 3> Rot;
        Eigen::Matrix<float, 3, 1> trans;
        Vec4f d_C_x, d_C_y;
        float drescale;

        int show_cnt = 0;

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
#endif

#if 0
        int count = 0;
        for (int idx = 0; idx < patternNum; idx++) {
          // pattern的坐标偏移
          int dx = patternP[idx][0];
          int dy = patternP[idx][1];

          //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
          /// Pj = [x y z]
          /// Pj' * Z = Pj
          /// Pj' = [x/z y/z 1]
          Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) +
                     t * point->idepth_new;

          // Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() *
          // Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

          Vec3f n_host = Vec3f((point->u + dx - cxl) * fxli,
                               (point->v + dy - cyl) * fyli, 1);
          Rot = refToNew.rotationMatrix().cast<float>();
          trans = refToNew.translation().cast<float>();
          Vec3f X_target_scaled =
              refToNew.rotationMatrix().cast<float>() * n_host +
              refToNew.translation().cast<float>() * point->idepth_new;
          Mat33f X_target_scaled_skew;
          X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
              X_target_scaled(2), (0), -X_target_scaled(0), -X_target_scaled(1),
              X_target_scaled(0), (0);
          d_uv_d_pt3d << fxl / X_target_scaled(2), 0,
              -fxl * X_target_scaled(0) / X_target_scaled(2) /
                  X_target_scaled(2),
              0, fyl / X_target_scaled(2),
              -fyl * X_target_scaled(1) / X_target_scaled(2) /
                  X_target_scaled(2);
          d_pt3d_d_pose.leftCols(3) = point->idepth_new * Mat33f::Identity();
          d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
          d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
          d_uv_host_d_n_host << fxl, 0, -fxl * n_host(0), 0, fyl,
              -fyl * n_host(1);
          d_uv_d_pose_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              d_pt3d_d_pose;

          Vec2f d_uv_d_d_inverse_comp;
          d_uv_d_d_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              refToNew.translation().cast<float>();

          d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
          d_uv_d_pose_fwd_jac = d_uv_target_d_x_target_scaled * d_pt3d_d_pose;

          Vec2f d_uv_d_d_fwd_jac;
          d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled *
                             refToNew.translation().cast<float>();
          // printf("xyz: %f\n",pt[2]);
          // 归一化坐标 Pj

          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          // 像素坐标pj
          float Ku = fxl * u + cxl;
          float Kv = fyl * v + cyl;

#ifdef SHOW_INIT_IMAGE
          if (show_image) {
            img_target->setPixel9(Ku, Kv, makeRainbow3B(1), target_cid);
          }
#endif

          // dpi/pz'
          /// 这2个相除应该没什么几何含义，相当于rou1/rou2吧，为了计算雅可比的
          /// idepth_new is the estimated z in host frame, and pt[2] is
          /// projected z in new frame.
          float new_idepth = point->idepth_new / pt[2];
          // 落在边缘附近，深度小于0, 则不好
          if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 &&
                new_idepth > 0)) {
            //                isGood = false;
            //                break;
            continue;
          }
          // 插值得到新图像中的 patch 像素值，(输入3维，输出3维像素值 +
          // x方向梯度 + y方向梯度)
          Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
          Vec3f hostColor = getInterpolatedElement33(colorRef, point->u + dx,
                                                     point->v + dy, wl);
          // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv,
          // wl);
          float host_value_corrected =
              (float)(r2new_aff[0] * hostColor[0] + r2new_aff[1]);

          // 参考帧上的 patch 上的像素值, 输出一维像素值
          // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
          float rlR = getInterpolatedElement31(colorRef, point->u + dx,
                                               point->v + dy, wl);
          // 像素值有穷, good
          if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
            //                isGood = false;
            //                break;
            continue;
          }
          hostColor[0] = host_value_corrected;
          host_info.conservativeResize(count + 1, 3);
          target_info.conservativeResize(count + 1, 3);
          host_info.row(count) = hostColor.transpose();
          target_info.row(count) = hitColor.transpose();
          count++;
        }

#ifdef SHOW_INIT_IMAGE
        //    std::cout << "idx: " << idx << ", hostColor: " <<
        //    hostColor.transpose()
        //              << ", hitColor: " << hitColor.transpose()
        //              << ", affLL: " << affLL.transpose()
        //              << ", color[idx]: " << color[idx] << std::endl;
        if (show_image) {
          IOWrap::displayImage("host", img_host);
          IOWrap::displayImage("target", img_target);
          IOWrap::waitKey(0);

          delete img_host;
          delete img_target;
        }
#endif

        int patch_num = host_info.rows();
        if (patch_num != 0) {

          host_val_mean = host_info.col(0).sum() / patch_num;
          target_val_mean = target_info.col(0).sum() / patch_num;

          ones.conservativeResize(patch_num, 1);
          ones.setOnes();
#ifdef USE_ZNCC
          host_info.col(0) = host_info.col(0) - host_val_mean * ones;
          target_info.col(0) = target_info.col(0) - target_val_mean * ones;
          host_sigma = host_info.col(0).norm();
          target_sigma = target_info.col(0).norm();
          host_info.col(0) /= host_sigma;
          target_info.col(0) /= target_sigma;

          Mat_ZNSSD_I.conservativeResize(patch_num, patch_num);
          Mat_ZNSSD_I.setIdentity();

          J_ZNSSD_mean = Mat_ZNSSD_I - (ones / static_cast<float>(patch_num)) *
                                           ones.transpose();

          J_ZNSSD_J_I_host = setting_variableScale *
                             ((Mat_ZNSSD_I - (host_info.col(0) *
                                              host_info.col(0).transpose())) /
                              host_sigma * J_ZNSSD_mean);
          J_ZNSSD_J_I_target =
              setting_variableScale *
              ((Mat_ZNSSD_I -
                (target_info.col(0) * target_info.col(0).transpose())) /
               target_sigma * J_ZNSSD_mean);

          grad_new_host =
              J_ZNSSD_J_I_host * host_info.rightCols(2); // "new" gradient: 8x2
          grad_new_target = J_ZNSSD_J_I_target *
                            target_info.rightCols(2); // "new" gradient: 8x2
          host_info.col(0) *= setting_variableScale;
          target_info.col(0) *= setting_variableScale;
#else
          grad_new_host = host_info.rightCols(2);     // "new" gradient: 8x2
          grad_new_target = target_info.rightCols(2); // "new" gradient: 8x2
#endif
          //            std::cout << "init, grad_new_host: \n" << grad_new_host
          //            << std::endl; std::cout << "init, grad_new_target: \n"
          //            << grad_new_target << std::endl;
          //
          //            std::cout << "init, grad_old_host: \n" <<
          //            host_info.rightCols(2) << std::endl; std::cout << "init,
          //            grad_old_target: \n" << target_info.rightCols(2) <<
          //            std::endl;
        }

#endif
        // int cnt = 0;
        int cnt_each_cam = 0;
        // float target_grad_sum = 0;
        for (int idx = 0; idx < patternNum; idx++) {
          int dx = patternP[idx][0];
          int dy = patternP[idx][1];

          //! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
          /// Pj = [x y z]
          /// Pj' * Z = Pj
          /// Pj' = [x/z y/z 1]
          Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) +
                     t * point->idepth_new;

          // Vec3f pt_scaled = refToNew.rotationMatrix().cast<float>() *
          // Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;

          Vec3f n_host = Vec3f((point->u + dx - cxl) * fxli,
                               (point->v + dy - cyl) * fyli, 1);
          Rot = refToNew.rotationMatrix().cast<float>();
          trans = refToNew.translation().cast<float>();
          Vec3f X_target_scaled =
              refToNew.rotationMatrix().cast<float>() * n_host +
              refToNew.translation().cast<float>() * point->idepth_new;
          Mat33f X_target_scaled_skew;
          X_target_scaled_skew << (0), -X_target_scaled(2), X_target_scaled(1),
              X_target_scaled(2), (0), -X_target_scaled(0), -X_target_scaled(1),
              X_target_scaled(0), (0);
          d_uv_d_pt3d << fxl_target / X_target_scaled(2), 0,
              -fxl_target * X_target_scaled(0) / X_target_scaled(2) /
                  X_target_scaled(2),
              0, fyl_target / X_target_scaled(2),
              -fyl_target * X_target_scaled(1) / X_target_scaled(2) /
                  X_target_scaled(2);
          d_pt3d_d_pose.leftCols(3) =
              point->idepth_new * Mat33f::Identity(); /// [t R]
          d_pt3d_d_pose.rightCols(3) = -X_target_scaled_skew;
          d_uv_d_pose = d_uv_d_pt3d * d_pt3d_d_pose;
          d_uv_host_d_n_host << fxl, 0, -fxl * n_host(0), 0, fyl,
              -fyl * n_host(1);
          d_uv_d_pose_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              d_pt3d_d_pose;

          Vec2f d_uv_d_d_inverse_comp;
          d_uv_d_d_inverse_comp =
              d_uv_host_d_n_host *
              refToNew.rotationMatrix().cast<float>().transpose() *
              refToNew.translation().cast<float>();

          d_uv_target_d_x_target_scaled = d_uv_d_pt3d;
          d_uv_d_pose_fwd_jac = (d_uv_target_d_x_target_scaled.cast<double>() *
                                 d_pt3d_d_pose.cast<double>())
                                    .cast<float>();

          Mat26f d_uv_d_pose_fwd_jac_use =
              (d_uv_target_d_x_target_scaled.cast<double>() *
               d_pt3d_d_pose.cast<double>() * extra_pose_jac)
                  .cast<float>();

          Vec2f d_uv_d_d_fwd_jac;
          d_uv_d_d_fwd_jac = d_uv_target_d_x_target_scaled *
                             refToNew.translation().cast<float>();

          Mat26f d_uv_d_pose_inverse_comp_use =
              (d_uv_host_d_n_host.cast<double>() *
               refToNew.rotationMatrix().transpose() *
               d_pt3d_d_pose.cast<double>() * extra_pose_jac)
                  .cast<float>();
          // printf("xyz: %f\n",pt[2]);
          // 归一化坐标 Pj

          float u = pt[0] / pt[2];
          float v = pt[1] / pt[2];
          float Ku = fxl_target * u + cxl_target;
          float Kv = fyl_target * v + cyl_target;
          // dpi/pz'
          /// 这2个相除应该没什么几何含义，相当于rou1/rou2吧，为了计算雅可比的
          /// idepth_new is the estimated z in host frame, and pt[2] is
          /// projected z in new frame.
          float new_idepth = point->idepth_new / pt[2];
#if 0
                drescale = 1/X_target_scaled(2);
                d_C_x[2] = drescale*(Rot(2,0)*u-Rot(0,0));
                d_C_x[3] = fxl * drescale*(Rot(2,1)*u-Rot(0,1)) * fyli;
                //TODO KliP: host帧归一化坐标
                d_C_x[0] = n_host[0]*d_C_x[2];
                d_C_x[1] = n_host[1]*d_C_x[3];

                d_C_y[2] = fyl * drescale*(Rot(2,0)*v-Rot(1,0)) * fxli;
                d_C_y[3] = drescale*(Rot(2,1)*v-Rot(1,1));
                d_C_y[0] = n_host[0]*d_C_y[2];
                d_C_y[1] = n_host[1]*d_C_y[3];

                d_C_x[0] = (d_C_x[0]+u);//TODO d_u2_d_fx
                d_C_x[1] *= 1;
                d_C_x[2] = (d_C_x[2]+1);//TODO d_u2_d_cx
                d_C_x[3] *= 1;

                d_C_y[0] *= 1;
                d_C_y[1] = (d_C_y[1]+v)*1;
                d_C_y[2] *= 1;
                d_C_y[3] = (d_C_y[3]+1)*1;
#endif
          // 落在边缘附近，深度小于0, 则不好
          if (!(Ku > 1 && Kv > 1 && Ku < wl_target - 2 && Kv < hl_target - 2 &&
                new_idepth > 0)) {
            // isGood = false;
            is_bad_res_count++;
            // break_inner_loop = true;
            // break;
            continue;
          }
          // 插值得到新图像中的 patch 像素值，(输入3维，输出3维像素值 +
          // x方向梯度 + y方向梯度)
          Vec3f hitColor =
              getInterpolatedElement33(colorNew, Ku, Kv, wl_target);
          Vec3f hostColor = getInterpolatedElement33(colorRef, point->u + dx,
                                                     point->v + dy, wl);
          // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv,
          // wl);

          // 参考帧上的 patch 上的像素值, 输出一维像素值
          // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
          float rlR = getInterpolatedElement31(colorRef, point->u + dx,
                                               point->v + dy, wl);

          // 像素值有穷, good
          if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
            // isGood = false;
            is_bad_res_count++;
            // break_inner_loop = true;
            // break;
            continue;
          }

#ifndef USE_ZNCC
          // 残差
          float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
          // Huber权重
          float hw =
              fabs(residual) < (setting_huberTH /*+ std::abs(r2new_aff[1])*/)
                  ? 1
                  : (setting_huberTH /*+ std::abs(r2new_aff[1])*/) /
                        fabs(residual);
          if (true) {
            float ws2 = a_zncc_mean(target_cid);
            ws2 *= ws2;
            assert(ws2 > 0);
            hw = ws2 > setting_huberTH_zncc ? 1 : ws2 / setting_huberTH_zncc;
          }
          // huberweight * (2-huberweight) = Objective Function
          // robust 权重和函数之间的关系
          energy += hw * residual * residual * (2 - hw);
#else
          float residual_bak = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
          // printf("gg\n");
          float residual = 1 * (a_target_info[target_cid](idx, 0) -
                                a_host_info[target_cid](idx, 0));
          if (lvl <= 1 && false) {
            printf("residual: %f, idx: %d, host_val: %f, target_val: %f\n",
                   residual, idx, a_host_info[target_cid](idx, 0),
                   a_target_info[target_cid](idx, 0));
          }

          // assert(!std::isnan(residual));

          if (std::isnan(residual)) {
            // isGood = false;
            is_bad_res_count++;
            // break_inner_loop = true;
            cnt++;
            cnt_each_cam++;
            continue;
          }

          // printf("residual: %f, hw: %f\n", residual, hw);
#ifndef USE_ZNCC_WEIGHT
          float hw =
              fabs(residual) < (setting_huberTH /*+ std::abs(r2new_aff[1])*/)
                  ? 1
                  : (setting_huberTH /*+ std::abs(r2new_aff[1])*/) /
                        fabs(residual);
#else
          float norm1 = a_host_info[target_cid].col(0).norm();
          float norm2 = a_target_info[target_cid].col(0).norm();

          float zncc = a_target_info[target_cid].col(0).normalized().dot(
              a_host_info[target_cid].col(0).normalized());
          float r2 = 2 - 2 * zncc;
          float ws2 = 2.0 / (r2 + 2.0);
          if (true) {
            // assert(std::abs(std::abs(zncc) - 1) < 0.0001);
            assert(std::abs(zncc) < 1.00001);
            float angle = (kOur_PI - std::acos(zncc)) / kOur_PI;
            angle = std::isnan(angle) ? 1 : angle;
            // assert(angle >= 0);
            assert(angle >= 0.0001);
            zncc = angle <= 1 ? angle : 1;
            // printf("zncc: %f, zncc_pre_calc: %f, diff: %f\n", zncc,
            // a_zncc_mean(target_cid), zncc - a_zncc_mean(target_cid));
            assert(std::abs(zncc - a_zncc_mean(target_cid)) < 0.00001);
            ws2 = zncc;
            ws2 *= ws2;
          }
          float hw =
              ws2; // std::sqrt(ws2);
                   // printf("norm12: [%f %f], hw: %f\n", norm1, norm2, hw);
          hw = ws2 > setting_huberTH_zncc ? 1 : ws2 / setting_huberTH_zncc;
#endif
          // printf("residual: %f, hw: %f\n", residual, hw);
          energy += hw * residual * residual * (2 - hw);
#endif
          target_grad_sum += hitColor.segment<2>(1).norm();

          // Pj 对 逆深度 di 求导
          //! 1/Pz * (tx - u*tz), u = px/pz
          float dxdd = (t[0] - t[2] * u) / pt[2];
          //! 1/Pz * (ty - v*tz), u = py/pz
          float dydd = (t[1] - t[2] * v) / pt[2];

          if (hw < 1)
            hw = sqrtf(hw); //?? 为啥开根号, 答: 鲁棒核函数等价于加权最小二乘
#ifndef USE_ZNCC
          //! dxfx, dyfy
          float dxInterp = hw * hitColor[1] * fxl_target;
          float dyInterp = hw * hitColor[2] * fyl_target;
#else
          float dxInterp =
              hw * a_grad_new_target[target_cid](idx, 0) * fxl_target;
          float dyInterp =
              hw * a_grad_new_target[target_cid](idx, 1) * fyl_target;
#endif
          // TODO* 残差对 j(新状态) 位姿求导, 6

          Eigen::Matrix<float, 3, 6> show;
          show(0, 0) = new_idepth * dxInterp;
          show(0, 1) = new_idepth * dyInterp;
          show(0, 2) = -new_idepth * (u * dxInterp + v * dyInterp);
          show(0, 3) = -u * v * dxInterp - (1 + v * v) * dyInterp;
          show(0, 4) = (1 + u * u) * dxInterp + u * v * dyInterp;
          show(0, 5) = -v * dxInterp + u * dyInterp;

#ifndef USE_ZNCC
          show.row(1) =
              show.row(0) -
              hw * Vec2f(hitColor[1], hitColor[2]).transpose() * d_uv_d_pose;
          Vec6f d_res_d_pose_inverse_comp =
              hw * Vec2f(hostColor[1], hostColor[2]).transpose() *
              d_uv_d_pose_inverse_comp;
          assert(grad_new_host(
                     index_to_count.at(idx + MAX_RES_PER_POINT * target_cid),
                     0) == a_grad_new_host[target_cid](idx, 0));
          assert(grad_new_host(
                     index_to_count.at(idx + MAX_RES_PER_POINT * target_cid),
                     1) == a_grad_new_host[target_cid](idx, 1));
          assert(grad_new_target(
                     index_to_count.at(idx + MAX_RES_PER_POINT * target_cid),
                     0) == a_grad_new_target[target_cid](idx, 0));
          assert(grad_new_target(
                     index_to_count.at(idx + MAX_RES_PER_POINT * target_cid),
                     1) == a_grad_new_target[target_cid](idx, 1));
          Vec6f d_res_d_pose_fwd_jac =
              hw *
              Vec2f(a_grad_new_target[target_cid](idx, 0),
                    a_grad_new_target[target_cid](idx, 1))
                  .transpose() *
              d_uv_d_pose_fwd_jac;
          show.row(2) = show.row(0) - d_res_d_pose_fwd_jac.transpose();
          Vec6f d_res_d_pose_fwd_jac_use =
              (static_cast<double>(hw) *
               Vec2(hitColor[1], hitColor[2]).transpose() *
               d_uv_d_pose_fwd_jac_use.cast<double>())
                  .cast<float>();
          Vec6f d_res_d_pose_inverse_comp_use =
              (static_cast<double>(hw) *
               Vec2(a_grad_new_host[target_cid](idx, 0),
                    a_grad_new_host[target_cid](idx, 1))
                   .transpose() *
               d_uv_d_pose_inverse_comp_use.cast<double>())
                  .cast<float>();
          assert(std::abs(a_grad_new_host[target_cid](idx, 0) - hostColor[1]) ==
                 0);
          assert(std::abs(a_grad_new_host[target_cid](idx, 1) - hostColor[2]) ==
                 0);
#else
          show.row(1) =
              show.row(0) - hw *
                                Vec2f(a_grad_new_target[target_cid](idx, 0),
                                      a_grad_new_target[target_cid](idx, 1))
                                    .transpose() *
                                d_uv_d_pose;
          Vec6f d_res_d_pose_inverse_comp =
              hw *
              Vec2f(a_grad_new_host[target_cid](idx, 0),
                    a_grad_new_host[target_cid](idx, 1))
                  .transpose() *
              d_uv_d_pose_inverse_comp;
          Vec6f d_res_d_pose_fwd_jac =
              hw *
              Vec2f(a_grad_new_target[target_cid](idx, 0),
                    a_grad_new_target[target_cid](idx, 1))
                  .transpose() *
              d_uv_d_pose_fwd_jac;
          Vec6f d_res_d_pose_fwd_jac_use =
              (static_cast<double>(hw) *
               Vec2(a_grad_new_target[target_cid](idx, 0),
                    a_grad_new_target[target_cid](idx, 1))
                   .transpose() *
               d_uv_d_pose_fwd_jac_use.cast<double>())
                  .cast<float>();
          Vec6f d_res_d_pose_inverse_comp_use =
              (static_cast<double>(hw) *
               Vec2(a_grad_new_host[target_cid](idx, 0),
                    a_grad_new_host[target_cid](idx, 1))
                   .transpose() *
               d_uv_d_pose_inverse_comp_use.cast<double>())
                  .cast<float>();
          // printf("hh, idx: %d\n", idx);
#endif

          Vec3f d_uv_d_idp_show;
          d_uv_d_idp_show(0) = dxInterp * dxdd + dyInterp * dydd;
#ifndef USE_ZNCC
          d_uv_d_idp_show(1) = d_uv_d_idp_show(0) -
                               hw *
                                   Vec2f(hitColor[1], hitColor[2]).transpose() *
                                   d_uv_d_pt3d * trans;
          // std::cout<<"d_uv_d_c_show:\n"<<d_uv_d_c_show<<std::endl;
          float d_res_d_idp_inverse_comp =
              hw * Vec2f(hostColor[1], hostColor[2]).transpose() *
              d_uv_d_d_inverse_comp;
          float d_res_d_idp_fwd_jac =
              hw *
              Vec2f(a_grad_new_target[target_cid](idx, 0),
                    a_grad_new_target[target_cid](idx, 1))
                  .transpose() *
              d_uv_d_d_fwd_jac;
          d_uv_d_idp_show(2) = d_uv_d_idp_show(0) - d_res_d_idp_fwd_jac;
#else
          d_uv_d_idp_show(1) = d_uv_d_idp_show(0) -
                               hw *
                                   Vec2f(a_grad_new_target[target_cid](idx, 0),
                                         a_grad_new_target[target_cid](idx, 1))
                                       .transpose() *
                                   d_uv_d_pt3d * trans;
          float d_res_d_idp_inverse_comp =
              hw *
              Vec2f(a_grad_new_host[target_cid](idx, 0),
                    a_grad_new_host[target_cid](idx, 1))
                  .transpose() *
              d_uv_d_d_inverse_comp;
          float d_res_d_idp_fwd_jac =
              hw *
              Vec2f(a_grad_new_target[target_cid](
                        idx /*index_to_count.at(idx + MAX_RES_PER_POINT *
                               target_cid)*/
                        ,
                        0),
                    a_grad_new_target[target_cid](idx, 1))
                  .transpose() *
              d_uv_d_d_fwd_jac;
#endif

          if (show_cnt < -10) {
            std::cout << "[du dv hw]: [" << dx << ", " << dy << ", " << hw
                      << "], tracking pose jac diff: \n"
                      << show << std::endl;
            std::cout << "d_uv_d_idp_show: \n"
                      << d_uv_d_idp_show.transpose() << std::endl
                      << std::endl
                      << std::endl;
            show_cnt++;
          }

#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC

          dp0[idx + MAX_RES_PER_POINT * target_cid_use] =
              new_idepth * dxInterp; //! dpi/pz' * dxfx
          dp1[idx + MAX_RES_PER_POINT * target_cid_use] =
              new_idepth * dyInterp; //! dpi/pz' * dyfy
          dp2[idx + MAX_RES_PER_POINT * target_cid_use] =
              -new_idepth *
              (u * dxInterp +
               v * dyInterp); //! -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
          dp3[idx + MAX_RES_PER_POINT * target_cid_use] =
              -u * v * dxInterp -
              (1 + v * v) *
                  dyInterp; //! - px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy
          dp4[idx + MAX_RES_PER_POINT * target_cid_use] =
              (1 + u * u) * dxInterp +
              u * v * dyInterp; //! (1+px'^2/pz'^2)*dxfx + px'py'/pz'^2*dxfy
          dp5[idx + MAX_RES_PER_POINT * target_cid_use] =
              -v * dxInterp + u * dyInterp; //! -py'/pz'*dxfx + px'/pz'*dyfy

          dp0[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(0);
          dp1[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(1);
          dp2[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(2);
          dp3[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(3);
          dp4[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(4);
          dp5[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(5);
#else
          dp0[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(0);
          dp1[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(1);
          dp2[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(2);
          dp3[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(3);
          dp4[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(4);
          dp5[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_fwd_jac_use(5);
#endif
#else
          dp0[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_inverse_comp_use(0);
          dp1[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_inverse_comp_use(1);
          dp2[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_inverse_comp_use(2);
          dp3[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_inverse_comp_use(3);
          dp4[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_inverse_comp_use(4);
          dp5[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_pose_inverse_comp_use(5);
#endif
          // TODO* 残差对光度参数求导, 2
          dp6[idx + MAX_RES_PER_POINT * target_cid_use] =
              -hw * r2new_aff[0] * rlR; //! exp(aj-ai)*I(pi)
          dp7[idx + MAX_RES_PER_POINT * target_cid_use] = -hw * 1; //! 对 b 导
          // TODO* 残差对 i(旧状态) 逆深度求导, 1
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
          dd[idx + MAX_RES_PER_POINT * target_cid_use] =
              dxInterp * dxdd +
              dyInterp * dydd; //! dxfx * 1/Pz * (tx - u*tz) +　dyfy *
                               //! 1/Pz * (tx - u*tz)
          // printf("i: %d, host_cid: %d, target_cid: %d, idx: %d, residual: %f,
          // J_idp: [%f %f %f], JtJ: %f, hw: %f, val[9]_accum: %f\n", i,
          // host_cid, target_cid, idx, residual, dd[idx + MAX_RES_PER_POINT *
          // target_cid_use], (float)(hw * Vec2f(hitColor[1],
          // hitColor[2]).transpose() *  d_uv_d_pt3d * trans)[0],
          // d_res_d_idp_fwd_jac, d_res_d_idp_fwd_jac * d_res_d_idp_fwd_jac, hw,
          // JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
#else
          dd[idx + MAX_RES_PER_POINT * target_cid_use] = d_res_d_idp_fwd_jac;
#endif
#else
          dd[idx + MAX_RES_PER_POINT * target_cid_use] =
              d_res_d_idp_inverse_comp;
#endif
          // TODO* 残差 res, 1
          r[idx + MAX_RES_PER_POINT * target_cid_use] =
              hw * residual; //! 残差 res

          //#else
          //            // Pj 对 逆深度 di 求导
          //			//! 1/Pz * (tx - u*tz), u = px/pz
          //			float dxdd = (t[0]-t[2]*u)/pt[2];
          //			//! 1/Pz * (ty - v*tz), u = py/pz
          //			float dydd = (t[1]-t[2]*v)/pt[2];
          //
          //			if(hw < 1) hw = sqrtf(hw); //?? 为啥开根号, 答:
          //鲁棒核函数等价于加权最小二乘
          //			//! dxfx, dyfy
          //			float dxInterp = hw*hitColor[1]*fxl;
          //			float dyInterp = hw*hitColor[2]*fyl;
          //			//TODO* 残差对 j(新状态) 位姿求导, 6
          //			dp0[idx] = new_idepth*dxInterp; //! dpi/pz' *
          // dxfx 			dp1[idx] = new_idepth*dyInterp; //!
          // dpi/pz'
          // * dyfy 			dp2[idx] = -new_idepth*(u*dxInterp +
          // v*dyInterp); //! -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
          // dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp; //! -
          // px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy dp4[idx] =
          // (1+u*u)*dxInterp + u*v*dyInterp; //! (1+px'^2/pz'^2)*dxfx +
          // px'py'/pz'^2*dxfy dp5[idx] = -v*dxInterp + u*dyInterp; //!
          // -py'/pz'*dxfx + px'/pz'*dyfy
          //			//TODO* 残差对光度参数求导, 2
          //			dp6[idx] = - hw*r2new_aff[0] * rlR; //!
          // exp(aj-ai)*I(pi) 			dp7[idx] = - hw*1;	//! 对 b
          // 导
          //			//TODO* 残差对 i(旧状态) 逆深度求导, 1
          //			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
          ////! dxfx * 1/Pz * (tx - u*tz) +　dyfy * 1/Pz * (tx - u*tz)
          //            //TODO* 残差 res, 1
          //			r[idx] = hw*residual; //! 残差 res
          //
          //#endif

          //* 像素误差对逆深度的导数，取模倒数
#ifndef USE_INVERSE_COMPOSITIONAL
#ifndef USE_ZNCC
          float maxstep = 1.0f / Vec2f(dxdd * fxl_target, dydd * fyl_target)
                                     .norm(); //? 为什么这么设置
#else
          float maxstep = 1.0f / d_uv_d_d_fwd_jac.norm();
#endif
#else
          float maxstep =
              1.0f / d_uv_d_d_inverse_comp.norm(); //? 为什么这么设置
#endif
          if (maxstep < point->maxstep)
            point->maxstep = maxstep;

            // immediately compute dp*dd' and dd*dd' in JbBuffer1.
            //* 计算Hessian的第一行(列), 及Jr 关于逆深度那一行
            // 用来计算舒尔补
            // |H11  H12| * |x1| = |b1|
            // |H21  H22|   |x2|   |b2|
            //
            //
            //
            //
            //
            //
            //
            //
            //
#ifdef USE_ZNCC
          dp6[idx + MAX_RES_PER_POINT * target_cid_use] = 0;
          -hw *r2new_aff[0] * rlR; //! exp(aj-ai)*I(pi)
          dp7[idx + MAX_RES_PER_POINT * target_cid_use] =
              0; //-hw * 1; //! 对 b 导
#endif
          JbBuffer_new[i + h[0] * w[0] * host_cid][0] +=
              dp0[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][1] +=
              dp1[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][2] +=
              dp2[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][3] +=
              dp3[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][4] +=
              dp4[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][5] +=
              dp5[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][6] +=
              dp6[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][7] +=
              dp7[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
              r[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT * target_cid_use];
          // TODO 10 = 6 + 2 + 1 + 1 = pose + affine + idepth + res
          // TODO H22 hessian约等于JTJ，当变量为1维时，hessian = J^2
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] +=
              dd[idx + MAX_RES_PER_POINT * target_cid_use] *
              dd[idx + MAX_RES_PER_POINT *
                           target_cid_use]; /// 1/(1+sum(dd*dd))=inverse depth
          /// hessian entry, while now is just
          /// sum(dd*dd), H_{\beta \beta}

          assert(index_to_count.at(idx + MAX_RES_PER_POINT * target_cid) ==
                 cnt);
          if (false &&
              (lvl == 2 && point->host_cid == 1 && i == 178 ||
               JbBuffer_new[i + h[0] * w[0] * host_cid].cwiseAbs().minCoeff() < 1e-8 /*&& JbBuffer_new[i + h[0] * w[0] * host_cid].cwiseAbs().minCoeff() > 0*/)) {
            //              std::cout << "refToNew:\n" << refToNew.matrix3x4()
            //              << std::endl;
            std::cout << "hitColor: " << hitColor.transpose()
                      << ", hostColor: " << rlR
                      << ", affline: " << r2new_aff.transpose() << std::endl;
            //              std::cout << "d_res_d_pose_fwd_jac_use;\n" <<
            //              d_res_d_pose_fwd_jac_use.transpose() << std::endl;
            //              std::cout << "dd: " << dd.transpose() << std::endl;
            std::cout << "JbBuffer_new[i + h[0] * w[0] * host_cid]: "
                      << JbBuffer_new[i + h[0] * w[0] * host_cid].transpose()
                      << std::endl;
            printf("idx: %d, residual: %f, i: %d, lvl: %d, cid: %d, uv: [%f "
                   "%f], iR: %f, isG0od: %d, isGoodNew: %d, idepth: %f, "
                   "isdepthNew: %f\n",
                   idx, residual, i, lvl, point->host_cid, point->u, point->v,
                   point->isGood, point->isGood_new, point->idepth,
                   point->idepth_new);
            //              std::exit(1);
          }
#if 1
          for (int i = 0; i + 3 < patternNum /** kCameraNumUsed*/;
               i += 4) // this for loop has 2 steps each step step 4 stride.
            // (align with SSE)
            acc9.updateSSE(_mm_load_ps(((float *)(&dp0)) +
                                       i), // _mm_load_ps load 4 float values
                                           // from pointer address at a time
                           _mm_load_ps(((float *)(&dp1)) + i),
                           _mm_load_ps(((float *)(&dp2)) + i),
                           _mm_load_ps(((float *)(&dp3)) + i),
                           _mm_load_ps(((float *)(&dp4)) + i),
                           _mm_load_ps(((float *)(&dp5)) + i),
                           _mm_load_ps(((float *)(&dp6)) + i),
                           _mm_load_ps(((float *)(&dp7)) + i),
                           _mm_load_ps(((float *)(&r)) + i));

          // 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
          // ((patternNum >> 2) << 2) this will align the patternNum to be n*4
          // which is required by SSE.
          // ((8 >> 2) << 2) = 8 so, this will jump this for loop directly.
          // this loop is prepared for the patternNum more than 8
          // for example if it's 10, then ((10 >> 2) << 2) = 8. i then loop
          // start from 8 to 10
          // TODO 这不是一路自加，pt+=4表示移位，其实算的就是Jt * J和-Jt*b
          // H += H_i
          // H = Jt * J,  b = -Jt * b
          // 老老实实对H和b进行累加
          for (int i = (((patternNum /* * kCameraNumUsed*/) >> 2) << 2);
               i < patternNum /* * kCameraNumUsed*/; i++) {
            acc9.updateSingle((float)dp0[i], (float)dp1[i], (float)dp2[i],
                              (float)dp3[i], (float)dp4[i], (float)dp5[i],
                              (float)dp6[i], (float)dp7[i], (float)r[i]);
          }
#endif
          cnt++;
          cnt_each_cam++;
        }
        // printf("ii\n");
        a_cnt[target_cid] = cnt_each_cam;
        assert(cnt_each_cam == a_count[target_cid]);
        assert(cnt_each_cam == a_target_info[target_cid].rows());
        assert(cnt_each_cam == a_grad_new_target[target_cid].rows());
        assert(cnt_each_cam == a_host_info[target_cid].rows());
        assert(cnt_each_cam == a_grad_new_host[target_cid].rows());
      }
#if 0
      assert(cnt == count);
#else
      assert(cnt == host_info.rows());
      assert(cnt % MAX_RES_PER_POINT == 0);
#endif
      // 如果点的pattern(其中一个像素)超出图像,像素值无穷, 或者残差大于阈值
      int threshold = MAX_RES_PER_POINT * (kCameraNumUsed - 1) + 4;
      float target_grad_mean =
          target_grad_sum / static_cast<float>(cnt /*/ MAX_RES_PER_POINT*/);
      assert(std::isfinite(energy));
      float zncc_avg = 0, err_avg = 0;
      int visible_cids = 0;
      if (ratio > ratio_thr1 && calc_stats_only) {
        if (false) {
          for (int cam = 0; cam < kCameraNumUsed; ++cam) {
            printf("a_cnt[%d]: %d\n", cam, a_cnt[cam]);
          }
          std::cout << "a_target_sigma_temp: "
                    << a_target_sigma_temp.transpose() << std::endl;
          std::cout << "a_err_mean: " << a_err_mean.transpose() << std::endl;
          std::cout << "a_err_max: " << a_err_max.transpose() << std::endl;
          std::cout << "a_err_min: " << a_err_min.transpose() << std::endl;

          std::cout << "a_grad_mean: " << a_grad_mean.transpose() << std::endl;
          std::cout << "a_grad_max: " << a_grad_max.transpose() << std::endl;
          std::cout << "a_grad_min: " << a_grad_min.transpose() << std::endl;

          std::cout << "a_zncc_mean: " << a_zncc_mean.transpose() << std::endl;
        }
        for (int cam = 0; cam < kCameraNumUsed; ++cam) {
          if (a_err_mean(cam) < -10) {
            assert(a_cnt[cam] == 0);
            assert(a_zncc_mean(cam) < -10);
          } else {
            if (false) {
              std::cout << "11 a_err_mean: " << a_err_mean.transpose()
                        << std::endl;
              std::cout << "11 a_zncc_mean: " << a_zncc_mean.transpose()
                        << std::endl;
              printf("a_zncc_mean(%d): %f\n", cam, a_zncc_mean(cam));
            }
            assert(a_zncc_mean(cam) > -10);
            // printf("a_cnt[cam]: %d\n", a_cnt[cam]);
            assert(a_cnt[cam] == MAX_RES_PER_POINT);
            err_avg += a_err_mean(cam);
            zncc_avg += a_zncc_mean(cam);
            visible_cids++;
          }
        }
        err_avg /= static_cast<float>(visible_cids);
        zncc_avg /= static_cast<float>(visible_cids);
        if (visible_cids > 0) {
          assert(std::isfinite(zncc_avg));
        }
      }
      // printf("target_mean_grad: %f, cnt: %d\n", target_grad_mean, cnt);
      int all_cnt = 0;
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        if (a_cnt[cid] > 0) {
          if (a_cnt[cid] != MAX_RES_PER_POINT) {
            printf("a_cnt[cid]: %d\n", a_cnt[cid]);
            std::exit(2);
          }
          all_cnt += MAX_RES_PER_POINT;
        }
      }
      assert(all_cnt == cnt);
      // printf("good_cam_num: %d, cnt: %d\n", good_cam_num, cnt);
      assert(cnt / MAX_RES_PER_POINT == good_cam_num);

      float energy_each_cam =
          energy / static_cast<float>(cnt / MAX_RES_PER_POINT);
      if (cnt > 0) {
        if (false) {
          point->v_energy_vec.emplace_back(energy_each_cam);
        } else {
          assert(calc_stats_only);
          point->v_energy_vec.emplace_back(zncc_avg);
        }
        // point->v_energy_vec[point->energy_size] = energy_each_cam;
        point->energy_size++;
        assert(point->v_energy_vec.size() == point->energy_size);
        if (point->min_energy > energy_each_cam) {
          point->min_energy = energy_each_cam;
        }
        if (point->max_zncc < zncc_avg) {
          point->max_zncc = zncc_avg;
        }
        point->median_energy = FindMedian(point->v_energy_vec);
        if (point->energy_size == 1) {
          assert(point->median_energy == point->v_energy_vec[0]);
        }
        MatXXf energy_vec;
        energy_vec.resize(point->energy_size, 1);
        for (int ii = 0; ii < point->energy_size; ++ii) {
          energy_vec(ii, 0) = point->v_energy_vec[ii];
        }
        //        std::cout << ", good_cam_num: " << good_cam_num << ",
        //        host_lvl: " << lvl
        //                  << ", target_lvl: " << lvl_target << ", iter: " <<
        //                  iter
        //                  << ", i: " << i << ", host_cid: " << point->host_cid
        //                  << ", every uv: " << Vec2f(point->u,
        //                  point->v).transpose()
        //                  << ", v_energy: " << energy_vec.transpose() <<
        //                  std::endl;
        // printf("visible_cids: %d, good_cam_num: %d, cnt: %d\n", visible_cids,
        // good_cam_num, cnt);
        if (ratio > ratio_thr1 && calc_stats_only) {
          assert(visible_cids == good_cam_num);
        }
        assert(std::isfinite(point->max_zncc));
      } else {
        //        std::cout << ", good_cam_num: " << good_cam_num << ",
        //        host_lvl: " << lvl
        //                  << ", target_lvl: " << lvl_target << ", iter: " <<
        //                  iter
        //                  << ", i: " << i << ", host_cid: " << point->host_cid
        //                  << ", every uv: " << Vec2f(point->u,
        //                  point->v).transpose()
        //                  << ", v_energy: 00" << std::endl;
      }
#ifndef USE_ZNCC
      const float photo_err_thr = point->outlierTH * 20;
#else
      const float photo_err_thr = 99999;
#endif
      if (cnt == 0 ||
          (false && ratio > ratio_thr1 && cnt > 0 && target_grad_mean < 3.f) ||
          ratio > ratio_thr2 /*iter >= 35*/ &&
              ((cnt > 0 &&
#if 0
                energy_each_cam >
                    1.5 * point->median_energy /* point->min_energy,
                                                  point->energy[0]*/
#else
                ((1.5 * zncc_avg < point->median_energy) ||
                 (point->max_zncc < 0.2))
#endif
                &&point->energy[0] > 0) ||
               (/* !isGood || energy / static_cast<float>(cnt /
                   MAX_RES_PER_POINT) */
                energy_each_cam > photo_err_thr /*point->outlierTH * 20, 2 20*/
                ||
                /*is_bad_res_count > threshold*/ good_cam_num ==
                    0) /*|| cnt == 0*/
               || (ratio > ratio_thr1 && calc_stats_only && visible_cids > 0 &&
                   (err_avg > 80.0 || zncc_avg < -0.1 /*zncc_thr[lvl]*/)))) {
        E.updateSingle((float)(point->energy[0])); // 上一帧的加进来 //
        // point->isGood = false;
        point->isGood_new = false;
        point->energy_new = point->energy; //上一次的给当前次的
        bad_pid_count++;
        if (bad_pid_count < 100 && cnt > 0) {
          printf("bbbbb, cnt: %d, energy: %f, good_cam_num: %d, host_cid: %d, "
                 "bad_pid_count: %d, target_grad_mean: %f, target_sigma_temp: "
                 "%f, ratio: %f, err_avg: %f, zncc_avg: %f, iter: "
                 "%d\nenergy_each_cam: %f, 20 * outlierTH: %f\n",
                 cnt, energy, good_cam_num, host_cid, bad_pid_count,
                 target_grad_mean, target_sigma_temp, ratio, err_avg, zncc_avg,
                 iter, energy_each_cam, 20 * point->outlierTH);
        } else {
          if (false) {
            printf("ccccc, cnt: %d, bad_pid_count: %d, [lvl_h lvl_t]: [%d %d], "
                   "iter: %d\n",
                   cnt, bad_pid_count, lvl, lvl_target, iter);
          }
        }
        if (bad_pid_count < 100 && cnt > 0) {
          MatXXf energy_vec;
          energy_vec.resize(point->energy_size, 1);
          for (int ii = 0; ii < point->energy_size; ++ii) {
            energy_vec(ii, 0) = point->v_energy_vec[ii];
          }
          std::cout << "i: " << i << ", host_cid: " << point->host_cid
                    << ", uv: " << Vec2f(point->u, point->v).transpose()
                    << ", v_energy: " << energy_vec.transpose() << std::endl;
          std::cout << "a_target_sigma_temp: "
                    << a_target_sigma_temp.transpose() << std::endl;
          std::cout << "a_err_mean: " << a_err_mean.transpose() << std::endl;
          std::cout << "a_err_max: " << a_err_max.transpose() << std::endl;
          std::cout << "a_err_min: " << a_err_min.transpose() << std::endl;

          std::cout << "a_grad_mean: " << a_grad_mean.transpose() << std::endl;
          std::cout << "a_grad_max: " << a_grad_max.transpose() << std::endl;
          std::cout << "a_grad_min: " << a_grad_min.transpose() << std::endl;

          std::cout << "a_zncc_mean: " << a_zncc_mean.transpose() << std::endl;
          if (point->energy[0] > 0 || true) {
            printf("cur err: %f, last err: %f, min_err: %f, med_err: %f, "
                   "zncc_cur: %f\nrate: %f, rate_min: %f, rate_med: %f, "
                   "rate_zncc_med: %f, max_zncc: %f\n\n\n",
                   energy_each_cam, point->energy[0], point->min_energy,
                   point->median_energy, zncc_avg,
                   energy_each_cam / point->energy[0],
                   energy_each_cam / point->min_energy,
                   energy_each_cam / point->median_energy,
                   point->median_energy / zncc_avg, point->max_zncc);
          }
        }
        // printf("i: %d, host: %d, target_sigma_temp: %f\n", i,
        // point->host_cid, target_sigma_temp);
        for (int target_cid_ = 0; target_cid_ < kCameraNumUsed; ++target_cid_) {
          point->is_valid_project[target_cid_] = false;
        }
        continue;
      }
      // printf("aaaaa\n");
      // 内点则加进能量函数
      // add into energy.
      /// energy = sum(weight * residual * residual * (2 - weight));
      for (int target_cid_ = 0; target_cid_ < kCameraNumUsed; ++target_cid_) {
        if (a_cnt[target_cid_] > 0) {
          if (a_cnt[target_cid_] != MAX_RES_PER_POINT) {
            printf("a_cnt[target_cid_]: %d\n", a_cnt[target_cid_]);
            std::exit(2);
          }
          point->is_valid_project[target_cid_] = true;
        } else {
          point->is_valid_project[target_cid_] = false;
        }
      }
      if (cnt > 0) {
        E.updateSingle(energy / static_cast<float>(cnt / MAX_RES_PER_POINT));
        point->isGood_new = true;
        point->energy_new[0] =
            energy / static_cast<float>(cnt / MAX_RES_PER_POINT);
        point->valid_cid_num = 1; // cnt / MAX_RES_PER_POINT;
      } else {
        printf("not valid projection\n");
      }
      //! 因为使用128位相当于每次加4个数, 因此i+=4, 妙啊!
      // update Hessian matrix.
      // update Hessian matrix.
      // update Hessian matrix.
      // acc += dp[0] + dp[4]
      // this trick unroll the loop into 4 blocks and speed up this for loop
      // in x86 SSE instruction set dp0 * dp0 + dp0*dp1 + .. + dp0*r
      //            dp1*dp1 + .. + dp1*r
      //                      ..
      //                           r * r
      // here ((float *) (&dp0)) + i we can see that &dp0 get the address of
      // dp0 convert this address into float * which occupy 4 size_of space.
      // and i is the offsets, which shift size_of 4 for each loop
      // this acc9 is aggregating inside each point, this is just summing up
      // the pattern, it will sum the points also
#if 0
      for (int i = 0; i + 3 < patternNum /** kCameraNumUsed*/;
           i += 4) // this for loop has 2 steps each step step 4 stride.
                   // (align with SSE)
        acc9.updateSSE(_mm_load_ps(((float *)(&dp0)) +
                                   i), // _mm_load_ps load 4 float values
                                       // from pointer address at a time
                       _mm_load_ps(((float *)(&dp1)) + i),
                       _mm_load_ps(((float *)(&dp2)) + i),
                       _mm_load_ps(((float *)(&dp3)) + i),
                       _mm_load_ps(((float *)(&dp4)) + i),
                       _mm_load_ps(((float *)(&dp5)) + i),
                       _mm_load_ps(((float *)(&dp6)) + i),
                       _mm_load_ps(((float *)(&dp7)) + i),
                       _mm_load_ps(((float *)(&r)) + i));

      // 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
      // ((patternNum >> 2) << 2) this will align the patternNum to be n*4
      // which is required by SSE.
      // ((8 >> 2) << 2) = 8 so, this will jump this for loop directly.
      // this loop is prepared for the patternNum more than 8
      // for example if it's 10, then ((10 >> 2) << 2) = 8. i then loop start
      // from 8 to 10
      // TODO 这不是一路自加，pt+=4表示移位，其实算的就是Jt * J和-Jt*b
      // H += H_i
      // H = Jt * J,  b = -Jt * b
      // 老老实实对H和b进行累加
      for (int i = (((patternNum /* * kCameraNumUsed*/) >> 2) << 2);
           i < patternNum /* * kCameraNumUsed*/; i++) {
        acc9.updateSingle((float)dp0[i], (float)dp1[i], (float)dp2[i],
                          (float)dp3[i], (float)dp4[i], (float)dp5[i],
                          (float)dp6[i], (float)dp7[i], (float)r[i]);
      }
#endif
      // printf("ppiidd: %d, host_cid: %d\n", i, host_cid);
    }
    printf("######## [bad / all] npts: [%d %d] , iter: %d, [h_lvl t_lvl]: [%d "
           "%d] ###########\n",
           bad_pid_count, npts, iter, lvl, lvl_target);
  }
  E.finish();
  // printf("E.A: %f\n", E.A);
  // std::exit(-1);
  acc9.finish();

  //????? 这是在干吗???
  // calculate alpha energy, and decide if we cap it.
  Accumulator11 EAlpha;
  EAlpha.initialize();
#if !defined(USE_MULTI_CAM) || defined(ALWAYS_USE_IDP_PRIOR)
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < 1 /*kCameraNumUsed*/; ++target_cid) {
      int npts = level_cid_to_numPoints[lvl][host_cid];
      Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][host_cid];
      for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (!point->isGood_new) // 点不好用之前的
        {
          EAlpha.updateSingle((float)(point->energy[1]));
          //          E.updateSingle(
          //              (float)(point->energy[1])); //!
          //              又是故意这样写的，没用的代码,
          //! it should be EAlpha, not E,
          //! stop bullshitting me!!
        } else {
          // 最开始初始化都是成1
          /// res = 1 - idepth_new
#if 1 // ndef USE_MULTI_CAM
          if (kCameraNumUsed == 1) {
            point->energy_new[1] =
                (point->idepth_new - 1) * (point->idepth_new - 1); //? 什么原理?
          } else {
            point->energy_new[1] =
                (point->idepth_new - point->iR_triangle) *
                (point->idepth_new - point->iR_triangle); //? 什么原理?
          }
#else
          point->energy_new[1] = 0;
#endif
          EAlpha.updateSingle((float)(point->energy_new[1]));
          // E.updateSingle((float)(point->energy_new[1]));
        }
      }
    }
  }
#endif
  EAlpha.finish(); //! 只是计算位移是否足够大
  int point_count = 0;
  for (int id = 0; id < kCameraNumUsed; ++id) {
    point_count += level_cid_to_numPoints[lvl][id];
  }
  //  assert(point_count ==
  //         level_cid_to_numPoints[lvl][kCameraNumUsed - 1] +
  //             level_cid_to_npts_success_offset[lvl][kCameraNumUsed - 1]);

  float alphaEnergy;
  // TODO roger, 此处逻辑是，当平移比较大时，不再满足纯平移假设了，
  //  所以场景平均深度为1的权重要弱一些了，但是iR的假设仍然要存在,
  //  但是多目时的场景平均深度是三角化来的，可以一直加这个ir_triangle的先验,
  //  可以永远不把平移加进惩罚项里面来
  if (kCameraNumUsed == 1 /*|| true*/) {
    alphaEnergy =
        alphaW * (EAlpha.A + refToNew_.translation().squaredNorm() *
                                 (point_count)); // 平移越大, 越容易初始化成功?
  } else {
    alphaEnergy = alphaW * (EAlpha.A); // 平移越大, 越容易初始化成功?
  }

  // printf("AE = %f * %f + %f\n", alphaW, EAlpha.A,
  // refToNew.translation().squaredNorm() * npts);

  // TODO roger, 此处逻辑是，当平移比较大时，不再满足纯平移假设了，
  //  所以场景平均深度为1的权重要弱一些了，但是iR的假设仍然要存在,
  //  但是多目时的场景平均深度是三角化来的，可以一直加这个ir_triangle的先验,
  //  可以永远不把平移加进惩罚项里面来

  // TODO roger, iR和iR_triangle要加两个地方，
  //  一个是energy[2]，统计EAlpha的，
  //  另一个是填在JbBuffer_new[i + h[0] * w[0] * host_cid][9]里的
  // compute alpha opt.
  float alphaOpt;
  if (kCameraNumUsed == 1) {
    if (alphaEnergy > alphaK * (point_count)) // 平移大于一定值
    {
      alphaOpt = 0;
      alphaEnergy = alphaK * (point_count);
    } else {
      alphaOpt = alphaW;
    }
  } else {
    alphaOpt = alphaW;
  }
  acc9SC.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < 1 /*kCameraNumUsed*/; ++target_cid) {
      int npts = level_cid_to_numPoints[lvl][host_cid];
      Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][host_cid];
      for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (!point->isGood_new)
          continue;

        /// hessian约等于JTJ，当变量为1维时，hessian = J^2
        /// 1/(1+sum(dd*dd))=inverse depth hessian entry, while now is just
        /// sum(dd*dd), H_{\beta \beta}
        assert(point->valid_cid_num > 0);
        point->lastHessian_new = JbBuffer_new[i + h[0] * w[0] * host_cid][9] /
                                 static_cast<float>(point->valid_cid_num);

        //? 这又是啥??? 对逆深度的值进行加权? 深度值归一化?
        // 前面Energe加上了（d-1)*(d-1), 所以dd = 1， r += (d-1)
        // TODO 因为初始化阶段idepth的目标值是1，所以idepth的残差就是 weight *
        // (1-idepth)^2了;
        // res = (point->idepth_new - 1); J = 1
        // b = Jt*res = 1 * res = res;
#if !defined(USE_MULTI_CAM) || defined(ALWAYS_USE_IDP_PRIOR)
        if (kCameraNumUsed == 1) {
          JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
              alphaOpt * (point->idepth_new - 1);
        } else {
          JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
              alphaOpt *
              (point->idepth_new -
               /*point->iR */ point->iR_triangle); // TODO Jt * w * res = 1 * w
                                                   // * res
        }
#endif
#if !defined(USE_MULTI_CAM) || defined(ALWAYS_USE_IDP_PRIOR)
        JbBuffer_new[i + h[0] * w[0] * host_cid][9] +=
            alphaOpt; // TODO Jt * w * J = 1 * w *1
                      // TODO, roger, 因为是三角化得来的，
        //  所以无论平移多大，都可以同时把ir_triangle和iR的prior加进来,
        //  单目时，当平移较大时，只能把iR的prior加进来，
        //  而不能把平均深度为1的prior加进来了,
        //  同理，单目时，当平移较大时，不能把iR的prior加进来，
        //  而只能把平均深度为1的prior加进来了
        if (alphaOpt == 0 || kCameraNumUsed > 1) {
          if (kCameraNumUsed == 1 || true) {
            JbBuffer_new[i + h[0] * w[0] * host_cid][8] +=
                couplingWeight * (point->idepth_new - point->iR);
          } else {
            JbBuffer_new[i + h[0] * w[0] * host_cid][8] += couplingWeight * 0;
          }
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] += couplingWeight;
        }
#endif
        /// refer to the equation (17) in DSO, here JbBuffer_new[i][9] is
        /// H^{-1}_{\beta \beta}
        if (kCameraNumUsed == 1) {
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] =
              1 / (1 + JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
        } else {
          // printf("val[9]: [%f %f], orig[9]: %f\n", 1/JbBuffer_new[i + h[0] *
          // w[0] * host_cid][9],1/(1+JbBuffer_new[i + h[0] * w[0] *
          // host_cid][9]), JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
          JbBuffer_new[i + h[0] * w[0] * host_cid][9] =
              1.f / (
#if 1 // def FIX_ZERO_TRANS_IN_INIT
                        1.f +
#endif
                        JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
        }
        //* 9做权重, 计算的是舒尔补项!
        //! dp*dd*(dd^2)^-1*dd*dp

        // H11_ = H11 - H12 * H22^-1 * H12.t ;
        // b1_ = b1 - H12 * H22^-1 * b2 ;
        // std::cout << "(float)JbBuffer_new[i + h[0] * w[0] * host_cid]: " <<
        // JbBuffer_new[i + h[0] * w[0] * host_cid].transpose() << std::endl;
        acc9SC.updateSingleWeighted(
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][0],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][1],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][2],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][3],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][4],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][5],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][6],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][7],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][8],
            (float)JbBuffer_new[i + h[0] * w[0] * host_cid][9]);
      }
    }
  }
  acc9SC.finish();

  // printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num,
  // (int)E.num, (int)acc9SC.num*9);
  H_out = acc9.H.topLeftCorner<8, 8>();       // / acc9.num;
  b_out = acc9.H.topRightCorner<8, 1>();      // / acc9.num;
  H_out_sc = acc9SC.H.topLeftCorner<8, 8>();  // / acc9.num;
  b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;

//  std::cout << "H_out:\n" << H_out << std::endl;
//  std::cout << "b_out:\n" << b_out.transpose() << std::endl;
//  std::cout << "H_out_sc:\n" << H_out_sc << std::endl;
//  std::cout << "b_out_sc:\n" << b_out_sc.transpose() << std::endl;
//??? 啥意思
// t*t*ntps
// 给 t 对应的Hessian, 对角线加上一个数, b也加上
// TODO 加权重
#if 1
  if (kCameraNumUsed == 1
#ifdef FIX_ZERO_TRANS_IN_INIT
      || true
#endif
  ) {
    H_out(0, 0) += alphaOpt * point_count;
    H_out(1, 1) += alphaOpt * point_count;
    H_out(2, 2) += alphaOpt * point_count;

    Vec3f tlog = refToNew_.log().head<3>().cast<float>();
    // TODO roger,
    // 要强制平移为0，因为默认平移初值是0，所以tlog其实是平移的残差，t_err =
    // t_new - t_orig, t_orig = zeros(3， 1)， J_d_terr_d_t = eye(3)，
    // H_d_err_d_t = eye(3) * eye(3) = eye(3)
    // TODO roger, b_new = H * delta_state + b_old, delta_state = tlog
    b_out[0] += tlog[0] * alphaOpt * point_count;
    b_out[1] += tlog[1] * alphaOpt * point_count;
    b_out[2] += tlog[2] * alphaOpt * point_count;
  }
  // Add zero prior to translation.
  // setting_weightZeroPriorDSOInitY is the squared weight of the prior
  // residual.
  if (kCameraNumUsed == 1
#ifdef FIX_ZERO_TRANS_IN_INIT
      || true
#endif
  ) {
    H_out(1, 1) += setting_weightZeroPriorDSOInitY;
    b_out(1) += setting_weightZeroPriorDSOInitY * refToNew_.translation().y();

    H_out(0, 0) += setting_weightZeroPriorDSOInitX;
    b_out(0) += setting_weightZeroPriorDSOInitX * refToNew_.translation().x();
  }
#endif
  /// E.A is actually equal to residual + (1-d)^2, this (1-d)^2 came from all
  /// good tracking points return vector is total error, alphaEnergy, and the
  /// good tracking number point E.num will be npts + num_good_points since the
  /// E += energy is for every point, but E += (d-1)^2 is only when the point is
  /// good so E.A = sum_npts(energy) + sum_isgood((d-1)^2)
  // 能量值, ? , 使用的点的个数
  return Vec3f(E.A, alphaEnergy, E.num);
}

float CoarseInitializer::rescale() {
  float factor = 20 * thisToNext.translation().norm();
  //	float factori = 1.0f/factor;
  //	float factori2 = factori*factori;
  //
  //	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
  //	{
  //		int npts = numPoints[lvl];
  //		Pnt* ptsl = points[lvl];
  //		for(int i=0;i<npts;i++)
  //		{
  //			ptsl[i].iR *= factor;
  //			ptsl[i].idepth_new *= factor;
  //			ptsl[i].lastHessian *= factori2;
  //		}
  //	}
  //	thisToNext.translation() *= factori;

  return factor;
}

//* 计算旧的和新的逆深度与iR的差值, 返回旧的差, 新的差, 数目
///? iR到底是啥呢     答：IR是逆深度的均值，尺度收敛到IR
Vec3f CoarseInitializer::calcEC(int lvl) {
  if (!snapped /*|| kCameraNumUsed > 1*/) {
    int point_count = 0;
    for (int id = 0; id < kCameraNumUsed; ++id) {
      point_count += level_cid_to_numPoints[lvl][id];
    }
#if 0
    return Vec3f(0, 0,
                 level_cid_to_npts_success_offset[lvl][kCameraNumUsed - 1] +
                     level_cid_to_numPoints[lvl][kCameraNumUsed - 1]);
#else
    return Vec3f(0, 0, static_cast<float>(point_count));
#endif
  }
  AccumulatorX<2> E;
  E.initialize();
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    int npts = level_cid_to_numPoints[lvl][cid];
    for (int i = 0; i < npts; i++) {
      Pnt *point = points[lvl] + i + level_cid_to_npts_success_offset[lvl][cid];
      if (!point->isGood_new)
        continue;
      float rOld = (point->idepth - point->iR);
      float rNew = (point->idepth_new - point->iR);
      if (kCameraNumUsed > 1
#ifdef ALWAYS_USE_IDP_PRIOR
          && false
#endif
      ) {
        rOld = 0;
        rNew = 0;
      }
      E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

      // printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
    }
  }
  E.finish();

  // printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1],
  // (float)E.num.numIn1m);
  return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
}

//* 使用最近点来更新每个点的iR, smooth的感觉
void CoarseInitializer::optReg(int lvl) {
  float dist_thr = 30.f * std::pow(2.0, -lvl);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    int npts = level_cid_to_numPoints[lvl][cid];
    Pnt *ptsl = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
    //* 位移不足够则设置iR是1
    if (kCameraNumUsed == 1
#ifdef FIX_ZERO_TRANS_IN_INIT
        || true
#endif
    ) {
      if (!snapped) {
        return;
      }
    }
    for (int i = 0; i < npts; i++) {
      Pnt *point = ptsl + i;
      if (!point->isGood)
        continue;

      float idnn[10];
      int nnn = 0;
      // 获得当前点周围最近10个点, 质量好的点的iR
      for (int j = 0; j < 10; j++) {
        if (point->neighbours[j] == -1) {
          printf("11 npts is less than 10, check detecion, traaceOn or depth "
                 "filter_DSM\n");
          std::exit(1);
          continue;
        }
        Pnt *other = ptsl + point->neighbours[j];
        if (!other->isGood)
          continue;
        if (point->neighboursDistL1[j] > dist_thr) {
          // printf("[%d/%d], level: %d, knn too far, dist: %f\n", j, 10, lvl,
          // point->neighboursDistL1[j] * std::pow(2.0, lvl));
          continue;
        }
        idnn[nnn] = other->iR;
        nnn++;
      }
      // 与最近点中位数进行加权获得新的iR
      if (nnn > 2) {
        std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); // 获得中位数
        point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
      }
    }
  }
}

//* 使用归一化积来更新高层逆深度值
/// from fine level to coarse level
void CoarseInitializer::propagateUp(int srcLvl) {
  assert(srcLvl + 1 < pyrLevelsUsed);
  // set idepth of target
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    int nptss = level_cid_to_numPoints[srcLvl][cid];
    int nptst = level_cid_to_numPoints[srcLvl + 1][cid];
    Pnt *ptss = points[srcLvl] + level_cid_to_npts_success_offset[srcLvl][cid];
    Pnt *ptst =
        points[srcLvl + 1] + level_cid_to_npts_success_offset[srcLvl + 1][cid];

    // set to zero.
    for (int i = 0; i < nptst; i++) {
      Pnt *parent = ptst + i;
      assert(parent->host_cid == cid);
      parent->host_cid = cid;
      parent->iR = 0;
      parent->iRSumNum = 0;
    }
    //* 更新在上一层的parent
    for (int i = 0; i < nptss; i++) {
      Pnt *point = ptss + i;
      assert(point->host_cid == cid);
      point->host_cid = cid;
      if (!point->isGood)
        continue;

      Pnt *parent = ptst + point->parent;
      assert(parent->host_cid == cid);
      parent->host_cid = cid;
      parent->iR +=
          point->iR * point->lastHessian; //! 均值*信息矩阵 ∑ (sigma*u)
      parent->iRSumNum += point->lastHessian; //! 新的信息矩阵 ∑ sigma
    }

    for (int i = 0; i < nptst; i++) {
      Pnt *parent = ptst + i;
      if (parent->iRSumNum > 0) {
        parent->idepth = parent->iR =
            (parent->iR / parent->iRSumNum); //! 高斯归一化积后的均值
        parent->isGood = true;
      }
    }
  }
  optReg(srcLvl + 1); // 使用附近的点来更新IR和逆深度
}

//@ 使用上层信息来初始化下层
//@ param: 当前的金字塔层+1
//@ note: 没法初始化顶层值
void CoarseInitializer::propagateDown(int srcLvl) {
  assert(srcLvl > 0);
  // set idepth of target
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    int nptst = level_cid_to_numPoints[srcLvl - 1][cid]; // 当前层的点数目
    /// source
    Pnt *ptss =
        points[srcLvl] +
        level_cid_to_npts_success_offset[srcLvl][cid]; // 当前层+1, 上一层的点集
    /// target
    Pnt *ptst = points[srcLvl - 1] +
                level_cid_to_npts_success_offset[srcLvl - 1][cid]; // 当前层点集

    for (int i = 0; i < nptst; i++) {
      Pnt *point = ptst + i;              // 遍历当前层的点
      Pnt *parent = ptss + point->parent; // 找到当前点的parrent
      assert(point->host_cid == cid);
      assert(point->host_cid == cid);
      point->host_cid = cid;
      parent->host_cid = cid;
      if (!parent->isGood || parent->lastHessian < 0.1)
        continue;
      if (!point->isGood) {
        // 当前点不好, 则把父点的值直接给它, 并且置位good
        point->iR = point->idepth = point->idepth_new = parent->iR;
        point->isGood = true;
        point->lastHessian = 0;
      } else {
        // 通过hessian给point和parent加权求得新的iR
        /// iR可以看做是深度的值, 使用的高斯归一化积, Hessian是信息矩阵
        /// fusion of father and son's idepth infomation
        float newiR = (point->iR * point->lastHessian * 2 +
                       parent->iR * parent->lastHessian) /
                      (point->lastHessian * 2 + parent->lastHessian);
        point->iR = point->idepth = point->idepth_new = newiR;
      }
    }
  }
  //? 为什么在这里又更新了iR, 没有更新 idepth
  // 感觉更多的是考虑附近点的平滑效果
  optReg(srcLvl - 1); // 当前层
}

//* 低层计算高层, 像素值和梯度
void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
  for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
    int lvlm1 = lvl - 1;
    int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

    Eigen::Vector3f *dINew_l = data[lvl];
    Eigen::Vector3f *dINew_lm = data[lvlm1];
    // 使用上一层得到当前层的值
    for (int y = 0; y < hl; y++)
      for (int x = 0; x < wl; x++)
        dINew_l[x + y * wl][0] =
            0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
                     dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                     dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                     dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
    // 根据像素计算梯度
    for (int idx = wl; idx < wl * (hl - 1); idx++) {
      dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
      dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
    }
  }
}

void CoarseInitializer::setFirst(CalibHessian *HCalib,
                                 FrameHessian *newFrameHessian) {
  //[ ***step 1*** ] 计算图像每层的内参
  makeK(HCalib);
  firstFrame = newFrameHessian;

  PixelSelector sel(w[0], h[0]); // 像素选择
  /// statusMap表示每个特征点在哪一层被提出来。0，2，4层，虽说是float，但其实int就行了吧
  float *statusMap = new float[w[0] * h[0] * kCameraNumUsed];
  bool *statusMapB = new bool[w[0] * h[0] * kCameraNumUsed];

  float densities[] = {0.03, 0.05, 0.15, 0.5, 1}; // 不同层取得点密度
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    for (int lvl = 0; lvl < pyrLevelsUsed;
         lvl++) { //[ ***step 2*** ] 针对不同层数选择大梯度像素,
                  //第0层比较复杂1d,
      // 2d, 4d大小block来选择3个层次的像素
      sel.currentPotential[cid] = 3; // 设置网格大小，3*3大小格
      int npts = 0;                  // 选择的像素数目
      // std::array<int, kCameraNumUsed> cid_to_npts;
      if (lvl == 0) { // 第0层提取特征像素
                      /// npts: the number of extracted points
        /// it will be significantly larger than 2000, we consider these "npts"
        /// as candidates or backups to make sure there will always be enough
        /// cadidate points to assemble the needed 2000 points
        //          for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        level_cid_to_npts[lvl][cid] =
            sel.makeMaps(firstFrame, statusMap + w[0] * h[0] * cid,
                         densities[lvl] * w[0] * h[0], 1, false, 2, cid);
        npts += level_cid_to_npts[lvl][cid];
        //          }
      } else {
        // 其它层则选出goodpoints
        //          for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        level_cid_to_npts[lvl][cid] =
            makePixelStatus(firstFrame->dIp[lvl] + w[lvl] * h[lvl] * cid,
                            statusMapB + w[0] * h[0] * cid, w[lvl], h[lvl],
                            densities[lvl] * w[0] * h[0], cid);
        npts += level_cid_to_npts[lvl][cid];
        //          }
      }
      // printf("lvl: %d, npts: %d\n", lvl, npts);

      // 如果点非空, 则释放空间, 创建新的
      if (points[lvl] != 0 && cid == 0) {
        delete[] points[lvl];
      }
      if (cid == 0) {
        assert(points[lvl] == 0);
        points[lvl] = new Pnt[npts * 200]; // TODO roger,
        // 因为现在只检测了cam0的，我不知道cam1，2，3会检测出多少点，我干脆开一个200倍的空间，防止不够用
      }
      // set idepth map to initially 1 everywhere.
      //      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      int wl = w[lvl], hl = h[lvl]; // 每一层的图像大小
      int offset = 0;
      for (int id = 0; id < cid; ++id) {
        offset += level_cid_to_npts[lvl][id];
      }
      level_cid_to_npts_offset[lvl][cid] = offset;
      Pnt *pl = points[lvl] + offset; // 每一层上的点
      int nl = 0;
      // 要留出pattern的空间, 2 border
      //[ ***step 3*** ] 在选出的像素中, 添加点信息
      /// 对全图做遍历，只有valid(!=0)的点才会执行相关操作
      for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++) {
        for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
          // if(x==2) printf("y=%d!\n",y);
          // 如果是被选中的像素
          if ((lvl != 0 && statusMapB[x + y * wl + w[0] * h[0] * cid]) ||
              (lvl == 0 && statusMap[x + y * wl + w[0] * h[0] * cid] != 0)) {
            // printf("aa, nl: %d\n", nl);
            // assert(patternNum==9);
            pl[nl].u = x + 0.1; //? 加0.1干啥
            pl[nl].v = y + 0.1;
            pl[nl].idepth = 1;
            pl[nl].iR = 1;
            pl[nl].isGood = true;
            pl[nl].energy.setZero();
            pl[nl].lastHessian = 0;
            pl[nl].lastHessian_new = 0;
            pl[nl].my_type =
                (lvl != 0) ? 1 : statusMap[x + y * wl + w[0] * h[0] * cid];
            pl[nl].host_cid = cid;
            Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl] +
                                   wl * hl * cid; // 该像素梯度
            float sumGrad2 = 0;
            // 计算pattern内像素梯度和
            for (int idx = 0; idx < patternNum; idx++) {
              int dx = patternP[idx][0]; // pattern 的偏移
              int dy = patternP[idx][1];
              float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
              sumGrad2 += absgrad;
            }

            //				float gth = setting_outlierTH *
            //(sqrtf(sumGrad2)+setting_outlierTHSumComponent);
            // pl[nl].outlierTH = patternNum*gth*gth;
            //
            //! 外点的阈值与pattern的大小有关, 一个像素是12*12
            //? 这个阈值怎么确定的...
            pl[nl].outlierTH =
                /*patternNum * kCameraNumUsed * */ setting_outlierTH;

            nl++;
            assert(nl <= level_cid_to_npts[lvl][cid] /*npts*/);
          }
        }
      }
      //      level_cid_to_npts_success[lvl][cid] = nl;
      level_cid_to_numPoints[lvl][cid] = nl; // 点的数目,  去掉了一些边界上的点
      //      int offset_success = 0;
      //      for (int id = 0; id < cid; ++id) {
      //        offset_success += level_cid_to_npts_success[lvl][id];
      //      }
      //      level_cid_to_npts_success_offset[lvl][cid] = offset_success;
      level_cid_to_npts_success_offset[lvl][cid] =
          level_cid_to_npts_offset[lvl][cid];
      //      }
    }
  }
  delete[] statusMap;
  delete[] statusMapB;
  //[ ***step 4*** ] 计算点的最近邻和父点
  /// nearest neighbours?
  /// // build kdtree of selected points in each lvl and find nearest neighbours
  /// in same lvl and parent lvl (smaller scaled layer)
  makeNN();
  // 参数初始化
  thisToNext = SE3();
  Rwb = Mat33::Identity();
  snapped = false;
  frameID = snappedAt = 0;

  for (int i = 0; i < pyrLevelsUsed; i++)
    dGrads[i].setZero();
}
float CoarseInitializer::FindMedian(const std::vector<float> &numbers) {
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
double CoarseInitializer::MultiViewTriangulation(
    const double &focal, const std::vector<Mat4> &poses,
    const std::vector<Vec3> &points,
    std::vector<std::pair<double, int>> &err_vec, VecX &errs, Vec3 &point_3d) {
  // TODO:Rewrite this for 3d point
  Eigen::MatrixXd design_matrix(poses.size() * 2, 4);
  assert(poses.size() > 0 && poses.size() == points.size() &&
         "We at least have 2 poses and number of pts and poses must equal");
  for (unsigned int i = 0; i < poses.size(); i++) {
    double p0x = points[i][0];
    double p0y = points[i][1];
    double p0z = points[i][2];
    design_matrix.row(i * 2) = p0x * poses[i].row(2) - p0z * poses[i].row(0);
    design_matrix.row(i * 2 + 1) =
        p0y * poses[i].row(2) - p0z * poses[i].row(1);
  }
  Vec4 triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);

  Eigen::MatrixXd pts(4, 1);
  pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
  errs = design_matrix * pts;

  //    std::cout << "design_matrix: " << design_matrix << std::endl;
  //    std::cout << "ERR: " << errs.sum() << ", bearing: " <<
  //    points[0].transpose() << ", pt3d: "<< point_3d.transpose() << std::endl;
  //    std::cout <<"err each: " << errs.transpose() << std::endl;
  double err_sum = 0;
  for (size_t i = 0; i < poses.size(); i++) {
    Vec3 rep = (poses[i].topLeftCorner<3, 3>() * point_3d +
                poses[i].topRightCorner<3, 1>())
                   .normalized();
    err_vec[i].first = focal * (rep - points[i]).norm();
    err_sum += err_vec[i].first;
    //      Vec3 rep2 = rep / rep(2);
    //      Vec3 obs = points[i] / points[i](2);
    //      std::cout<<"rep: " << rep.transpose() << ", obs: " <<
    //      points[i].transpose()<<", err:
    //      "<<static_cast<double>(kFocalLength)*(rep - points[i]).norm()<<",
    //      err2:
    //      "<<static_cast<double>(kFocalLength)*(rep2 - obs).norm()<<std::endl;
  }
  // return static_cast<double>(kFocalLength) * errs.norm() /
  // static_cast<double>(errs.rows());
  return err_sum / static_cast<double>(poses.size());
}
void CoarseInitializer::convert_to_ImageData(cv::Mat &data,
                                             ImageDataAM &image_data,
                                             uint8_t camera_id) {

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
//#define SHOW_DETECTION_RES
void CoarseInitializer::setFirstStereo(CalibHessian *HCalib,
                                       FrameHessian *newFrameHessian) {
  assert(kCameraNumUsed > 1);
  //[ ***step 1*** ] 计算图像每层的内参
  makeK(HCalib);
  firstFrame = newFrameHessian;

  PixelSelector sel(w[0], h[0]); // 像素选择
  /// statusMap表示每个特征点在哪一层被提出来。0，2，4层，虽说是float，但其实int就行了吧
  float *statusMap = new float[w[0] * h[0] * kCameraNumUsed];
  bool *statusMapB = new bool[w[0] * h[0] * kCameraNumUsed];
  Vec2f aff = AffLight::fromToVecExposure(
                  firstFrame->ab_exposure, firstFrame->ab_exposure,
                  firstFrame->aff_g2l(), firstFrame->aff_g2l())
                  .cast<float>();
  float densities[] = {0.03, 0.05, 0.15, 0.5, 1}; // 不同层取得点密度
  Vec3 point_3d_wcs;
  Vec3 point_3d_ccs;
  std::vector<std::pair<double, int>> err_vec = {std::pair<double, int>(0, 1),
                                                 std::pair<double, int>(0, 2),
                                                 std::pair<double, int>(0, 3)};
#ifdef SHOW_DETECTION_RES
  MinimalImageB3 *img_host[pyrLevelsUsed];
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    img_host[lvl] = new MinimalImageB3(wG[lvl], hG[lvl]);
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(firstFrame->dIp[lvl] + wG[lvl] * hG[lvl] * cid + i))[0];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        img_host[lvl]->at(i, cid) = Vec3b(colL, colL, colL);
      }
    }
  }

#endif
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    for (int lvl = 0; lvl < pyrLevelsUsed;
         lvl++) { //[ ***step 2*** ] 针对不同层数选择大梯度像素,
                  //第0层比较复杂1d,
      // 2d, 4d大小block来选择3个层次的像素
      sel.currentPotential[cid] = 3; // 设置网格大小，3*3大小格
      int npts = 0;                  // 选择的像素数目
      // std::array<int, kCameraNumUsed> cid_to_npts;
      if (lvl == 0) { // 第0层提取特征像素
        /// npts: the number of extracted points
        /// it will be significantly larger than 2000, we consider these "npts"
        /// as candidates or backups to make sure there will always be enough
        /// cadidate points to assemble the needed 2000 points
        //      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        level_cid_to_npts[lvl][cid] =
            sel.makeMaps(firstFrame, statusMap + w[0] * h[0] * cid,
                         densities[lvl] * w[0] * h[0], 1, false, 2, cid);
        npts += level_cid_to_npts[lvl][cid];
        //      }
      } else {
        // 其它层则选出goodpoints
        //      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        //        printf("enter\n");
        level_cid_to_npts[lvl][cid] =
            makePixelStatus(firstFrame->dIp[lvl] + w[lvl] * h[lvl] * cid,
                            statusMapB + w[0] * h[0] * cid, w[lvl], h[lvl],
                            densities[lvl] * w[0] * h[0], cid);
        //          printf("exit\n");
        npts += level_cid_to_npts[lvl][cid];
        //      }
      }
      printf("lvl: %d, npts: %d\n", lvl, npts);
      // 如果点非空, 则释放空间, 创建新的
      if (points[lvl] != 0 && cid == 0) {
        delete[] points[lvl];
        //        points[lvl] = new Pnt
        //            [npts *
        //             200]; // TODO roger,
        // 因为现在只检测了cam0的，我不知道cam1，2，3会检测出多少点，我干脆开一个200倍的空间，防止不够用
      }
      if (cid == 0) {
        assert(points[lvl] == 0);
        points[lvl] = new Pnt[npts * 10 * kCameraNumUsed]; // TODO roger,
        // 因为现在只检测了cam0的，我不知道cam1，2，3会检测出多少点，我干脆开一个200倍的空间，防止不够用
      }
      // set idepth map to initially 1 everywhere.
      //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      int wl = w[lvl], hl = h[lvl]; // 每一层的图像大小
      dso::ImagesBuffer::Initial(40, wl, hl);
#if 1
      std::array<ImageDataAM, kCameraNumUsed> cid_to_image_data;

      Point point;
      bool is_corner;
      std::array<std::shared_ptr<AlgsImage>, kCameraNumUsed> cid_to_img;
      std::array<cv::Mat, kCameraNumUsed> cid_to_cv_img;
      MinimalImageB *img_target;
      img_target = new MinimalImageB(wl, hl);
      for (int cam = 0; cam < kCameraNumUsed; ++cam) {
        firstFrame->p_multi_camera->cid_to_K_temp[cam].setIdentity();
        firstFrame->p_multi_camera->cid_to_K_temp[cam](0, 0) = fx[lvl];
        firstFrame->p_multi_camera->cid_to_K_temp[cam](1, 1) = fy[lvl];
        firstFrame->p_multi_camera->cid_to_K_temp[cam](0, 2) = cx[lvl];
        firstFrame->p_multi_camera->cid_to_K_temp[cam](1, 2) = cy[lvl];
        std::vector<number_t> param = {fx[lvl], fy[lvl], cx[lvl], cy[lvl]};
        firstFrame->p_multi_camera->cid_to_Kinv_temp[cam] =
            firstFrame->p_multi_camera->cid_to_K_temp[cam].inverse();
        firstFrame->p_multi_camera->cid_to_cam_pinhole[cam] =
            new PinholeCamera(cam, wl, hl, param.data());

        for (size_t i = 0; i < kCameraNumUsed; ++i) {
          p_depth_filter_DSM_->px_err_angle_vec_[i] =
              std::atan(p_depth_filter_DSM_->px_noise_ / fx[lvl]);
        }

        assert(estimator_config_.search_level == 0);
        number_t search_level_focal_length =
            fx[lvl] * std::pow(2.0f, -estimator_config_.search_level);

        p_depth_filter_DSM_->p_multi_cam_epipolar_search_->rad_step_ =
            estimator_config_.pixel_step *
            std::asin(1.0 / search_level_focal_length / 2.0) * 2.0;

        Eigen::Vector3f *colorCur = firstFrame->dIp[lvl] + cam * wl * hl;
        for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
          // BRIGHTNESS TRANSFER
          float colL = (*(colorCur + i))[0];
          if (colL < 0)
            colL = 0;
          if (colL > 255)
            colL = 255;
          img_target->at(i, cam) = static_cast<unsigned char>(colL);
        }
        cid_to_cv_img[cam] =
            cv::Mat(img_target->h, img_target->w, CV_8UC1,
                    img_target->data + img_target->w * img_target->h * cam);

        convert_to_ImageData(cid_to_cv_img[cam], cid_to_image_data[cam],
                             static_cast<uint8_t>(cam));

        cid_to_img[cam] = dso::ImagesBuffer::Acquire(wl, hl);
        cid_to_img[cam]->DangerouslyCopyFrom(
            cid_to_image_data[cam].width, cid_to_image_data[cam].height,
            cid_to_image_data[cam].step, cid_to_image_data[cam].data,
            cid_to_image_data[cam].exposure_ts,
            cid_to_image_data[cam].tuning_index,
            cid_to_image_data[cam].shutter_speed_ns,
            cid_to_image_data[cam].gain);
        // cv::imshow("img", cid_to_img[cam]);
        // cv::waitKey(0);
      }

#endif
      int offset = 0;
      for (int id = 0; id < cid; ++id) {
        offset += level_cid_to_npts[lvl][id];
      }
      level_cid_to_npts_offset[lvl][cid] = offset;
      Pnt *pl = points[lvl] + offset; // 每一层上的点
      int nl = 0;
      // 要留出pattern的空间, 2 border
      //[ ***step 3*** ] 在选出的像素中, 添加点信息
      /// 对全图做遍历，只有valid(!=0)的点才会执行相关操作
      int trials = 0;
      for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++) {
        for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
          trials++;
          // if(x==2) printf("y=%d!\n",y);
          // 如果是被选中的像素
          if ((lvl != 0 && statusMapB[x + y * wl + w[0] * h[0] * cid]) ||
              (lvl == 0 && statusMap[x + y * wl + w[0] * h[0] * cid] != 0)) {
            float my_type =
                (lvl != 0) ? 1 : statusMap[x + y * wl + w[0] * h[0] * cid];
            //            std::cout << "x: " << x << ", y: " << y << ", nl: " <<
            //            nl << ", nums: " << level_cid_to_npts[lvl][cid]<<",
            //            all pts: " << npts << ", trials: "<< trials<<
            //            std::endl;
            ImmaturePoint *pt =
                new ImmaturePoint(x, y, firstFrame, my_type, HCalib, cid, lvl);
            // pt->idepth_min = 0.5;
            if (pt->energyTH == NAN) {
              delete pt;
              continue;
            }
            int success_count = 0;
            std::vector<Vec2f> best_uvs;
            std::vector<Mat4> Tcws = {
                firstFrame->p_multi_camera->cid_to_T01[cid].inverse()};
            std::vector<Vec3> bearings = {
                (Ki[lvl] * Vec3(x, y, 1.0)).normalized()};
            if (kCameraNumUsed > 1) {
#if 1
              bool good_point = true;
              for (int target_cid = 0; target_cid < kCameraNumUsed;
                   ++target_cid) {
                if (target_cid == cid) {
                  continue;
                }
                SE3 hostToNew =
                    firstFrame->p_multi_camera->cid_to_T01_SE3[target_cid]
                        .inverse() *
                    firstFrame->p_multi_camera->cid_to_T01_SE3[cid];
                Mat33f KRKi =
                    (K[lvl] * hostToNew.rotationMatrix() * K[lvl].inverse())
                        .cast<float>();
                Vec3f Kt = (K[lvl] * hostToNew.translation()).cast<float>();
                ImmaturePointStatus stat =
                    pt->traceOn(target_cid, firstFrame, KRKi, Kt, aff, HCalib,
                                false, lvl, true, lvl == 100);
                if (stat == ImmaturePointStatus::IPS_GOOD) {
                  best_uvs.emplace_back(pt->lastTraceUV[target_cid]);
                  Tcws.emplace_back(
                      firstFrame->p_multi_camera->cid_to_T01[target_cid]
                          .inverse());
                  bearings.emplace_back(
                      (Ki[lvl] *
                       Vec3(best_uvs.back().x(), best_uvs.back().y(), 1.0))
                          .normalized());
                  success_count++;
                }
              }
              if (success_count >= 1) {
                VecX errs;
                double triang_arr = 0;
                triang_arr = MultiViewTriangulation(
                    K[lvl](0, 0), Tcws, bearings, err_vec, errs, point_3d_wcs);
                point_3d_ccs =
                    firstFrame->p_multi_camera->cid_to_T01_SE3_inv[cid] *
                    point_3d_wcs;
                //                printf(
                //                    "success_count: %d, lvl: %d, triang_arr:
                //                    %f, [mean_depth " "triang_depth]: [%f
                //                    %f]\n", success_count, lvl, triang_arr,
                //                    1.0 / (0.5 * (pt->idepth_max +
                //                    pt->idepth_min)), point_3d_ccs[2]);
                float idepth_mean = 0.5 * (pt->idepth_max + pt->idepth_min);
                if (point_3d_ccs[2] < 0.01 || idepth_mean < 0.01) {
                  // printf("failed epipolar search\n");
                  delete pt;
                  continue;
                }
                if (false) {
                  pl[nl].iR_triangle = idepth_mean;
                  pl[nl].idepth_new_triangle = idepth_mean;
                  //  0.5 * (pt->idepth_max + pt->idepth_min);
                } else {
                  pl[nl].iR_triangle = 1.0 / point_3d_ccs[2];
                  pl[nl].idepth_new_triangle = 1.0 / point_3d_ccs[2];
                }
                pl[nl].idepth = pl[nl].iR_triangle;
                pl[nl].iR = pl[nl].iR_triangle;
                // printf("success epipolar search\n");
              } else {
                // printf("failed in crosss image epipolar search\n");
                // pl[nl].idepth = pl[nl].iR_triangle = 1;
                // pl[nl].iR = pl[nl].iR_triangle = 1;
                delete pt;
                continue;
              }
#else
              Vec2i px = Vec2i(x, y);
              point.n =
                  (firstFrame->p_multi_camera->cid_to_Kinv_temp.at(cid) *
                   Vec3(static_cast<number_t>(x), static_cast<number_t>(y), 1.))
                      .normalized();
              bool is_success = point.pyramid_patch.SetFromImg(
                  cid_to_img[cid], px.cast<number_t>(), cid, is_corner,
                  firstFrame->p_multi_camera);
              // printf("is_success: %d, is_corner: %d\n", is_success,
              // is_corner); std::exit(1);
              if (is_success) {
                // number_t init_idp = 0.5;
                Seed seed;
                DF_Frame::InitSeedDepth(seed, false, 0);
                number_t res_idp;
                std::array<MultiCameraEpipolarSearch::MatchRes, kCameraNumUsed>
                    cid_to_output;
                MultiCameraEpipolarSearch::State state =
                    p_depth_filter_DSM_->p_multi_cam_epipolar_search_
                        ->FindEpipolarMatch(point, cid, cid_to_img, 1, seed.rho,
                                            seed.sigma2, 1, cid_to_output,
                                            res_idp, -1, true);
                if (state == MultiCameraEpipolarSearch::kReject) {
                  // printf("reject\n");
                  delete pt;
                  continue;
                } else if (state == MultiCameraEpipolarSearch::kUnVisible) {
                  // printf("not visible\n");
                  delete pt;
                  continue;
                } else if (state == MultiCameraEpipolarSearch::kFail) {
                  // printf("fail\n");
                  delete pt;
                  continue;
                } else if (state == MultiCameraEpipolarSearch::kSuccess) {
                  // printf("success_triangulation\n");
                  Vec3 xyz = point.n / res_idp;
                  pl[nl].iR_triangle = 1.0 / xyz[2];
                  pl[nl].idepth_new_triangle = 1.0 / xyz[2];

                  pl[nl].idepth = pl[nl].iR_triangle;
                  pl[nl].iR = pl[nl].iR_triangle;
                } else {
                  printf("you should never see this\n");
                  std::exit(1);
                }
              } else {
                delete pt;
                continue;
              }
#endif
            } else {
              printf("you should never see this, only in multi-cam mode\n");
              std::exit(1);
              pl[nl].idepth = 1;
              pl[nl].iR = 1;
            }

            // assert(patternNum==9);
            pl[nl].u = x + 0.1; //? 加0.1干啥
            pl[nl].v = y + 0.1;
            //            pl[nl].idepth = 1;
            //            pl[nl].iR = 1;
            pl[nl].isGood = true;
            pl[nl].energy.setZero();
            pl[nl].lastHessian = 0;
            pl[nl].lastHessian_new = 0;
            pl[nl].my_type = my_type;

#ifdef SHOW_DETECTION_RES
            img_host[lvl]->setPixel9(x + 0.5, y + 0.5,
                                     makeRainbow3B(pl[nl].idepth), cid);
#endif

            pl[nl].host_cid = cid;
            Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl] +
                                   wl * hl * cid; // 该像素梯度
            float sumGrad2 = 0;
            // 计算pattern内像素梯度和
            for (int idx = 0; idx < patternNum; idx++) {
              int dx = patternP[idx][0]; // pattern 的偏移
              int dy = patternP[idx][1];
              float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
              sumGrad2 += absgrad;
            }

            //				float gth = setting_outlierTH *
            //(sqrtf(sumGrad2)+setting_outlierTHSumComponent);
            // pl[nl].outlierTH = patternNum*gth*gth;
            //
            //! 外点的阈值与pattern的大小有关, 一个像素是12*12
            //? 这个阈值怎么确定的...
            pl[nl].outlierTH =
                /*patternNum * kCameraNumUsed * */ setting_outlierTH;
            //              printf("reaching end, nl: %d\n", nl);
            nl++;
            assert(nl <= level_cid_to_npts[lvl][cid] /*npts*/);
            delete pt;
          }
        }
      }
      printf("lvl: %d, cid: %d, valid seeds: %d\n", lvl, cid, nl);
      // level_cid_to_npts_success[lvl][cid] = nl;
      level_cid_to_numPoints[lvl][cid] = nl; // 点的数目,  去掉了一些边界上的点
      //      int offset_success = 0;
      //      for (int id = 0; id < cid; ++id) {
      //        offset_success += level_cid_to_npts_success[lvl][id];
      //      }
      // level_cid_to_npts_success_offset[lvl][cid] = offset_success;
      level_cid_to_npts_success_offset[lvl][cid] =
          level_cid_to_npts_offset[lvl][cid];
      //    }
      //      printf("level end\n");
      // std::exit(1);

      if (false) {
        std::vector<cv::Mat> colored_mat_vec(kCameraNumUsed);
        cv::Mat gray_mat;
        for (size_t cam_id = 0; cam_id < kCameraNumUsed; ++cam_id) {
          std::shared_ptr<AlgsImage> p_img = cid_to_img[cam_id];
          gray_mat = cv::Mat(p_img->height, p_img->width, CV_8UC1, p_img->data,
                             p_img->stride);
          cv::cvtColor(gray_mat, colored_mat_vec[cam_id], cv::COLOR_GRAY2BGR);
        }
        cv::Mat temp1, temp2, res;
        cv::vconcat(colored_mat_vec[1], colored_mat_vec[0], temp1);
        cv::vconcat(colored_mat_vec[2], colored_mat_vec[3], temp2);
        cv::hconcat(temp1, temp2, res);
        cv::imshow("VM Result", res);
        cv::waitKey(0);
      }
      dso::ImagesBuffer::SetInitial(false);
      for (int cam = 0; cam < kCameraNumUsed; ++cam) {
        cid_to_img[cam].reset();
      }
      delete img_target;
    }
  }
  // std::exit(1);
#ifdef SHOW_DETECTION_RES
  for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl) {
    std::array<cv::Mat, kCameraNumUsed> show_mat_vec;
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      show_mat_vec[cid] = cv::Mat(
          img_host[lvl]->h, img_host[lvl]->w, CV_8UC3,
          img_host[lvl]->data + img_host[lvl]->w * img_host[lvl]->h * cid);
      //            cv::cvtColor(show_mat_vec[cid], show_mat_vec[cid],
      //            cv::COLOR_GRAY2BGR);
    }
    cv::Mat img1, img2, img_show;
    cv::hconcat(show_mat_vec[1], show_mat_vec[2], img1);
    cv::hconcat(show_mat_vec[0], show_mat_vec[3], img2);
    cv::vconcat(img1, img2, img_show);
    cv::imshow("level_" + std::to_string(lvl), img_show);
  }
  cv::waitKey(0);
  printf("before release\n");
  for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl) {
    delete img_host[lvl];
  }
  printf("after release\n");
  // delete img_host;
#endif
  delete[] statusMap;
  delete[] statusMapB;
  //[ ***step 4*** ] 计算点的最近邻和父点
  /// nearest neighbours?
  /// // build kdtree of selected points in each lvl and find nearest neighbours
  /// in same lvl and parent lvl (smaller scaled layer)
  makeNN();
  // 参数初始化
  thisToNext = SE3();
  Rwb = Mat33::Identity();
  snapped = false;
  frameID = snappedAt = 0;

  for (int i = 0; i < pyrLevelsUsed; i++)
    dGrads[i].setZero();
}

//@ 重置点的energy, idepth_new参数
/// 把每个点的深度值按10个neighbor做归一化
void CoarseInitializer::resetPoints(int lvl) {
  float dist_thr = 30.f * std::pow(2.0, -lvl);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    Pnt *pts = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
    // int npts = level_cid_to_npts[lvl][cid];
    int npts = level_cid_to_numPoints[lvl][cid];
    for (int i = 0; i < npts; i++) { // 重置
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        pts[i].is_valid_project[target_cid] = false;
      }
      pts[i].v_energy_vec.clear();
      pts[i].energy_size = 0;
      pts[i].min_energy = pts[i].median_energy = 999999;
      pts[i].max_zncc = -999999;
      pts[i].energy.setZero();
      pts[i].idepth_new = pts[i].idepth;
      assert(pts[i].host_cid == cid);
      pts[i].host_cid = cid;
      // 如果是最顶层, 则使用周围点平均值来重置
      /// the lowest resolution
      if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
        float snd = 0, sn = 0;
        for (int n = 0; n < 10; n++) {
          if (pts[i].neighbours[n] == -1) {
            printf("22 npts is less than 10, check detecion, traaceOn or depth "
                   "filter_DSM\n");
            std::exit(2);
            continue;
          }
          if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood)
            continue;

          if (pts[i].neighboursDistL1[n] > dist_thr) {
            // printf("22 [%d/%d], level: %d, knn too far, dist: %f\n", j, 10,
            // lvl, point->neighboursDistL1[j] * std::pow(2.0, lvl));
            continue;
          }
          /// 逆深度求和
          snd += pts[pts[i].neighbours[n]].iR;
          sn += 1;
        }
#if 1 // ndef USE_MULTI_CAM
        if (sn > 0) {
          pts[i].isGood = true;
          /// normalize idepth to 1
          pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
        }
#else
        pts[i].isGood = true;
        pts[i].iR = pts[i].idepth;
#endif
      }
    }
  }
}

//* 求出状态增量后, 计算被边缘化掉的逆深度, 更新逆深度
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {

  const float maxPixelStep = 0.25;
  const float idMaxStep = 1e10;

  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    Pnt *pts = points[lvl] + level_cid_to_npts_success_offset[lvl][target_cid];
    int npts = level_cid_to_numPoints[lvl][target_cid];
    for (int i = 0; i < npts; i++) {
      if (!pts[i].isGood)
        continue;

      //! dd*r + (dp*dd)^T*delta_p
      // linearized plus with prior
      // todo roger, b2 = b2_sc + H12.transpose() * dx1;
      if (pts[i].host_cid != target_cid) {
        printf("cid not consistent, sth wrong\n");
        std::exit(1);
      }
      float b = (JbBuffer[i + w[0] * h[0] * pts[i].host_cid /* *
                                  kCameraNumUsed*/
                          /*+ target_cid)*/])[8] +
                JbBuffer[i].head<8>().dot(inc);
      //! dd * delta_d = dd*r - (dp*dd)^T*delta_p = b
      //! delta_d = b * dd^-1
      // todo roger, dx2 = -b2 / H22;
      float step =
          -b *
          JbBuffer[i + w[0] * h[0] * pts[i].host_cid /* * kCameraNumUsed*/][9] /
          (1 + lambda);
      // TODO
      // 这应该只是粗略的更新depth，要严谨点应该用[三角化]，可能是出于计算量考虑吧，毕竟还在初始化阶段，本身也算不了太准（毕竟idepth都全被初始化成1的正态分布了，能准到哪里去）

      float maxstep = maxPixelStep * pts[i].maxstep; // 逆深度最大只能增加这些
      if (maxstep > idMaxStep)
        maxstep = idMaxStep;

#if 1 // ndef USE_MULTI_CAM
      if (step > maxstep) {
        // printf("+max step\n");
        step = maxstep;
      }
      if (step < -maxstep) {
        // printf("-max step\n");
        step = -maxstep;
      }
#endif
      // 更新得到新的逆深度
      float newIdepth = pts[i].idepth + step;
      // TODO 不是说idepth全假设在1附近吗，怎么又有0.001和50差别这么大范围？？
      // TODO 遇到这种情况再不济也应该把isGood置成0呀
      if (newIdepth < 1e-3)
        newIdepth = 1e-3;
      if (newIdepth > 50)
        newIdepth = 50;
      if (!std::isfinite(newIdepth)) {
        printf("newIdepth: %f, oldIdepth: %f,step: %f, i: %d, lvl: %d, cid: "
               "%d, uv: [%f %f]\n",
               newIdepth, pts[i].idepth, step, i, lvl, pts[i].host_cid,
               pts[i].u, pts[i].v);
        //                  std::cout <<
        //                  "JbBuffer[i + w[0] * h[0] * pts[i].host_cid *
        //                  kCameraNumUsed]: " << JbBuffer[i + w[0] * h[0] *
        //                  pts[i].host_cid * kCameraNumUsed].transpose() <<
        //                  std::endl; std::cout << "JbBuffer_new[i + w[0] *
        //                  h[0] * pts[i].host_cid
        //                  * kCameraNumUsed]: " << JbBuffer_new[i + w[0] * h[0]
        //                  * pts[i].host_cid * kCameraNumUsed].transpose() <<
        //                  std::endl;
        std::cout << "JbBuffer[i + w[0] * h[0] * pts[i].host_cid]: "
                  << JbBuffer[i + w[0] * h[0] * pts[i].host_cid].transpose()
                  << std::endl;
        std::cout << "JbBuffer_new[i + w[0] * h[0] * pts[i].host_cid]: "
                  << JbBuffer_new[i + w[0] * h[0] * pts[i].host_cid].transpose()
                  << std::endl;
        std::exit(1);
      }
      pts[i].idepth_new = newIdepth;
    }
  }
}

//* 新的值赋值给旧的 (能量, 点状态, 逆深度, hessian)
void CoarseInitializer::applyStep(int lvl) {
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    Pnt *pts = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
    int npts = level_cid_to_numPoints[lvl][cid];
    for (int i = 0; i < npts; i++) {
      if (!pts[i].isGood) {
        // TODO roger,
        // iR比iR_triangulate更好，因为iR是融合了周围idp均值的，会更稳定一些
#if 1 // ndef USE_MULTI_CAM
        pts[i].idepth = pts[i].idepth_new = pts[i].iR;
#else
        pts[i].idepth = pts[i].idepth_new = pts[i].idepth_new_triangle =
            pts[i].iR_triangle;
#endif
        continue;
      }
      pts[i].energy = pts[i].energy_new;
      pts[i].isGood = pts[i].isGood_new;
      pts[i].idepth = pts[i].idepth_new;
#if 0 // def USE_MULTI_CAM
      pts[i].iR_triangle = pts[i].idepth_new_triangle = pts[i].idepth_new;
#endif
      pts[i].lastHessian = pts[i].lastHessian_new;
    }
  }
  //  std::cout
  //      << "aa JbBuffer[i + w[0] * h[0] * pts[i].host_cid * kCameraNumUsed]: "
  //      << JbBuffer[178 + w[0] * h[0] * 1 * kCameraNumUsed].transpose()
  //      << std::endl;
  //  std::cout
  //      << "bb JbBuffer_new[i + w[0] * h[0] * pts[i].host_cid *
  //      kCameraNumUsed]: "
  //      << JbBuffer_new[178 + w[0] * h[0] * 1 * kCameraNumUsed].transpose()
  //      << std::endl;
  std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
}

//@ 计算每个金字塔层的相机参数
void CoarseInitializer::makeK(CalibHessian *HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();
  // 求各层的K参数
  for (int level = 1; level < pyrLevelsUsed; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    //* 0.5 offset 看README是设定0.5到1.5之间积分表示1的像素值？
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
    printf("lvl: %d, w: %d, h: %d\n", level, w[level], h[level]);
  }
  // 求K_inverse参数
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

// find the nearest neighbor of selected points (in the smaller scale space)
// from the it's larger parent scale space. detailed original code can be found
// from:
// https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_example.cpp
// which is the author's repo of nanoflann
//@ 生成每一层点的KDTree, 并用其找到邻近点集和父点
//#define SHOW_NN
#ifdef SHOW_NN
//#define SHOW_NN_DETAIL
#endif
void CoarseInitializer::makeNN() {
  const float NNDistFactor = 0.05;
  // 第一个参数为distance, 第二个是datasetadaptor, 第三个是维数
  /// construct a kd-tree index:
  /// idex take 4 type of templates as parameter: 1. the distance class, 2. the
  /// data-source class 3. dimension (default -1), 4. index type (default size_t
  /// is type returned by the sizeof operator). so this KDTree index is a 2d
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>, FLANNPointcloud, 2>
      KDTree;
#ifdef SHOW_NN
  MinimalImageB3 *img_big;
  MinimalImageB3 *img_small;
#endif
  // build indices
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    FLANNPointcloud pcs[PYR_LEVELS]; // 每层建立一个点云
    KDTree *indexes[PYR_LEVELS];     // 点云建立KDtree
    //* 每层建立一个KDTree索引二维点云
    for (int i = 0; i < pyrLevelsUsed; i++) {
      pcs[i] = FLANNPointcloud(
          level_cid_to_numPoints[i][cid],
          points[i] + level_cid_to_npts_success_offset[i][cid]); // 二维点点云
      // 参数: 维度, 点数据, 叶节点中最大的点数(越大build快, query慢)
      indexes[i] =
          new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
      indexes[i]->buildIndex();
    }

    const int nn = 10;

    // find NN & parents
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
#ifdef SHOW_NN
      img_big = new MinimalImageB3(wG[lvl], hG[lvl]);
      Vec3f *colorRef = firstFrame->dIp[lvl] + wG[lvl] * hG[lvl] * cid;
      for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(colorRef + i))[0];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        img_big->at(i, cid) = Vec3b(colL, colL, colL);
      }
      int lvl_small = lvl < pyrLevelsUsed - 1 ? lvl + 1 : lvl;
      img_small = new MinimalImageB3(wG[lvl_small], hG[lvl_small]);
      Vec3f *colorSmall =
          firstFrame->dIp[lvl_small] + wG[lvl_small] * hG[lvl_small] * cid;
      for (int i = 0; i < wG[lvl_small] * hG[lvl_small]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(colorSmall + i))[0];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        img_small->at(i, cid) = Vec3b(colL, colL, colL);
      }
      Pnt *pts_big = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
      int npts_big = level_cid_to_numPoints[lvl][cid];
      for (int i = 0; i < npts_big; i++) {
        img_big->setPixel1(pts_big[i].u + 0.5, pts_big[i].v + 0.5,
                           makeRainbow3B(0.1), cid);
      }
      Pnt *pts_small =
          points[lvl_small] + level_cid_to_npts_success_offset[lvl_small][cid];
      int npts_small = level_cid_to_numPoints[lvl_small][cid];
      for (int i = 0; i < npts_small; i++) {
        img_small->setPixel1(pts_small[i].u + 0.5, pts_small[i].v + 0.5,
                             makeRainbow3B(0.1), cid);
      }
#endif
      Pnt *pts = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
      int npts = level_cid_to_numPoints[lvl][cid];

      int ret_index[nn], ret_index_big[nn]; // 搜索到的临近点
      float ret_dist[nn], ret_dist_big[nn]; // 搜索到点的距离
      // 搜索结果, 最近的nn个和1个
      nanoflann::KNNResultSet<float, int, int> resultSet(nn);
      nanoflann::KNNResultSet<float, int, int> resultSet1(1);

      for (int i = 0; i < npts; i++) {
#ifdef SHOW_NN
        img_big->setPixel4(pts[i].u + 0.5, pts[i].v + 0.5, makeRainbow3B(1),
                           cid);
#endif
        // resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
        resultSet.init(ret_index, ret_dist);
        Vec2f pt = Vec2f(pts[i].u, pts[i].v); // 当前点
        // 使用建立的KDtree, 来查询最近邻
        /// 点云数据在初始化KdTree时已经load进去了
        indexes[lvl]->findNeighbors(resultSet, (float *)&pt,
                                    nanoflann::SearchParams());
        int myidx = 0;
        float sumDF = 0;
        //* 给每个点的neighbours赋值
        for (int k = 0; k < nn; k++) {
          pts[i].neighbours[myidx] = ret_index[k]; // 最近的索引
          if (ret_index[k] < 0) {
            printf("no close point, dist: %d\n", ret_index[k]);
          }
          pts[i].neighboursDistL1[myidx] = std::sqrt(ret_dist[k]);
          float df = expf(-ret_dist[k] * NNDistFactor); // 距离使用指数形式
          sumDF += df;                                  // 距离和
          pts[i].neighboursDist[myidx] = df;
          //          printf("ret_index[k]: %d, ret_index[k]: %d, npts: %d\n",
          //          ret_index[k],
          //                 ret_index[k], npts);
          assert(ret_index[k] >= 0 && ret_index[k] < npts);
          myidx++;
        }
        // 对距离进行归10化,,,,,
        for (int k = 0; k < nn; k++)
          pts[i].neighboursDist[k] *= 10 / sumDF;

        //* 高一层的图像中找到该点的父节点
        if (lvl < pyrLevelsUsed - 1) {
          for (int ind = 0; ind < nn; ++ind) {
            ret_index_big[ind] = ret_index[ind];
            ret_dist_big[ind] = ret_dist[ind];
          }
          // TODO roger, 细节，这是复用同一个变量地址了
          resultSet1.init(ret_index, ret_dist);
          /// 父节点是在更模糊的一层中找的
          Vec2f pt_big = pt;
          pt = pt * 0.5f - Vec2f(0.25f, 0.25f); // 换算到高一层
          indexes[lvl + 1]->findNeighbors(resultSet1, (float *)&pt,
                                          nanoflann::SearchParams());

          pts[i].parent = ret_index[0]; // 父节点
          pts[i].parentDist =
              expf(-ret_dist[0] * NNDistFactor); // 到父节点的距离(在高层中)
          // printf("ret_index[0]: %d, level_cid_to_numPoints[%d + 1][%d]:
          // %d\n", ret_index[0], lvl, cid, level_cid_to_numPoints[lvl +
          // 1][cid]);
          assert(ret_index[0] >= 0 &&
                 ret_index[0] < level_cid_to_numPoints[lvl + 1][cid]);

#ifdef SHOW_NN_DETAIL
          Pnt *pts_small = points[lvl_small] +
                           level_cid_to_npts_success_offset[lvl_small][cid];
          int npts_small = level_cid_to_numPoints[lvl_small][cid];
          Pnt *pts_big =
              points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
          int npts_big = level_cid_to_numPoints[lvl][cid];
          MatXXf diff_vec_small;
          diff_vec_small.resize(npts_small, 1);
          diff_vec_small.setOnes();
          diff_vec_small *= 999;
          std::vector<float> v_diff_small;
          for (int id = 0; id < npts_small; ++id) {
            float diff =
                (pt - Vec2f(pts_small[id].u, pts_small[id].v)).squaredNorm();
            diff_vec_small(id, 0) = diff;
            v_diff_small.emplace_back(diff);
          }
          std::sort(v_diff_small.begin(), v_diff_small.end(),
                    [](const float &p1, const float &p2) { return p1 < p2; });
          int nearest_id = 0;
          printf("--------------------------------------------------------\n");
          for (int id = 0; id < npts_small; ++id) {
            float diff =
                (pt - Vec2f(pts_small[id].u, pts_small[id].v)).squaredNorm();
            if (std::abs(diff - diff_vec_small.minCoeff()) < 0.00001) {
              printf("nearest_id: %d, dist: %f, uv: [%f %f]\n", id, diff,
                     pts_small[id].u, pts_small[id].v);
            }
          }
          for (int index = 0; index < nn; ++index) {
            float diff = (pt - Vec2f(pts_small[ret_index[index]].u,
                                     pts_small[ret_index[index]].v))
                             .squaredNorm();
            if (index == 0) {
              assert(std::abs(diff - ret_dist[index]) < 0.0001);
              assert(std::abs(v_diff_small[index] - ret_dist[index]) < 0.0001);
            }
            printf("small level neighbour, index: %d, diff: %f, sorted: %f, "
                   "dist: %f\n",
                   ret_index[index], diff, v_diff_small[index],
                   ret_dist[index]);
          }
          MatXXf diff_vec_big;
          diff_vec_big.resize(npts_big, 1);
          diff_vec_big.setOnes();
          diff_vec_big *= 999;
          std::vector<float> v_diff_big;
          printf("--------------\n");
          for (int id = 0; id < npts_big; ++id) {
            float diff =
                (pt_big - Vec2f(pts_big[id].u, pts_big[id].v)).squaredNorm();
            diff_vec_big(id, 0) = diff;
            v_diff_big.emplace_back(diff);
          }
          std::sort(v_diff_big.begin(), v_diff_big.end(),
                    [](const float &p1, const float &p2) { return p1 < p2; });
          for (int id = 0; id < npts_big; ++id) {
            float diff =
                (pt_big - Vec2f(pts_big[id].u, pts_big[id].v)).squaredNorm();
            if (std::abs(diff - diff_vec_big.minCoeff()) < 0.00001) {
              printf("big level, nearest_id: %d, dist: %f, uv: [%f %f]\n", id,
                     diff, pts_big[id].u, pts_big[id].v);
            }
          }
          for (int index = 0; index < nn; ++index) {
            float diff = (pt_big - Vec2f(pts_big[ret_index_big[index]].u,
                                         pts_big[ret_index_big[index]].v))
                             .squaredNorm();
            assert(std::abs(diff - ret_dist_big[index]) < 0.0001);
            assert(std::abs(v_diff_big[index] - ret_dist_big[index]) < 0.0001);
            printf("big level neighbour, index: %d, diff: %f, sorted: %f, "
                   "dist: %f\n",
                   ret_index_big[index], diff, v_diff_big[index],
                   ret_dist_big[index]);
          }
          img_small->setPixel4(pt(0) + 0.5, pt(1) + 0.5, makeRainbow3B(1), cid);
          IOWrap::displayImage("big", img_big, true);
          IOWrap::displayImage("small", img_small, true);
          IOWrap::waitKey(0);
#endif
        } else { // 最高层没有父节点
          pts[i].parent = -1;
          pts[i].parentDist = -1;
        }
      }
#ifdef SHOW_NN
#ifndef SHOW_NN_DETAIL
      IOWrap::displayImage("big detection", img_big, true);
      IOWrap::waitKey(0);
#endif
      delete img_big;
      delete img_small;
#endif
    }

    // done.

    for (int i = 0; i < pyrLevelsUsed; i++)
      delete indexes[i];
  }
}
} // namespace dso
