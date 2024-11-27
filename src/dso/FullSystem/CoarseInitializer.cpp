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
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "util/nanoflann.h"

#include <opencv2/highgui/highgui.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh)
    : thisToNext_aff(0, 0), thisToNext(SE3()) {
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    points[lvl] = 0;
    // numPoints[lvl] = 0;
  }

  JbBuffer =
      new Vec10f[ww * hh * kCameraNumUsed /* * kCameraNumUsed * kCameraNumUsed*/
  ]; // todo roger, align with host uv, rounding pixel
  JbBuffer_new =
      new Vec10f[ww * hh *
                 kCameraNumUsed /* * kCameraNumUsed * kCameraNumUsed*/];

  frameID = -1;
  fixAffine = true;
  printDebug = false;
  //! 这是
  wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
  wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
  wM.diagonal()[6] = SCALE_A;
  wM.diagonal()[7] = SCALE_B;
}

CoarseInitializer::~CoarseInitializer() {
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
    if (points[lvl] != 0)
      delete[] points[lvl];
  }

  delete[] JbBuffer;
  delete[] JbBuffer_new;
}

bool CoarseInitializer::trackFrame(
    FrameHessian *newFrameHessian,
    std::vector<IOWrap::Output3DWrapper *> &wraps) {
  newFrame = newFrameHessian;
  //[ ***step 1*** ] 先显示新来的帧
  // 新的一帧, 在跟踪之前显示的
  for (IOWrap::Output3DWrapper *ow : wraps)
    ow->pushLiveFrame(newFrameHessian);

  int maxIterations[] = {5, 5, 10, 30, 50};
//? 调参
#ifndef USE_ZNCC
  alphaK = 2.5 * 2.5; //*freeDebugParam1*freeDebugParam1;
  alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
#else
  alphaK =
      2.5 *
      2.5; // 2.5*2.5;//0.0150*0.0150;//0.005*0.005;//0.010*0.010;//0.0150*0.0150;//*freeDebugParam1*freeDebugParam1;
  alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
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
          if (kCameraNumUsed == 1) {
            ptsl[i].iR = 1; // TODO 每个点的深度初值都赋1，hessian赋0
            ptsl[i].idepth_new = 1;
          } else {
            ptsl[i].iR =
                ptsl[i].iR_triangle; // TODO 每个点的深度初值都赋1，hessian赋0
            ptsl[i].idepth_new = ptsl[i].idepth_new_triangle;
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
  for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {
    printf("---------------------------------------------\n");
    //[ ***step 3*** ] 使用计算过的上一层来初始化下一层
    // 顶层未初始化到, reset来完成
    if (lvl < pyrLevelsUsed - 1 && kCameraNumUsed == 1) {
      /// from coarse image to fine image, hence "down"
      propagateDown(lvl + 1);
    }

    Mat88f H, Hsc;
    Vec8f b, bsc;
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
    //      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid)
    //      {
    resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current,
                          refToNew_aff_current, false, N, lvl <= 0);
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
          b - bsc * (1 / (1 + lambda)); // 因为dd必定是对角线上的, 所以也乘倒数
      //? wM为什么这么乘, 它对应着状态的SCALE
      //? (0.01f/(w[lvl]*h[lvl]))是为了减小数值, 更稳定?
      Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
      bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

      //[ ***step 5.2*** ] 求解增量
      Vec8f inc;
      SE3 refToNew_new;
      if (fixAffine) // 固定光度参数
      {
        // Note as we set the weights of rotation and translation to 1 the wM is
        // just the identity in this case.
        inc.head<6>() =
            -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
              (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
        inc.tail<2>().setZero();
      } else
        inc = -(wM * (Hl.ldlt().solve(bl))); //=-H^-1 * b.

      double incNorm = inc.norm();

      //[ ***step 5.3*** ] 更新状态, doStep中更新逆深度
      /// lifting
      refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;

      AffLight refToNew_aff_new = refToNew_aff_current;
      refToNew_aff_new.a += inc[6];
      refToNew_aff_new.b += inc[7];
      doStep(lvl, lambda, inc);

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
      resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new,
                            refToNew_aff_new, false, N2, lvl <= 0);
      //        }
      //      }
      Vec3f regEnergy = calcEC(lvl);

      //      std::cout << "resNew: " << resNew.transpose()
      //                << ", regEnergy: " << regEnergy.transpose() <<
      //                std::endl;

      float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
      float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

      bool accept = eTotalOld > eTotalNew;

      printf("accept: %d, level: %d, [eTotalOld / eTotalNew]: [%f / %f], diff: "
             "%f\n",
             accept, lvl, eTotalOld, eTotalNew, eTotalNew - eTotalOld);
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
      if (accept) {
        int point_count = 0;
        for (int id = 0; id < kCameraNumUsed; ++id) {
          point_count += level_cid_to_numPoints[lvl][id];
        }
        //        printf("alphaK: %f, level_cid_to_numPoints[lvl]: %d,
        //        resNew[1]: %f\n",
        //               alphaK, point_count, resNew[1]);

        //? 这是啥   答：应该是位移足够大，才开始优化IR
        if (resNew[1] ==
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
        if (kCameraNumUsed == 1) {
          optReg(lvl); // 更新iR
        }
        lambda *= 0.5;
        fails = 0;
        if (lambda < 0.0001)
          lambda = 0.0001;
      } else {
        fails++;
        lambda *= 4;
        if (lambda > 10000)
          lambda = 10000;
      }

      bool quitOpt = false;
      // 迭代停止条件, 收敛/大于最大次数/失败2次以上
      if (!(incNorm > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
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

  //[ ***step 6*** ] 优化后赋值位姿, 从底层计算上层点的深度
  std::cout << "refToNew_current: \n"
            << refToNew_current.matrix3x4() << std::endl;
  thisToNext = refToNew_current;
  thisToNext_aff = refToNew_aff_current;

  for (int i = 0; i < pyrLevelsUsed - 1; i++)
    propagateUp(i);

  frameID++;
  if (!snapped)
    snappedAt = 0;

  if (snapped && snappedAt == 0)
    snappedAt = frameID; // 位移足够的帧数

  debugPlot(0, wraps);

  // 位移足够大, 再优化5帧才行
  if (kCameraNumUsed == 1) {
    return snapped && frameID > snappedAt + 5;
  } else {
    // snapped = true;
    return snapped && frameID >= snappedAt + 0;
  }
}

void CoarseInitializer::debugPlot(
    int lvl, std::vector<IOWrap::Output3DWrapper *> &wraps) {
  bool needCall = false;
  for (IOWrap::Output3DWrapper *ow : wraps)
    needCall = needCall || ow->needPushDepthImage();
  if (!needCall)
    return;

  int wl = w[lvl], hl = h[lvl];

  MinimalImageB3 iRImg(wl, hl);

  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + hG[lvl] * wG[lvl] * cid;
    for (int i = 0; i < wl * hl; i++)
      iRImg.at(i, cid) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);
  }
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    int npts = level_cid_to_numPoints[lvl][cid];

    float nid = 0, sid = 0;
    for (int i = 0; i < npts; i++) {
      Pnt *point = points[lvl] + i + level_cid_to_npts_success_offset[lvl][cid];
      if (point->isGood) {
        nid++;
#ifndef USE_MULTI_CAM
        sid += point->iR;
#else
        sid += point->idepth;
#endif
      }
    }
    float fac = nid / sid;

    for (int i = 0; i < npts; i++) {
      Pnt *point = points[lvl] + level_cid_to_npts_success_offset[lvl][cid] + i;

      if (!point->isGood)
        iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0),
                        point->host_cid);

      else
        iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f,
#ifndef USE_MULTI_CAM
                        makeRainbow3B(point->iR * fac),
#else
                        makeRainbow3B(point->idepth * fac),
#endif
                        point->host_cid);
    }
  }
  // IOWrap::displayImage("idepth-R", &iRImg, false);
  for (IOWrap::Output3DWrapper *ow : wraps) {
    ow->pushDepthImage(&iRImg);
  }

  if (false) {
    Mat3 intr = Mat3::Identity();
    intr(0, 0) = fx[0];
    intr(1, 1) = fy[0];
    intr(0, 2) = cx[0];
    intr(1, 2) = cy[0];
    Mat3 intr_inv = intr.inverse();

    MinimalImageB3 *img_host;
    MinimalImageB3 *img_target;

    img_host = new MinimalImageB3(wG[0], hG[0]);
    img_target = new MinimalImageB3(wG[0], hG[0]);

    for (int cam = 0; cam < kCameraNumUsed; ++cam) {
      Vec3f *colorRef = firstFrame->dI + wG[0] * hG[0] * cam;
      for (int i = 0; i < wG[0] * hG[0]; i++) {
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
      Eigen::Vector3f *colorCur = newFrame->dI + cam * wG[0] * hG[0];
      for (int i = 0; i < wG[0] * hG[0]; i++) {
        // BRIGHTNESS TRANSFER
        float colL = (*(colorCur + i))[0];
        if (colL < 0)
          colL = 0;
        if (colL > 255)
          colL = 255;
        img_target->at(i, cam) = Vec3b(colL, colL, colL);
      }
    }

    /////////////////////////////////////////////////////////////////////
    for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
      for (int i = 0; i < level_cid_to_numPoints[0][host_cid]; i++) {
        Pnt *point =
            points[0] + i + level_cid_to_npts_success_offset[0][host_cid];
        // TODO roger, like SetFromImage in orca, 判断这个坐标纹理是否充分
        if (!point->isGood) {
          // continue;
        }
        if (point->isGood) {
          img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                              host_cid);
          img_host->setPixelCirc(point->u + 0.5, point->v + 0.5,
                                 makeRainbow3B(1), host_cid);
        } else {
          img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(10),
                              host_cid);
          img_host->setPixelCirc(point->u + 0.5, point->v + 0.5,
                                 makeRainbow3B(10), host_cid);
        }
        for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
          SE3 refToNew =
              newFrame->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
              thisToNext * newFrame->p_multi_camera->cid_to_T01_SE3[host_cid];
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
          if (!(Ku > 15 && Kv > 15 && Ku < wG[0] - 15 && Kv < hG[0] - 15 &&
                new_idepth > 0)) {
            //                isGood = false;
            //                break;
            continue;
          }
          if (point->isGood) {
            img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(1),
                                  target_cid);
            img_target->setPixelCirc(Ku + 0.5, Kv + 0.5, makeRainbow3B(1),
                                     target_cid);
          } else {
            img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(10),
                                  target_cid);
            //            img_target->setPixelCirc(Ku + 0.5, Kv + 0.5,
            //            makeRainbow3B(10),
            //                                     target_cid);
          }
        }
      }
    }
    IOWrap::displayImage("host", img_host);
    IOWrap::displayImage("target", img_target);
    IOWrap::waitKey(0);
    delete img_host;
    delete img_target;
  }
}

//* 计算能量函数和Hessian矩阵, 以及舒尔补, sc代表Schur
// calculates residual, Hessian and Hessian-block neede for re-substituting
// depth.
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
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
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
          if (kCameraNumUsed == 1) {
            point->energy_new[1] =
                (point->idepth_new - 1) * (point->idepth_new - 1); //? 什么原理?
          } else {
            point->energy_new[1] =
                (point->idepth_new - point->iR_triangle) *
                (point->idepth_new - point->iR_triangle); //? 什么原理?
          }
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
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
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
//#define SHOW_INIT_IMAGE
#define USE_8_RES_MULTI_CAM
Vec3f CoarseInitializer::calcResAndGS(int lvl, MatStatef &H_out,
                                      VecStatef &b_out, MatStatef &H_out_sc,
                                      VecStatef &b_out_sc, const SE3 &refToNew_,
                                      AffLight refToNew_aff, bool plot, int &N,
                                      bool show_image) {
  int wl = w[lvl], hl = h[lvl];
  // 当前层图像及梯度
  Accumulator11 E;   // 1*1 的累加器
  acc9.initialize(); // 初始值, 分配空间
  E.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
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

    //  Accumulator11 E;   // 1*1 的累加器
    //  acc9.initialize(); // 初始值, 分配空间
    //  E.initialize();

    int npts = level_cid_to_numPoints[lvl][host_cid];
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
      Eigen::MatrixXf ones;
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
        img_target = new MinimalImageB3(wG[lvl], hG[lvl]);

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
          Eigen::Vector3f *colorCur = newFrame->dIp[lvl] + cam * wl * hl;
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

        img_host->setPixel9(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                            host_cid);
        img_host->setPixelCirc(point->u + 0.5, point->v + 0.5, makeRainbow3B(1),
                               host_cid);
      }

#endif
      int host_target_info_size = 0;
      for (int target_cam_id = 0; target_cam_id < kCameraNumUsed;
           ++target_cam_id) {
        // Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl *
        // hl;
        Eigen::Vector3f *colorNew =
            newFrame->dIp[lvl] + target_cam_id * wl * hl;
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

          uv_draw.emplace_back(Vec2f(Ku, Kv));

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
      assert(host_target_info_size == host_info.rows());
      int patch_num = host_info.rows();
      // assert(patch_num == MAX_RES_PER_POINT * kCameraNumUsed);
      if (patch_num != 0) {

        for (int id = 0; id < kCameraNumUsed; ++id) {
          if (cam_info[id] > 0) {
            host_val_mean = a_host_info[id].col(0).sum() /
                            static_cast<float>(a_target_info[id].rows());
            target_val_mean = a_target_info[id].col(0).sum() /
                              static_cast<float>(a_target_info[id].rows());

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
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        if (host_cid != target_cid) {
          // continue;
        }
        if (cam_info[target_cid] < 0) {
          continue;
        }
        int target_cid_use = 0; // target_cid;
        const Mat66 &extra_pose_jac =
            newFrame->p_multi_camera->cid_to_T01_inv_Adj[target_cid];
        // Eigen::Vector3f *colorRef = firstFrame->dIp[lvl] + host_cid * wl *
        // hl;
        Eigen::Vector3f *colorNew = newFrame->dIp[lvl] + target_cid * wl * hl;
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
            // isGood = false;
            is_bad_res_count++;
            // break_inner_loop = true;
            // break;
            continue;
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
          float hw = fabs(residual) < setting_huberTH
                         ? 1
                         : setting_huberTH / fabs(residual);
          // huberweight * (2-huberweight) = Objective Function
          // robust 权重和函数之间的关系
          energy += hw * residual * residual * (2 - hw);
#else
          float residual_bak = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
          float residual = 1 * (a_target_info[target_cid](idx, 0) -
                                a_host_info[target_cid](idx, 0));
          // printf("residual: %f\n", residual);

          assert(!std::isnan(residual));

          if (std::isnan(residual)) {
            // isGood = false;
            is_bad_res_count++;
            // break_inner_loop = true;
            continue
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
          float dxInterp = hw * a_grad_new_target[target_cid](idx, 0) * fxl;
          float dyInterp = hw * a_grad_new_target[target_cid](idx, 1) * fyl;
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
                    a_grad_new_target[target + cid](idx, 1))
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
                                   Vec2f(agrad_new_target[target_cid](idx, 0),
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
                        index_to_count.at(idx + MAX_RES_PER_POINT * target_cid),
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
      if (/* !isGood || */ energy / static_cast<float>(cnt) >
              point->outlierTH * 20 ||
          /*is_bad_res_count > threshold*/ good_cam_num == 0) {
        E.updateSingle((float)(point->energy[0])); // 上一帧的加进来 //
        point->isGood = false;
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
  }
  E.finish();
  acc9.finish();

  //????? 这是在干吗???
  // calculate alpha energy, and decide if we cap it.
  Accumulator11 EAlpha;
  EAlpha.initialize();
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
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
          if (kCameraNumUsed == 1) {
            point->energy_new[1] =
                (point->idepth_new - 1) * (point->idepth_new - 1); //? 什么原理?
          } else {
            point->energy_new[1] =
                (point->idepth_new - point->iR_triangle) *
                (point->idepth_new - point->iR_triangle); //? 什么原理?
          }
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
  //  assert(point_count ==
  //         level_cid_to_numPoints[lvl][kCameraNumUsed - 1] +
  //             level_cid_to_npts_success_offset[lvl][kCameraNumUsed - 1]);

  float alphaEnergy;
  if (kCameraNumUsed == 1) {
    alphaEnergy =
        alphaW * (EAlpha.A + refToNew_.translation().squaredNorm() *
                                 (point_count)); // 平移越大, 越容易初始化成功?
  } else {
    alphaEnergy = alphaW * (EAlpha.A); // 平移越大, 越容易初始化成功?
  }

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
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
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
  if (!snapped) {
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
      if (kCameraNumUsed > 1 && false) {
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
        if (point->neighbours[j] == -1)
          continue;
        Pnt *other = ptsl + point->neighbours[j];
        if (!other->isGood)
          continue;
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
  snapped = false;
  frameID = snappedAt = 0;

  for (int i = 0; i < pyrLevelsUsed; i++)
    dGrads[i].setZero();
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
      // printf("lvl: %d, npts: %d\n", lvl, npts);
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
        points[lvl] = new Pnt[npts * 200]; // TODO roger,
        // 因为现在只检测了cam0的，我不知道cam1，2，3会检测出多少点，我干脆开一个200倍的空间，防止不够用
      }
      // set idepth map to initially 1 everywhere.
      //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
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
                //                printf("success_count: %d, lvl: %d,
                //                triang_arr: %f, [mean_depth "
                //                       "triang_depth]: [%f %f]\n",
                //                       success_count, lvl, triang_arr,
                //                       1.0 / (0.5 * (pt->idepth_max +
                //                       pt->idepth_min)), point_3d_ccs[2]);
                float idepth_mean = 0.5 * (pt->idepth_max + pt->idepth_min);
                if (point_3d_ccs[2] < 0.01 || idepth_mean < 0.01) {
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
              } else {
                // pl[nl].idepth = pl[nl].iR_triangle = 1;
                // pl[nl].iR = pl[nl].iR_triangle = 1;
                delete pt;
                continue;
              }

            } else {
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
    }
  }
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
  snapped = false;
  frameID = snappedAt = 0;

  for (int i = 0; i < pyrLevelsUsed; i++)
    dGrads[i].setZero();
}

//@ 重置点的energy, idepth_new参数
/// 把每个点的深度值按10个neighbor做归一化
void CoarseInitializer::resetPoints(int lvl) {
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    Pnt *pts = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
    // int npts = level_cid_to_npts[lvl][cid];
    int npts = level_cid_to_numPoints[lvl][cid];
    for (int i = 0; i < npts; i++) { // 重置
      pts[i].energy.setZero();
      pts[i].idepth_new = pts[i].idepth;
      assert(pts[i].host_cid == cid);
      pts[i].host_cid = cid;
      // 如果是最顶层, 则使用周围点平均值来重置
      /// the lowest resolution
      if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
        float snd = 0, sn = 0;
        for (int n = 0; n < 10; n++) {
          if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood)
            continue;
          /// 逆深度求和
          snd += pts[pts[i].neighbours[n]].iR;
          sn += 1;
        }

        if (sn > 0) {
          pts[i].isGood = true;
          /// normalize idepth to 1
          pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
        }
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
      float b = (JbBuffer[i + w[0] * h[0] * pts[i].host_cid *
                                  kCameraNumUsed /*+ target_cid)*/])[8] +
                JbBuffer[i].head<8>().dot(inc);
      //! dd * delta_d = dd*r - (dp*dd)^T*delta_p = b
      //! delta_d = b * dd^-1
      // todo roger, dx2 = -b2 / H22;
      float step =
          -b * JbBuffer[i + w[0] * h[0] * pts[i].host_cid * kCameraNumUsed][9] /
          (1 + lambda);
      // TODO
      // 这应该只是粗略的更新depth，要严谨点应该用[三角化]，可能是出于计算量考虑吧，毕竟还在初始化阶段，本身也算不了太准（毕竟idepth都全被初始化成1的正态分布了，能准到哪里去）

      float maxstep = maxPixelStep * pts[i].maxstep; // 逆深度最大只能增加这些
      if (maxstep > idMaxStep)
        maxstep = idMaxStep;

      if (step > maxstep)
        step = maxstep;
      if (step < -maxstep)
        step = -maxstep;
      // 更新得到新的逆深度
      float newIdepth = pts[i].idepth + step;
      // TODO 不是说idepth全假设在1附近吗，怎么又有0.001和50差别这么大范围？？
      // TODO 遇到这种情况再不济也应该把isGood置成0呀
      if (newIdepth < 1e-3)
        newIdepth = 1e-3;
      if (newIdepth > 50)
        newIdepth = 50;
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
        pts[i].idepth = pts[i].idepth_new = pts[i].iR;
        continue;
      }
      pts[i].energy = pts[i].energy_new;
      pts[i].isGood = pts[i].isGood_new;
      pts[i].idepth = pts[i].idepth_new;
      pts[i].lastHessian = pts[i].lastHessian_new;
    }
  }
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
      Pnt *pts = points[lvl] + level_cid_to_npts_success_offset[lvl][cid];
      int npts = level_cid_to_numPoints[lvl][cid];

      int ret_index[nn];  // 搜索到的临近点
      float ret_dist[nn]; // 搜索到点的距离
      // 搜索结果, 最近的nn个和1个
      nanoflann::KNNResultSet<float, int, int> resultSet(nn);
      nanoflann::KNNResultSet<float, int, int> resultSet1(1);

      for (int i = 0; i < npts; i++) {
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
          pts[i].neighbours[myidx] = ret_index[k];      // 最近的索引
          float df = expf(-ret_dist[k] * NNDistFactor); // 距离使用指数形式
          sumDF += df;                                  // 距离和
          pts[i].neighboursDist[myidx] = df;
          assert(ret_index[k] >= 0 && ret_index[k] < npts);
          myidx++;
        }
        // 对距离进行归10化,,,,,
        for (int k = 0; k < nn; k++)
          pts[i].neighboursDist[k] *= 10 / sumDF;

        //* 高一层的图像中找到该点的父节点
        if (lvl < pyrLevelsUsed - 1) {
          resultSet1.init(ret_index, ret_dist);
          /// 父节点是在更模糊的一层中找的
          pt = pt * 0.5f - Vec2f(0.25f, 0.25f); // 换算到高一层
          indexes[lvl + 1]->findNeighbors(resultSet1, (float *)&pt,
                                          nanoflann::SearchParams());

          pts[i].parent = ret_index[0]; // 父节点
          pts[i].parentDist =
              expf(-ret_dist[0] * NNDistFactor); // 到父节点的距离(在高层中)

          assert(ret_index[0] >= 0 &&
                 ret_index[0] < level_cid_to_numPoints[lvl + 1][cid]);
        } else { // 最高层没有父节点
          pts[i].parent = -1;
          pts[i].parentDist = -1;
        }
      }
    }

    // done.

    for (int i = 0; i < pyrLevelsUsed; i++)
      delete indexes[i];
  }
}
} // namespace dso
