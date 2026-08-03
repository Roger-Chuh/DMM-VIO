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

#include "FullSystem/FullSystem.h"

#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/TimeMeasurement.h"

#include <cmath>

#include <algorithm>

#include "IOWrapper/ImageRW.h"

namespace dso {

//@ 对残差进行线性化
//@ 参数: [true是applyRes, 并去掉不好的残差] [false不进行固定线性化]
#define SHOW_CUR_FRAME_RES
void FullSystem::linearizeAll_Reductor(int iter_num, bool fixLinearization, bool reset_backup_value,
                                       std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats,
                                       int tid) {
  double thr_draw = setting_huberTH_LBA;
  std::array<Vec2f, kCameraNumUsed> other_residual{Vec2f(0, 0)};
#if 1  // def USE_EDGE_ALIGN
  std::array<Vec2f*, kCameraNumUsed> p_other_residual;
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    p_other_residual[cid] = &other_residual[cid];
  }
#else
  std::array<Vec2f*, kCameraNumUsed> p_other_residual{nullptr};
#endif
  FrameHessian* newFrame = frameHessians.back();
  std::array<int, kCameraNumUsed> inliner_count{0};
#ifdef SHOW_CUR_FRAME_RES
  MinimalImageB3* img_target;
  img_target = new MinimalImageB3(wG[0], hG[0]);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    const Eigen::Vector3f* dIl_gray = newFrame->dI + wG[0] * hG[0] * cid;
    for (int i = 0; i < wG[0] * hG[0]; i++) {
      float colL = dIl_gray[i][0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_target->at(i, cid) = Vec3b(colL, colL, colL);
    }
  }
#endif
  for (int k = min; k < max; k++) {  /// 对每一个host点（landmark）遍历，算出他们的光度误差
    PointFrameResidual* r = activeResiduals[k];
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      Vec6f res = r->linearize(&Hcalib, cid, nullptr /*p_other_residual[cid]*/);  // 线性化得到能量;
      if (r->target == newFrame && r->state_NewState[cid] == ResState::IN) {
        if (r->state_residual_residual_gray[cid].hasNaN()) {
          printf("r->state_residual_residual_gray[cid].hasNaN()\n");
          std::exit(1);
        }
#ifdef SHOW_CUR_FRAME_RES
        // img_target->setPixel9((int)r->centerProjectedTo[cid][0], (int)r->centerProjectedTo[cid][1], makeRainbow3B(1),
        // cid);

        double color_draw = (r->state_residual_residual_gray[cid][0] - 0) / (thr_draw - 0);
        if (!std::isfinite(color_draw)) {
          color_draw = 0.0;
          printf("!std::isfinite(color_draw)\n");
          std::exit(1);
        }
        if (color_draw < 0.0) color_draw = 0.0;
        if (color_draw > 1.0) color_draw = 1.0;
        Vec3 bgr_map = color_map.GetBgr(color_draw) * 255;
        img_target->setPixelCirc((int)(r->projectedTo[cid][0][0] + 0.5f), (int)(r->projectedTo[cid][0][1] + 0.5f),
                                 Vec3b(bgr_map[0], bgr_map[1], bgr_map[2]), cid);
#endif
        inliner_count[cid]++;
      }
      if (reset_backup_value) {
        r->point->step = r->point->step_backup = 0.f;
        r->point->idepth_backup = r->point->idepth_zero_scaled;
      }
      if (k < 5 && cid == 0) {
        Vec3f twc0 = r->host->PRE_camToWorld.translation().cast<float>();
        Vec3f twc1 = r->target->PRE_camToWorld.translation().cast<float>();
        printf(
            "linearizeAll, reset_backup_value: %d, k: %d, host_fid: %d, "
            "host_target_cid: [%d %d], host_uv: [%f %f], step: %f, "
            "step_bakup: %f, idp: %f, idp_bakup: %f, "
            "target_uv: (res->[%f %f], center->[%f %f]), factor_res: %f, other_res: %f, twc0: [%f "
            "%f %f], twc1: [%f %f %f]\n",
            reset_backup_value, k, r->host->frameID, r->host_cid, cid, r->point->u, r->point->v, r->point->step,
            r->point->step_backup, r->point->idepth_zero_scaled, r->point->idepth_backup, r->projectedTo[cid][0][0],
            r->projectedTo[cid][0][1], r->centerProjectedTo[cid][0], r->centerProjectedTo[cid][1], res[0],
            res[1] /*other_residual[cid]*/, twc0[0], twc0[1], twc0[2], twc1[0], twc1[1], twc1[2]);
        std::cout << "J[target_cid_now]->JIdx: " << r->J[cid]->JIdx[0].transpose() << std::endl;
        std::cout << "J[target_cid_now]->JIdy: " << r->J[cid]->JIdx[1].transpose() << std::endl << std::endl;
      }
      (*stats).head(6) += res.head(6).cast<double>();
    }
    if (fixLinearization) {  // 固定线性化（优化后执行）
      int active_count = 0;
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        r->applyRes(true, cid);  // 把值给efResidual
      }
      // TODO roger, 重写，把对相机的循环放到最外围，
      // 不行，这里要决定该ph是否为outlier，
      // 必须要先在内部对所有cid遍历完，统计active的个数。再决定它是不是外点
      // TODO roger, 像这种倒是可以先在最外围把cid遍历，
      // 尽量模仿原来不用bunddled_res的处理逻辑，不容易出错，
      // 因为有可能Hb堆叠的顺序也有讲究
      // 像AccumulatedSCHessianSSE::addPoint(EFPoint这种倒是可以先在最外围把cid遍历，
      // 尽量模仿原来不用bunddled_res的处理逻辑，不容易出错，
      // 因为有可能Hb堆叠的顺序也有讲究，
      // 就像AccumulatedSCHessianSSE::addPoint(EFPoint 一样，
      // 这个就是模仿原作者的叠加顺序写的【虽然我最开始的写法可能也没错】
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        if (r->efResidual->isActive(cid)) {  // 残差是in的
          active_count++;
          if (r->isNew[cid]) {  // TODO 理解无穷远点
            PointHessian* p = r->point;
#if 0
                Vec3f ptp_inf =
                    r->host->targetPrecalc[r->target->idx].PRE_KRKiTll *
                    Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
                Vec3f ptp = ptp_inf +
                            r->host->targetPrecalc[r->target->idx].PRE_KtTll *
                                p->idepth_scaled; // projected point with real depth.
#else
            // printf("[host target] : [%d %d]\n", r->host_cid, r->target_cid);
            // std::cout<<"r->host->targetPrecalc[r->target->idx]\n"
            //           "                  .a_PRE_KRKiTll[r->host_cid *
            //           kCameraNumUsed +
            //           r->target_cid]:\n"<<r->host->targetPrecalc[r->target->idx]
            //        .a_PRE_KRKiTll[r->host_cid * kCameraNumUsed +
            //        r->target_cid]<<std::endl;
            //            std::cout<<"delta
            //            Tc0:\n"<<r->host->targetPrecalc[r->target->idx]
            //                             .PRE_KRKiTll<<std::endl;
            Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].a_PRE_KRKiTll[r->host_cid * kCameraNumUsed + cid] *
                            Vec3f(p->u, p->v,
                                  1);  // projected point assuming infinite depth.
            Vec3f ptp =
                ptp_inf + r->host->targetPrecalc[r->target->idx].a_PRE_KtTll[r->host_cid * kCameraNumUsed + cid] *
                              p->idepth_scaled;  // projected point with real depth.
#endif
            float relBS =
                0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm();  // 0.01 = one pixel.

            if (relBS > p->maxRelBaseline) {
              p->maxRelBaseline = relBS;  // 正比于点的基线长度
            }
            p->numGoodResiduals++;
          }
        }
      }
      if (active_count == 0) {  //* tid线程的id
        // 删除OOB, Outlier
        // printf("possible outlier??\n");
        toRemove[tid].push_back(activeResiduals[k]);  // 残差太大则移除
      }
    }
  }
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    printf("!@#$ cid: %d, newFrame inliner num: %d, all num: %d\n", cid, inliner_count[cid], activeResiduals.size());
  }
#ifdef SHOW_CUR_FRAME_RES

  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec2i* edge_pixel_start = newFrame->edge_pixels[0] + wG[0] * hG[0] * cam;
    for (int i = 0; i < newFrame->edge_pixel_num[0][cam]; ++i) {
      int epx = edge_pixel_start[i][0];
      int epy = edge_pixel_start[i][1];
      if (epx < 10 || epx >= wG[0] - 10 || epy < 10 || epy >= hG[0] - 10) continue;
      img_target->setPixel1((float)epx + 0.5, (float)epy + 0.5, Vec3b(255, 0, 255), cam);
    }
  }

  IOWrap::displayImage("lba newFrame res", img_target);
#ifdef SAVE_IMAGES
  if (newFrame && newFrame->shell) {
    // printf("time: %f, %f, %f\n", newFrame->shell->timestamp, newFrame->timestamp, newFrame->shell->timestamp_eval);
    char buf[100];
    snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/lba_new_frame_res_%015lu_%d.png",
             (uint64_t)(newFrame->shell->timestamp_eval * 1e9), iter_num);
    IOWrap::writeImage(buf, img_target);
  }
#endif
  IOWrap::waitKey(1);

  delete img_target;
#endif
}

//@ 把线性化结果传给能量函数efResidual, copyJacobians [true: 更新jacobian]
//[false: 不更新]
void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid) {
  for (int k = min; k < max; k++) {
    // todo
    // 因为是刚把activeResidual线性化的，所以这里还是为activeResidual拷贝雅可比和residual
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      PointFrameResidual* r = activeResiduals[k];
      Vec3f twc = r->host->targetPrecalc[r->target->idx].a_PRE_KtTll[r->host_cid * kCameraNumUsed + cid];
      if (k < 5 && cid == 0) {
        printf(
            "before: k: %d, host_fid: %d, host_cid: %d, host_uv: [%f %f], "
            "idp: %f, target_cid: %d, factor_res: %f, twc: [%f %f %f]\n",
            k, r->host->frameID, r->host_cid, r->point->u, r->point->v, r->point->idepth_zero_scaled, cid, -1.0, twc[0],
            twc[1], twc[2]);
      }
      activeResiduals[k]->applyRes(true, cid);
      if (k < 5 && cid == 0) {
        printf(
            "after: k: %d, host_fid: %d, host_cid: %d, host_uv: [%f %f], "
            "idp: %f, target_cid: %d, factor_res: %f, twc: [%f %f %f]\n",
            k, r->host->frameID, r->host_cid, r->point->u, r->point->v, r->point->idepth_zero_scaled, cid, -1.0, twc[0],
            twc[1], twc[2]);
      }
    }
  }
}

//@ 计算当前最新帧的能量阈值, 太玄学了
void FullSystem::setNewFrameEnergyTH() {
  // collect all residuals and make decision on TH.
  allResVec.clear();
  allResVec.reserve(activeResiduals.size() * 2);
  FrameHessian* newFrame = frameHessians.back();

  for (PointFrameResidual* r : activeResiduals) {
    float energy_count = 0;
    float energy_sum = 0;
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      if (r->state_NewEnergyWithOutlier[cid] >= 0 && r->target == newFrame) {  // 新的帧上残差
        energy_sum += r->state_NewEnergyWithOutlier[cid];
        energy_count += 1;
      }
    }
    if (energy_count > 0) {
      allResVec.push_back(energy_sum / energy_count);
    }
  }
  if (allResVec.size() == 0) {
#ifndef USE_ZNCC
    newFrame->frameEnergyTH = 20 * 20 * patternNum;  // 12 * 12 * patternNum;
    newFrame->frameEnergyTH = 1.5 * 1.5 * setting_outlierTH_LBA *
                              setting_outlierTH_LBA /*setting_coarseCutoffTH * setting_coarseCutoffTH
                                                       setting_outlierTH_epi_trace_on * setting_outlierTH_epi_trace_on*/
                              * patternNum;
#else
    newFrame->frameEnergyTH = (1 * setting_variableScale) * (1 * setting_variableScale);
#endif
    return;  // should never happen, but lets make sure.
  }

  int nthIdx = setting_frameEnergyTHN * allResVec.size();  // 以 setting_frameEnergyTHN 的能量为阈值

  assert(nthIdx < (int)allResVec.size());
  assert(setting_frameEnergyTHN < 1);

  std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx,
                   allResVec.end());            // 排序
  float nthElement = sqrtf(allResVec[nthIdx]);  // 70% 的值都小于这个值

  //? 这阈值为啥这么设置
  //* 先扩大, 在乘上一个鲁棒函数? , 再算平方得到阈值
  newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH =
      26.0f * setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

  float frame_energyTh_dso = newFrame->frameEnergyTH;
  if (setting_useIMU) {
    // Used to enforce a maximum energy threshold.
    imuIntegration.newFrameEnergyTH(newFrame->frameEnergyTH);
  }

  (*frameEnergyThLog) << std::fixed << static_cast<double>(newFrame->shell->timestamp_eval) << " " << frame_energyTh_dso
                      << " " << newFrame->frameEnergyTH << std::endl;
  //
  //	int good=0,bad=0;
  //	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else
  // bad++; 	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)!
  // \n", 			meanElement, nthElement,
  // sqrtf(newFrame->frameEnergyTH), good, bad);
  // printf("fid: %d, newFrame->frameEnergyTH: %f\n", newFrame->frameID,
  // newFrame->frameEnergyTH);
}

//@ 对残差进行线性化, 并去掉不在图像内, 并且残差大的
Vec7 FullSystem::linearizeAll(int iter_num, bool fixLinearization, bool reset_backup_value) {
  double lastEnergyP = 0;
  double lastEnergyP_gray = 0;
  double znccP = 0;
  double hwP = 0;
  double hwP_gray = 0;
  double lastEnergyR = 0;
  double num = 0;
  printf("linearizeAll, fid: %d, cur newFrame->frameEnergyTH: %f\n", frameHessians.back()->frameID,
         frameHessians.back()->frameEnergyTH);
  std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) toRemove[i].clear();

  if (multiThreading) {  // TODO 看多线程这个IndexThreadReduce
    treadReduce.stats.head(5).setZero();
    treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, iter_num, fixLinearization,
                                   reset_backup_value, toRemove, _1, _2, _3, _4),
                       0, activeResiduals.size(), 0);
    lastEnergyP = treadReduce.stats[0];
    lastEnergyP_gray = treadReduce.stats[1];
    znccP = treadReduce.stats[2];
    hwP = treadReduce.stats[3];
    hwP_gray = treadReduce.stats[4];
    num = treadReduce.stats[5];
  } else {
    Vec10 stats = Vec10::Zero();
    linearizeAll_Reductor(iter_num, fixLinearization, reset_backup_value, toRemove, 0, activeResiduals.size(), &stats,
                          0);
    lastEnergyP = stats[0];
    lastEnergyP_gray = stats[1];
    znccP = stats[2];
    hwP = stats[3];
    hwP_gray = stats[4];
    num = stats[5];
  }

  setNewFrameEnergyTH();
  printf("linearizeAll, fid: %d, updated newFrame->frameEnergyTH: %f\n", frameHessians.back()->frameID,
         frameHessians.back()->frameEnergyTH);
  if (fixLinearization) {
    //* 前面线性化, apply之后更新了state_state, 如果有相同的, 就更新状态
    /// state_state只有oob，in这些吧，都是enum，不是具体数值
    for (PointFrameResidual* r : activeResiduals) {
      PointHessian* ph = r->point;
      if (ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].second = r->state_state;
      else if (ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].second = r->state_state;
    }
    //! residual创建时候都创建, 再去掉不好的
    int nResRemoved = 0;
    for (int i = 0; i < NUM_THREADS; i++)  // 线程数
    {
      for (PointFrameResidual* r : toRemove[i]) {
        PointHessian* ph = r->point;
        // 删除不好的lastResiduals
        if (ph->lastResiduals[0].first == r)
          ph->lastResiduals[0].first = 0;
        else if (ph->lastResiduals[1].first == r)
          ph->lastResiduals[1].first = 0;
        int target_fid = r->target->idx;
        int remaining_good_res_on_this_fid = 0;
        for (unsigned int k = 0; k < ph->residuals.size(); k++) {
          if (ph->residuals[k]->target->idx == target_fid) {
            if (ph->residuals[k] == r) {
            } else {
              remaining_good_res_on_this_fid++;
            }
          }
        }
        for (unsigned int k = 0; k < ph->residuals.size(); k++)
          // TODO roger,
          // 要删除这个残差，
          // 所以只有这个残差是这个fid上的最后一个res时才要触发delete_connection
          // 且能break
          if (ph->residuals[k] == r) {
            ef->dropResidual(r->efResidual, remaining_good_res_on_this_fid == 0);
            deleteOut<PointFrameResidual>(ph->residuals,
                                          k);  // residuals删除第k个
            nResRemoved++;
            break;
          }
      }
    }
    // printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved,
    // (int)activeResiduals.size());
  }
  Vec7 ret = Vec7::Zero();
  ret << lastEnergyP, lastEnergyR, num, lastEnergyP_gray, znccP, hwP, hwP_gray;
  return ret;  // 后面两个变量都没用
}

// applies step to linearization point.
//@ 更新各个状态, 并且判断是否可以停止优化
// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD) {
  //	float meanStepC=0,meanStepP=0,meanStepD=0;
  //	meanStepC += Hcalib.step.norm();
  //* 相当于步长了
  VecState pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<2 /* * kCameraNumUsed*/>(6).setConstant(stepfacA);

  float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

  float sumNID = 0;

  if (setting_solverMode & SOLVER_MOMENTUM) {
    Hcalib.setValue(Hcalib.value_backup + Hcalib.step);  // 内参的值进行update
    for (FrameHessian* fh : frameHessians) {
      VecState step = fh->step;
      step.head<6>() += 0.5f * (fh->step_backup.head<6>());  //? 为什么加一半 答：这种解法很奇怪。。不管了

      fh->setState(fh->state_backup + step);  // 位姿 光度 update
      for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
        sumA += step[6 + cid * 2] * step[6 + cid * 2];  // 光度增量平方
        sumB += step[7 + cid * 2] * step[7 + cid * 2];
      }
      sumT += step.segment<3>(0).squaredNorm();  // 平移增量
      sumR += step.segment<3>(3).squaredNorm();  // 旋转增量

      for (PointHessian* ph : fh->pointHessians) {
        float step = ph->step + 0.5f * (ph->step_backup);  //? 为啥加一半
        ph->setIdepth(ph->idepth_backup + step);
        sumID += step * step;                // 逆深度增量平方
        sumNID += fabsf(ph->idepth_backup);  // 逆深度求和
        numID++;
        //* 逆深度没有使用FEJ
        ph->setIdepthZero(ph->idepth_backup + step);
      }
    }
  } else {  //* 相机内参更新状态
    Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);
    //* 相机内参, 光度参数更新
    for (FrameHessian* fh : frameHessians) {
      VecState newState = fh->state_backup + pstepfac.cwiseProduct(fh->step);
      NAN_CHECK_EIGEN(newState, "doStep newState");
      fh->setState(newState);
      for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
        sumA += fh->step[6 + cid * 2] * fh->step[6 + cid * 2];
        sumB += fh->step[7 + cid * 2] * fh->step[7 + cid * 2];
      }
      sumT += fh->step.segment<3>(0).squaredNorm();
      sumR += fh->step.segment<3>(3).squaredNorm();
      //* 点的逆深度更新, 注意点逆深度没使用FEJ
      for (PointHessian* ph : fh->pointHessians) {
        float newIdepth = ph->idepth_backup + stepfacD * ph->step;
        NAN_CHECK_SCALAR(newIdepth, "doStep newIdepth");
        NAN_CHECK_SCALAR(ph->step, "doStep ph->step");
        ph->setIdepth(newIdepth);
        sumID += ph->step * ph->step;
        sumNID += fabsf(ph->idepth_backup);
        numID++;

        ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
      }
    }
  }

  sumA /= frameHessians.size();  // / kCameraNumUsed;
  sumB /= frameHessians.size();  // / kCameraNumUsed;
  sumR /= frameHessians.size();
  sumT /= frameHessians.size();
  sumID /= numID;
  sumNID /= numID;

  if (!setting_debugout_runquiet)
    printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t", sqrtf(sumA) / (0.0005 * setting_thOptIterations),
           sqrtf(sumB) / (0.00005 * setting_thOptIterations), sqrtf(sumR) / (0.00005 * setting_thOptIterations),
           sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations));

  EFDeltaValid = false;
  setPrecalcValues();  // 更新相对位姿, 光度

  // 步长小于阈值则可以停止了
  printf("[sumA sumB sumR sumT*sumNID]: [%f %f %f %f]\n", sumA, sumB, sumR, sumT * sumNID);
  return sqrtf(sumA) < 0.0005 * setting_thOptIterations && sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumR) < 0.00005 * setting_thOptIterations && sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
  //
  //	printf("mean steps: %f %f %f!\n",
  //			meanStepC, meanStepP, meanStepD);
}

// sets linearization point.
//@ 对帧, 点, 内参的step和state进行备份
void FullSystem::backupState(bool backupLastStep) {
  if (setting_solverMode & SOLVER_MOMENTUM)  // TODO 是否备份 step 有啥区别
  {
    if (backupLastStep)  // 不是第0步
    {
      Hcalib.step_backup = Hcalib.step;
      Hcalib.value_backup = Hcalib.value;
      for (FrameHessian* fh : frameHessians) {
        fh->step_backup = fh->step;
        fh->state_backup = fh->get_state();
        for (PointHessian* ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          printf("ccc, host_uv: [%f %f], idp: [%f %f %f]\n", ph->u, ph->v, ph->idepth, ph->idepth_scaled,
                 ph->idepth_zero_scaled);
          ph->step_backup = ph->step;
        }
      }
    } else  // 迭代前初始化
    {
      Hcalib.step_backup.setZero();
      Hcalib.value_backup = Hcalib.value;
      for (FrameHessian* fh : frameHessians) {
        fh->step_backup.setZero();
        fh->state_backup = fh->get_state();
        for (PointHessian* ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = 0;
        }
      }
    }
  } else {
    Hcalib.value_backup = Hcalib.value;
    for (FrameHessian* fh : frameHessians) {
      fh->state_backup = fh->get_state();
      for (PointHessian* ph : fh->pointHessians) ph->idepth_backup = ph->idepth;
    }
  }
}

//@ 恢复为原来的值
// sets linearization point.
void FullSystem::loadSateBackup() {
  Hcalib.setValue(Hcalib.value_backup);
  for (FrameHessian* fh : frameHessians) {
    fh->setState(fh->state_backup);
    for (PointHessian* ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup);

      ph->setIdepthZero(ph->idepth_backup);  // 没用FEJ
      ph->step = ph->step_backup;
    }
  }

  EFDeltaValid = false;
  setPrecalcValues();  // 更新当前的状态
}

//@ 计算能量, 计算的是绝对的能量
double FullSystem::calcMEnergy(bool useNewValues) {
  if (setting_forceAceptStep) return 0;
  // calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
  // ef->makeIDX();
  // ef->setDeltaF(&Hcalib);

  return ef->calcMEnergyF(useNewValues);
}

void FullSystem::printOptRes(const Vec7& res, double resL, double resM, double resPrior, double LExact, float a,
                             float b) {
  printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n", res[0],
         sqrtf((float)(res[0] / (patternNum * ef->resInA))), ef->resInA, ef->resInM, a, b);
}

//@ 对当前的关键帧进行GN优化
#ifdef USE_MULTI_CAM
//#define CHANGE_RES_NUM
#endif
float FullSystem::optimize(int mnumOptIts) {
  dmvio::TimeMeasurement timeMeasurement("FullSystemOptimize");
  if (frameHessians.size() < 2) return 0;
  if (frameHessians.size() < 3) mnumOptIts = 20;  // 迭代次数
  if (frameHessians.size() < 4) mnumOptIts = 15;
  if (mnumOptIts < setting_minOptIterations) {
    mnumOptIts = setting_minOptIterations + 3;
  }
  if (allKeyFramesHistory.size() > 10) {
    printf("FQ\n");
    // std::exit(1);
  }
  int offset = 2;
  // get statistics and active residuals.
  //[ ***step 1*** ] 找出未线性化(边缘化)的残差, 加入activeResiduals
  /// this includes newest keyframe, and newly triangulated PointHesiian in
  /// older keyframes 所谓的activeResiduals是指允许他自由relinearize，而不是fix
  /// linearization point
  activeResiduals.clear();  // TODO 使用前先清零，所以在applyRes时swap也没毛病，不影响啥
  int numPoints = 0;
  int numLRes = 0;
  for (FrameHessian* fh : frameHessians)
    for (PointHessian* ph : fh->pointHessians) {
      for (PointFrameResidual* r : ph->residuals) {
        // printf("new_res!!!\n");
        int active_count = 0;
        for (int cid = 0; cid < kCameraNumUsed; ++cid) {
          if (!r->efResidual->isLinearized[cid]) {
            active_count++;
          }
        }
        assert(active_count == 0 || active_count == kCameraNumUsed);
        if (active_count > 0 /* !r->efResidual->isLinearized*/) {  // 没有求线性误差
                                                                   // TODO
          // 这个会一直进入这个判断，只有要marg的点才会线性化残差，它的残差使用fej状态求的，不需要用最新状态
          activeResiduals.push_back(r);  // 新加入的残差 //TODO r中包含host
                                         // target帧id，uv，idepth，Jac这些信息
          // printf("cc\n");
          for (int cid_ = 0; cid_ < kCameraNumUsed; ++cid_) {
            r->resetOOB(cid_);  // residual状态重置
          }
        } else {
          numLRes++;  //已经线性化过得计数
        }
      }
      numPoints++;
    }

  if (!setting_debugout_runquiet)
    printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->nPoints, (int)activeResiduals.size(), numLRes);
  // std::exit(-1);
  //[ ***step 2*** ] 线性化activeResiduals的残差, 计算边缘化的能量值
  //(然而这里都设成0了)
  //* 线性化, 参数: [true是进行固定线性化, 并去掉不好的残差]
  //[false不进行固定线性化]
  // TODO linearizeAll()的操作对象是 activeResiduals
  printf("+++STEP -2, first time calc res and jac\n");
  Vec7 lastEnergy = linearizeAll(
      -1 + offset, false,
      true);  // TODO
              // 这里是第一次线性化，后面还没有优化，所以还没有剔除点的过程，对应pdf里“先第一次统一构建，再第二次里提出误差大的点”的说辞
  //? 和linearizeAll计算的有啥区别
  // printf("check!!\n");
  NAN_CHECK_EIGEN(lastEnergy, "initial lastEnergy");
  double lastEnergyL =
      calcLEnergy();  // islinearized的量的能量 //TODO
                      // 还能通过显式的残差构建来算energy，部分状态用的是Fej（idp，pose，camera），部分状态用的是最新估计（gradient，ab），但host帧的b0用的是fej，算是一个比较强的prior吧
  double lastEnergyM = calcMEnergy(false);  // HM部分的能量 //TODO
                                            // 指被marg掉的那些帧的残差，已经没有显式的残差构建了，只有H和m，

  printf("********* energies: [%f %f %f %f]\n", lastEnergy(0), lastEnergyL, lastEnergyM, lastEnergy[3]);

  // 把线性化的结果给efresidual //TODO
  // 刚刚首次计算的那些Jac，因为刚刚fixLinearization = false，都是船新的状态
  printf("+++STEP -1, first time pass newly calced jac\n");
  if (multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                       activeResiduals.size(), 50);
  else
    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

  if (!setting_debugout_runquiet) {
    printf("Initial Error       \t");
    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a,
                frameHessians.back()->aff_g2l().b);
  }

  debugPlotTracking();

  double dynamicGTSAMWeight = 1.0;
#ifndef USE_EDGE_ALIGN
  double minLambda = 1e-5;
#else
  double minLambda = 1e-1;
#endif
  //[ ***step 3*** ] 迭代求解
  //	double lambda = 1e-1;
  double lambda = minLambda;
  float stepsize = 1;
  VecX previousX = VecX::Constant(CPARS + STATE_DIM * frameHessians.size(), NAN);
  int numIterations = 0;
  for (int iteration = 0; iteration < mnumOptIts; iteration++) {
    dmvio::TimeMeasurement timeMeasurement("baIteration");
    // solve!
    //[ ***step 3.1*** ] 备份当前的各个状态值
    backupState(iteration != 0);
    // solveSystemNew(0);

    if (imuIntegration.getImuSettings().updateDynamicWeightDuringOptimization || iteration == 0) {
      // Update dynamic weight before solving (where the active DSO factor will
      // be scaled accordingly).
      dynamicGTSAMWeight =
          baIntegration->updateDynamicWeight(lastEnergy[0], sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))),
                                             frameHessians.back()->shell->trackingWasGood);
      if (!setting_debugout_runquiet) {
        std::cout << "Dynamic weight: " << dynamicGTSAMWeight << ", lastEnergy: " << lastEnergy[0]
                  << ", rmse: " << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) << std::endl;
        (*rmseLog) << std::fixed << static_cast<double>(frameHessians.back()->shell->timestamp_eval) << " "
                   << dynamicGTSAMWeight << " " << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)))
                   << std::endl;
      }
      if (std::isfinite(sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)))) &&
          sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) > 70000) {
        MinimalImageB3* img_target;
        img_target = new MinimalImageB3(wG[0], hG[0]);
        for (int cid = 0; cid < kCameraNumUsed; ++cid) {
          Vec3f* image = frameHessians.back()->dIp[0] + wG[0] * hG[0] * cid;
          for (int i = 0; i < wG[0] * hG[0]; i++) {
            float colL = image[i][0];
            if (colL < 0) colL = 0;
            if (colL > 255) colL = 255;
            img_target->at(i, cid) = Vec3b(colL, colL, colL);
          }
        }
        IOWrap::displayImage("target", img_target);
        IOWrap::waitKey(0);
      }
    }

    //[ ***step 3.2*** ] 求解系统
    // TODO 根据最新的状态重新计算新的残差
    solveSystem(iteration, lambda);
    NAN_CHECK_EIGEN(ef->lastX, "ef->lastX after solveSystem");
    NAN_PRINT("solveSystem: iter=%d, lambda=%g, lastX_norm=%g\n", iteration, lambda, ef->lastX.norm());
    double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
    NAN_CHECK_SCALAR(incDirChange, "incDirChange");
    previousX = ef->lastX;
    std::cout << "\n+++STEP 0, cur_iter: " << iteration << ", cur_dx: " << ef->lastX.transpose() << std::endl;
    //? TUM自己的解法???
    if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
      float newStepsize = exp(incDirChange * 1.4);
      if (incDirChange < 0 && stepsize > 1) stepsize = 1;

      stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
      if (stepsize > 2) stepsize = 2;
      if (stepsize < 0.25) stepsize = 0.25;
    }
    //[ ***step 3.3*** ] 更新状态
    //* 更新变量, 判断是否停止
    bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

    canbreak = canbreak && baIntegration->canBreak();

    // eval new energy!
    // TODO * 更新后重新计算, 根据新的状态更新残差，同时更新雅可比
    // 因为优化迭代还未完成，要继续更新线性化点（对梯度，ab是这样，pose idp
    // camera的雅可比不会变）【NEED TO PRINT SOM LOG】
    // 哪些时inlier，outler也不确定，毕竟优化还未完成
    printf("+++STEP 1, apply dx, re-calc res and jac\n");
    Vec7 newEnergy = linearizeAll(iteration + offset, false, false);
    NAN_CHECK_EIGEN(newEnergy, "newEnergy after linearizeAll");
    double newEnergyL = calcLEnergy();
    NAN_CHECK_SCALAR(newEnergyL, "newEnergyL");
    double newEnergyM = calcMEnergy(true);
    NAN_CHECK_SCALAR(newEnergyM, "newEnergyM");

    if (imuIntegration.getImuSettings().updateDynamicWeightDuringOptimization) {
      // Update dynamic weight before deciding whether to accept the step.
      dynamicGTSAMWeight =
          baIntegration->updateDynamicWeight(lastEnergy[0], sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))),
                                             frameHessians.back()->shell->trackingWasGood);
    }

    if (!setting_debugout_runquiet) {
      printf(
          "+++STEP 2, previousX: %f, num: %0.3f, resInA: %d, newEnergy: %0.3f, newEnergy_gray: %0.3f, zncc_angle: "
          "%0.3f, hw_lba: %0.3f, hw_gray_lba: %0.3f [|||] %s %d (L %.2f, dir "
          "%.2f, ss %.1f): \t",
          previousX.norm(), newEnergy[2], ef->resInA, std::sqrt(newEnergy[0] / newEnergy[2]),
          std::sqrt(newEnergy[3] / newEnergy[2]), newEnergy[4] / newEnergy[2], newEnergy[5] / newEnergy[2],
          newEnergy[6] / newEnergy[2],
          (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM / dynamicGTSAMWeight <
           lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM / dynamicGTSAMWeight)
              ? "++++++ACCEPT"
              : "REJECT",
          iteration, log10(lambda), incDirChange, stepsize);
      printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frameHessians.back()->aff_g2l().a,
                  frameHessians.back()->aff_g2l().b);
    }
    //[ ***step 4*** ] 判断是否接受这次计算
    if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM / dynamicGTSAMWeight <
                                   lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM / dynamicGTSAMWeight)) {
      // TODO 接受更新后的量, 把最新的雅可比状态传递给energyFunctional
      printf("+++STEP 3.1.1, cur_iter: %d, pass newly calced jac\n", iteration);
      if (multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                           activeResiduals.size(), 50);
      else
        applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

#ifndef USE_EDGE_ALIGN
      lambda *= 0.25;
#else
      lambda *= 0.75;
#endif
      lambda = std::max(lambda, minLambda);

      if (setting_useGTSAMIntegration) {
        baIntegration->acceptBAUpdate(lastEnergy[0]);
      }
    } else {
      // TODO 不接受, roll back
      printf(
          "+++STEP 3.2.1, cur_iter: %d, fallback state and restore last "
          "state jac\n",
          iteration);
      loadSateBackup();
      lastEnergy = linearizeAll(
          -2 + offset, false,
          false);  // TODO
                   // 理论上为了少算一次，应该把上次迭代的这部分变量也buffer住的，这里因为已经覆盖，只能重新计算一次
      NAN_CHECK_EIGEN(lastEnergy, "REJECT lastEnergy after linearizeAll");
#if defined(SHOW_CUR_FRAME_RES) && defined(SAVE_IMAGES)
      FrameHessian* newFrame = frameHessians.back();
      if (newFrame && newFrame->shell) {
        char buf[100];
        snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/lba_new_frame_res_%015lu_%d.png",
                 (uint64_t)(newFrame->shell->timestamp_eval * 1e9), iteration + offset);
        std::system((std::string("rm \"") + buf + "\"").c_str());
      }
#endif
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy(false);
      lambda *= 1e2;
    }
    numIterations++;

    if (canbreak && iteration >= setting_minOptIterations) break;
  }

  if (!setting_debugout_runquiet) {
    std::cout << "Num BA Iterations done: " << numIterations << "\n";
  }

  // Update again!
  baIntegration->updateDynamicWeight(lastEnergy[0], sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))),
                                     frameHessians.back()->shell->trackingWasGood);

  //[ ***step 5*** ] 把最新帧的位姿设为线性化点
  //* 最新一帧的位姿设为线性化点, 0-5是位姿增量因此是0, 6-7是值, 直接赋值
  // TODO 因为时relative
  // pose，所以只有01，12，23，34，45，以及56，56的connection是新加的，所以不用fix
  // linearization point，理解的对吗？ (scratch that)
  // TODO 代码和我的注释好像南辕北辙啊
  VecState newStateZero = VecState ::Zero();
  // TODO 至此最新进来的一个关键帧应该设置线性化点了，就像ppt里的f30状态那样
  newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

  frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
                                  newStateZero);  // TODO 最新帧设置为线性化点, 待估计量, and calculate
                                                  // nullspace
  EFDeltaValid = false;
  EFAdjointsValid = false;
  ef->setAdjointsF(&Hcalib);  // TODO 重新计算adj 因为新加入帧了，重新计算一下伴随的东西
  setPrecalcValues();         // TODO 更新增量 after optimization
                              // 只要位姿更新了，就一定要重新计算以下这些相对位姿的值

  // 更新之后的能量
  lastEnergy = linearizeAll(mnumOptIts + offset, true, false);

  //* 能量函数太大, 投影的不好, 跟丢
  if (!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) ||
      !std::isfinite((double)lastEnergy[2]) || !std::isfinite((double)lastEnergy[3])) {
    std::cout << "Tracking lost after bundle adjustment! lastEnergy: " << lastEnergy.transpose() << std::endl;
    NAN_PRINT("TRACKING_LOST: lastEnergy=[%g %g %g %g %g %g %g]\n", lastEnergy[0], lastEnergy[1], lastEnergy[2],
              lastEnergy[3], lastEnergy[4], lastEnergy[5], lastEnergy[6]);
    isLost = true;
  }

  statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

  if (calibLog != 0) {
    (*calibLog) << Hcalib.value_scaled.transpose() << " " << frameHessians.back()->get_state_scaled().transpose() << " "
                << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) << " " << ef->resInM << "\n";
    calibLog->flush();
  }

  //[ ***step 6*** ] 把优化的结果, 给每个帧的shell,
  //注意这里其他帧的线性点是不更新的
  //* 把优化结果给shell
  {
    /// 出了函数块，自动解锁
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    for (FrameHessian* fh : frameHessians) {
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l = fh->aff_g2l();  // TODO 存的不是线性化点的位姿，而是最新估计的状态
      // TODO
      // 存的不是线性化点的位姿，而是最新估计的状态，frameShell可理解成frameHessian的简化版
    }
  }

  baIntegration->postOptimization(ef->frames);

  // std::exit(-1);

  debugPlotTracking();
  //* 返回平均误差rmse
  return statistics_lastFineTrackRMSE;
}

//@ 求解系统
void FullSystem::solveSystem(int iteration, double lambda) {
  ef->lastNullspaces_forLogging = getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
                                                ef->lastNullspaces_affA, ef->lastNullspaces_affB);

  ef->solveSystemF(iteration, lambda, &Hcalib);
}

//@ 计算能量E (chi2), 是相对的
double FullSystem::calcLEnergy() {
  if (setting_forceAceptStep) return 0;

  double Ef = ef->calcLEnergyF_MT();
  return Ef;
}

//@ 去除外点(残差数目变为0的)
#define SHOW_DROPPED_POINTS_IN_CUR_FRAME
void FullSystem::removeOutliers() {
  dmvio::TimeMeasurement timeMeasurement("removeOutliers");
#ifdef SHOW_DROPPED_POINTS_IN_CUR_FRAME
  MinimalImageB3* img_target;
  int lvl = 0;
  img_target = new MinimalImageB3(wG[lvl], hG[lvl]);
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(frameHessians.back()->dIp[lvl] + wG[lvl] * hG[lvl] * cid + i))[0];
      if (colL < 0) colL = 0;
      if (colL > 255) colL = 255;
      img_target->at(i, cid) = Vec3b(colL, colL, colL);
    }
  }
#endif
  int numPointsDropped = 0;
  int allPoints = 0;
  Mat33f K = Mat33f::Identity();
  K(0, 0) = Hcalib.fxl();
  K(1, 1) = Hcalib.fyl();
  K(0, 2) = Hcalib.cxl();
  K(1, 2) = Hcalib.cyl();
  FrameHessian* new_frame = frameHessians.back();

  for (FrameHessian* fh : frameHessians) {
    for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
      PointHessian* ph = fh->pointHessians[i];
      if (ph == 0) continue;
      // std::cout << "ph->residuals: " << ph->residuals.size() << std::endl;
      allPoints++;
#ifdef SHOW_DROPPED_POINTS_IN_CUR_FRAME
      SE3 hostToNew_ = new_frame->PRE_worldToCam * fh->PRE_camToWorld;
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        SE3 hostToNew = fh->p_multi_camera->cid_to_T01_SE3[cid].inverse() * hostToNew_ *
                        fh->p_multi_camera->cid_to_T01_SE3[ph->host_cid];
        Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
        Vec3f Kt = K * hostToNew.translation().cast<float>();
        Vec3f pr = KRKi * Vec3f(ph->u, ph->v, 1);
        Vec3f proj = pr + Kt * (ph->idepth_zero_scaled);
        proj /= proj[2];

        if (ph->residuals.size() == 0) {
          img_target->setPixelCirc(proj[0], proj[1], Vec3b(0, 0, 255), cid);
        } else {
          for (int id = 0; id < ph->residuals.size(); ++id) {
            if (ph->residuals[id]->target == new_frame && std::isfinite(ph->residuals[id]->state_zncc_angle[cid])) {
              // yellow
              img_target->setPixelCirc(proj[0], proj[1], Vec3b(0, 255, 255), cid);
              if (ph->residuals[id]->state_NewState[cid] == ResState::IN) {
                // green inlier
                img_target->setPixelCirc(proj[0], proj[1], Vec3b(0, 255, 0), cid);
              }
            }
          }
        }
      }
#endif
      if (ph->residuals.size() == 0)  // 如果该点的残差数为0, 则丢掉
      {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
      }
    }
  }
  ef->dropPointsF();
  printf("numPointsDropped: [%d / %d]\n", numPointsDropped, allPoints);
#ifdef SHOW_DROPPED_POINTS_IN_CUR_FRAME
  for (int cam = 0; cam < kCameraNumUsed; ++cam) {
    Vec2i* edge_pixel_start = new_frame->edge_pixels[0] + wG[0] * hG[0] * cam;
    for (int i = 0; i < new_frame->edge_pixel_num[0][cam]; ++i) {
      int epx = edge_pixel_start[i][0];
      int epy = edge_pixel_start[i][1];
      if (epx < 10 || epx >= wG[0] - 10 || epy < 10 || epy >= hG[0] - 10) continue;
      img_target->setPixel4((float)epx + 0.5, (float)epy + 0.5, Vec3b(255, 0, 255), cam);
    }
  }
  IOWrap::displayImage("dropped points", img_target);
#ifdef SAVE_IMAGES
  if (frameHessians.back() && frameHessians.back()->shell) {
    char buf[100];
    snprintf(buf, 100, "/media/roger/Elements_SE/CI/dm_vio_results/dropped_points_%015lu.png",
             (uint64_t)(frameHessians.back()->shell->timestamp_eval * 1e9));
    IOWrap::writeImage(buf, img_target);
  }
#endif
  IOWrap::waitKey(1);
  delete img_target;
#endif
}

//@ 得到各个状态的零空间
std::vector<VecX> FullSystem::getNullspaces(std::vector<VecX>& nullspaces_pose, std::vector<VecX>& nullspaces_scale,
                                            std::vector<VecX>& nullspaces_affA, std::vector<VecX>& nullspaces_affB) {
  nullspaces_pose.clear();   // size: 6; vec: 4+8*n
  nullspaces_scale.clear();  // size: 1;
  nullspaces_affA.clear();   // size: 1
  nullspaces_affB.clear();   // size: 1

  int n = CPARS + frameHessians.size() * (STATE_DIM);
  std::vector<VecX> nullspaces_x0_pre;  // 所有的零空间
  //* 位姿的零空间
  for (int i = 0; i < 6; i++)  // 第i个变量的零空间
  {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian* fh : frameHessians) {
      nullspace_x0.segment<6>(CPARS + fh->idx * (STATE_DIM)) = fh->nullspaces_pose.col(i);
      nullspace_x0.segment<3>(CPARS + fh->idx * (STATE_DIM)) *= SCALE_XI_TRANS_INVERSE;  // 去掉scale
      nullspace_x0.segment<3>(CPARS + fh->idx * (STATE_DIM) + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_pose.push_back(nullspace_x0);
  }
  //* 光度参数a b的零空间
  for (int i = 0; i < 2 /* * kCameraNumUsed*/; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian* fh : frameHessians) {
      nullspace_x0.segment<2 /* * kCameraNumUsed*/>(CPARS + fh->idx * (STATE_DIM) + 6) =
          fh->nullspaces_affine.col(i).head<2 /* * kCameraNumUsed*/>();
      for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
        nullspace_x0[CPARS + fh->idx * (STATE_DIM) + 6 + cid * 2] *= SCALE_A_INVERSE;
        nullspace_x0[CPARS + fh->idx * (STATE_DIM) + 7 + cid * 2] *= SCALE_B_INVERSE;
      }
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    if (i == 0) nullspaces_affA.push_back(nullspace_x0);
    if (i == 1) nullspaces_affB.push_back(nullspace_x0);
  }
  //* 尺度零空间
  VecX nullspace_x0(n);
  nullspace_x0.setZero();
  for (FrameHessian* fh : frameHessians) {
    nullspace_x0.segment<6>(CPARS + fh->idx * (STATE_DIM)) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS + fh->idx * (STATE_DIM)) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS + fh->idx * (STATE_DIM) + 3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.push_back(nullspace_x0);
  nullspaces_scale.push_back(nullspace_x0);

  return nullspaces_x0_pre;
}

dmvio::IMUIntegration& FullSystem::getImuIntegration() { return imuIntegration; }

}  // namespace dso
