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
#include <Eigen/LU>
#include <algorithm>

#include "FullSystem/ImmaturePoint.h"
#include "math.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace dso {

//@ 优化未成熟点逆深度, 并创建成PointHessian
/// 是优化滑窗内所有关键帧上的未成熟点
/// 然后往最新关键帧上投构造photometric error？(seems like it) (scratch that)
/// it's multiple view triangulate for one point
#define SHOW_MULTI_VIEW_OTP
PointHessian *
FullSystem::optimizeImmaturePoint(ImmaturePoint *point, int minObs,
                                  ImmaturePointTemporaryResidual *residuals,
                                  bool add_to_residuals, bool print_info) {
  ///[ ***step 1*** ] 初始化和其它关键帧的res(点在其它关键帧上投影)
  int nres = 0;
  std::map<int, int> nres_to_target_cid;
  for (FrameHessian *fh : frameHessians) {
    // TODO roger,
    // 现在的实现是只在最新帧极线搜索，就不在host帧的其他cid上搜了，当然也可以搜，但感觉没什么必要
    if (fh != point->host) {
      for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
        residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
        residuals[nres].state_NewState = ResState::OUTLIER;
        residuals[nres].state_state = ResState::IN;
        residuals[nres].target = fh;
        nres_to_target_cid.emplace(std::make_pair(nres, target_cid));
        nres++;
      }
    }
  }
  assert(nres == kCameraNumUsed * (((int)frameHessians.size()) - 1));

  bool print = print_info; // false; // !add_to_residuals ; // rand()%50==0;

  float lastEnergy = 0;
  float lastHdd = 0;
  float lastbd = 0;
  float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

#ifdef SHOW_MULTI_VIEW_OTP

#endif
  for (int lvl_target = pyrLevelsUsed - 1; lvl_target >= 0; lvl_target--) {
    lastEnergy = 0;
    lastHdd = 0;
    lastbd = 0;
    for (int i = 0; i < nres; i++) {
      residuals[i].state_NewEnergy = residuals[i].state_energy = 0;
      residuals[i].state_NewState = ResState::OUTLIER;
      residuals[i].state_state = ResState::IN;
      lastEnergy += point->linearizeResidual(nres_to_target_cid.at(i), &Hcalib,
                                             1000, residuals + i, lastHdd,
                                             lastbd, currentIdepth, lvl_target);
      residuals[i].state_state = residuals[i].state_NewState;
      residuals[i].state_energy = residuals[i].state_NewEnergy;
    }

    if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
      if (print)
        printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
               nres, lastHdd, lastEnergy);
      return 0;
    }

    if (print)
      printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial "
             "Id=%f\n",
             nres, lastHdd, lastEnergy, currentIdepth);

    float lambda = 0.1;
    for (int iteration = 0; iteration < setting_GNItsOnPointActivation;
         iteration++) {
      float H = lastHdd;
      H *= 1 + lambda;
      float step = (1.0 / H) * lastbd;
      float newIdepth = currentIdepth - step;

      float newHdd = 0;
      float newbd = 0;
      float newEnergy = 0;
      for (int i = 0; i < nres; i++) {
        newEnergy += point->linearizeResidual(nres_to_target_cid.at(i), &Hcalib,
#if 1 // ndef USE_MULTI_CAM //TODO roger, 不能完全不做deoutlier，outlier_thr =
      // 40太大了，因为可能会有遮挡的情况
                                              1
#else
                                              1000
#endif
                                              ,
                                              residuals + i, newHdd, newbd,
                                              newIdepth, lvl_target);
      }
      if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
        if (print)
          printf(
              "OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
              nres, newHdd, lastEnergy);
        return 0;
      }

      if (print /*|| true*/) {
        printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!, step: %f\n",
               (newEnergy < lastEnergy) ? "ACCEPT" : "REJECT", iteration,
               log10(lambda), "", lastEnergy, newEnergy, newIdepth, step);
      }
      if (newEnergy < lastEnergy) {
        currentIdepth = newIdepth;
        lastHdd = newHdd;
        lastbd = newbd;
        lastEnergy = newEnergy;
        for (int i = 0; i < nres; i++) {
          residuals[i].state_state = residuals[i].state_NewState;
          residuals[i].state_energy = residuals[i].state_NewEnergy;
        }

        lambda *= 0.5;
      } else {
        lambda *= 5;
      }

      if (fabsf(step) < 0.0001 * currentIdepth)
        break;
    }

    if (!std::isfinite(currentIdepth)) {
      printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n",
             currentIdepth);
      return (PointHessian *)((
          long)(-1)); // yeah I'm like 99% sure this is OK on 32bit systems.
    }
  }
  int numGoodRes = 0;
  for (int i = 0; i < nres; i++)
    if (residuals[i].state_state == ResState::IN)
      numGoodRes++;

  if (numGoodRes < minObs) {
    if (print)
      printf("OptPoint: OUTLIER!\n");
    return (PointHessian *)((
        long)(-1)); // yeah I'm like 99% sure this is OK on 32bit systems.
  }

  PointHessian *p = new PointHessian(point, &Hcalib, point->host_cid);
  if (!std::isfinite(p->energyTH)) {
    delete p;
    return (PointHessian *)((long)(-1));
  }
  std::array<ResState, kCameraNumUsed> res_state_out{};
  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    res_state_out[target_cid] = ResState::OOB;
  }

  p->lastResiduals[0].first = 0;
  p->lastResiduals[0].second = res_state_out;
  p->lastResiduals[1].first = 0;
  p->lastResiduals[1].second = res_state_out;

  p->setIdepthZero(currentIdepth);
  p->setIdepth(currentIdepth);
  p->setPointStatus(PointHessian::ACTIVE);

  if (!add_to_residuals) {
    if (print) {
      printf("point activated! numGoodRes: %d\n", numGoodRes);
    }
    return p;
  }

  std::array<ResState, kCameraNumUsed> res_state{};
  for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
    res_state[target_cid] = ResState::IN;
  }

#ifndef USE_BUNDLED_RES
  for (int i = 0; i < nres; i++) {
    if (residuals[i].state_state == ResState::IN) {
#ifdef DISABLE_CROSS_CID_ALIGN
      if (p->host_cid != nres_to_target_cid.at(i)) {
        continue;
      }
#endif
      // TODO roger, 这个res是可能属于同一个pid的，要注意
      PointFrameResidual *r = new PointFrameResidual(
          p, p->host, residuals[i].target,
          point->host_cid /*, nres_to_target_cid.at(i)*/);
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        r->state_NewEnergy[cid] = r->state_energy[cid] = 0;
        r->state_NewState[cid] = ResState::OUTLIER;
        r->setState(ResState::IN, cid);
        // TODO roger, 新建residuals，并且推到新激活的点里
        // p->residuals.push_back(r);

        //            if (r->target == frameHessians.back()) {
        //                p->lastResiduals[nres_to_target_cid.at(i)][0].first =
        //                r;
        //                p->lastResiduals[nres_to_target_cid.at(i)][0].second =
        //                ResState::IN;
        //            } else if (r->target == (frameHessians.size() < 2
        //                                     ? 0
        //                                     :
        //                                     frameHessians[frameHessians.size()
        //                                     - 2])) {
        //                p->lastResiduals[nres_to_target_cid.at(i)][1].first =
        //                r;
        //                p->lastResiduals[nres_to_target_cid.at(i)][1].second =
        //                ResState::IN;
        //            }
      }
      if (r->target == frameHessians.back()) {
        p->lastResiduals[0].first = r;
        p->lastResiduals[0].second = res_state; // ResState::IN;
      } else if (r->target == (frameHessians.size() < 2
                                   ? 0
                                   : frameHessians[frameHessians.size() - 2])) {
        p->lastResiduals[1].first = r;
        p->lastResiduals[1].second = res_state; // ResState::IN;
      }
      p->residuals.push_back(r);
    }
  }
#else
  for (int i = 0; i < nres; i += kCameraNumUsed) {
    int inlier_count = 0;
    for (int idx = 0; idx < kCameraNumUsed; ++idx) {
      if (residuals[i + idx].state_state == ResState::IN) {
        assert(residuals[i].target == residuals[i + idx].target);
        inlier_count++;
      }
    }
    if (inlier_count > 0 /*residuals[i].state_state == ResState::IN*/) {
      // TODO roger, 这个res是可能属于同一个pid的，要注意
      PointFrameResidual *r = new PointFrameResidual(
          p, p->host, residuals[i].target,
          point->host_cid /*, nres_to_target_cid.at(i)*/);
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        r->state_NewEnergy[cid] = r->state_energy[cid] = 0;
        r->state_NewState[cid] = ResState::OUTLIER;
        r->setState(ResState::IN, cid);
        // TODO roger, 新建residuals，并且推到新激活的点里
        // p->residuals.push_back(r);

        //            if (r->target == frameHessians.back()) {
        //                p->lastResiduals[nres_to_target_cid.at(i)][0].first =
        //                r;
        //                p->lastResiduals[nres_to_target_cid.at(i)][0].second =
        //                ResState::IN;
        //            } else if (r->target == (frameHessians.size() < 2
        //                                     ? 0
        //                                     :
        //                                     frameHessians[frameHessians.size()
        //                                     - 2])) {
        //                p->lastResiduals[nres_to_target_cid.at(i)][1].first =
        //                r;
        //                p->lastResiduals[nres_to_target_cid.at(i)][1].second =
        //                ResState::IN;
        //            }
      }
      if (r->target == frameHessians.back()) {
        // printf("point with good res\n");
        p->lastResiduals[0].first = r;
        p->lastResiduals[0].second = res_state; // ResState::IN;
      } else if (r->target == (frameHessians.size() < 2
                                   ? 0
                                   : frameHessians[frameHessians.size() - 2])) {
        p->lastResiduals[1].first = r;
        p->lastResiduals[1].second = res_state; // ResState::IN;
      }
      p->residuals.push_back(r);
    }
  }
#endif
  if (print)
    printf("point activated! numGoodRes: %d\n", numGoodRes);
  if (add_to_residuals) {
    statistics_numActivatedPoints++;
  }
  return p;
}

} // namespace dso
