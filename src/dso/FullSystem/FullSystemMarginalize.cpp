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

#include "FullSystem/FullSystem.h"
#include <util/TimeMeasurement.h>

#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/ResidualProjections.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso {

// note that marginalization is using the Schur complete in the original paper
// which basically is using the H matrix special arrow shaped structure to
// eliminate off diagnal elements and solve camera pose delta update at first
// and then solve the point hessian depth depth updates shur elimination is just
// one way, we can definitely use cholsky method to decompose the sparse H
// matrix.
///@ 对于关键帧的边缘化策略 1. 活跃点只剩下5%的; 2.
///和最新关键帧曝光变化大于0.7; 3. 距离最远的关键帧

void FullSystem::flagFramesForMarginalization(
    FrameHessian *newFH) { //? 怎么会有这种情况呢?
  dmvio::TimeMeasurement timeMeasurement("flagFramesForMarginalization");
  if (setting_minFrameAge > setting_maxFrames) {
    for (int i = setting_maxFrames; i < (int)frameHessians.size(); i++) {
      FrameHessian *fh = frameHessians[i - setting_maxFrames];
      fh->flaggedForMarginalization = true;
    }
    return;
  }

  int flagged = 0; // 标记为边缘化的个数
  // marginalize all frames that have not enough points.
  /// 虽说最老的不一定被marg，但遍历还是从最老帧开始的，最老帧应该是最有可能marg的
  for (int i = 0; i < (int)frameHessians.size(); i++) {
    FrameHessian *fh = frameHessians[i];
    ///           active                      inactive
    int in = fh->pointHessians.size() + fh->immaturePoints.size(); // 还在的点
    ///              marged inlier                         dropped outlier
    int out = fh->pointHessiansMarginalized.size() +
              fh->pointHessiansOut.size(); // 边缘化和丢掉的点

    Vec2 refToFh = AffLight::fromToVecExposure(
        frameHessians.back()->ab_exposure, fh->ab_exposure,
        frameHessians.back()->aff_g2l(), fh->aff_g2l());

    //* 这一帧里的内点少, 曝光时间差的大, 并且边缘化掉后还有5-7帧, 则边缘化
    if ((in < setting_minPointsRemaining * (in + out) ||
         fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) &&
        ((int)frameHessians.size()) - flagged > setting_minFrames) {
      //			printf("MARGINALIZE frame %d, as only %'d/%'d
      // points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d,
      // activated %d!\n", 					fh->frameID, in,
      // in+out,
      // (int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
      //					(int)fh->pointHessiansMarginalized.size(),
      //(int)fh->pointHessiansOut.size(),
      // visInLast, outInLast,
      // fh->statistics_tracesCreatedForThisFrame,
      // fh->statistics_pointsActivatedForThisFrame);
      fh->flaggedForMarginalization = true;
      flagged++;
    } else {
      //			printf("May Keep frame %d, as %'d/%'d points
      // remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated
      //%d!\n", 					fh->frameID, in, in+out,
      //(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
      //					(int)fh->pointHessiansMarginalized.size(),
      //(int)fh->pointHessiansOut.size(),
      // visInLast, outInLast,
      // fh->statistics_tracesCreatedForThisFrame,
      // fh->statistics_pointsActivatedForThisFrame);
    }
  }

  // marginalize one.
  if ((int)frameHessians.size() - flagged >= setting_maxFrames) {
    double smallestScore = 1;
    FrameHessian *toMarginalize = 0;
    FrameHessian *latest = frameHessians.back();

    for (FrameHessian *fh : frameHessians) {
      ///* 至少是setting_minFrameAge个之前的帧 (保留了当前帧)
      if (fh->frameID > latest->frameID - setting_minFrameAge ||
          fh->frameID == 0)
        continue;
      // if(fh==frameHessians.front() == 0) continue;

      double distScore = 0;
      for (FrameFramePrecalc &ffh : fh->targetPrecalc) {
        if (ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 ||
            ffh.target == ffh.host)
          continue;
        distScore += 1 / (1e-5 + ffh.distanceLL);
      }
      //* 有负号, 与最新帧距离占所有目标帧最大的被边缘化掉, 离得最远的,
      // 论文有提到, 启发式的良好的3D空间分布, 关键帧更接近
      distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

      if (distScore < smallestScore) {
        smallestScore = distScore;
        toMarginalize = fh;
      }
    }

    //		printf("MARGINALIZE frame %d, as it is the closest (score
    //%.2f)!\n", 				toMarginalize->frameID,
    // smallestScore);
    toMarginalize->flaggedForMarginalization = true;
    flagged++;
  }

  //	printf("FRAMES LEFT: ");
  //	for(FrameHessian* fh : frameHessians)
  //		printf("%d ", fh->frameID);
  //	printf("\n");
}

//@ 边缘化一个关键帧, 删除该帧上的残差
void FullSystem::marginalizeFrame(FrameHessian *frame) {
  dmvio::TimeMeasurement timeMeasurement("marginalizeFrame");
  // marginalize or remove all this frames points.
  //! marginalize or remove all this frames points.
  assert((int)frame->pointHessians.size() == 0);

  ef->marginalizeFrame(frame->efFrame);

  dmvio::TimeMeasurement timeMeasurementEnd("marginalizeFrameOverhead");

  // drop all observations of existing points in that frame.
  //* 删除其它帧在被边缘化帧上的残差
  for (FrameHessian *fh : frameHessians) {
    if (fh == frame)
      continue;

    for (PointHessian *ph : fh->pointHessians) {
      for (unsigned int i = 0; i < ph->residuals.size(); i++) {
        PointFrameResidual *r = ph->residuals[i];
        if (r->target == frame) {
          if (ph->lastResiduals[r->target_cid][0].first == r) {
            ph->lastResiduals[r->target_cid][0].first = 0;
          } else if (ph->lastResiduals[r->target_cid][1].first == r) {
            ph->lastResiduals[r->target_cid][1].first = 0;
          }
          if (r->host->frameID < r->target->frameID) {
            statistics_numForceDroppedResFwd++;
          } else {
            statistics_numForceDroppedResBwd++;
          }

          ef->dropResidual(r->efResidual);
          deleteOut<PointFrameResidual>(ph->residuals, i);
          break;
        }
      }
    }
  }

  // ef->marginalizeFrame does not delete efFrame anymore, because it's field
  // frameID was needed in dropResidual.
  delete frame->efFrame;
  frame->efFrame = nullptr;

  {
    std::vector<FrameHessian *> v;
    v.push_back(frame);
    for (IOWrap::Output3DWrapper *ow : outputWrapper)
      ow->publishKeyframes(v, true, &Hcalib);
  }

  frame->shell->marginalizedAt = frameHessians.back()->shell->id;
  frame->shell->movedByOpt = frame->w2c_leftEps().norm();

  auto frameID = frame->frameID;

  deleteOutOrder<FrameHessian>(frameHessians, frame);
  for (unsigned int i = 0; i < frameHessians.size(); i++)
    frameHessians[i]->idx = i;

  int numDel = 0;
  for (auto it = ef->connectivityMap.begin();
       it != ef->connectivityMap.end();) {
    int host = (int)(it->first >> 32);
    int target = (int)(it->first & (uint64_t)0xFFFFFFFF);
    if (host == frameID || target == frameID) {
      numDel++;
      it = ef->connectivityMap.erase(it);
    } else {
      it++;
    }
  }

  setPrecalcValues();
  ef->setAdjointsF(&Hcalib);
}

} // namespace dso
