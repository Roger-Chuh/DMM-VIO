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

#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <util/TimeMeasurement.h>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

//@ 计算adHost(F), adTarget(F)
// 传的参数也没用啊
void EnergyFunctional::setAdjointsF(CalibHessian *Hcalib) {

  if (adHost != 0)
    delete[] adHost;
  if (adTarget != 0)
    delete[] adTarget;
  adHost = new Mat88[nFrames *
                     nFrames]; // TODO nFrames*nFrames 2帧可以互为host target
  adTarget = new Mat88[nFrames * nFrames];

  for (int h = 0; h < nFrames; h++) // 主帧  //TODO 2帧可以互为host target
    for (int t = 0; t < nFrames; t++) // 目标帧
    {
      FrameHessian *host = frames[h]->data;
      FrameHessian *target = frames[t]->data;

      SE3 hostToTarget = target->get_worldToCam_evalPT() *
                         host->get_worldToCam_evalPT().inverse();
      if (0) {
        // TODO
        // 通过打印信息证明了这确实是fej的状态，只要frame被marg后，相同id的host
        // target pair的相对位姿就不可能变了
        printf("[host target]: [%d %d], [%d %d]\n", host->idx, target->idx,
               host->shell->keyframeId, target->shell->keyframeId);
        std::cout << "FEJ hostToTarget:\n"
                  << hostToTarget.matrix3x4() << std::endl;
      }
      /// Host
      Mat88 AH = Mat88::Identity();
      /// Target
      Mat88 AT = Mat88::Identity();
      // 见笔记推导吧, 或者https://www.cnblogs.com/JingeTU/p/9077372.html
      // TODO 相对pose增量转成绝对pose增量 类似Twc的增量和Twb的增量是一样的
      AH.topLeftCorner<6, 6>() =
          -hostToTarget.Adj()
               .transpose(); //* 转置是因为后面stitchDoubleInternal计算hessian时候就不转了
      AT.topLeftCorner<6, 6>() = Mat66::Identity();

      // 光度参数, 合并项对参数求导
      //! E = Ij - tj*exp(aj) / ti*exp(ai) * Ii - (bj - tj*exp(aj) / ti*exp(ai)
      //! * bi) a = - tj*exp(aj) / ti*exp(ai),  b = - (bj - tj*exp(aj) /
      //! ti*exp(ai) * bi)
      // TODO a21 = (t2 * exp(a2)) / (t1 * exp(a1))
      // b21 = b2 - a21 * b1
      // affLL = [a21 b21];
      Vec2f affLL =
          AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure,
                                      host->aff_g2l_0(), target->aff_g2l_0())
              .cast<float>();
      AT(6, 6) = -affLL[0]; //! a'(aj)  //TODO a21对a2的雅可比 = -a21 (负负得正
                            //! scratch that)
      AH(6, 6) = affLL[0]; //! a'(ai)   //TODO a21对a1的雅可比 = a21
      AT(7, 7) = -1;       //! b'(bj)   //TODO b21对b2的雅可比 = -1
      AH(7, 7) = affLL[0]; //! b'(bi)   //TODO b21对b1的雅可比 = a21

      AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
      AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
      AH.block<1, 8>(6, 0) *= SCALE_A;
      AH.block<1, 8>(7, 0) *= SCALE_B;
      AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
      AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
      AT.block<1, 8>(6, 0) *= SCALE_A; //? 已经是乘过的, 怎么又乘一遍
      AT.block<1, 8>(7, 0) *= SCALE_B;

      /// adHost, adTarget是一个很长的1维数组
      // TODO precompute
      adHost[h + t * nFrames] =
          AH; // TODO 跟我处理essential matrix的组合时很像
              // 难怪每来一个新帧都要更新一下伴随关系（因为每一帧pose
              // ab的估计都有细微变化）
      adTarget[h + t * nFrames] = AT;
    }
  /// 相机内参的信息矩阵很大(5e9)，导致它不会轻易改变，相当于设了一个很大的weight（也不能说是weight，只能说让它的变化不要太剧烈）
  cPrior = VecC::Constant(setting_initialCalibHessian);

  // float型
  if (adHostF != 0)
    delete[] adHostF;
  if (adTargetF != 0)
    delete[] adTargetF;
  adHostF = new Mat88f[nFrames * nFrames];
  adTargetF = new Mat88f[nFrames * nFrames];

  for (int h = 0; h < nFrames; h++)
    for (int t = 0; t < nFrames; t++) {
      adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
      adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
    }

  cPriorF = cPrior.cast<float>();

  EFAdjointsValid = true;
}

EnergyFunctional::EnergyFunctional(dmvio::BAGTSAMIntegration &gtsamIntegration)
    : gtsamIntegration(gtsamIntegration) {
  adHost = 0;
  adTarget = 0;

  red = 0;

  adHostF = 0;
  adTargetF = 0;
  adHTdeltaF = 0;

  nFrames = nResiduals = nPoints = 0;

  HM = MatXX::Zero(CPARS, CPARS); // 初始的, 后面增加frame改变
  HMForGTSAM = MatXX::Zero(CPARS, CPARS);
  bM = VecX::Zero(CPARS);
  bMForGTSAM = VecX::Zero(CPARS);

  accSSE_top_L = new AccumulatedTopHessianSSE();
  accSSE_top_A = new AccumulatedTopHessianSSE();
  accSSE_bot = new AccumulatedSCHessianSSE();

  resInA = resInL = resInM = 0;
  currentLambda = 0;
}

EnergyFunctional::~EnergyFunctional() {
  for (EFFrame *f : frames) {
    for (EFPoint *p : f->points) {
      for (EFResidual *r : p->residualsAll) {
        r->data->efResidual = 0;
        delete r;
      }
      p->data->efPoint = 0;
      delete p;
    }
    f->data->efFrame = 0;
    delete f;
  }

  if (adHost != 0)
    delete[] adHost;
  if (adTarget != 0)
    delete[] adTarget;

  if (adHostF != 0)
    delete[] adHostF;
  if (adTargetF != 0)
    delete[] adTargetF;
  if (adHTdeltaF != 0)
    delete[] adHTdeltaF;

  delete accSSE_top_L;
  delete accSSE_top_A;
  delete accSSE_bot;
}

//@ 计算各种状态的相对量的增量
/// 这是为了后面使用固定线性化点吧（FEJ）
void EnergyFunctional::setDeltaF(CalibHessian *HCalib) {
  if (adHTdeltaF != 0)
    delete[] adHTdeltaF;
  adHTdeltaF = new Mat18f[nFrames * nFrames];
  for (int h = 0; h < nFrames; h++)
    for (int t = 0; t < nFrames; t++) { // TODO ------------> host
      // TODO |
      //     |
      //     |
      //     |
      //
      //   target
      int idx = h + t * nFrames;
      //! delta_th = Adj * delta_t or delta_th = Adj * delta_h
      // 加一起应该是, 两帧之间位姿变换的增量, 因为h变一点, t变一点
      // TODO
      // 因为是小量，所以可以用李代数直接加减，又因为是绝对pose（T_world_host和T_world_target的增量，所以增量也是在同一个world坐标系下的，所以应该直接相加）
      // adj of relative pose incremental cast as float
      adHTdeltaF[idx] = frames[h]
                                ->data->get_state_minus_stateZero()
                                .head<8>()
                                .cast<float>()
                                .transpose() *
                            adHostF[idx] +
                        frames[t]
                                ->data->get_state_minus_stateZero()
                                .head<8>()
                                .cast<float>()
                                .transpose() *
                            adTargetF[idx];
      // TODO 理论上应该是8x1，这里为了方便写成了1x8, (absolute_inc.transpose *
      // adj.transpose) = (adj * absolute_inc).transpose;
      // get_state_minus_stateZero 是世界系下绝对pose
      // Tcw的增量，所以要用伴随矩阵把绝对增量转化成相对pose的增量
    }

  cDeltaF = HCalib->value_minus_value_zero.cast<float>(); // 相机内参增量
  for (EFFrame *f : frames) {
    f->delta = f->data->get_state_minus_stateZero().head<8>(); // 帧绝对位姿增量
    f->delta_prior =
        (f->data->get_state() - f->data->getPriorZero())
            .head<8>(); // 先验增量//TODO 据视频说只有第一帧才有，初始逆深度等等

    for (EFPoint *p : f->points)
      p->deltaF = p->data->idepth - p->data->idepth_zero; // 逆深度的增量
  }

  EFDeltaValid = true;
}

// accumulates & shifts L.
// accumulates & shifts L.
//@ 计算能量方程内帧点构成的 正规方程
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
  if (MT) {
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A,
                            nFrames, _1, _2, _3, _4),
                0, 0, 0);
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                            accSSE_top_A, &allPoints, this, _1, _2, _3, _4),
                0, allPoints.size(), 50);
    accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
    resInA = accSSE_top_A->nres[0];
  } else {
    accSSE_top_A->setZero(nFrames);
    for (EFFrame *f : frames)
      for (EFPoint *p : f->points)
        accSSE_top_A->addPoint<0>(p, this); //! mode 0 增加EF点
    accSSE_top_A->stitchDoubleMT(red, H, b, this, false,
                                 false); // 不加先验, 得到H, b
    // accSSE_top_A->stitchDoubleMT(red,H,b,this,true,false); // 加先验, 得到H,
    // b
    resInA = accSSE_top_A->nres[0]; // 所有残差计数
  }
}

//@ 计算 H 和 b , 加先验, res是减去线性化残差
// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
  if (MT) {
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L,
                            nFrames, _1, _2, _3, _4),
                0, 0, 0);
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                            accSSE_top_L, &allPoints, this, _1, _2, _3, _4),
                0, allPoints.size(), 50);
    accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
    resInL = accSSE_top_L->nres[0];
  } else {
    accSSE_top_L->setZero(nFrames);
    for (EFFrame *f : frames)
      for (EFPoint *p : f->points)
        accSSE_top_L->addPoint<1>(p, this); //! mode 1
    accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
    resInL = accSSE_top_L->nres[0];
  }
}

//@ 计算边缘化掉逆深度的Schur complement部分
void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
  if (MT) {
    red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot,
                            nFrames, _1, _2, _3, _4),
                0, 0, 0);
    red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
                            accSSE_bot, &allPoints, true, _1, _2, _3, _4),
                0, allPoints.size(), 50);
    accSSE_bot->stitchDoubleMT(red, H, b, this, true);
  } else {
    accSSE_bot->setZero(nFrames);
    for (EFFrame *f : frames)
      for (EFPoint *p : f->points)
        accSSE_bot->addPoint(p, true);
    accSSE_bot->stitchDoubleMT(red, H, b, this, false);
  }
}

//@ 计算相机内参和位姿, 光度的增量
void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT) {
  assert(x.size() == CPARS + nFrames * 8);

  VecXf xF = x.cast<float>();
  HCalib->step = -x.head<CPARS>();
  std::cout << "###############################################################"
               "################################# HCalib incremental: "
            << HCalib->step.transpose() << std::endl;
  Mat18f *xAd = new Mat18f[nFrames * nFrames];
  VecCf cstep = xF.head<CPARS>();
  for (EFFrame *h : frames) {
    h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
    h->data->step.tail<2>().setZero();
    // TODO * 绝对位姿增量变相对的, xAd用于更新逆深度
    for (EFFrame *t : frames)
      xAd[nFrames * h->idx + t->idx] =
          xF.segment<8>(CPARS + 8 * h->idx).transpose() *
              adHostF[h->idx + nFrames * t->idx] +
          xF.segment<8>(CPARS + 8 * t->idx).transpose() *
              adTargetF[h->idx + nFrames * t->idx];
  }
  //* 计算点的逆深度增量
  if (MT)
    red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt, this, cstep,
                            xAd, _1, _2, _3, _4),
                0, allPoints.size(), 50);
  else
    resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

  delete[] xAd;
}

//@ 计算点逆深度的增量
void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min,
                                       int max, Vec10 *stats, int tid) {
  for (int k = min; k < max; k++) {
    EFPoint *p = allPoints[k];

    int ngoodres = 0;
    for (EFResidual *r : p->residualsAll)
      if (r->isActive())
        ngoodres++;
    if (ngoodres == 0) {
      p->data->step = 0;
      continue;
    }
    float b = p->bdSumF;
    b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF); //* 减去逆深度和内参
    // b -= xc.dot(p->Hcd_accAF); //* 减去逆深度和内参

    for (EFResidual *r : p->residualsAll) {
      if (!r->isActive())
        continue;
      //* 减去逆深度和位姿 光度参数
      b -= xAd[r->hostIDX * nFrames + r->targetIDX] *
           r->JpJdF; //! 绝对变相对的, xAd是转置了的
    }

    p->data->step = -b * p->HdiF; // 逆深度的增量
    assert(std::isfinite(p->data->step));
  }
}

//@ 也是求能量, 使用HM和bM求的, delta是绝对的
double EnergyFunctional::calcMEnergyF(bool useNewValues) {

  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  VecX delta = getStitchedDeltaF();

  // TODO res = res0 + J * delta
  // res^2 = (res0 + J * delta)^2
  //       = res0 * res0 + delta * Jt * J * delta + 2 * res0 * J * delta
  //       = res0 * res0 + delta * Hm     * delta + 2 * bM       * delta;
  //       = res0 * res0 + delta * (Hm * delta + 2* bM)

  double firstVal = delta.dot(2 * bM + HM * delta);

  if (setting_useGTSAMIntegration) {
    if (!useNewValues) {
      gtsamIntegration.updateBAValues(frames);
    }
    double secondVal = gtsamIntegration.getBAEnergy(useNewValues) +
                       delta.dot(2 * bMForGTSAM + HMForGTSAM * delta);
    return secondVal;
  }

  return firstVal;
}

//@ 计算所有点的能量E之和, delta是相对的
void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

  Accumulator11 E;
  E.initialize();
  VecCf dc = cDeltaF;
  printf("min: %d, max: %d\n", min, max);
  for (int i = min; i < max; i++) {
    EFPoint *p = allPoints[i];
    float dd = p->deltaF;
    // printf("dd: %f, ", dd);
    /*dd: 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd:
     * 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd:
     * 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd:
     * 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd:
     * 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd: 0.000000, dd:
     * 0.000000, dd: 0.000000, dd: 0.000000, */
    for (
        EFResidual *r :
        p->residualsAll) // TODO
                         // 遍历这一个host点在滑窗内所有帧上形成的残差，增量重投影点加权求和
    {
      if (!r->isLinearized || !r->isActive())
        continue; // 同时满足
      // TODO 1x8 vector
      Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
      RawResidualJacobian *rJ = r->J;

      // compute Jp*delta
      // TODO 在idp，pose，内参的扰动下，像素投影点的增量
      float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>()) +
                           rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd;

      float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>()) +
                           rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd;

      __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
      __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
      __m128 delta_a = _mm_set1_ps((float)(dp[6]));
      __m128 delta_b = _mm_set1_ps((float)(dp[7]));

      for (int i = 0; i + 3 < patternNum; i += 4) {
        // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
        //! PATTERN: E = (2*resb_toZeroF + J*delta) * J*delta.
        //! E = (f(x0)+J*dx)^2 = dx*H*dx + 2*J*dx*f(x0) + f(x0)^2 丢掉常数
        //! f(x0)^2 E = (f(x0)+J*dx)^2 = dx*JtJ*dx + 2*J*dx*f(x0) + f(x0)^2
        //! //TODO 丢掉常数 f(x0)^2
        //! ,难怪，或者我们认为f(x0)=0，我们认为在线性化点处的残差（能量）为0
        __m128 Jdelta =
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x);
        Jdelta = _mm_add_ps(
            Jdelta,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
        Jdelta = _mm_add_ps(
            Jdelta,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
        Jdelta = _mm_add_ps(
            Jdelta,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));

        __m128 r0 = _mm_load_ps(((float *)&r->res_toZeroF) + i);
        r0 = _mm_add_ps(r0, r0);
        r0 = _mm_add_ps(r0, Jdelta);
        Jdelta = _mm_mul_ps(Jdelta, r0);
        E.updateSSENoShift(Jdelta);
      }
      // 128位对齐, 多出来部分
      for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
        // TODO 在像素投影点的扰动下，梯度，ab的变化带来的光度残差的增量
        float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 +
                       rJ->JIdx[1][i] * Jp_delta_y_1 + rJ->JabF[0][i] * dp[6] +
                       rJ->JabF[1][i] * dp[7];
        // TODO 丢掉常数 f(x0)^2
        // ,难怪，或者我们认为f(x0)=0，我们认为在线性化点处的残差（能量）为0
        E.updateSingleNoShift(
            (float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
      }
    }
    E.updateSingle(p->deltaF * p->deltaF * p->priorF); // 逆深度先验
  }
  E.finish();
  (*stats)[0] += E.A;
}

//@ MT是多线程, 计算能量, 包括 先验 + 点残差平方
double EnergyFunctional::calcLEnergyF_MT() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  double E = 0;
  //! 先验的能量 (x-x_prior)^T * ∑ * (x-x_prior)
  //* 因为 f->prior 是hessian的对角线, 使用向量表示,
  //所以使用cwiseProduct进行逐个相乘
  for (EFFrame *f : frames) {
    E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior); // 位姿先验
  }
  E += cDeltaF.cwiseProduct(cPriorF).dot(
      cDeltaF); // 相机内参先验 //TODO 估计值和先验值之间残差的平方

  red->reduce(
      boost::bind(&EnergyFunctional::calcLEnergyPt, this, _1, _2, _3, _4), 0,
      allPoints.size(), 50);

  return E + red->stats[0];
}

//@ 向能量函数中插入一残差, 更新连接图关系
EFResidual *EnergyFunctional::insertResidual(PointFrameResidual *r) {
  EFResidual *efr = new EFResidual(r, r->point->efPoint, r->host->efFrame,
                                   r->target->efFrame);
  efr->idxInAll =
      r->point->efPoint->residualsAll.size(); // 在这个点的所有残差的id
  r->point->efPoint->residualsAll.push_back(
      efr); /// 这个点的所有残差, this meas [8 pattern photometric error] or
            /// [pose + idepth + affine] or [covisible?]
  /// asked and answered, residual is just between 2 keyframes, should be the
  /// first one

  // 两帧之间的res计数加一
  // TODO
  // 这个点是刚通过pose固定的光度优化得到的，把这个残差记下来可以在大优化中少计算一点吧？甚至还可以用它计算一些分布用于判断outlier或关键帧什么的?
  connectivityMap[(((uint64_t)efr->host->frameID) << 32) +
                  ((uint64_t)efr->target->frameID)][0]++;

  nResiduals++;
  r->efResidual = efr;
  return efr;
}

// TODO@ 向能量函数中增加一帧, 进行的操作:
// 改变正规方程(绝对pose的雅可比转成相对pose的雅可比), 重新排ID, 共视关系
EFFrame *EnergyFunctional::insertFrame(FrameHessian *fh, CalibHessian *Hcalib) {
  // 建立优化用的能量函数帧. 并加进能量函数frames中
  EFFrame *eff = new EFFrame(fh);
  // printf("eff->frameId: %d\n", eff->frameID);
  eff->idx = frames.size();
  frames.push_back(eff);

  nFrames++;
  fh->efFrame = eff; // FrameHessian 指向能量函数帧
  /// 6dof pose + 2dof affine + 4dof intr
  assert(HM.cols() == 8 * nFrames + CPARS - 8); // 边缘化掉一帧, 缺8个
  // 一个帧8个参数 + 相机内参
  bM.conservativeResize(8 * nFrames + CPARS);
  bMForGTSAM.conservativeResize(8 * nFrames + CPARS);
  HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
  HMForGTSAM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
  // 新帧的块为0
  /// 矩阵增广 阔维
  bM.tail<8>().setZero();
  bMForGTSAM.tail<8>().setZero();
  HM.rightCols<8>().setZero();
  HM.bottomRows<8>().setZero();
  HMForGTSAM.rightCols<8>().setZero();
  HMForGTSAM.bottomRows<8>().setZero();

  // TODO 默认是false，因为刚加进来还没更新，更新之后会设置成true
  EFIndicesValid = false;
  EFAdjointsValid = false;
  EFDeltaValid = false;

  /// 绝对pose的雅可比转成相对pose的雅可比
  // TODO relinearize
  setAdjointsF(Hcalib); // TODO 设置伴随矩阵
  makeIDX(); // 设置ID 告诉优化器每个残差是哪个host和哪个target

  //!< 能量函数中的帧
  for (EFFrame *fh2 : frames) // TODO 这里frames里已经包含最新帧eff了
  {
    // 前32位是host帧的历史ID, 后32位是Target的历史ID
    /// >>表示右移，除以2^32，<<表示左移，乘以2^32
    ///!< connectivityMap: 关键帧之间的连接关系, first: 前32表示host ID,
    ///后32位表示target ID; second:数目 [0] 普通的, [1] 边缘化的
    connectivityMap[(((uint64_t)eff->frameID) << 32) +
                    ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
    // connectivityMap[1][1];
    if (fh2 != eff) { // TODO 避免重复添加
      connectivityMap[(((uint64_t)fh2->frameID) << 32) +
                      ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
    }
  }

  return eff;
}

//@ 向能量函数中插入一个点, 放入对应的EFframe
EFPoint *EnergyFunctional::insertPoint(PointHessian *ph) {
  EFPoint *efp = new EFPoint(ph, ph->host->efFrame);
  efp->idxInPoints = ph->host->efFrame->points.size();
  ph->host->efFrame->points.push_back(efp);

  nPoints++;
  ph->efPoint = efp;

  EFIndicesValid = false; // 有插入需要重新梳理残差的ID

  return efp;
}

//@ 丢掉一个residual, 并更新关系
void EnergyFunctional::dropResidual(EFResidual *r) {
  EFPoint *p = r->point;
  assert(r == p->residualsAll[r->idxInAll]);

  p->residualsAll[r->idxInAll] = p->residualsAll.back(); // 最后一个给当前的
  p->residualsAll[r->idxInAll]->idxInAll =
      r->idxInAll;            // 当前的id变成现在位置的
  p->residualsAll.pop_back(); // 弹出最有一个

  // 计数
  if (r->isActive())
    r->host->data->shell->statistics_goodResOnThis++;
  else
    r->host->data->shell->statistics_outlierResOnThis++;

  // residual关键减一
  connectivityMap[(((uint64_t)r->host->frameID) << 32) +
                  ((uint64_t)r->target->frameID)][0]--;
  nResiduals--;
  r->data->efResidual = 0; // pointframehessian指向该残差的指针
  delete r;
}

//@ 边缘化掉一帧 fh
void EnergyFunctional::marginalizeFrame(EFFrame *fh) {
  dmvio::TimeMeasurement timeMeasurement("EF-marginalizeFrame");
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  assert((int)fh->points.size() == 0);
  int ndim = nFrames * 8 + CPARS - 8; // new dimension
  int odim = nFrames * 8 + CPARS;     // old dimension

  //	VecX eigenvaluesPre = HM.eigenvalues().real();
  //	std::sort(eigenvaluesPre.data(),
  // eigenvaluesPre.data()+eigenvaluesPre.size());
  //

  if (setting_useGTSAMIntegration) {
    // When adding additional factors with GTSAM they need to be accounted for
    // during keyframe marginalization. Hence we move the whole keyframe
    // marginalization to the GTSAMIntegration.
    dmvio::TimeMeasurement innerMeas("MainMarginalization");
    assert(odim == (int)HM.rows());
    assert(odim == (int)HM.cols());
    assert(odim == (int)bM.size());

    // Adds H and b from the last points to the graph. Needs the current
    // evaluation point for each frames.
    gtsamIntegration.addMarginalizedPointsBA(HMForGTSAM, bMForGTSAM, frames);

    Vec8 priorH;
    Vec8 priorB;
    priorH = fh->prior;
    priorB = fh->prior.cwiseProduct(fh->delta_prior);

    gtsamIntegration.addPriorBA(fh, priorH, priorB);

    // Marginalizes out the frame. Adds the symbols of this frame and then calls
    // marginalize out.
    gtsamIntegration.marginalizeBAFrame(fh);

    HMForGTSAM.resize(ndim, ndim);
    bMForGTSAM.resize(ndim);

    HMForGTSAM.setZero();
    bMForGTSAM.setZero();
  }

  //    if(!setting_useGTSAMIntegration) // enable to remove the redundant
  //    visual only marginalization.
  if (true) {
    dmvio::TimeMeasurement measVis("VisualMarginalization");
    //[ ***step 1*** ] 把边缘化的帧挪到最右边, 最下边
    //* HM bM就是边缘化点得到的
    if ((int)fh->idx != (int)frames.size() - 1) {
      int io = fh->idx * 8 + CPARS;            // index of frame to move to end
      int ntail = 8 * (nFrames - fh->idx - 1); // 边缘化帧后面的变量数
      assert((io + 8 + ntail) == nFrames * 8 + CPARS);

      Vec8 bTmp = bM.segment<8>(io); // 被边缘化的8个变量
      VecX tailTMP = bM.tail(ntail); // 后面的挪到前面
      bM.segment(io, ntail) = tailTMP;
      bM.tail<8>() = bTmp;

      //* 边缘化帧右侧挪前面
      MatXX HtmpCol = HM.block(0, io, odim, 8);
      MatXX rightColsTmp = HM.rightCols(ntail);
      HM.block(0, io, odim, ntail) = rightColsTmp;
      HM.rightCols(8) = HtmpCol;
      //* 边缘化帧下边挪上面
      MatXX HtmpRow = HM.block(io, 0, 8, odim);
      MatXX botRowsTmp = HM.bottomRows(ntail);
      HM.block(io, 0, ntail, odim) = botRowsTmp;
      HM.bottomRows(8) = HtmpRow;
    }

    //[ ***step 2*** ] 加上先验
    //* 如果是初始化得到的帧有先验, 边缘化时需要加上. 光度也有先验
    // marginalize. First add prior here, instead of to active.
    /// 这里的prior和inertial
    /// mode下ba，gravity的prior一样，需要预先设置一个权重的，没啥花里胡哨的
    HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
    bM.tail<8>() += fh->prior.cwiseProduct(
        fh->delta_prior); /// b_new = b_old + H * delta_state

    //	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

    //[ ***step 3*** ] 先scaled 然后计算Schur complement
    VecX SVec =
        (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();

    //	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() <<
    //"\n\n"; 	std::cout << std::setprecision(16) << "SVecI: " <<
    // SVecI.transpose() << "\n\n";

    // scale!
    MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
    VecX bMScaled = SVecI.asDiagonal() * bM;

    // invert bottom part!
    Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
    hpi = 0.5f * (hpi + hpi);
    hpi = hpi.inverse();
    hpi = 0.5f * (hpi + hpi);

    // schur-complement!
    MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
    HMScaled.topLeftCorner(ndim, ndim).noalias() -=
        bli * HMScaled.bottomLeftCorner(8, ndim);
    bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

    // unscale!
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    // set.
    HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) +
                HMScaled.topLeftCorner(ndim, ndim).transpose());
    bM = bMScaled.head(ndim);

    // With the imu-integration this cannot be used, because there are other
    // variables that have to be considered.
  } else {
    // Just update the dimensions without actually marginalizing, as these are
    // not used in practice.
    HM = HM.topLeftCorner(ndim, ndim);
    bM = bM.head(ndim);
  }

  //[ ***step 4*** ] 改变EFFrame的ID编号, 并删除
  // remove from vector, without changing the order!
  for (unsigned int i = fh->idx; i + 1 < frames.size(); i++) {
    frames[i] = frames[i + 1];
    frames[i]->idx = i;
  }
  frames.pop_back();
  nFrames--;
  fh->data->efFrame = 0;

  assert((int)frames.size() * 8 + CPARS == (int)HM.rows());
  assert((int)frames.size() * 8 + CPARS == (int)HM.cols());
  assert((int)frames.size() * 8 + CPARS == (int)bM.size());
  assert((int)frames.size() == (int)nFrames);

  //	VecX eigenvaluesPost = HM.eigenvalues().real();
  //	std::sort(eigenvaluesPost.data(),
  // eigenvaluesPost.data()+eigenvaluesPost.size());

  //	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

  //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
  //	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

  // TODO 下面这些量也都是变了的，现在先置成false , 伴随，delta，编号都会变
  EFIndicesValid = false;
  EFAdjointsValid = false;
  EFDeltaValid = false;
  // TODO 当marg帧时，要更新一下编号，
  makeIDX();
}

//@ 边缘化掉一个点
void EnergyFunctional::marginalizePointsF() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  //[ ***step 1*** ] 记录被边缘化的点
  allPointsToMarg.clear();
  for (EFFrame *f : frames) {
    for (int i = 0; i < (int)f->points.size(); i++) {
      EFPoint *p = f->points[i];
      if (p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
        p->priorF *= setting_idepthFixPriorMargFac; //? 这是干啥 ???
        for (EFResidual *r : p->residualsAll)
          if (r->isActive()) // 边缘化残差计数
            connectivityMap[(((uint64_t)r->host->frameID) << 32) +
                            ((uint64_t)r->target->frameID)][1]++;
        allPointsToMarg.push_back(p);
      }
    }
  }
  //[ ***step 2*** ] 计算该点相连的残差构成的H, b, HSC, bSC
  accSSE_bot->setZero(nFrames);
  accSSE_top_A->setZero(nFrames);
  for (EFPoint *p : allPointsToMarg) {
    accSSE_top_A->addPoint<2>(p, this); // 这个点的残差, 计算 H b
    accSSE_bot->addPoint(p, false);     // 边缘化部分
    removePoint(p);
  }
  MatXX M, Msc;
  VecX Mb, Mbsc;
  accSSE_top_A->stitchDouble(M, Mb, this, false, false); // 不加先验, 在后面加了
  accSSE_bot->stitchDouble(Msc, Mbsc, this);

  resInM += accSSE_top_A->nres[0];

  MatXX H = M - Msc; // TODO  H: b: Marg后得到的新先验
  VecX b = Mb - Mbsc;
  //[ ***step 3*** ] 处理零空间
  // 减去零空间部分
  if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
    // have a look if prior is there.
    bool haveFirstFrame = false;
    for (EFFrame *f : frames)
      if (f->frameID == 0)
        haveFirstFrame = true;

    if (!haveFirstFrame)
      orthogonalize(&b, &H);
  }
  //! 给边缘化的量加了个权重，不准确的线性化
  HM += setting_margWeightFac * H; //* 所以边缘化的部分直接加在HM bM了
  bM += setting_margWeightFac * b;

  HMForGTSAM += setting_margWeightFac * H;
  bMForGTSAM += setting_margWeightFac * b;

  if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
    orthogonalize(&bM, &HM); // TODO  做正交化之前要先执行get
                             // nullspace这个操作，不然难为无米之炊

  EFIndicesValid = false;
  makeIDX(); // 梳理ID
}

//@ 直接丢掉点, 不边缘化
void EnergyFunctional::dropPointsF() {

  for (EFFrame *f : frames) {
    for (int i = 0; i < (int)f->points.size(); i++) {
      EFPoint *p = f->points[i];
      if (p->stateFlag == EFPointStatus::PS_DROP) {
        removePoint(p);
        i--;
      }
    }
  }

  EFIndicesValid = false;
  makeIDX();
}

//@ 从EFFrame中移除一个点p
void EnergyFunctional::removePoint(EFPoint *p) {
  for (EFResidual *r : p->residualsAll)
    dropResidual(r); // 丢掉改点的所有残差

  EFFrame *h = p->host;
  h->points[p->idxInPoints] = h->points.back();
  h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
  h->points.pop_back();

  nPoints--;
  p->data->efPoint = 0;

  EFIndicesValid = false;

  delete p;
}

//@ 计算零空间矩阵伪逆, 从 H 和 b 中减去零空间, 相当于设相应的Jacob为0
void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {
  //	VecX eigenvaluesPre = H.eigenvalues().real();
  //	std::sort(eigenvaluesPre.data(),
  // eigenvaluesPre.data()+eigenvaluesPre.size()); 	std::cout << "EigPre:: "
  // << eigenvaluesPre.transpose() << "\n";

  // decide to which nullspaces to orthogonalize.
  std::vector<VecX> ns;
  ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
  ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
  //	if(setting_affineOptModeA <= 0)
  //		ns.insert(ns.end(), lastNullspaces_affA.begin(),
  // lastNullspaces_affA.end()); 	if(setting_affineOptModeB <= 0)
  //		ns.insert(ns.end(), lastNullspaces_affB.begin(),
  // lastNullspaces_affB.end());

  // make Nullspaces matrix
  //! 7自由度不可观
  MatXX N(ns[0].rows(), ns.size()); //! size (4+8*n)×7
  for (unsigned int i = 0; i < ns.size(); i++)
    N.col(i) = ns[i].normalized();

  //* 求伪逆
  // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
  Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

  VecX SNN = svdNN.singularValues();
  double minSv = 1e10, maxSv = 0;
  for (int i = 0; i < SNN.size(); i++) {
    if (SNN[i] < minSv)
      minSv = SNN[i];
    if (SNN[i] > maxSv)
      maxSv = SNN[i];
  }
  // 比最大奇异值小setting_solverModeDelta(e-5)倍, 则认为是0
  for (int i = 0; i < SNN.size(); i++) {
    if (SNN[i] > setting_solverModeDelta * maxSv)
      SNN[i] = 1.0 / SNN[i];
    else
      SNN[i] = 0;
  } // 求逆

  MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() *
              svdNN.matrixV().transpose(); // [dim] x 7.	// [dim] x 9.
  //! Npi.transpose()是N的伪逆
  MatXX NNpiT = N * Npi.transpose();                // [dim] x [dim].
  MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose()); // = N * (N' * N)^-1 * N'.

  //*****************add by gong********************
  // std::vector<VecX> ns;
  // ns.insert(ns.end(), lastNullspaces_pose.begin(),
  // lastNullspaces_pose.end()); ns.insert(ns.end(),
  // lastNullspaces_scale.begin(), lastNullspaces_scale.end());
  std::cout
      << "//=====================Test null space start=====================/ "
      << std::endl;
  // make Nullspaces matrix
  //! 7自由度不可观
  // MatXX N(ns[0].rows(), ns.size());  //! size (4+8*n)×7
  // for(unsigned int i=0;i<ns.size();i++)
  // 	N.col(i) = ns[i].normalized();

  VecX zero_x = *b;

  // MatXX zero = (lastHS) * zero_x;
  for (int i = 0; i < zero_x.cols(); i++) {
    VecX xHx = 0.5 * zero_x.col(i).transpose() * lastHS * zero_x.col(i);
    VecX xb = zero_x.col(i).transpose() * lastbS;

    std::cout << "Before nullspace process " << i << " : " << xHx << " + " << xb
              << std::endl;
  }

  // std::cout<<"//=====================Test null space
  // start=====================/ "<<std::endl; std::cout<<"HA_top * nullspace
  // matrix = " << zero << std::endl; std::cout<<"//=====================Test
  // null space end=====================/ "<<std::endl;

  // TODO 为什么这么做?
  // TODO * 把变量投影到零空间，再减去, 类似边缘化的过程
  // 把零空间从H和b中减去??? 以免乱飘?
  if (b != 0)
    *b -= NNpiTS * *b;
  if (H != 0)
    *H -= NNpiTS * *H * NNpiTS;

  zero_x = *b;
  for (int i = 0; i < zero_x.cols(); i++) {
    VecX xHx = 0.5 * zero_x.col(i).transpose() * lastHS * zero_x.col(i);
    VecX xb = zero_x.col(i).transpose() * lastbS;

    std::cout << "After nullspace process " << i << " : " << xHx << " + " << xb
              << std::endl;
  }
  std::cout
      << "//=====================Test null space end=====================/ "
      << std::endl;

  //	std::cout << std::setprecision(16) << "Orth SV: " <<
  // SNN.reverse().transpose() << "\n";

  //	VecX eigenvaluesPost = H.eigenvalues().real();
  //	std::sort(eigenvaluesPost.data(),
  // eigenvaluesPost.data()+eigenvaluesPost.size()); 	std::cout << "EigPost::
  // " << eigenvaluesPost.transpose() << "\n";
}

//@ 计算正规方程, 并求解
void EnergyFunctional::solveSystemF(int iteration, double lambda,
                                    CalibHessian *HCalib) {
  if (setting_solverMode & SOLVER_USE_GN)
    lambda = 0; // 不同的位控制不同的模式
  if (setting_solverMode & SOLVER_FIX_LAMBDA)
    lambda = 1e-5; //! 还真他娘的用的GN, 只是一个小阻尼

  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);
  //[ ***step 1*** ] 先计算正规方程, 涉及边缘化, 先验, 舒尔补等
  MatXX HL_top, HA_top, H_sc;
  VecX bL_top, bA_top, bM_top, b_sc;
  //* 针对新的残差, 使用的当前残差, 没有逆深度的部分
  accumulateAF_MT(HA_top, bA_top, multiThreading);

  //* 边缘化fix的残差, 有边缘化对的, 使用的res_toZeroF减去线性化部分, 加上先验,
  //没有逆深度的部分
  // TODO bug: 这里根本就没有点参与了, 只有先验信息,
  // 因为边缘化的和删除的点都不在了
  //! 这里唯一的作用就是 把 p相关的置零, useless
  accumulateLF_MT(HL_top, bL_top, multiThreading); // 计算的是之前计算过得

  // p->Hdd_accLF = 0;
  // p->bd_accLF = 0;
  // p->Hcd_accLF =0 ;

  //* 关于逆深度的Schur部分

  accumulateSCF_MT(H_sc, b_sc, multiThreading);

  // TODO HM 和 bM是啥啊
  // TODO * 由于固定线性化点, 每次迭代更新残差,
  // 对于marg部分，残差bM进行一阶泰勒更新, HM stays unchanged
  bM_top = (bM + HM * getStitchedDeltaF());
  VecX bMGTSAM_top = (bMForGTSAM + HMForGTSAM * getStitchedDeltaF());

  // printf("HA_top: \n");
  // for(int i = 0; i < HA_top.rows(); i++)
  // {
  // 	// for(int j = 0; j < HA_top.cols(); j++)
  // 	{
  // 		printf("  %f", HA_top(i,i));
  // 	}
  // 	// printf("; \n");
  // }

  // printf("HL_top: \n");
  // for(int i = 0; i < HL_top.rows(); i++)
  // {
  // 	// for(int j = 0; j < HL_top.cols(); j++)
  // 	{
  // 		printf("  %f", HL_top(i,i));
  // 	}
  // 	// printf("; \n");
  // }

  MatXX HFinal_top;
  VecX bFinal_top;
  //[ ***step 2*** ] 如果是设置求解正交系统,
  //则把相对应的零空间部分Jacobian设置为0, 否则正常计算schur
  if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
    // have a look if prior is there.
    bool haveFirstFrame = false;
    for (EFFrame *f : frames)
      if (f->frameID == 0)
        haveFirstFrame = true;

    // 计算Schur之后的
    // MatXX HT_act =  HL_top + HA_top - H_sc;
    MatXX HT_act = HL_top + HA_top - H_sc;
    // VecX bT_act =   bL_top + bA_top - b_sc;
    VecX bT_act = bL_top + bA_top - b_sc;

    //! 包含第一帧则不减去零空间
    //! 不包含第一帧, 因为要固定第一帧, 和第一帧统一, 减去零空间,
    //! 防止在零空间乱飘
    if (!haveFirstFrame)
      orthogonalize(&bT_act, &HT_act);

    HFinal_top = HT_act + HM;
    bFinal_top = bT_act + bM_top;

    lastHS = HFinal_top;
    lastbS = bFinal_top;
    // LM
    //* 这个阻尼也是加在Schur complement计算之后的
    for (int i = 0; i < 8 * nFrames + CPARS; i++)
      HFinal_top(i, i) *= (1 + lambda);

  } else {

    // HFinal_top = HL_top + HM + HA_top;
    HFinal_top = HL_top + HM + HA_top;
    // bFinal_top = bL_top + bM_top + bA_top - b_sc;
    bFinal_top = bL_top + bM_top + bA_top - b_sc;

    lastHS = HFinal_top - H_sc;
    lastbS = bFinal_top;
    //* 而这个就是阻尼加在了整个Hessian上
    //? 为什么呢, 是因为减去了零空间么  ??
    for (int i = 0; i < 8 * nFrames + CPARS; i++)
      HFinal_top(i, i) *= (1 + lambda);
    HFinal_top -=
        H_sc * (1.0f / (1 + lambda)); // 因为Schur里面有个对角线的逆, 所以是倒数
  }

  //[ ***step 3*** ] 使用SVD求解, 或者ldlt直接求解
  VecX x;
  if (setting_solverMode & SOLVER_SVD) {
    //* 为数值稳定进行缩放
    VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
    MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
    VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
    //! Hx=b --->  U∑V^T*x = b
    Eigen::JacobiSVD<MatXX> svd(HFinalScaled,
                                Eigen::ComputeThinU | Eigen::ComputeThinV);

    VecX S = svd.singularValues(); // 奇异值
    double minSv = 1e10, maxSv = 0;
    for (int i = 0; i < S.size(); i++) {
      if (S[i] < minSv)
        minSv = S[i];
      if (S[i] > maxSv)
        maxSv = S[i];
    }
    //! Hx=b --->  U∑V^T*x = b  --->  ∑V^T*x = U^T*b
    VecX Ub = svd.matrixU().transpose() * bFinalScaled;
    int setZero = 0;
    for (int i = 0; i < Ub.size(); i++) {
      if (S[i] < setting_solverModeDelta * maxSv) //* 奇异值小的设置为0
      {
        Ub[i] = 0;
        setZero++;
      }

      if ((setting_solverMode & SOLVER_SVD_CUT7) &&
          (i >= Ub.size() - 7)) //* 留出7个不可观的, 零空间
      {
        Ub[i] = 0;
        setZero++;
      }
      //! V^T*x = ∑^-1*U^T*b
      else
        Ub[i] /= S[i];
    }
    //! x = V*∑^-1*U^T*b   把scaled的乘回来
    x = SVecI.asDiagonal() * svd.matrixV() * Ub;

  } else {
    VecX myX;
    if (setting_useGTSAMIntegration) {
      // Instead of directly solving the system we instead pass it to the
      // GTSAMIntegration which will add more factors and then solve it for us.
      // This is mathematically correct as long as the new residuals are
      // independent of the DSO residuals (which usually they are) and as long
      // as they don't depend on the points (as otherwise the Schur-complement
      // trick doesn't work like this anymore).
      MatXX HPassed = HL_top + HMForGTSAM + HA_top;
      for (int i = 0; i < 8 * nFrames + CPARS; i++)
        HPassed(i, i) *= (1 + lambda);
      HPassed -= H_sc * (1.0f / (1 + lambda));
      x = gtsamIntegration.computeBAUpdate(
          HPassed, bL_top + bMGTSAM_top + bA_top - b_sc, lambda, frames,
          HL_top + HMForGTSAM + HA_top - H_sc);
    } else {
      VecX SVecI =
          (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10))
              .cwiseSqrt()
              .cwiseInverse();
      MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
      x = SVecI.asDiagonal() *
          HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);
    }
    // Important: x is -step !
  }

  //[ ***step 4*** ] 如果设置的是直接对解进行处理, 直接去掉解x中的零空间
  if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
      (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
    VecX xOld = x;
    orthogonalize(&x, 0);
    // //********************* check nullspace added by gong
    // *********************** VecX new_b = HA_top * x; VecX old_b = HA_top *
    // xOld; std::cout<<"//=====================Test null space
    // start=====================/ "<<std::endl; std::cout<<"new_b - old_b: "<<
    // (new_b - old_b).transpose() << std::endl;
    // // xHx
    // std::cout<<"//=====================Test null space
    // end=====================/ "<<std::endl;
  }

  lastX = x;

  //[ ***step 5*** ] 分别求出各个待求量的增量值
  // resubstituteF(x, HCalib);
  currentLambda = lambda;
  // TODO 刚求的是posegraph，求现在要idp点的增量
  resubstituteF_MT(x, HCalib, multiThreading);
  currentLambda = 0;
}

//@ 设置EFFrame, EFPoint, EFResidual对应的 ID 号
void EnergyFunctional::makeIDX() {
  // 重新赋值ID
  for (unsigned int idx = 0; idx < frames.size(); idx++)
    frames[idx]->idx = idx;

  allPoints.clear();

  for (EFFrame *f : frames)      // TODO all fids
    for (EFPoint *p : f->points) // TODO pids hosted in this fid
    {
      allPoints.push_back(p);
      // 残差的ID号
      for (EFResidual *r : p->residualsAll) // TODO all vms of this pid
      {
        r->hostIDX = r->host->idx; // EFFrame的idx
        r->targetIDX =
            r->target->idx; // TODO 告诉优化器每个残差是哪个host和哪个target
      }
    }

  EFIndicesValid = true;
}

//@ 返回状态增量, 这里帧位姿和光度参数, 使用的是每一帧绝对的
VecX EnergyFunctional::getStitchedDeltaF() const {
  VecX d = VecX(CPARS + nFrames * 8);
  d.head<CPARS>() = cDeltaF.cast<double>(); // 相机内参增量
  for (int h = 0; h < nFrames; h++)
    d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
  return d;
}

} // namespace dso
