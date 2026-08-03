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

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctional.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

void EFResidual::takeDataF(int cid) {  // TODO welcome FEJ
  std::swap<RawResidualJacobian*>(J[cid],
                                  data->J[cid]);  // TODO 确实是把刚计算的给拿过来了
  //! 图像导数 * 图像导数 * 逆深度导数
  // TODO Jgx * Jgx * Jdd = [2x1]
  Vec2f JI_JI_Jd = J[cid]->JIdx2 * J[cid]->Jpdd;
  //! 位姿导数 * 图像导数 * 图像导数 * 逆深度导数
  for (int i = 0; i < 6; i++) JpJdF[cid][i] = J[cid]->Jpdxi[0][i] * JI_JI_Jd[0] + J[cid]->Jpdxi[1][i] * JI_JI_Jd[1];
  //! 图像导数 * 逆深度导数 * 光度导数
  // TODO JpJdF 表示Jpose_ab.transpose() * Jdd []
  // pose_ab跟idp对应的右上角的block
  JpJdF[cid].segment<2>(6) = J[cid]->JabJIdx * J[cid]->Jpdd;
  // TODO like "H12" in orca
  //     | H_poseab_poseab  H_poseab_dd | = | [8x8] [8x1] | = | [8x8] [JpJdF] |
  //     | J_dd_poseab      H_dd        |   | [1x8] [1x1] |   | [JpJdF.t]  [1x1]
  //     |
  //
  //
  //
  //
}

//@ 从 FrameHessian 中提取数据
void EFFrame::takeData() {
  prior = data->getPrior().head<STATE_DIM>();                   // 得到先验状态, 主要是光度仿射变换
  delta = data->get_state_minus_stateZero().head<STATE_DIM>();  // 状态与FEJ零状态之间差
  delta_prior = (data->get_state() - data->getPriorZero())
                    .head<STATE_DIM>();  // 状态与先验之间的差 //?
                                         // 可先验是0啊?
                                         // 可能因为只有第一帧才会用prior吧，而第一帧的真值就是Vec8::Zero()

  //	Vec10 state_zero =  data->get_state_zero();
  //	state_zero.segment<3>(0) = SCALE_XI_TRANS * state_zero.segment<3>(0);
  //	state_zero.segment<3>(3) = SCALE_XI_ROT * state_zero.segment<3>(3);
  //	state_zero[6] = SCALE_A * state_zero[6];
  //	state_zero[7] = SCALE_B * state_zero[7];
  //	state_zero[8] = SCALE_A * state_zero[8];
  //	state_zero[9] = SCALE_B * state_zero[9];
  //
  //	std::cout << "state_zero: " << state_zero.transpose() << "\n";

  assert(data->frameID != -1);

  frameID = data->frameID;  // 所有帧的ID序号
}

//@ 从PointHessian读取先验和当前状态信息
void EFPoint::takeData() {
  priorF = data->hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
  if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR) {
    printf("never use idepth prior!!!\n");
    priorF = 0;
  }
  // TODO 每次都更新线性化点，这不一直是零？？
  deltaF = data->idepth - data->idepth_zero;
}

//@ 计算线性化更新后的残差,
//! 没平方叫残差, 平方叫能量
void EFResidual::fixLinearizationF(EnergyFunctional* ef, int cid) {
  Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];  // 得到hostIDX -->
                                                                 // targetIDX的状态增量

  // compute Jp*delta
  __m128 Jp_delta_x = _mm_set1_ps(J[cid]->Jpdxi[0].dot(dp.head<6>()) + J[cid]->Jpdc[0].dot(ef->cDeltaF) +
                                  J[cid]->Jpdd[0] * point->deltaF);
  __m128 Jp_delta_y = _mm_set1_ps(J[cid]->Jpdxi[1].dot(dp.head<6>()) + J[cid]->Jpdc[1].dot(ef->cDeltaF) +
                                  J[cid]->Jpdd[1] * point->deltaF);
  __m128 delta_a = _mm_set1_ps((float)(dp[6]));
  __m128 delta_b = _mm_set1_ps((float)(dp[7]));

  for (int i = 0; i < patternNum * eachErrDim; i += 4) {
    // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
    // TODO  PATTERN: rtz = resF - [JI*Jp Ja]*delta.
    // TODO J->resF
    // 是最新状态下的残差，并不是fej状态下的残差，现在要把残差恢复到fej状态，（所以用减号），如果打印出来会发现恢复到fej状态的残差后，残差会变大
    // 残差 ← 残差 − J × Δx，
    __m128 rtz = _mm_load_ps(((float*)&J[cid]->resF) + i);
    //! res - J * delta_x
    // TODO 这是减法，subtract
    rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(J[cid]->JIdx)) + i), Jp_delta_x));
    rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(J[cid]->JIdx + 1)) + i), Jp_delta_y));
    rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(J[cid]->JabF)) + i), delta_a));
    rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float*)(J[cid]->JabF + 1)) + i), delta_b));
    _mm_store_ps(((float*)&res_toZeroF[cid]) + i,
                 rtz);  // TODO 存储在res_toZeroF
    // if(res_toZeroF[i] > J->resF[i])
    // printf("true");
  }
  // std::cout<<"resF: "<<J->resF<<" ||   res_toZeroF "<<res_toZeroF<<std::endl;
  isLinearized[cid] = true;
}

}  // namespace dso
