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

#include "FullSystem/ImmaturePoint.h"
#include "../camera_model/vio_math_0.h"
#include "FullSystem/ResidualProjections.h"
#include "util/FrameShell.h"

namespace dso { //! 这里u_ v_ 是加了0.5的
// ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian *host_, float type,
//                             CalibHessian *HCalib, const int &host_cid_)
//    : u(u_), v(v_), host(host_), my_type(type), host_cid(host_cid_),
//    idepth_min(0), idepth_max(NAN),
//      lastTraceStatus(IPS_UNINITIALIZED) {
//
//  gradH.setZero();
//
//  for (int idx = 0; idx < patternNum; idx++) {
//    int dx = patternP[idx][0];
//    int dy = patternP[idx][1];
//    // 由于+0.5导致积分, 插值得到值3个 [像素值, dx, dy]
//    Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy,
//    wG[0]);
//
//    color[idx] = ptc[0];
//    if (!std::isfinite(color[idx])) {
//      energyTH = NAN;
//      return;
//    }
//
//    // 梯度矩阵[dx*2, dxdy; dydx, dy^2]
//    gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
//    //! 点的权重 c^2 / ( c^2 + ||grad||^2 )
//    weights[idx] =
//        sqrtf(setting_outlierTHSumComponent /
//              (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
//  }
//
//  energyTH = patternNum * setting_outlierTH;
//  energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
//
//  idepth_GT = 0;
//  quality = 10000;
//}
#if 1//ndef USE_EDGE_ALIGN
#define USE_ZNCC_SEARCH
#endif
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian *host_, float type,
                             CalibHessian *HCalib, const int &host_cid_,
                             const int &host_level_)
    : u(u_), v(v_), host(host_), my_type(type), host_cid(host_cid_),
      host_level(host_level_), idepth_min(0), idepth_max(NAN) {
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    lastTraceStatus[cid] = IPS_UNINITIALIZED;
    quality[cid] = 10000;
  }

  gradH.setZero();
  gradH_converged.setZero();
  for (int idx = 0; idx < patternNumSeed; idx++) {
    float dx = patternPSeed[idx][0] * pattern_scale_seed_point_opt;
    float dy = patternPSeed[idx][1] * pattern_scale_seed_point_opt;
    // 由于+0.5导致积分, 插值得到值3个 [像素值, dx, dy]
    // Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy,
    // wG[0]);
    Vec3f ptc = getInterpolatedElement33BiLin(
        host->dIp[host_level_] + wG[host_level] * hG[host_level] * host_cid,
        u + dx, v + dy, wG[host_level]);

    color[idx] = ptc[0];
#ifdef USE_EDGE_ALIGN
    Vec3f dt_dx_dy = getInterpolatedElement33BiLin(
        host->dt_dx_dy[host_level_] + wG[host_level] * hG[host_level] * host_cid,
        u + dx, v + dy, wG[host_level]);
    distance_transform[idx] = dt_dx_dy[0];
#endif
    if (!std::isfinite(color[idx])
#ifdef USE_EDGE_ALIGN
    || !std::isfinite(distance_transform[idx])
#endif
    ) {
      energyTH = NAN;
      energyTH_converged = NAN;
      return;
    }

    // 梯度矩阵[dx*2, dxdy; dydx, dy^2]
    gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
    //! 点的权重 c^2 / ( c^2 + ||grad||^2 )
    weights_gray[idx] =
        sqrtf(setting_outlierTHSumComponent /
              (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
#if 1//ndef USE_EDGE_ALIGN
    weights[idx] = weights_gray[idx];
#else
    weights[idx] =
        sqrtf(setting_outlierTHSumComponent /
              (setting_outlierTHSumComponent + dt_dx_dy.tail<2>().squaredNorm()));
#endif
  }
  for (int idx = 0; idx < patternNum; idx++) {
    float dx = patternP[idx][0];
    float dy = patternP[idx][1];
    // 由于+0.5导致积分, 插值得到值3个 [像素值, dx, dy]
    // Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy,
    // wG[0]);
    Vec3f ptc = getInterpolatedElement33BiLin(
        host->dIp[host_level_] + wG[host_level] * hG[host_level] * host_cid,
        u + dx, v + dy, wG[host_level]);
#ifdef USE_EDGE_ALIGN
    Vec3f dt_dx_dy = getInterpolatedElement33BiLin(
        host->dt_dx_dy[host_level_] +
            wG[host_level] * hG[host_level] * host_cid,
        u + dx, v + dy, wG[host_level]);
#endif
    color_converged[idx] = ptc[0];
    if (!std::isfinite(color_converged[idx])
#ifdef USE_EDGE_ALIGN
        || dt_dx_dy.tail(2).norm() < 0.001f
#endif
    ) {
      energyTH = NAN;
      energyTH_converged = NAN;
      return;
    }

    // 梯度矩阵[dx*2, dxdy; dydx, dy^2]
    gradH_converged += ptc.tail<2>() * ptc.tail<2>().transpose();
    //! 点的权重 c^2 / ( c^2 + ||grad||^2 )
    weights_converged_gray[idx] =
        sqrtf(setting_outlierTHSumComponent /
              (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
#if 1//ndef USE_EDGE_ALIGN
    weights_converged[idx] = weights_converged_gray[idx];
#else
    weights_converged[idx] =
        sqrtf(setting_outlierTHSumComponent /
              (setting_outlierTHSumComponent + dt_dx_dy.tail<2>().squaredNorm()));
#endif
  }

  energyTH = patternNumSeed * setting_outlierTH_epi_trace_on * setting_outlierTH_epi_trace_on;
  energyTH_converged = patternNum * setting_outlierTH_init * setting_outlierTH_init;// 只被用来判断是不是finite，没用具体数值
  energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
  energyTH_converged *=
      setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

  idepth_GT = 0;
  //   quality[target_cid] = 10000;
}

ImmaturePoint::~ImmaturePoint() {}

/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
float ImmaturePoint::CalcZncc(const Eigen::MatrixXf &host_,
                              const Eigen::MatrixXf &target_) {
  if (host_.rows() != target_.rows()) {
    printf("zncc size doesn't match, sth wrong\n");
    std::exit(4);
  }
  Eigen::MatrixXf host = host_;
  Eigen::MatrixXf target = target_;
  const int patch_num = host.rows();
  Eigen::MatrixXf ones;
  ones.conservativeResize(patch_num, 1);
  ones.setOnes();

  const float host_val_mean = host.col(0).sum() / patch_num;
  const float target_val_mean = target.col(0).sum() / patch_num;

  host.col(0) = host.col(0) - host_val_mean * ones;
  target.col(0) = target.col(0) - target_val_mean * ones;

  const float host_sigma = host.col(0).norm();
  const float target_sigma = target.col(0).norm();
  host.col(0) /= host_sigma;
  target.col(0) /= target_sigma;

  float zncc = (target.col(0).dot(host.col(0)));

  float angle = (kOur_PI - std::acos(zncc)) / kOur_PI;
  angle = std::isnan(angle) ? 1 : angle;
  float r2 = 2 - 2 * zncc;
  float ws2 = 2.0 / (r2 + 2.0);
  return angle * std::sqrt(ws2)/*zncc*/;
}
///@ 使用深度滤波对未成熟点进行深度估计
//#define SHOW_TRACEON

ImmaturePointStatus ImmaturePoint::traceOn(
    const int &target_cid, FrameHessian *frame, const Mat33f &hostToFrame_KRKi,
    const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
    CalibHessian *HCalib, bool debugPrint, int lvl, bool is_first_frame,
    bool show_image) {
  hw_use[target_cid] = NAN;
  if (lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_OOB)
    return lastTraceStatus[target_cid];
  // float setting_huberTH_use = setting_huberTH_loose;
  debugPrint = false; // rand()%100==0;
#ifdef USE_MULTI_CAM
  float maxPixSearch = (wG[lvl] + hG[lvl]) * setting_maxPixSearch *
                       (is_first_frame ? 2.0f : 1.0f); // 极限搜索的最大长度
#else
  float maxPixSearch =
      (wG[lvl] + hG[lvl]) * setting_maxPixSearch; // 极限搜索的最大长度
#endif

  if (debugPrint) {
    printf("is_first_frame: %d, cid: [%d %d], level: [%d], trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f "
           "%f!\n",is_first_frame, host_cid, target_cid, lvl,
           u, v, host->shell->id, frame->shell->id, idepth_min, idepth_max,
           hostToFrame_Kt[0], hostToFrame_Kt[1], hostToFrame_Kt[2]);
  }
  //	const float stepsize = 1.0;				// stepsize for
  // initial discrete search.
  //	const int GNIterations = 3;				// max # GN
  // iterations
  //	const float GNThreshold = 0.1;				// GN stop after
  // this stepsize. 	const float extraSlackOnTH = 1.2; // for energy-based
  // outlier check, be slightly more relaxed by this factor. const float
  // slackInterval = 0.8;			// if pixel-interval is smaller
  // than this, leave it be. 	const float minImprovementFactor = 2;
  // // if pixel-interval is smaller than this, leave it be.
  // ============== project min and max. return if one of them is OOB
  // ===================
  //[ ***step 1*** ] 计算出来搜索的上下限, 对应idepth_max, idepth_min
  Vec3f pr = hostToFrame_KRKi * Vec3f(u, v, 1);
  Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;
  float uMin = ptpMin[0] / ptpMin[2];
  float vMin = ptpMin[1] / ptpMin[2];

  Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();
  int maxRotPatX = 0;
  int maxRotPatY = 0;

#ifdef SHOW_TRACEON
  MinimalImageB3 *img_host;
  MinimalImageB3 *img_target;
  if (show_image) {
    img_host = new MinimalImageB3(wG[lvl], hG[lvl]);
    img_target = new MinimalImageB3(wG[lvl], hG[lvl]);

    for (int i = 0; i < wG[lvl] * hG[lvl]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(host->dIp[lvl] + wG[lvl] * hG[lvl] * host_cid + i))[0];
      if (colL < 0)
        colL = 0;
      if (colL > 255)
        colL = 255;
      img_host->at(i, host_cid) = Vec3b(colL, colL, colL);
      colL = (*(frame->dIp[lvl] + wG[lvl] * hG[lvl] * target_cid + i))[0];
      if (colL < 0)
        colL = 0;
      if (colL > 255)
        colL = 255;
      img_target->at(i, target_cid) = Vec3b(colL, colL, colL);
    }

    img_host->setPixel9(u + 0.5, v + 0.5, makeRainbow3B(1), host_cid);
  }
#endif

  //* pattern在新的帧上的偏移量
  Vec2f rotatetPattern[MAX_RES_PER_POINT_SEED];
  for (int idx = 0; idx < patternNumSeed; idx++) {
    rotatetPattern[idx] =
        Rplane * Vec2f(patternPSeed[idx][0] * pattern_scale_seed_point_opt, patternPSeed[idx][1] * pattern_scale_seed_point_opt);
    int absX = (int)abs(rotatetPattern[idx][0]);
    int absY = (int)abs(rotatetPattern[idx][1]);
    maxRotPatX = std::max(absX, maxRotPatX);
    maxRotPatY = std::max(absY, maxRotPatY);
  }
  int realBoundU = maxRotPatX + 2;
  int realBoundV = maxRotPatY + 2;
  int boundU = 4;
  int boundV = 4;
  boundU = std::max(boundU, realBoundU);
  boundV = std::max(boundV, realBoundV);
  // 如果超出图像范围则设为 OOB
  if (!(uMin > boundU && vMin > boundV && uMin < wG[lvl] - boundU - 1 &&
        vMin < hG[lvl] - boundV - 1)) {
    if (debugPrint)
      printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n", u, v, uMin, vMin,
             ptpMin[2], idepth_min, idepth_max);
    lastTraceUV[target_cid] = Vec2f(-1, -1);
    lastTracePixelInterval[target_cid] = 0;
#ifdef SHOW_TRACEON
    if (show_image) {
      delete img_host;
      delete img_target;
      printf("return 1\n");
    }
#endif
    return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
  }

  float dist;
  float uMax;
  float vMax;
  Vec3f ptpMax;
  if (std::isfinite(idepth_max)) {
    ptpMax = pr + hostToFrame_Kt * idepth_max;
    uMax = ptpMax[0] / ptpMax[2];
    vMax = ptpMax[1] / ptpMax[2];

    if (!(uMax > boundU && vMax > boundV && uMax < wG[lvl] - boundU - 1 &&
          vMax < hG[lvl] - boundV - 1)) {
      if (debugPrint)
        printf("OOB uMax  %f %f - %f %f!\n", u, v, uMax, vMax);
      lastTraceUV[target_cid] = Vec2f(-1, -1);
      lastTracePixelInterval[target_cid] = 0;
#ifdef SHOW_TRACEON
      if (show_image) {
        delete img_host;
        delete img_target;
        printf("return 2\n");
      }
#endif
      return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
    }

    // ============== check their distance. everything below 2px is OK (->
    // skip). ===================
    dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
    dist = sqrtf(dist);
    /// 说明此时深度已经收敛的比较好了
    if (dist < setting_trace_slackInterval) {
      if (debugPrint)
        printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

      lastTraceUV[target_cid] =
          Vec2f(uMax + uMin, vMax + vMin) * 0.5; // 直接设为中值
      lastTracePixelInterval[target_cid] = dist;
#ifdef SHOW_TRACEON
      if (show_image) {
        delete img_host;
        delete img_target;
        printf("return 3\n");
      }
#endif
      return lastTraceStatus[target_cid] =
                 ImmaturePointStatus::IPS_SKIPPED; //跳过
    }
    assert(dist > 0);
  } else { //* 上限无穷大, 则设为最大值
    dist = maxPixSearch;

    // project to arbitrary depth to get direction.
    ptpMax = pr + hostToFrame_Kt * 0.01;
    uMax = ptpMax[0] / ptpMax[2];
    vMax = ptpMax[1] / ptpMax[2];

    // direction.
    float dx = uMax - uMin;
    float dy = vMax - vMin;
    float d = 1.0f / sqrtf(dx * dx + dy * dy);
    //* 根据比例得到最大值
    // set to [setting_maxPixSearch].
    uMax = uMin + dist * dx * d;
    vMax = vMin + dist * dy * d;

    // may still be out!
    if (!(uMax > boundU && vMax > boundV && uMax < wG[lvl] - boundU - 1 &&
          vMax < hG[lvl] - boundV - 1)) {
      if (debugPrint)
        printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax, ptpMax[2]);
      lastTraceUV[target_cid] = Vec2f(-1, -1);
      lastTracePixelInterval[target_cid] = 0;
#ifdef SHOW_TRACEON
      if (show_image) {
        delete img_host;
        delete img_target;
        printf("return 4\n");
      }
#endif
      return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
    }
    assert(dist > 0);
  }

  //? 为什么是这个值呢??? 0.75 - 1.5
  /// 这个值是两个帧上深度的比值, 它的变化太大就是前后尺度变化太大了
  // set OOB if scale change too big.
  if (!(idepth_min < 0 ||
        (ptpMin[2] > 0.1 /*0.5 0.75*/ && ptpMin[2] < 10.0 /*2 .01.5*/))) {
    if (debugPrint)
      printf("OOB SCALE %f %f %f!\n", uMax, vMax, ptpMin[2]);
    lastTraceUV[target_cid] = Vec2f(-1, -1);
    lastTracePixelInterval[target_cid] = 0;
#ifdef SHOW_TRACEON
    if (show_image) {
      delete img_host;
      delete img_target;
      printf("return 5\n");
    }
#endif
    return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
  }

  //[ ***step 2*** ] 计算误差大小(图像梯度和极线夹角大小), 夹角大,
  //小的几何误差会有很大影响
  // ============== compute error-bounds on result in pixel. if the new interval
  // is not at least 1/2 of the old, SKIP ===================
  float dx = setting_trace_stepsize * (uMax - uMin);
  float dy = setting_trace_stepsize * (vMax - vMin);
  //! (dIx*dx + dIy*dy)^2
  float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
  //! (dIx*dy - dIy*dx)^2
  float b = (Vec2f(dy, -dx).transpose() * gradH *
             Vec2f(dy, -dx)); // (dx, dy)垂直方向的乘积
  // 计算的是极线方向和梯度方向的夹角大小，90度则a=0,
  // errorInPixel变大；平行时候b=0
  float errorInPixel =
      0.2f +
      0.2f * (a + b) / a; /// 没有使用LSD的方法, 估计是能有效防止位移小的情况
  //* errorInPixel大说明垂直, 这时误差会很大, 视为bad
  if (errorInPixel * setting_trace_minImprovementFactor > dist &&
      std::isfinite(idepth_max)) {
    if (debugPrint)
      printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
    lastTraceUV[target_cid] = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
    lastTracePixelInterval[target_cid] = dist;
#ifdef SHOW_TRACEON
    if (show_image) {
      delete img_host;
      delete img_target;
      printf("return 6\n");
    }
#endif
    return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_BADCONDITION;
  }

  if (errorInPixel > 10)
    errorInPixel = 10;

  // ============== do the discrete search ===================
  //[ ***step 3*** ] 在极线上找到最小的光度误差的位置,
  //并计算和第二次的比值作为质量
  dx /= dist; // cos
  dy /= dist; // sin

  if (debugPrint)
    printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> "
           "%f (%.1f %.1f)! ErrorInPixel %.1f!\n",
           u, v, host->shell->id, frame->shell->id, idepth_min, uMin, vMin,
           idepth_max, uMax, vMax, errorInPixel);

  if (dist > maxPixSearch) {
    uMax = uMin + maxPixSearch * dx;
    vMax = vMin + maxPixSearch * dy;
    dist = maxPixSearch;
  }

  int numSteps = 1.9999f + dist / (setting_trace_stepsize *
                                   std::pow(2.0, 0.0 /*-lvl*/)); // 步数

  float randShift =
      uMin * 1000 - floorf(uMin * 1000); // 	取小数点后面的做随机数??
  float ptx = uMin - randShift * dx;
  float pty = vMin - randShift * dy;

  // 这个判断太多了, 学习学习, 全面考虑
  if (!std::isfinite(dx) || !std::isfinite(dy)) {
    // printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

    lastTracePixelInterval[target_cid] = 0;
    lastTraceUV[target_cid] = Vec2f(-1, -1);
#ifdef SHOW_TRACEON
    if (show_image) {
      delete img_host;
      delete img_target;
      printf("return 7\n");
    }
#endif
    return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
  }

  //* 沿着级线搜索误差最小的位置
  int step_num;
  if (is_first_frame) {
    step_num = 200;
  } else {
    step_num = 100;
  }
  float errors[step_num];      //[150];
  float errors_zncc[step_num]; //[150];
  // float errors_edge[step_num]; //[150];
  float bestU = 0, bestV = 0, bestEnergy = 1e10, bestEnergy_zncc = 0, best_mean_hw = 0;
  int bestIdx = -1;
#ifdef SHOW_TRACEON
  if (show_image) {
    printf("numSteps: %d\n", numSteps);
  }
#endif
  if (numSteps >= step_num /*150*/)
    numSteps = step_num - 1; //[149]

  float zncc_each = 0;
  // float dt_each = 0;
  for (int i = 0; i < numSteps; i++) {
    float energy = 0;
    Eigen::MatrixXf host_val_each, target_val_each;
    int valid_count_each = 0;
    float hw_sum = 0;
    float hw_count = 0;
    for (int idx = 0; idx < patternNumSeed; idx++) {
      float hitColor;
      float hitColor_edge;
      if (!is_first_frame) {
        hitColor = getInterpolatedElement31(
            frame->dI + wG[lvl] * hG[lvl] * target_cid,
            (float)(ptx + rotatetPattern[idx][0]),
            (float)(pty + rotatetPattern[idx][1]), wG[lvl]);
        hitColor_edge = getInterpolatedElement31(
            frame->dt_dx_dy_0 + wG[lvl] * hG[lvl] * target_cid,
            (float)(ptx + rotatetPattern[idx][0]),
            (float)(pty + rotatetPattern[idx][1]), wG[lvl]);
      } else {
        hitColor = getInterpolatedElement31(
            frame->dIp[lvl] + wG[lvl] * hG[lvl] * target_cid,
            (float)(ptx + rotatetPattern[idx][0]),
            (float)(pty + rotatetPattern[idx][1]), wG[lvl]);
        hitColor_edge = getInterpolatedElement31(
            frame->dt_dx_dy[lvl] + wG[lvl] * hG[lvl] * target_cid,
            (float)(ptx + rotatetPattern[idx][0]),
            (float)(pty + rotatetPattern[idx][1]), wG[lvl]);
      }
#ifdef SHOW_TRACEON
      if (show_image) {
        img_target->setPixel9(ptx + rotatetPattern[idx][0],
                              pty + rotatetPattern[idx][1], makeRainbow3B(1),
                              target_cid);
          if (i == 0){
            img_target->setPixelCirc(ptx+ rotatetPattern[idx][0], pty+ rotatetPattern[idx][1], Vec3b(255, 0, 0), target_cid);
          }
          if (i == numSteps - 1){
            img_target->setPixelCirc(ptx+ rotatetPattern[idx][0], pty+ rotatetPattern[idx][1], Vec3b(0, 255, 255), target_cid);
          }
      }
#endif
      if (!std::isfinite(hitColor)) {
        energy += 1e5;
        continue;
      }
#ifdef USE_ZNCC_SEARCH
      float residual =
          hitColor - (float)(hostToFrame_affine[0] *
                                 (color[idx /* + wG[0] * hG[0] *  host_cid*/]) +
                             hostToFrame_affine[1]);
#else
      float residual = hitColor_edge;
#endif
      float hw = fabs(residual) < setting_huberTH_trace_on
                     ? 1
                     : setting_huberTH_trace_on / fabs(residual);
      if (debugPrint) {
        printf("step: %d, idx: %d, residual: %f, setting_huberTH_trace_on: %f, hw: %f\n", i, idx, residual,setting_huberTH_trace_on, hw);
      }
      hw_sum += hw;
      hw_count += 1;
      energy += hw * residual * residual * (2 - hw);
      host_val_each.conservativeResize(valid_count_each + 1, 1);
      target_val_each.conservativeResize(valid_count_each + 1, 1);
      host_val_each(valid_count_each, 0) =
          (float)(hostToFrame_affine[0] *
                      (color[idx /* + wG[0] * hG[0] *  host_cid*/]) +
                  hostToFrame_affine[1]);
      target_val_each(valid_count_each, 0) = hitColor;
      valid_count_each++;
    }
    zncc_each = CalcZncc(host_val_each, target_val_each);
    if (debugPrint) {
      printf("step %.1f %.1f (id %f): energy = %f! zncc_each: %f\n", ptx, pty, 0.0f, energy, zncc_each);
    }
    errors[i] = energy;
    errors_zncc[i] = zncc_each;
#ifndef USE_ZNCC_SEARCH
    if (energy < bestEnergy) {
      bestU = ptx;
      bestV = pty;
      bestEnergy = energy;
      bestIdx = i;
      bestEnergy_zncc = zncc_each;
      best_mean_hw = hw_sum / hw_count;
    }
#else
    if (zncc_each > bestEnergy_zncc) {
      bestU = ptx;
      bestV = pty;
      bestEnergy = energy;
      bestIdx = i;
      bestEnergy_zncc = zncc_each;
      best_mean_hw = hw_sum / hw_count;
    }
#endif
    // 每次走1 dist对应大小
    ptx += dx;
    pty += dy;
  }
#ifdef SHOW_TRACEON
  if (show_image) {
    img_target->setPixelCirc(bestU, bestV, Vec3b(0, 0, 255), target_cid);
  }
#endif
  ///* 在一定的半径内找最到误差第二小的, 差的足够大, 才更好(这个常用)
  // find best score outside a +-2px radius.
  float secondBest = 1e10;
  float secondBest_zncc = 0;
  for (int i = 0; i < numSteps; i++) {
    if ((i < bestIdx - setting_minTraceTestRadius ||
         i > bestIdx + setting_minTraceTestRadius) &&
#ifndef USE_ZNCC_SEARCH
        errors[i] < secondBest
#else
        errors_zncc[i] > secondBest_zncc
#endif
    ) {
      secondBest = errors[i];
      secondBest_zncc = errors_zncc[i];
    }
  }
  float newQuality = secondBest / bestEnergy;
  float newQuality_zncc = bestEnergy_zncc / secondBest_zncc;
#ifndef USE_ZNCC_SEARCH
  float newQuality_all = std::min(newQuality, newQuality_zncc);
#else
  float newQuality_all = std::max(newQuality, newQuality_zncc);
#endif
  if (debugPrint) {
    printf("\n++++++ best_step: %d, numSteps: %d, best_energy: %f, best_zncc: %f, best_mean_hw: %f, newQuality: %f, newQuality_zncc: %f\n",bestIdx, numSteps, bestEnergy, bestEnergy_zncc, best_mean_hw, newQuality, newQuality_zncc);
  }
#ifndef USE_ZNCC_SEARCH
  if (newQuality < quality[target_cid] || numSteps > 10) {
    quality[target_cid] = newQuality;
  }
#else
  if (newQuality_all < quality[target_cid] || numSteps > 10) {
    quality[target_cid] = newQuality_all;
  }
#endif
  //[ ***step 4*** ] 在上面的最优位置进行线性搜索, 进行求精
  // ============== do GN optimization ===================
  float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0, zncc = 0;
  if (setting_trace_GNIterations > 0)
    bestEnergy = 1e5;
  int gnStepsGood = 0, gnStepsBad = 0;
  float hw_sum = 0, hw_count = 0;;
  for (int it = 0; it < setting_trace_GNIterations; it++) {
    float H = 1, b = 0, energy = 0, zncc = 0;
    Eigen::MatrixXf host_val, target_val;
    int valid_count = 0;
    for (int idx = 0; idx < patternNumSeed; idx++) {
      float posU = (float)(bestU + rotatetPattern[idx][0]);
      float posV = (float)(bestV + rotatetPattern[idx][1]);
      if (posU < 0 || posV < 0 || posU >= wG[lvl] - 1 || posV >= hG[lvl] - 1) {
        if (debugPrint)
          printf("OOB uMax  %f %f - %f %f!\n", posU, posV, uMax, vMax);
        lastTraceUV[target_cid] = Vec2f(-1, -1);
        lastTracePixelInterval[target_cid] = 0;
#ifdef SHOW_TRACEON
        if (show_image) {
          delete img_host;
          delete img_target;
          printf("return 8\n");
        }
#endif
        return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
      }

      Vec3f hitColor;
      Vec3f hitColor_edge;
      if (!is_first_frame) {
        hitColor = getInterpolatedElement33(
            frame->dI + wG[lvl] * hG[lvl] * target_cid, posU, posV, wG[lvl]);
        hitColor_edge = getInterpolatedElement33(
            frame->dt_dx_dy_0 + wG[lvl] * hG[lvl] * target_cid, posU, posV, wG[lvl]);
      } else {
        hitColor = getInterpolatedElement33(frame->dIp[lvl] +
                                                wG[lvl] * hG[lvl] * target_cid,
                                            posU, posV, wG[lvl]);
        hitColor_edge = getInterpolatedElement33(frame->dt_dx_dy[lvl] +
                                                wG[lvl] * hG[lvl] * target_cid,
                                            posU, posV, wG[lvl]);
      }
      /// 1维搜索，自变量只有一个，H和b都是一个数
      if (!std::isfinite((float)hitColor[0])
#ifndef USE_ZNCC_SEARCH
      || !std::isfinite((float)hitColor_edge[0])
#endif
      ) {
        energy += 1e5;
        continue;
      }
#ifdef USE_ZNCC_SEARCH
      float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] +
                                      hostToFrame_affine[1]);
      float dResdDist =
          dx * hitColor[1] + dy * hitColor[2]; /// 极线方向梯度, jacobian
#else
      float residual = hitColor_edge[0];
      float dResdDist =
          dx * hitColor_edge[1] + dy * hitColor_edge[2]; /// 极线方向梯度, jacobian
#endif

      float hw = fabs(residual) < setting_huberTH_trace_on
                     ? 1
                     : setting_huberTH_trace_on / fabs(residual);
      if (debugPrint) {
        printf("iter: %d, idx: %d, res: %f, hw: %f, setting_huberTH_trace_on: %f\n", it, idx, residual, hw, setting_huberTH_trace_on);
      }
      /// 跟一维光流一样，只是不再是正方形邻域，变成了环形邻域
      H += hw * dResdDist * dResdDist;
      b += hw * residual * dResdDist;
      hw_sum += hw;
      hw_count += 1.0;
#ifdef USE_ZNCC_SEARCH
      energy +=
          weights_gray[idx] * weights_gray[idx] * hw * residual * residual * (2 - hw);
#else
      energy +=
          weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
#endif
      host_val.conservativeResize(valid_count + 1, 1);
      target_val.conservativeResize(valid_count + 1, 1);
      host_val(valid_count, 0) =
          (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
      target_val(valid_count, 0) = hitColor[0];
      valid_count++;
    }
    zncc = CalcZncc(host_val, target_val);
    if (debugPrint) {
      printf("\n===== iter: %d, energy: %f, zncc: %f, best_energy: %f, best_zncc: %f\n", it, energy, zncc,bestEnergy,bestEnergy_zncc);
    }
#ifndef USE_ZNCC_SEARCH
    if ((zncc < bestEnergy_zncc) || (energy > bestEnergy)) {
      gnStepsBad++;

      // do a smaller step from old point.
      stepBack *= 0.5; //* 减小步长再进行计算
      bestU = uBak + stepBack * dx;
      bestV = vBak + stepBack * dy;
      if (debugPrint)
        printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
               it, energy, H, b, stepBack, uBak, vBak, bestU, bestV);
    }
#else
    if ((zncc < bestEnergy_zncc) || (energy > bestEnergy)) {
      gnStepsBad++;

      // do a smaller step from old point.
      stepBack *= 0.5; //* 减小步长再进行计算
      bestU = uBak + stepBack * dx;
      bestV = vBak + stepBack * dy;
      if (debugPrint)
        printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
               it, energy, H, b, stepBack, uBak, vBak, bestU, bestV);
    }
#endif
    else {
      gnStepsGood++;

      float step = -gnstepsize * b / H;
      //* 步长最大才0.5
      /// 最大0.5, 防止点跟飞掉
      if (step < -0.5)
        step = -0.5;
      else if (step > 0.5)
        step = 0.5;

      if (!std::isfinite(step))
        step = 0;

      uBak = bestU; // 备份
      vBak = bestV;
      stepBack = step;

      bestU += step * dx;
      bestV += step * dy;
      //#ifndef USE_ZNCC_SEARCH
      bestEnergy = energy;
      //#else
      bestEnergy_zncc = zncc;
      //#endif
      if (debugPrint)
        printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
               it, energy, H, b, step, uBak, vBak, bestU, bestV);
    }

    if (fabsf(stepBack) < setting_trace_GNThreshold)
      break;
  }

#ifdef SHOW_TRACEON
  if (show_image) {
    img_target->setPixelCirc(bestU, bestV, Vec3b(0, 0, 255), target_cid);
  }
  if (debugPrint){
    printf("\n##### bestEnergy: %f, energyTH: %f, setting_trace_extraSlackOnTH: %f, bestEnergy_zncc_angle: %f, setting_outlierTH_zncc_tracker: %f, setting_outlierTH_zncc_angle_epi_trace_on: %f\n", bestEnergy, energyTH, setting_trace_extraSlackOnTH,bestEnergy_zncc, setting_outlierTH_zncc_tracker, setting_outlierTH_zncc_angle_epi_trace_on);
  }
#endif
  // ============== detect energy-based outlier. ===================
  //	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU,
  // bestV, wG[0]); 	float absGrad1 =
  // getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25,
  // bestV*0.5-0.25, wG[1]); 	float absGrad2 =
  // getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375,
  // bestV*0.25-0.375, wG[2]);
  if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH) ||
      bestEnergy_zncc < setting_outlierTH_zncc_angle_epi_trace_on/*setting_outlierTH_zncc_tracker*/)
  //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
  //		     && absGrad1*areaGradientSlackFactor <
  // host->frameGradTH*0.75f
  //			 && absGrad2*areaGradientSlackFactor <
  // host->frameGradTH*0.50f))
  {
    if (debugPrint)
      printf("OUTLIER!\n");

    lastTracePixelInterval[target_cid] = 0;
    lastTraceUV[target_cid] = Vec2f(-1, -1);
    if (lastTraceStatus[target_cid] == ImmaturePointStatus::IPS_OUTLIER) {
#ifdef SHOW_TRACEON
      if (show_image) {
        printf("return 9\n");
        IOWrap::displayImage("host", img_host);
        IOWrap::displayImage("target", img_target);
        IOWrap::waitKey(0);
        delete img_host;
        delete img_target;
      }
#endif
      return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OOB;
    } else {
#ifdef SHOW_TRACEON
      if (show_image) {
        printf("return 10, is_first_frame: %d,  host_cid: %d, target_cid: %d, idepth_min: %f, idepth_max: %f\n", is_first_frame, host_cid, target_cid, idepth_min, idepth_max);
        IOWrap::displayImage("host", img_host);
        IOWrap::displayImage("target", img_target);
        IOWrap::waitKey(0);
        delete img_host;
        delete img_target;
      }
#endif
      return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OUTLIER;
    }
  }

  //[ ***step 5*** ] 根据得到的最优位置重新计算逆深度的范围
  // TODO 这里是三角化了，跟LSD-SLAM一样, use the same formular in LSD-SLAM to
  // update new inverse depth search interval
  // ============== set new interval ===================
  //! u = (pr[0] + Kt[0]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (u*pr[2]
  //! - pr[0]) / (Kt[0] - u*Kt[2]) v = (pr[1] + Kt[1]*idepth) / (pr[2] +
  //! Kt[2]*idepth) ==> idepth = (v*pr[2] - pr[1]) / (Kt[1] - v*Kt[2])
  //* 取误差最大的
  if (dx * dx > dy * dy) {
    idepth_min =
        (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
        (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
    idepth_max =
        (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
        (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
  } else {
    idepth_min =
        (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
        (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
    idepth_max =
        (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
        (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
  }
  if (idepth_min > idepth_max)
    std::swap<float>(idepth_min, idepth_max);

  if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) ||
      (idepth_max < 0)) {
    // printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min,
    // idepth_max);

    lastTracePixelInterval[target_cid] = 0;
    lastTraceUV[target_cid] = Vec2f(-1, -1);
#ifdef SHOW_TRACEON
    if (show_image) {
      printf("return 11\n");
      IOWrap::displayImage("host", img_host);
      IOWrap::displayImage("target", img_target);
      IOWrap::waitKey(0);
      delete img_host;
      delete img_target;
    }
#endif
    return lastTraceStatus[target_cid] = ImmaturePointStatus::IPS_OUTLIER;
  }
#ifdef SHOW_TRACEON
  //    std::cout << "idx: " << idx << ", hostColor: " << hostColor.transpose()
  //              << ", hitColor: " << hitColor.transpose()
  //              << ", affLL: " << affLL.transpose()
  //              << ", color[idx]: " << color[idx] << std::endl;
  if (show_image) {
    printf("GOOD !!! is_first_frame: %d,  host_cid: %d, target_cid: %d, idepth_min: %f, idepth_max: %f\n", is_first_frame, host_cid, target_cid, idepth_min, idepth_max);
    IOWrap::displayImage("host", img_host);
    IOWrap::displayImage("target", img_target);
    IOWrap::waitKey(0);

    delete img_host;
    delete img_target;
  }
#endif
  lastTracePixelInterval[target_cid] = 2 * errorInPixel; // 搜索的范围
  lastTraceUV[target_cid] = Vec2f(bestU, bestV); // 上一次得到的最有位置
  hw_use[target_cid] = hw_sum / hw_count;
  return lastTraceStatus[target_cid] =
             ImmaturePointStatus::IPS_GOOD; //上一次的位置
}

float ImmaturePoint::getdPixdd(CalibHessian *HCalib,
                               ImmaturePointTemporaryResidual *tmpRes,
                               float idepth) {
  FrameFramePrecalc *precalc = &(host->targetPrecalc[tmpRes->target->idx]);
  const Vec3f &PRE_tTll = precalc->PRE_tTll;
  float drescale, u = 0, v = 0, new_idepth;
  float Ku, Kv;
  Vec3f KliP;

  projectPoint(this->u, this->v, idepth, 0, 0, HCalib, precalc->PRE_RTll,
               PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

  float dxdd = (PRE_tTll[0] - PRE_tTll[2] * u) * HCalib->fxl();
  float dydd = (PRE_tTll[1] - PRE_tTll[2] * v) * HCalib->fyl();
  return drescale * sqrtf(dxdd * dxdd + dydd * dydd);
}
// function not used
float ImmaturePoint::calcResidual(CalibHessian *HCalib,
                                  const float outlierTHSlack,
                                  ImmaturePointTemporaryResidual *tmpRes,
                                  float idepth) {
  FrameFramePrecalc *precalc = &(host->targetPrecalc[tmpRes->target->idx]);
  // float setting_huberTH_use = setting_huberTH_loose;
  float energyLeft = 0;
  const Eigen::Vector3f *dIl = tmpRes->target->dI;
  const Eigen::Vector3f *dIl_edge = tmpRes->target->dt_dx_dy_0;
  const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
  const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
  Vec2f affLL = precalc->PRE_aff_mode;

  for (int idx = 0; idx < patternNumSeed; idx++) {
    float Ku, Kv;
    if (!projectPoint(this->u + patternPSeed[idx][0] * pattern_scale_seed_point_opt,
                      this->v + patternPSeed[idx][1] * pattern_scale_seed_point_opt, idepth, PRE_KRKiTll,
                      PRE_KtTll, Ku, Kv)) {
      return 1e10;
    }

    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
    Vec3f hitColor_edge = (getInterpolatedElement33(dIl_edge, Ku, Kv, wG[0]));
    if (!std::isfinite((float)hitColor[0])
#ifndef USE_ZNCC_SEARCH
    || !std::isfinite((float)hitColor_edge[0])
#endif
    ) {
      return 1e10;
    }
    // if(benchmarkSpecialOption==5) hitColor =
    // (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));
#ifdef USE_ZNCC_SEARCH
    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);
#else
    float residual = hitColor_edge[0];
#endif
    float hw = fabsf(residual) < setting_huberTH_trace_on
                   ? 1
                   : setting_huberTH_trace_on / fabsf(residual);
#ifdef USE_ZNCC_SEARCH
    energyLeft +=
        weights_gray[idx] * weights_gray[idx] * hw * residual * residual * (2 - hw);
#else
    energyLeft +=
        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
#endif
  }

  if (energyLeft > energyTH * outlierTHSlack) {
    energyLeft = energyTH * outlierTHSlack;
  }
  return energyLeft;
}

///@ 计算当前点逆深度的残差, 正规方程(H和b), 残差状态
//#define SHOW_POINT_OPT
double ImmaturePoint::linearizeResidual(const int &target_cid,
                                        CalibHessian *HCalib,
                                        const float outlierTHSlack,
                                        ImmaturePointTemporaryResidual *tmpRes,
                                        float &Hdd, float &bd, float idepth,
                                        int lvl_target, bool print_info) {

  float zncc = 0;
  zncc_opt = 0;
  hw_use[target_cid] = NAN;
  if (tmpRes->state_state == ResState::OOB) {
    tmpRes->state_NewState = ResState::OOB;
    return tmpRes->state_energy;
  }
  // float setting_huberTH_use = setting_huberTH_loose;
  FrameFramePrecalc *precalc = &(host->targetPrecalc[tmpRes->target->idx]);

  // check OOB due to scale angle change.

  float energyLeft = 0;
  // TODO roger, only opt in level 0
  const Eigen::Vector3f *dIl = tmpRes->target->dIp[lvl_target] +
                               wG[lvl_target] * hG[lvl_target] * target_cid;
  const Eigen::Vector3f *dIl_edge = tmpRes->target->dt_dx_dy[lvl_target] +
                               wG[lvl_target] * hG[lvl_target] * target_cid;
  const Mat33f &PRE_RTll =
      precalc->a_PRE_RTll[host_cid * kCameraNumUsed + target_cid];
  const Vec3f &PRE_tTll =
      precalc->a_PRE_tTll[host_cid * kCameraNumUsed + target_cid];
  // const float * const Il = tmpRes->target->I;

  Vec2f affLL = precalc->PRE_aff_mode;

#ifdef SHOW_POINT_OPT
  bool show_image = true;
  MinimalImageB3 *img_host;
  MinimalImageB3 *img_target;
  if (show_image) {
    img_host = new MinimalImageB3(wG[0], hG[0]);
    img_target = new MinimalImageB3(wG[0], hG[0]);

    for (int i = 0; i < wG[0] * hG[0]; i++) {
      // BRIGHTNESS TRANSFER
      float colL = (*(host->dI + wG[0] * hG[0] * host_cid + i))[0];
      if (colL < 0)
        colL = 0;
      if (colL > 255)
        colL = 255;
      img_host->at(i, host_cid) = Vec3b(colL, colL, colL);
      colL = (*(dIl + i))[0];
      if (colL < 0)
        colL = 0;
      if (colL > 255)
        colL = 255;
      img_target->at(i, target_cid) = Vec3b(colL, colL, colL);
    }

    img_host->setPixel9(this->u + 0.5, this->u + 0.5, makeRainbow3B(1),
                        host_cid);
  }

#endif
  Eigen::MatrixXf host_val, target_val;
  int valid_count = 0;
  float hw_sum = 0;
  float hw_count = 0;
  for (int idx = 0; idx < patternNumSeed; idx++) {
    float dx = patternPSeed[idx][0] * pattern_scale_seed_point_opt;
    float dy = patternPSeed[idx][1] * pattern_scale_seed_point_opt;

    float drescale, u, v, new_idepth; /// u,v metric coordinate
    float Ku, Kv;                     /// pixel coordinate
    Vec3f KliP;
    /// kidding me ? new_idepth was never used, not to mention it's not even the
    /// ACTUAL new_idepth in target frame
    bool projectedd = projectPoint(this->u, this->v, idepth, dx, dy, HCalib,
                                   PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv,
                                   KliP, new_idepth, lvl_target);
#ifdef SHOW_POINT_OPT
    if (show_image) {
      if (idx == 0 && (Ku > 15 && Kv > 15 && Ku < wG[0] - 15 &&
                       Kv < hG[0] - 15 && new_idepth > 0)) {
        img_target->setPixel9(Ku + 0.5, Kv + 0.5, makeRainbow3B(1), target_cid);
        IOWrap::displayImage("host", img_host);
        IOWrap::displayImage("target", img_target);
        IOWrap::waitKey(0);
      }
      if (idx == 0) {
        delete img_host;
        delete img_target;
      }
    }
#endif

    if (!projectedd) {
      tmpRes->state_NewState = ResState::OOB;
      return tmpRes->state_energy;
    }

    /// dIl传进来只是为了得到数据的地址,用以指向角点4邻域内的数据，它的内容并不参与计算，指针妙用
    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[lvl_target]));
    Vec3f hitColor_edge = (getInterpolatedElement33(dIl_edge, Ku, Kv, wG[lvl_target]));

    if (!std::isfinite((float)hitColor[0])
#ifndef USE_ZNCC_SEARCH
    || !std::isfinite((float)hitColor_edge[0])
#endif
    ) {
      tmpRes->state_NewState = ResState::OOB;
      return tmpRes->state_energy;
    }
    /// pattern of 8, so there should be 8 residuals contributing to 1
    /// photometric factor
#ifdef USE_ZNCC_SEARCH
    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);
#else
    float residual = hitColor_edge[0];
#endif
    // printf("idx: %d, residual: %f\n", idx, residual);

    float hw = fabsf(residual) < setting_huberTH_linearize
                   ? 1
                   : setting_huberTH_linearize / fabsf(residual);
#ifdef USE_ZNCC_SEARCH
    energyLeft +=
        weights_gray[idx] * weights_gray[idx] * hw * residual * residual * (2 - hw);
#else
    energyLeft +=
        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
#endif
    hw_sum += hw;
    hw_count += 1;
    // depth derivatives.
    /// assume the 8 neighbours share the same idepth
#ifdef USE_ZNCC_SEARCH
    float dxInterp = hitColor[1] * fxG[lvl_target]; // HCalib->fxl();
    float dyInterp = hitColor[2] * fyG[lvl_target]; // HCalib->fyl();
#else
    float dxInterp = hitColor_edge[1] * fxG[lvl_target]; // HCalib->fxl();
    float dyInterp = hitColor_edge[2] * fyG[lvl_target]; // HCalib->fyl();
#endif
    float d_idepth =
        derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);
    if (print_info) {
      printf("idx: %d, weights[idx]: %f,weights_gray[idx]: %f, residual: %f, setting_huberTH_linearize: %f\n", idx, weights[idx],weights_gray[idx], residual, setting_huberTH_linearize);
    }
#ifdef USE_ZNCC_SEARCH
    hw *= weights_gray[idx] * weights_gray[idx];
#else
    hw *= weights[idx] * weights[idx];
#endif
    /// simply add em' up
    Hdd += (hw * d_idepth) * d_idepth; // 对逆深度的hessian
    bd += (hw * residual) * d_idepth;  // 对逆深度的Jres
    host_val.conservativeResize(valid_count + 1, 1);
    target_val.conservativeResize(valid_count + 1, 1);
    host_val(valid_count, 0) = (affLL[0] * color[idx] + affLL[1]);
    target_val(valid_count, 0) = hitColor[0];
    valid_count++;
  }
  zncc = CalcZncc(host_val, target_val);
  if (energyLeft > energyTH * outlierTHSlack || zncc < setting_outlierTH_zncc_angle_epi_linearize/*setting_outlierTH_zncc_tracker*/) {
    energyLeft = energyTH * outlierTHSlack;
    tmpRes->state_NewState = ResState::OUTLIER;
  } else {
    tmpRes->state_NewState = ResState::IN;
  }

  if (print_info) {
    printf("hw_mean: %f, energyLeft: %f, zncc: %f, energyTH: %f, outlierTHSlack: %f, setting_outlierTH_zncc_tracker: %f, setting_outlierTH_zncc_angle_epi_linearize: %f\n", hw_sum / hw_count, energyLeft, zncc, energyTH, outlierTHSlack, setting_outlierTH_zncc_tracker, setting_outlierTH_zncc_angle_epi_linearize);
  }
  tmpRes->state_NewEnergy = energyLeft;
  zncc_opt = zncc;
  hw_use[target_cid] = hw_sum / hw_count;
  return energyLeft;
}

} // namespace dso
