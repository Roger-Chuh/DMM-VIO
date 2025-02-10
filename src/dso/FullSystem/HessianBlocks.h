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

#pragma once
#define MAX_ACTIVE_FRAMES 100

#include "util/globalCalib.h"
#include "vector"

#include "FullSystem/Residuals.h"
#include "util/ImageAndExposure.h"
#include "util/NumType.h"
#include <fstream>
#include <iostream>

namespace dso {
class MultiCamera;
//* 求得两个参考帧之间的光度仿射变换系数
// 设from是 i->j(ref->tar);  to是 k->j; 则结果是 i->k 的变换系数.
inline Vec2 affFromTo(const Vec2 &from,
                      const Vec2 &to) // contains affine parameters as XtoWorld.
{
  return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}

//提前声明下
struct FrameHessian;
struct PointHessian;

class ImmaturePoint;

class FrameShell;

class EFFrame;

class EFPoint;
//? 这是干什么用的? 是为了求解时候的数值稳定?
#define SCALE_IDEPTH 1.0f // scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
// #define SCALE_XI_TRANS 0.5f
#define SCALE_XI_TRANS 1.0f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f
// TODO 数值稳定性，尽量把数量级拉到一样
//上面的逆
#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

// TODO* 其中带0的是FEJ用的初始状态, 不带0的是更新的状态
struct FrameFramePrecalc {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // static values
  static int instanceCounter;
  FrameHessian *host;   // defines row
  FrameHessian *target; // defines column

  std::array<FrameHessian *, kCameraNumUsed> a_host;   // defines row
  std::array<FrameHessian *, kCameraNumUsed> a_target; // defines column

  // precalc values
  Mat33f PRE_RTll; // host 到 target 之间优化后旋转矩阵 R
  std::array<Mat33f, kCameraNumUsed * kCameraNumUsed>
      a_PRE_RTll;     // host 到 target 之间优化后旋转矩阵 R
  Mat33f PRE_KRKiTll; // k*R*k_inv
  Mat33f PRE_RKiTll;  // R*k_inv
  Mat33f PRE_RTll_0; // host 到 target之间初始的旋转矩阵, 优化更新前
  std::array<Mat33f, kCameraNumUsed * kCameraNumUsed>
      a_PRE_RTll_0; // host 到 target之间初始的旋转矩阵, 优化更新前

  Vec2f PRE_aff_mode; // 能量函数对仿射系数处理后的, 总系数
  float PRE_b0_mode;  // host的光度仿射系数b

  std::array<Vec2f, kCameraNumUsed> a_PRE_aff_mode;
  Vec4f PRE_b0_mode_vec;

  Vec3f PRE_tTll; //  host 到 target之间优化后的平移 t
  std::array<Vec3f, kCameraNumUsed * kCameraNumUsed>
      a_PRE_tTll;   //  host 到 target之间优化后的平移 t
  Vec3f PRE_KtTll;  // K*t
  Vec3f PRE_tTll_0; //  host 到 target之间初始的平移, 优化更新前
  std::array<Vec3f, kCameraNumUsed * kCameraNumUsed>
      a_PRE_tTll_0; //  host 到 target之间初始的平移, 优化更新前

  std::array<Mat33f, kCameraNumUsed * kCameraNumUsed> a_PRE_KRKiTll;
  std::array<Vec3f, kCameraNumUsed * kCameraNumUsed> a_PRE_KtTll;

  float distanceLL; // 两帧间距离

  inline ~FrameFramePrecalc() {}

  inline FrameFramePrecalc() { host = target = 0; }

  void set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib);
  //  void set(std::array<FrameHessian *, kCameraNumUsed> a_host,
  //           std::array<FrameHessian *, kCameraNumUsed> a_target,
  //           std::array<CalibHessian *, kCameraNumUsed> a_HCalib);
};

//* 相机位姿+相机光度Hessian
struct FrameHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EFFrame *efFrame; //!< 帧的能量函数

  // constant info & pre-calculated values
  // DepthImageWrap* frame;
  FrameShell *shell; //!< 帧的"壳", 保存一些不变的,要留下来的量
  //* 图像导数[0]:辐照度  [1]:x方向导数  [2]:y方向导数, （指针表示图像）
  Eigen::Vector3f *dI; //[PYR_LEVELS * kCameraNumUsed]; //!< 图像导数 // trace,
                       // fine tracking. Used for direction
  //!< select (not for gradient histograms etc.)
  Eigen::Vector3f
      *dIp[PYR_LEVELS]; // * kCameraNumUsed]; //!< 各金字塔层的图像导数   //
                        // coarse tracking / coarse
  //!< initializer. NAN in [0] only.
  float *absSquaredGrad[PYR_LEVELS]; // * kCameraNumUsed]; //!< x,y
                                     // 方向梯度的平方和  // only used for
  //!< pixel select (histograms etc.). no NAN.

  bool addCamPrior;

  //* 都是ID
  int frameID; //!< 所有关键帧的序号(FrameShell)	// incremental ID for
  //!< keyframes only!
  static int instanceCounter; //!< 计数器
  int idx;                    //!< 激活关键帧的序号(FrameHessian)

  // Photometric Calibration Stuff
  float frameEnergyTH; //!< 阈值 //// set dynamically depending on tracking
  //!< residual
  float ab_exposure;
  // VecCamNum ab_exposure_vec;

  bool flaggedForMarginalization;

  MultiCamera *p_multi_camera;
  std::vector<PointHessian *> pointHessians; // contains all ACTIVE points.
  std::vector<PointHessian *>
      pointHessiansMarginalized; // contains all MARGINALIZED points (= fully
  // marginalized, usually because point went
  // OOB.)
  std::array<std::vector<PointHessian *>, kCameraNumUsed> cid_to_pointHessians;
  std::vector<PointHessian *>
      pointHessiansOut; // contains all OUTLIER points (= discarded.).
  std::array<std::vector<PointHessian *>, kCameraNumUsed>
      cid_to_pointHessiansOut;
  std::vector<ImmaturePoint *>
      immaturePoints; // contains all OUTLIER points (= discarded.).
  std::array<std::vector<ImmaturePoint *>, kCameraNumUsed>
      cid_to_immaturePoints;

  //* 零空间, 好奇怎么求???
  Mat66 nullspaces_pose;
  MatStateAffine2 nullspaces_affine;
  Vec6 nullspaces_scale;

  // variable info.
  SE3 worldToCam_evalPT; //!< 在估计的相机位姿, //TODO 这是线性化点
                         // todo roger, apply extrinsic

  std::array<SE3, kCameraNumUsed> cid_to_worldToCam_evalPT;
  // [0-5: 位姿左乘小量. 6-7: a,b 光度仿射系数]
  // TODO* 这三个是与线性化点的增量(相对于固定线性化点的增量),
  // 而光度参数不是增量, state就是值 !!!
  VecState state_zero; //!< 固定的线性化点的状态增量, 为了计算进行缩放, //TODO

  //!< 是世界系下绝对pose Tcw的增量
  VecState state_scaled; //!< 乘上比例系数的状态增量, 这个是真正求的值!!!
  VecState state; //!< 计算的状态增量// [0-5: worldToCam-leftEps. 6-7: a,b]
  //* step是与上一次优化结果的状态增量, [8 ,9]直接就设置为0了
  VecState step;         //!< 求解正规方程得到的增量
  VecState step_backup;  //!< 上一次的增量备份
  VecState state_backup; //!< 上一次状态的备份

  //内联提高效率, 返回上面的值
  EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const {
    return worldToCam_evalPT;
  }

  EIGEN_STRONG_INLINE const VecState &get_state_zero() const {
    return state_zero;
  } // the first 6 parameters of state_zero seem to be always 0 (as this part
  // ist represented by the worldToCam_evalPT. The last two parameters on the
  // other hand are not zero.
  EIGEN_STRONG_INLINE const VecState &get_state() const { return state; }

  EIGEN_STRONG_INLINE const VecState &get_state_scaled() const {
    return state_scaled;
  }

  EIGEN_STRONG_INLINE const VecState get_state_minus_stateZero() const {
    return get_state() - get_state_zero();
  } // TODO x小量可以直接减 lie algebra

  // precalc values
  SE3 PRE_worldToCam; //!< 预计算的, 位姿状态增量更新到位姿上
  SE3 PRE_camToWorld;
  std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>>
      targetPrecalc;          //!< 对于其它帧的预运算值
  MinimalImageB3 *debugImage; //!< 小图???

  inline Vec6 w2c_leftEps() const {
    return get_state_scaled().head<6>();
  } //* 返回位姿状态增量
  inline AffLight aff_g2l() const {
    return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
  } //* 返回光度仿射系数
  inline AffLight aff_g2l_0() const {
    return AffLight(get_state_zero()[6] * SCALE_A,
                    get_state_zero()[7] * SCALE_B);
  } //* 返回线性化点处的仿射系数增量
    //  inline AffLight aff_g2l(int cid) const {
    //    return AffLight(get_state_scaled()[6 + 2 * cid],
    //                    get_state_scaled()[7 + 2 * cid]);
    //  } //* 返回光度仿射系数
    //  inline AffLight aff_g2l_0(int cid) const {
    //    return AffLight(get_state_zero()[6 + 2 * cid] * SCALE_A,
    //                    get_state_zero()[7 + 2 * cid] * SCALE_B);
    //  } //* 返回线性化点处的仿射系数增量

  //* 设置FEJ点状态增量
  void setStateZero(const VecState &state_zero);

  //* 设置增量, 同时复制state和state_scale
  inline void setState(const VecState &state) {

    this->state = state;
    state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
    state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
    for (int cid = 0; cid < 1 /*kCameraNumUsed**/; ++cid) {
      state_scaled[6 + 2 * cid] = SCALE_A * state[6 + 2 * cid];
      state_scaled[7 + 2 * cid] = SCALE_B * state[7 + 2 * cid];
    }
    //    state_scaled[8] = SCALE_A * state[8];
    //    state_scaled[9] = SCALE_B * state[9];
    //位姿更新
    PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
    PRE_camToWorld = PRE_worldToCam.inverse();
    // setCurrentNullspace();
  };

  //* 设置增量, 传入state_scaled
  inline void setStateScaled(const VecState &state_scaled) {

    this->state_scaled = state_scaled;
    state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
    state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
    for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
      state[6 + 2 * cid] = SCALE_A_INVERSE * state_scaled[6 + 2 * cid];
      state[7 + 2 * cid] = SCALE_B_INVERSE * state_scaled[7 + 2 * cid];
    }
    //    state[8] = SCALE_A_INVERSE * state_scaled[8];
    //    state[9] = SCALE_B_INVERSE * state_scaled[9];

    PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
    PRE_camToWorld = PRE_worldToCam.inverse();
    // setCurrentNullspace();
  };

  //* 设置当前位姿, 和状态增量, 同时设置了FEJ点
  inline void setEvalPT(const SE3 &worldToCam_evalPT, const VecState &state) {
    // std::cout<<"worldToCam_evalPT:
    // \n"<<worldToCam_evalPT.matrix3x4()<<std::endl;
    this->worldToCam_evalPT = worldToCam_evalPT;
    setState(state);
    setStateZero(state);
  };

  //* 设置当前位姿, 光度仿射系数, FEJ点
  inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT,
                               const AffLight &aff_g2l) {
    VecState initial_state = VecState::Zero();
    initial_state[6] = aff_g2l.a; // 直接设置光度系数a和b
    initial_state[7] = aff_g2l.b;
    this->worldToCam_evalPT = worldToCam_evalPT;
    setStateScaled(initial_state); // TODO pose增量必须是0
    setStateZero(this->get_state());
  };
  //  //* 设置当前位姿, 光度仿射系数, FEJ点
  //  inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT,
  //                               const std::array<AffLight, kCameraNumUsed>
  //                               &cid_to_aff_g2l) {
  //    VecState initial_state = VecState::Zero();
  //    for (int cid = 0; cid < 1/*kCameraNumUsed*/; ++cid) {
  //      initial_state[6 + 2 * cid] = cid_to_aff_g2l[cid].a; //
  //      直接设置光度系数a和b initial_state[7 + 2 * cid] =
  //      cid_to_aff_g2l[cid].b;
  //    }
  //    this->worldToCam_evalPT = worldToCam_evalPT;
  //    setStateScaled(initial_state); // TODO pose增量必须是0
  //    setStateZero(this->get_state());
  //  };

  //* 释放该帧内存
  void release();

  inline ~FrameHessian() {
    assert(efFrame == 0);
    release();
    instanceCounter--;
    for (int i = 0; i < pyrLevelsUsed; i++) {
      delete[] dIp[i];
      delete[] absSquaredGrad[i];
    }

    if (debugImage != 0)
      delete debugImage;
  };

  inline FrameHessian(MultiCamera *p_multi_camera_)
      : p_multi_camera(p_multi_camera_) {
    instanceCounter++; //! 若是发生拷贝, 就不会增加了
    flaggedForMarginalization = false;
    frameID = -1;
    efFrame = 0;
    frameEnergyTH = 8 * 8 * patternNum;

    debugImage = 0;

    addCamPrior = false;
  };

  void makeImages(float *color, CalibHessian *HCalib);

  //* 获得先验信息矩阵， 怎么感觉除了第一帧没什么用
  inline VecState getPrior() // TODO 得到帧位姿的先验
  // 第0帧，pose和ab都有先验，非0帧只有ab有先验，pose没有
  {
    VecState p = VecState::Zero();
    if (frameID == 0) //* 第一帧就用初始值做先验
    {
      p.head<3>() = Vec3::Constant(setting_initialTransPrior);
      p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
      // 用位运算, 有点东西
      if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
        p.head<6>().setZero();
      for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
        p[6 + cid * 2] = setting_initialAffAPrior; // 1e14
        p[7 + cid * 2] = setting_initialAffBPrior; // 1e14
      }
    } else //* 否则根据模式决定
    {
      for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
        if (setting_affineOptModeA < 0) //* 小于零是固定的不优化
          p[6 + cid * 2] = setting_initialAffAPrior;
        else
          p[6 + cid * 2] = setting_affineOptModeA; // 1e12

        if (setting_affineOptModeB < 0)
          p[7 + cid * 2] = setting_initialAffBPrior;
        else
          p[7 + cid * 2] = setting_affineOptModeB; // 1e8
      }
    }
    //? 8,9是干嘛的呢???  没用....
    // p[8] = setting_initialAffAPrior;
    // p[9] = setting_initialAffBPrior;

    if (addCamPrior) {
      p.head<3>() = Vec3::Constant(setting_initialTransPrior);
      p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
      if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
        p.head<6>().setZero();
    }

    return p;
  }

  inline VecState getPriorZero() { return VecState::Zero(); }
};

//* 相机内参Hessian, 响应函数
struct CalibHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static int instanceCounter;
  // * 4×1的向量
  VecC value_zero;     //!< FEJ固定点
  VecC value_scaled;   //!< 乘以scale的内参
  VecCf value_scaledf; //!< float型的内参
  VecCf value_scaledi; //!< 逆, 应该是求导用为, 1/fx, 1/fy, -cx/fx, -cy/fy
  VecC value;          //!< 没乘scale的
  VecC step;           //!< 迭代中的增量
  VecC step_backup;    //!< 上一次增量备份
  VecC value_backup;   //!< 上一次值的备份
  VecC value_minus_value_zero; //!< 减去线性化点

  inline ~CalibHessian() { instanceCounter--; }

  inline CalibHessian() {

    VecC initial_value = VecC::Zero();
    initial_value[0] = fxG[0];
    initial_value[1] = fyG[0];
    initial_value[2] = cxG[0];
    initial_value[3] = cyG[0];

    setValueScaled(initial_value);
    value_zero = value;
    value_minus_value_zero.setZero();

    intr.setIdentity();
    intr(0, 0) = fxG[0];
    intr(1, 1) = fyG[0];
    intr(0, 2) = cxG[0];
    intr(1, 2) = cyG[0];

    intr_inv = intr.inverse();

    instanceCounter++;
    //响应函数
    for (int i = 0; i < 256; i++)
      Binv[i] = B[i] = i; // set gamma function to identity
  };

  inline void setMultiCamera(MultiCamera *p_multi_camera_) {
    p_multi_camera = p_multi_camera_;
  }

  // normal mode: use the optimized parameters everywhere!
  inline float &fxl() { return value_scaledf[0]; }

  inline float &fyl() { return value_scaledf[1]; }

  inline float &cxl() { return value_scaledf[2]; }

  inline float &cyl() { return value_scaledf[3]; }

  inline float &fxli() { return value_scaledi[0]; }

  inline float &fyli() { return value_scaledi[1]; }

  inline float &cxli() { return value_scaledi[2]; }

  inline float &cyli() { return value_scaledi[3]; }

  //* 通过value设置
  inline void setValue(const VecC &value) {
    // [0-3: Kl, 4-7: Kr, 8-12: l2r] what's this, stereo camera???
    this->value = value;
    value_scaled[0] = SCALE_F * value[0];
    value_scaled[1] = SCALE_F * value[1];
    value_scaled[2] = SCALE_C * value[2];
    value_scaled[3] = SCALE_C * value[3];

    this->value_scaledf = this->value_scaled.cast<float>();
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
    //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    //      value_scaled[0 + 4 * cid] = SCALE_F * value[0 + 4 * cid];
    //      value_scaled[1 + 4 * cid] = SCALE_F * value[1 + 4 * cid];
    //      value_scaled[2 + 4 * cid] = SCALE_C * value[2 + 4 * cid];
    //      value_scaled[3 + 4 * cid] = SCALE_C * value[3 + 4 * cid];
    //    }
    //
    //    this->value_scaledf = this->value_scaled.cast<float>();
    //    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    //      this->value_scaledi[0 + 4 * cid] =
    //          1.0f / this->value_scaledf[0 + 4 * cid];
    //      this->value_scaledi[1 + 4 * cid] =
    //          1.0f / this->value_scaledf[1 + 4 * cid];
    //      this->value_scaledi[2 + 4 * cid] =
    //          -this->value_scaledf[2 + 4 * cid] / this->value_scaledf[0 + 4 *
    //          cid];
    //      this->value_scaledi[3 + 4 * cid] =
    //          -this->value_scaledf[3 + 4 * cid] / this->value_scaledf[1 + 4 *
    //          cid];
    //    }
    this->value_minus_value_zero = this->value - this->value_zero;
  };

  //* 通过value_scaled赋值
  inline void setValueScaled(const VecC &value_scaled) {
    this->value_scaled = value_scaled;
    this->value_scaledf = this->value_scaled.cast<float>();
    value[0] = SCALE_F_INVERSE * value_scaled[0];
    value[1] = SCALE_F_INVERSE * value_scaled[1];
    value[2] = SCALE_C_INVERSE * value_scaled[2];
    value[3] = SCALE_C_INVERSE * value_scaled[3];

    this->value_minus_value_zero = this->value - this->value_zero;
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
  };

  //* gamma函数, 相机的响应函数G和G^-1, 映射到0~255
  float Binv[256];
  float B[256];

  //* 响应函数的导数
  EIGEN_STRONG_INLINE float getBGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5)
      c = 5;
    if (c > 250)
      c = 250;
    return B[c + 1] - B[c];
  }
  //* 响应函数逆的导数
  EIGEN_STRONG_INLINE float getBInvGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5)
      c = 5;
    if (c > 250)
      c = 250;
    return Binv[c + 1] - Binv[c];
  }
  MultiCamera *p_multi_camera;
  Mat3 intr;
  Mat3 intr_inv;
};

//* 点Hessian
// hessian component associated with one point.
struct PointHessian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static int instanceCounter;
  EFPoint *efPoint; //!< 点的能量函数

  // static values
  float color[MAX_RES_PER_POINT]; // * kCameraNumUsed]; // colors in host frame
  float weights[MAX_RES_PER_POINT]; // * kCameraNumUsed]; // host-weights for
                                    // respective residuals.

  float u, v; //!< 像素点的位置
  int idx;
  float energyTH;     //!< 光度误差阈值
  FrameHessian *host; //!< 主帧
  bool hasDepthPrior; //!< 初始化得到的点是有深度先验的, 其它没有

  float my_type; //不同类型点, 显示用

  float idepth_scaled;      //!< target还是host上点逆深度 ??
  float idepth_zero_scaled; //!< FEJ使用, 点在host上x=0初始逆深度
  float idepth_zero;        //!< 缩放了scale倍的固定线性化点逆深度
  float idepth;             //!< 缩放scale倍的逆深度
  float idepth_before = 0.f; //!< 缩放scale倍的逆深度
  float step;                //!< 迭代优化每一步增量
  float step_backup;         //!< 迭代优化上一步增量的备份
  float idepth_backup;       //!< 上一次的逆深度值

  float nullspaces_scale; //!< 零空间 ?
  float idepth_hessian;   //!< 对应的hessian矩阵值
  float maxRelBaseline;   //!< 衡量该点的最大基线长度
  int numGoodResiduals;

  int host_cid;

  enum PtStatus {
    ACTIVE = 0,
    INACTIVE,
    OUTLIER,
    OOB,
    MARGINALIZED
  }; // 这些状态都没啥用.....
  PtStatus status;

  inline void setPointStatus(PtStatus s) { status = s; }

  inline void setIdepth(float idepth) {
    this->idepth = idepth;
    this->idepth_scaled = SCALE_IDEPTH * idepth;
  }

  inline void setIdepthScaled(float idepth_scaled) {
    this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
    this->idepth_scaled = idepth_scaled;
  }

  inline void setIdepthZero(float idepth) { // TODO
    // 调完setIdepthZero会立马调setIdepthScaled，所以idp的线性化点是一直在变的，等于没固定线性化点,
    // idp的状态是在一直更新的，不是固定的
    idepth_zero = idepth;
    idepth_zero_scaled = SCALE_IDEPTH * idepth;
    nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500; //? 为啥这么求
  }

  std::vector<PointFrameResidual *>
      residuals; // only contains good residuals (not OOB and not OUTLIER).
  // Arbitrary order.
  //  std::array<std::array<std::pair<PointFrameResidual *, ResState>, 2>,
  //             kCameraNumUsed>
  //      lastResiduals; //[2]; // contains information about residuals to the
  //      last
  std::array<
      std::pair<PointFrameResidual *, std::array<ResState, kCameraNumUsed>>, 2>
      lastResiduals; //[2]; // contains information about residuals to the last
                     // two
  // (!) frames. ([0] = latest, [1] = the one before).

  void release();

  //  PointHessian(const ImmaturePoint *const rawPoint, CalibHessian *Hcalib);

  PointHessian(const ImmaturePoint *const rawPoint, CalibHessian *Hcalibconst,
               int &host_cid_);

  inline ~PointHessian() {
    assert(efPoint == 0);
    release();
    instanceCounter--;
  }

  //@ 判断其它帧上的点是否不值得要了
  inline bool isOOB(const std::vector<FrameHessian *> &toKeep,
                    const std::vector<FrameHessian *> &toMarg) const {
    int visInToMarg = 0;
    for (PointFrameResidual *r : residuals) {
      int in_count = kCameraNumUsed;
      for (int cid = 0; cid < kCameraNumUsed; ++cid) {
        if (r->state_state[cid] != ResState::IN) {
          in_count--;
          // continue;
        }
      }
      if (in_count == 0) {
        continue;
      }
      // TODO roger, only need good residuals
      for (FrameHessian *k : toMarg) {
        if (r->target == k) {
          visInToMarg++; // 在要边缘化掉的帧被观测的数量
        }
      }
    }
    //[1]: 原本是很好的一个点，但是边缘化一帧后，残差变太少了, 边缘化or丢掉
    if ((int)residuals.size() >=
            setting_minGoodActiveResForMarg && // 残差数大于一定数目
        numGoodResiduals > setting_minGoodResForMarg + 10 &&
        (int)residuals.size() - visInToMarg <
            setting_minGoodActiveResForMarg) //剩余残差足够少
      return true;

    //[2]: 最新一帧的投影在图像外了, 看不见了, 边缘化or丢掉
    // 或者满足以下条件,
    int oob_count = 0;
    int outlier_count = 0;
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      if (lastResiduals[0].second[cid] == ResState::OOB) {
        oob_count++;
      }
      if (lastResiduals[0].second[cid] == ResState::OUTLIER &&
          lastResiduals[1].second[cid] == ResState::OUTLIER) {
        outlier_count++;
      }
    }
    if (oob_count ==
        kCameraNumUsed /*lastResiduals[0].second == ResState::OOB*/)
      return true; //上一帧是OOB
    //[3]: 残差比较少, 新加入的, 不边缘化
    if (residuals.size() < 2
#ifndef USE_BUNDLED_RES
                               * kCameraNumUsed
#endif
    ) {
      return false;
    }
    //[4]: 前两帧投影都是外点, 边缘化or丢掉
    if (outlier_count == kCameraNumUsed/*lastResiduals[0].second == ResState::OUTLIER &&
        lastResiduals[1].second == ResState::OUTLIER*/) {
      return true; //前两帧都是外点
    }
    return false;
  }

  //内点条件
  inline bool isInlierNew() {
    return (int)residuals.size() >= setting_minGoodActiveResForMarg &&
           numGoodResiduals >= setting_minGoodResForMarg;
  }
};

} // namespace dso
