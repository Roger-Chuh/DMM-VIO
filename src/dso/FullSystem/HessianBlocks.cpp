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

#include "FullSystem/HessianBlocks.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/FrameShell.h"

namespace dso {

//@ 从ImmaturePoint构造函数, 不成熟点变地图点
// PointHessian::point_counter_ = 0;
// PointHessian::PointHessian(const ImmaturePoint *const rawPoint,
//                           CalibHessian *Hcalib) {
//  instanceCounter++;
//  host = rawPoint->host; // 主帧
//  hasDepthPrior = false;
//
//  idepth_hessian = 0;
//  maxRelBaseline = 0;
//  numGoodResiduals = 0;
//
//  // set static values & initialization.
//  u = rawPoint->u;
//  v = rawPoint->v;
//  assert(std::isfinite(rawPoint->idepth_max));
//  // idepth_init = rawPoint->idepth_GT;
//
//  my_type = rawPoint->my_type; //似乎是显示用的
//
//  setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) *
//                  0.5); //深度均值
//  setPointStatus(PointHessian::INACTIVE);
//
//  int n = patternNum;
//  memcpy(color, rawPoint->color, sizeof(float) * n); // 一个点对应8个像素
//  memcpy(weights, rawPoint->weights, sizeof(float) * n);
//  energyTH = rawPoint->energyTH;
//
//  efPoint = 0; // 指针=0
//}

PointHessian::PointHessian(const ImmaturePoint *const rawPoint,
                           CalibHessian *Hcalib, int &host_cid_) {
  host_cid = host_cid_;
  instanceCounter++;
  host = rawPoint->host; // 主帧
  hasDepthPrior = false;

  idepth_hessian = 0;
  maxRelBaseline = 0;
  numGoodResiduals = 0;

  // set static values & initialization.
  u = rawPoint->u;
  v = rawPoint->v;
  assert(std::isfinite(rawPoint->idepth_max));
  // idepth_init = rawPoint->idepth_GT;

  my_type = rawPoint->my_type; //似乎是显示用的

  setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) *
                  0.5); //深度均值
  setPointStatus(PointHessian::INACTIVE);

  int n = patternNum;
  memcpy(color, rawPoint->color_converged,
         sizeof(float) * n); // 一个点对应8个像素
  memcpy(weights, rawPoint->weights_converged, sizeof(float) * n);
  energyTH = rawPoint->energyTH_converged;

  efPoint = 0; // 指针=0
}

//@ 释放residual
void PointHessian::release() {
  for (unsigned int i = 0; i < residuals.size(); i++)
    delete residuals[i];
  residuals.clear();
}

//@ 设置固定线性化点位置的状态
// TODO 后面求nullspaces地方没看懂, 回头再看<2019.09.18> 数学原理是啥?
void FrameHessian::setStateZero(
    const VecState &state_zero) { //! 前六维位姿必须是0
  assert(state_zero.head<6>().squaredNorm() < 1e-20);

  this->state_zero = state_zero;

  //! 感觉这个nullspaces_pose就是 Adj_T
  //! Exp(Adj_T*zeta)=T*Exp(zeta)*T^{-1}
  // 全局转为局部的，左乘边右乘
  //! T_c_w * delta_T_g * T_c_w_inv = delta_T_l
  // TODO 这个是数值求导的方法么???
  for (int i = 0; i < 6;
       i++) { // TODO 一个整扰动，一个负扰动，然后把它变换到local系下, w.r.t.
    // pose
    Vec6 eps;
    eps.setZero();
    eps[i] = 1e-3;
    SE3 EepsP = Sophus::SE3::exp(eps);
    SE3 EepsM = Sophus::SE3::exp(-eps);
    SE3 w2c_leftEps_P_x0 =
        (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
    SE3 w2c_leftEps_M_x0 =
        (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
    nullspaces_pose.col(i) =
        (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
  }
  // nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
  // nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

  // scale change
  //? rethink
  // scale change
  // TODO 一个整扰动，一个负扰动，然后把它变换到local系下 w.r.t scale
  SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
  w2c_leftEps_P_x0.translation() *= 1.00001;
  w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
  SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
  w2c_leftEps_M_x0.translation() /= 1.00001;
  w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
  nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

  nullspaces_affine.setZero();
  assert(ab_exposure > 0);
  for (int cid = 0; cid < 1 /*kCameraNumUsed*/; ++cid) {
    // assert(ab_exposure_vec(cid) > 0);
    nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
    nullspaces_affine.topRightCorner<2, 1>() =
        Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
    //    nullspaces_affine.topRightCorner<2, 1>() =
    //        Vec2(0, expf(aff_g2l_0(cid).a) * ab_exposure_vec(cid));
    //    nullspaces_affine.block<2, 1>(0 + cid * 2, 0) = Vec2(1, 0);
    //    nullspaces_affine.block<2, 1>(0 + cid * 2, 1) =
    //        Vec2(0, expf(aff_g2l_0(cid).a) * ab_exposure_vec(cid));
  }
};

void FrameHessian::release() {
  // DELETE POINT
  // DELETE RESIDUAL
  for (unsigned int i = 0; i < pointHessians.size(); i++)
    delete pointHessians[i];
  for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
    delete pointHessiansMarginalized[i];
  for (unsigned int i = 0; i < pointHessiansOut.size(); i++)
    delete pointHessiansOut[i];
  for (unsigned int i = 0; i < immaturePoints.size(); i++)
    delete immaturePoints[i];

  pointHessians.clear();
  pointHessiansMarginalized.clear();
  pointHessiansOut.clear();
  immaturePoints.clear();
}

//* 计算各层金字塔图像的像素值和梯度
void FrameHessian::makeImages(float *color, CalibHessian *HCalib) {
  // 每一层创建图像值, 和图像梯度的存储空间
  for (int i = 0; i < pyrLevelsUsed; i++) {
    ///* 图像导数[0]:辐照度  [1]:x方向导数  [2]:y方向导数, （指针表示图像）
    dIp[i] = new Eigen::Vector3f
        [wG[i] * hG[i] * kCameraNumUsed]; // TODO image size at each pyr level
    absSquaredGrad[i] = new float[wG[i] * hG[i] * kCameraNumUsed];
    edge_label_image[i] = new Eigen::Vector2i[wG[i] * hG[i] * kCameraNumUsed];
    dt_dx_dy[i] = new Eigen::Vector3f[wG[i] * hG[i] * kCameraNumUsed];
    label2xy[i] = new Eigen::Vector2i[wG[i] * hG[i] * kCameraNumUsed];
    edge_pixels[i] = new Eigen::Vector2i[wG[i] * hG[i] * kCameraNumUsed];
    for (int cid = 0; cid < kCameraNumUsed; ++cid) {
      label_num[i][cid] = 0;
      edge_pixel_num[i][cid] = 0;
    }
  }
  //  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //      dI[cid * kCameraNumUsed] = dIp[0]; // TODO assign pointer //
  //      原来他们指向同一个地方
  //  }
  dI = dIp[0]; // TODO assign pointer // 原来他们指向同一个地方
  // make d0
  int w = wG[0]; // 零层weight
  int h = hG[0]; // 零层height
  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
    for (int i = 0; i < w * h; i++) {
      /// here assign the color image inside dI. this color only takes one
      /// dimension, grey scale?
      ///       // note that dI is wG*hG*3. this dI[i][0] only assigned color to
      ///       the first dimension.
      /// overwrite第一个channel的数据
      dI[i + w * h * cid][0] = color[i + w * h * cid];
    }

    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
      int wl = wG[lvl], hl = hG[lvl]; // 该层图像大小
      Eigen::Vector3f *dI_l = dIp[lvl] + wl * hl * cid;

      float *dabs_l = absSquaredGrad[lvl] + wl * hl * cid;
      if (lvl > 0) {
        int lvlm1 = lvl - 1;
        int wlm1 = wG[lvlm1]; // 列数
        int hlm1 = hG[lvlm1]; // 列数
        Eigen::Vector3f *dI_lm = dIp[lvlm1] + wlm1 * hlm1 * cid;

        // 像素4合1, 生成金字塔
        // row major
        for (int y = 0; y < hl; y++)
          for (int x = 0; x < wl; x++) {
            dI_l[x + y * wl][0] =
                0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                         dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                         dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                         dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1]
                              [0]); // TODO filter image noise?[scratch that],
            // generate pyramid
          }
      }
      std::vector<uint8_t> image_data(wl * hl);
      for (int i = 0; i < wl * hl; ++i) {
        if (dI_l[i][0] + 0.6 >= 255) {
          image_data[i] = 255;
        } else if (dI_l[i][0] - 0.6 <= 0) {
          image_data[i] = 0;
        } else {
          image_data[i] = static_cast<uint8_t>(dI_l[i][0]);
        }
      }
      cv::Mat cv_img = cv::Mat(hl, wl, CV_8UC1, image_data.data()).clone();
      cv::Mat output, edge;
      float threshold;
      if (adaptiveCannyThreshold) {
        threshold = cv::threshold(cv_img, output, 0, 255, cv::THRESH_OTSU);
        printf("canny_threshold: %f\n", threshold);
        cv::Canny(cv_img, edge, std::max(3, (int)threshold - 130),
                  std::min(245, (int)threshold - 80), 3, true);
      } else {
        cv::Canny(cv_img, edge, cannyThreshold1, cannyThreshold2, 3, true);
      }
      cv::Mat inverted = 255 - edge;
      cv::Mat labels = cv::Mat::zeros(edge.size(), CV_32SC1);
      cv::Mat distanceTransformMap;
      // inverted 中的 0 表示 edge 像素
      cv::distanceTransform(inverted, distanceTransformMap, labels, cv::DIST_L2,
                            cv::DIST_MASK_PRECISE, cv::DIST_LABEL_PIXEL);

      // cv::imwrite("img_small.png", cv_img);
      // cv::imwrite("edge.png", edge);
      // cv::imwrite("dt.tiff", distanceTransformMap);
      // cv::imwrite("labels.tiff", labels);
      // std::exit(1);
      int labelNum = cv::countNonZero(edge);
      edge_pixel_num[lvl][cid] = labelNum;
      label_num[lvl][cid] = labelNum;
      Eigen::Vector2i *label2xy_start = label2xy[lvl] + wl * hl * cid;
      Eigen::Vector2i *edge_pixels_start = edge_pixels[lvl] + wl * hl * cid;
      Eigen::Vector2i *edge_label_image_start =
          edge_label_image[lvl] + wl * hl * cid;
      Eigen::Vector3f *dt_dx_dy_start = dt_dx_dy[lvl] + wl * hl * cid;

      int labelNumCheck = 0;
      max_dt_dx_dy[lvl][cid] = -99999 * Vec3f::Ones();
      min_dt_dx_dy[lvl][cid] = 99999 * Vec3f::Ones();
      for (int r = 0; r < hl; ++r) {
        for (int c = 0; c < wl; ++c) {
          if (labels.at<int>(r, c) < 1) {
            std::cerr << "lable < 1, sth wrong" << std::endl;
            std::exit(1);
          }
          edge_label_image_start[c + r * wl] = Eigen::Vector2i(
              (int)edge.at<uchar>(r, c), (int)labels.at<int>(r, c) - 1);
          float dist = (float)distanceTransformMap.at<float>(r, c);
          dt_dx_dy_start[c + r * wl][0] = dist;
          if (dist > max_dt_dx_dy[lvl][cid][0]) {
            max_dt_dx_dy[lvl][cid][0] = dist;
          }
          if (dist < min_dt_dx_dy[lvl][cid][0]) {
            min_dt_dx_dy[lvl][cid][0] = dist;
          }
          if (edge.at<uchar>(r, c) > 0) {
            label2xy_start[labels.at<int>(r, c) - 1] = Eigen::Vector2i(c, r);
            edge_pixels_start[labels.at<int>(r, c) - 1] = Eigen::Vector2i(c, r);
            labelNumCheck++;
          }
        }
      }
      if (labelNumCheck != labelNum) {
        std::cerr << "labelNumCheck != labelNum, sth wrong" << std::endl;
        std::exit(1);
      }
      for (int c = 1; c < wl - 1; ++c) {   // 第二行开始
        for (int r = 1; r < hl - 1; ++r) { // 第二行开始
          int idx = c + r * wl;
          // for (int idx = wl; idx < wl * (hl - 1); idx++) {// 第二行开始
          float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
          float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

          float dx_dt =
              0.5f * (dt_dx_dy_start[idx + 1][0] - dt_dx_dy_start[idx - 1][0]);
          float dy_dt = 0.5f * (dt_dx_dy_start[idx + wl][0] -
                                dt_dx_dy_start[idx - wl][0]);

          if (!std::isfinite(dx))
            dx = 0;
          if (!std::isfinite(dy))
            dy = 0;
          if (!std::isfinite(dx_dt))
            dx_dt = 0;
          if (!std::isfinite(dy_dt))
            dy_dt = 0;

          dI_l[idx][1] = dx; // 梯度
          dI_l[idx][2] = dy;
          dt_dx_dy_start[idx][1] = dx_dt;
          dt_dx_dy_start[idx][2] = dy_dt;

          if (dx_dt > max_dt_dx_dy[lvl][cid][1]) {
            max_dt_dx_dy[lvl][cid][1] = dx_dt;
          }
          if (dx_dt < min_dt_dx_dy[lvl][cid][1]) {
            min_dt_dx_dy[lvl][cid][1] = dx_dt;
          }

          if (dy_dt > max_dt_dx_dy[lvl][cid][2]) {
            max_dt_dx_dy[lvl][cid][2] = dy_dt;
          }
          if (dy_dt < min_dt_dx_dy[lvl][cid][2]) {
            min_dt_dx_dy[lvl][cid][2] = dy_dt;
          }

          dabs_l[idx] = dx * dx + dy * dy; // 梯度平方

          if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0) {
            //! 乘上响应函数, 变换回正常的颜色, 因为光度矫正时 I = G^-1(I) /
            //! V(x)
            float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
            dabs_l[idx] *=
                gw *
                gw; // TODO convert to gradient of original color space (before
            // removing response, i.e. before compensate affine param a b).
          }
          // }
        }
      }
    }
  }
}

//@ 计算优化前和优化后的相对位姿, 相对光度变化, 及中间变量
void FrameFramePrecalc::set(FrameHessian *host, FrameHessian *target,
                            CalibHessian *HCalib) {
  this->host = host; // 这个是赋值, 计数会增加, 不是拷贝
  this->target = target;
  if (host->frameID == target->frameID) {
    printf(" host and target has the same frameID\n");
  }
  //? 实在不懂leftToleft_0这个名字怎么个含义
  // 优化前host target间位姿变换
  // TODO also known as "Tth"
  SE3 leftToLeft_0 =
      target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
  PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
  PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();
  // std::cout<<"PRE_tTll_0: "<<PRE_tTll_0.transpose()<<std::endl;

  // 优化后host到target间位姿变换
  SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
  PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
  PRE_tTll = (leftToLeft.translation()).cast<float>();
  distanceLL = leftToLeft.translation().norm();

  // 乘上内参, 中间量?
  Mat33f K = Mat33f::Zero();
  K(0, 0) = HCalib->fxl();
  K(1, 1) = HCalib->fyl();
  K(0, 2) = HCalib->cxl();
  K(1, 2) = HCalib->cyl();
  K(2, 2) = 1;
  PRE_KRKiTll = K * PRE_RTll * K.inverse();
  PRE_RKiTll = PRE_RTll * K.inverse();
  PRE_KtTll = K * PRE_tTll;
  for (int host_cid = 0; host_cid < kCameraNumUsed; ++host_cid) {
    for (int target_cid = 0; target_cid < kCameraNumUsed; ++target_cid) {
      SE3 Tcjci = target->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
                  leftToLeft * host->p_multi_camera->cid_to_T01_SE3[host_cid];
      SE3 Tcjci_0 =
          target->p_multi_camera->cid_to_T01_SE3[target_cid].inverse() *
          leftToLeft_0 * host->p_multi_camera->cid_to_T01_SE3[host_cid];
      a_PRE_RTll[host_cid * kCameraNumUsed + target_cid] =
          Tcjci.rotationMatrix().cast<float>();
      a_PRE_tTll[host_cid * kCameraNumUsed + target_cid] =
          Tcjci.translation().cast<float>();
      a_PRE_RTll_0[host_cid * kCameraNumUsed + target_cid] =
          Tcjci_0.rotationMatrix().cast<float>();
      a_PRE_tTll_0[host_cid * kCameraNumUsed + target_cid] =
          Tcjci_0.translation().cast<float>();
      // std::cout << "Tcjci:\n" << Tcjci.matrix3x4() << std::endl;
      a_PRE_KRKiTll[host_cid * kCameraNumUsed + target_cid] =
          K * Tcjci.rotationMatrix().cast<float>() * K.inverse();
      a_PRE_KtTll[host_cid * kCameraNumUsed + target_cid] =
          K * Tcjci.translation().cast<float>();
    }
  }

  // 光度仿射值
  // TODO 这是两帧相对的a和b，
  //  for (int cid = 0; cid < kCameraNumUsed; ++cid) {
  //    a_PRE_aff_mode[cid] =
  //        AffLight::fromToVecExposure(host->ab_exposure_vec[cid],
  //                                    target->ab_exposure_vec[cid],
  //                                    host->aff_g2l(cid),
  //                                    target->aff_g2l(cid))
  //            .cast<float>();
  //
  //    // TODO 这是host帧绝对的b
  //    PRE_b0_mode_vec[cid] =
  //        host->aff_g2l_0(cid).b; // TODO host帧的相对于第一帧的绝对的b
  //  }
  // 光度仿射值
  // TODO 这是两帧相对的a和b，
  PRE_aff_mode =
      AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure,
                                  host->aff_g2l(), target->aff_g2l())
          .cast<float>();
  // TODO 这是host帧绝对的b
  PRE_b0_mode = host->aff_g2l_0().b; // TODO host帧的相对于第一帧的绝对的b
}

} // namespace dso
