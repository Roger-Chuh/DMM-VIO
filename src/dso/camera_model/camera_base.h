#pragma once
#include "../util/NumType.h"
#include "vio_def.h"
#include <Eigen/Core>
#include <memory>
#include <vector>
#define USE_EXP_IN_KB20
//#define USE_TILT_IN_KB20
#ifndef USE_TILT_IN_KB20
#define USE_JR
#endif
//#define FIX_BETA_IN_KB20
namespace dso {

class CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<CameraBase>;

  enum CameraModel {
    kUnknown = -1,
    kPinhole,
    kKB8,
    kKB16,
    kRadialTangential,
    kDoubleSphere,
    kUcmRTP,
    kUcmRTPth,
    kKB20,
    kKBL16
  };

  static inline std::string ModelAsString(CameraModel camera_model) {
    if (camera_model == kPinhole) return "Pinhole";
    if (camera_model == kKB8) return "KB8";
    if (camera_model == kKB16) return "KB16";
    if (camera_model == kRadialTangential) return "RT";
    if (camera_model == kDoubleSphere) return "DS";
    if (camera_model == kUcmRTP) return "UcmRTP";
    if (camera_model == kKB20) return "KB20";
    if (camera_model == kKBL16) return "KBL16";

    return "UNKNOWN";
  }

  static inline CameraModel FromString(const std::string& camera_model) {
    if (camera_model == "Pinhole") return kPinhole;
    if (camera_model == "KB8") return kKB8;
    if (camera_model == "KB16") return kKB16;
    if (camera_model == "RT") return kRadialTangential;
    if (camera_model == "DS") return kDoubleSphere;
    if (camera_model == "UcmRTP") return kUcmRTP;
    if (camera_model == "KB20") return kKB20;
    if (camera_model == "KBL16") return kKBL16;
    return kUnknown;
  }

  CameraBase(CamId camera_id, int width, int height) : camera_id_(camera_id), width_(width), height_(height) {}
  CameraBase(CamId camera_id, int width, int height, const number_t* parameters, const int& param_length)
      : camera_id_(camera_id), width_(width), height_(height) {
    kParamLength = param_length;
    std::memcpy(parameters_, parameters, kParamLength * sizeof(number_t));
    // printf("kParamLength %d\n", kParamLength);
  }

  virtual ~CameraBase() {}

  virtual bool Project(
      const Vec3& p_3d, Vec2& p_img, LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d = nullptr,
      LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param = nullptr) const = 0;

  virtual bool Project(
      const Vec3& p_3d, LinearAlgebraLib::Ref<Vec2>& p_img,
      LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d = nullptr,
      LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param = nullptr) const = 0;

  //输入图像坐标，输出单位方向向量
  virtual bool UnProject(
      const Vec2& p_img, Vec3& p_3d, LinearAlgebraLib::Matrix<number_t, 3, 2>* d_p3d_d_img = nullptr,
      LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>* d_p3d_d_param = nullptr) const = 0;

  number_t GetParamByIndex(const size_t index) const { return parameters_[index]; }

  bool SetParamByIndex(const number_t value, const size_t index) {
    //    assert(index < parameters_.size());
    parameters_[index] = value;
    return true;
  }

  const number_t* parameters_ptr() { return parameters_; }

  const int GetLevel() { return level_; }

  CamId camera_id() { return camera_id_; }

  CameraModel camera_model() { return camera_model_; };

  int width() { return width_; }
  int height() { return height_; }

  void SetWidth(const int& w) { width_ = w; }
  void SetHeight(const int& h) { height_ = h; }

  void SetIntrinsic(const VecX& intri_vec) {
    for (int i = 0; i < kParamLength; ++i) {
      parameters_[i] = intri_vec[i];
      parameters_bak_[i] = intri_vec[i];
    }
    //    if (camera_model_ == CameraBase::kKB20 && !opt_ofst_xy0) {
    //      parameters_[24] = 0;
    //      parameters_[25] = 0;
    //    }
    //    if (camera_model_ == CameraBase::kKB20 && !opt_ofst_xy1) {
    //      parameters_[26] = 0;
    //      parameters_[27] = 0;
    //    }
    //    if (camera_model_ == CameraBase::kKB20 && !opt_ofst_xy2) {
    //      parameters_[28] = 0;
    //      parameters_[29] = 0;
    //    }
    //    if (camera_model_ == CameraBase::kKB20 && !opt_extra_p) {
    //      parameters_[30] = 0;
    //      parameters_[31] = 0;
    //      parameters_[32] = 0;
    //    }
    //    if (camera_model_ == CameraBase::kKB20 && !opt_p1) {
    //      parameters_[33] = 0;
    //      parameters_[34] = 0;
    //    }
    //    if (camera_model_ == CameraBase::kKB20 && !opt_p2) {
    //      parameters_[35] = 0;
    //      parameters_[36] = 0;
    //    }
    //#ifdef FIX_BETA_IN_KB20
    //    if (camera_model_ == CameraBase::kKB20) {
    //#ifdef USE_EXP_IN_KB20
    //      parameters_[21] = 0;
    //#else
    //      parameters_[21] = 1;
    //#endif
    //    }
    //#endif
  }

  void PlusIntrinsic(const VecX& intri_ksai) {
    for (int i = 0; i < kParamLength; ++i) {
      parameters_[i] += intri_ksai[i];
    }
  }

  void PrintIntri() const {
    for (int i = 0; i < kParamLength; ++i) {
      printf("%.15f, ", parameters_[i]);
    }
    printf("\n");
  }

  void SetUseExtraParam(const bool& use_extra_param, bool use_full = false, bool fix_fc_ = false) {
    extra_param = use_extra_param;
    if (camera_model_ == CameraBase::kKB20) {
      opt_ofst_xy0 = false;
      opt_ofst_xy1 = false;
      opt_ofst_xy2 = false;
      opt_p1 = false;
      opt_p2 = false;
      opt_extra_p = false;
      fix_k = false;
      fix_fc = fix_fc_;
      k_nums_used = 6;
      if (extra_param && use_full) {
        opt_ofst_xy0 = true;
        opt_ofst_xy1 = true;
        opt_ofst_xy2 = true;
        opt_p1 = true;
        opt_p2 = true;
        opt_extra_p = true;
      }

      int param_size_old = kParamLength;
      SetParamSize();
      int param_size_new = kParamLength;
      printf("param, old size: %d, new size: %d\n", param_size_old, param_size_new);
      if (param_size_new <= param_size_old) {
        return;
      } else {
        for (int i = param_size_old; i < param_size_new; ++i) {
          parameters_[i] = 0;
        }
      }
    }
  }

  virtual void SetParamSize() = 0;

  void SetFixK(const bool& is_fix_k) { fix_k = is_fix_k; }

  int kParamLength = 0;
  bool extra_param = false;
  bool fix_k = false;
  bool fix_fc = false;
  int k_nums_used = 4;  // 4;
  bool opt_ofst_xy0 = true;
  bool opt_ofst_xy1 = true;
  bool opt_ofst_xy2 = true;
  bool opt_extra_p = true;
  bool opt_p1 = true;
  bool opt_p2 = true;

 protected:
  CamId camera_id_;
  int width_;
  int height_;
  CameraModel camera_model_;
  number_t parameters_[50];
  number_t parameters_bak_[50];

  int level_ = -1;
};
}  // namespace dso