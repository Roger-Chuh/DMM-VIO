#pragma once
#include "camera_base.h"
//#define UCMK56
namespace dso {
class UCMRTPCamera : public CameraBase {
 public:
  using Ptr = std::shared_ptr<UCMRTPCamera>;
  bool use_exp_ = true;
  /*use_exp:
   * if opti ucmrtp params, setting true to acc optimization;
   * else if load ucmrtp model for project func, setting false to avoid exp
   * alpha;
   * */
  explicit UCMRTPCamera(CamId camera_id, int width, int height, bool use_exp) : CameraBase(camera_id, width, height) {
    camera_model_ = CameraModel::kUcmRTP;
    kParamLength = 17;
    level_ = 0;
    use_exp_ = use_exp;
  }

  void SetExpAlpha() {
    const number_t& temp = parameters_[4];
    parameters_[4] = 1. / (1. + exp(-temp));
  }

  // parameters: fx, fy, cx, cy, alpha, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2,
  // s3, s4
  UCMRTPCamera(CamId camera_id, int width, int height, const number_t* parameters, const int& level,
               const bool& use_exp)
      : CameraBase(camera_id, width, height, parameters, 17) {
    camera_model_ = CameraModel::kUcmRTP;
    level_ = level;
    use_exp_ = use_exp;
  }

  virtual bool Project(
      const Vec3& p_3d, Vec2& p_img, LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d = nullptr,
      LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param = nullptr) const override;

  virtual bool Project(
      const Vec3& p_3d, LinearAlgebraLib::Ref<Vec2>& p_img,
      LinearAlgebraLib::Matrix<number_t, 2, 3>* d_img_d_p3d = nullptr,
      LinearAlgebraLib::Matrix<number_t, 2, LinearAlgebraLib::Dynamic>* d_img_d_param = nullptr) const override;

  virtual bool UnProject(
      const Vec2& p_img, Vec3& p_3d, LinearAlgebraLib::Matrix<number_t, 3, 2>* d_p3d_d_img = nullptr,
      LinearAlgebraLib::Matrix<number_t, 3, LinearAlgebraLib::Dynamic>* d_p3d_d_param = nullptr) const override;
  virtual void SetParamSize() override {}

 public:
  void Distortion(const number_t& p_x, const number_t& p_y, number_t& d_u, number_t& d_v, Mat2* p_d_uv_d_xy = nullptr,
                  Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_uv_d_params = nullptr) const;

  void UnDistortion(const number_t& d_u, const number_t& d_v, number_t& p_x, number_t& p_y, Mat2* d_xy_d_uv = nullptr,
                    Eigen::Matrix<number_t, 2, Eigen::Dynamic>* d_xy_d_params = nullptr) const;
};
}  // namespace dso