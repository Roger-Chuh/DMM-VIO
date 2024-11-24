#pragma once
#include "../camera_model/vio_def.h"

namespace Eigen {

template <typename T> using Vector2 = Matrix<T, 2, 1>;

template <typename T> using Vector3 = Matrix<T, 3, 1>;

template <typename T> using Vector4 = Matrix<T, 4, 1>;

template <typename T> using VectorX = Matrix<T, Dynamic, 1>;

template <typename T> using MatrixX = Matrix<T, Dynamic, Dynamic>;

template <typename T> using Matrix3 = Matrix<T, 3, 3>;

template <typename T, int N> using Vector = Matrix<T, N, 1>;

} // namespace Eigen

namespace yvr::yvr_calib {

template <typename T> class Pose3 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pose3() {
    q_.setIdentity();
    t_.setZero();
  }

  Pose3(const Eigen::Quaternion<T> &q, const Eigen::Vector3<T> &t)
      : q_(q), t_(t) {}

  Pose3(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Vector3<T> &t) {
    q_ = Eigen::Quaternion<T>(R);
    if (std::abs(q_.norm() - 1) > 1e-6) {
      // throw std::runtime_error("The quaternion extracted is not a unit
      // quaternion.");
      std::cerr
          << "The quaternion extracted is not a unit quaternion difference is "
          << std::abs(q_.norm() - 1) << std::endl;
      q_.normalize();
    }
    t_ = t;
  }

  Pose3(const Eigen::Matrix<T, 4, 4> &mat) {
    Eigen::Matrix<T, 3, 3> R = mat.template block<3, 3>(0, 0);
    Eigen::Vector3<T> t = mat.template block<3, 1>(0, 3);
    q_ = Eigen::Quaternion<T>(R);

    if (std::abs(q_.norm() - 1) > 1e-6) {
      // throw std::runtime_error("The quaternion extracted is not a unit
      // quaternion.");
      std::cerr
          << "The quaternion extracted is not a unit quaternion difference is "
          << std::abs(q_.norm() - 1) << std::endl;
      q_.normalize();
    }

    t_ = t;
  }

  Eigen::Quaternion<T> &rotation() { return q_; }
  const Eigen::Quaternion<T> &rotation() const { return q_; }

  Eigen::Vector3<T> &translation() { return t_; }
  const Eigen::Vector3<T> &translation() const { return t_; }

  /// Rotation setter as a 4-vector quaternion for python bindings. Quaternion
  /// vector must be in the order \f$\left[w, x, y, z\right]\f$
  void SetRotation(const Eigen::Vector4<T> &q) {
    const Eigen::Vector4<T> qn = q.normalized();
    q_.w() = qn(0);
    q_.x() = qn(1);
    q_.y() = qn(2);
    q_.z() = qn(3);
  }

  /// Rotation getter as a 4-vector for python bindings. Returned quaternion
  /// will be in the order \f$\left[w, x, y, z\right]\f$
  Eigen::Vector4<T> GetRotation() const {
    return Eigen::Vector4<T>(q_.w(), q_.x(), q_.y(), q_.z());
  }

  /// Translation setter for python bindings.
  void SetTranslation(const Eigen::Vector3<T> &t) { t_ = t; }

  /// Translation getter for python bindings.
  Eigen::Vector3<T> GetTranslation() const { return t_; }

  Pose3 operator*(const Pose3<T> &T_b_a) const {
    // this = T_c_b
    const Eigen::Quaternion<T> &q_c_b = this->rotation();
    const Eigen::Quaternion<T> &q_b_a = T_b_a.rotation();
    const Eigen::Vector3<T> &t_c_b = this->translation();
    const Eigen::Vector3<T> &t_b_a = T_b_a.translation();
    const Eigen::Quaternion<T> q_c_a = q_c_b * q_b_a;
    const Eigen::Vector3<T> t_c_a = q_c_b * t_b_a + t_c_b;
    return Pose3(q_c_a, t_c_a);
  }

  Eigen::Vector3<T> operator*(const Eigen::Vector3<T> &p) const {
    return this->rotation() * p + this->translation();
  }

  Pose3<T> inverse() const {
    const Eigen::Quaternion<T> q_inv = this->rotation().conjugate();
    const Eigen::Vector3<T> t_inv = -(q_inv * this->translation());
    return Pose3(q_inv, t_inv);
  }

  Eigen::Matrix<T, 4, 4> ToMat4() const {
    Eigen::Matrix<T, 4, 4> H;
    H.setIdentity();
    H.template block<3, 3>(0, 0) = q_.toRotationMatrix();
    H.template block<3, 1>(0, 3) = t_;

    return H;
  }

  Eigen::Matrix<T, 6, 1> ToVec6() const {
    Eigen::Matrix<T, 6, 1> vec6;
    auto Mat4 = this->ToMat4();
    vec6 = Log(Mat4);
    return vec6;
  }

  bool isApprox(const Pose3<T> &pose) const {
    return (pose.rotation().isApprox(q_) && pose.translation().isApprox(t_));
  }

  friend std::ostream &operator<<(std::ostream &os, const Pose3<T> &pose) {
    os << "q: " << pose.rotation().w() << " " << pose.rotation().x() << " "
       << pose.rotation().y() << " " << pose.rotation().z()
       << ", t: " << pose.translation().transpose();
    return os;
  }

private:
  Eigen::Quaternion<T> q_;
  Eigen::Vector3<T> t_;
};

using Pose3t = Pose3<double>;

template <typename Derived>
std::string EigenToString(const Eigen::MatrixBase<Derived> &matrix) {
  std::stringstream ss;
  if (matrix.rows() > 1)
    ss << std::endl;
  ss << matrix;
  return ss.str();
}

} // namespace yvr::yvr_calib
