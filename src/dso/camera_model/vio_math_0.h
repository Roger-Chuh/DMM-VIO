#pragma once

#include "../util/NumType.h"

namespace dso {

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> Skew(const TVec3 &v) {
  using T = typename TVec3::Scalar;
  Matrix3<T> mat = Matrix3<T>::Zero();
  mat(0, 1) = -v(2);
  mat(0, 2) = v(1);
  mat(1, 0) = v(2);
  mat(1, 2) = -v(0);
  mat(2, 0) = -v(1);
  mat(2, 1) = v(0);
  return mat;
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> ExpSO3(const TVec3 &omega) {
  using std::abs;
  using std::cos;
  using std::sin;
  using std::sqrt;
  using T = typename TVec3::Scalar;
  T theta_sq = omega.squaredNorm();
  T imag_factor;
  T real_factor;
  T theta;
  // todo: test when theta_sq is extremely small
  if (theta_sq <
      std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon()) {
    theta = T(0);
    T theta_po4 = theta_sq * theta_sq;
    imag_factor =
        T(0.5) - T(1.0 / 48.0) * theta_sq + T(1.0 / 3840.0) * theta_po4;
    real_factor = T(1) - T(1.0 / 8.0) * theta_sq + T(1.0 / 384.0) * theta_po4;
  } else {
    theta = sqrt(theta_sq);
    T half_theta = T(0.5) * (theta);
    T sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta / (theta);
    real_factor = cos(half_theta);
  }
  Eigen::Quaternion<T> q(real_factor, imag_factor * omega.x(),
                         imag_factor * omega.y(), imag_factor * omega.z());
  return q.toRotationMatrix();
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix4<typename TVec3::Scalar> ExpSE3(const TVec3 &w, const TVec3 &v) {
  using std::cos;
  using std::sin;
  using T = typename TVec3::Scalar;
  using TMat3 = Matrix3<T>;
  using TMat4 = Matrix4<T>;

  const TVec3 &omega = w;
  T theta_sq = omega.squaredNorm();

  T theta;

  if (theta_sq <
      std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon())
    theta = T(0);
  else
    theta = std::sqrt(theta_sq);

  const TMat3 so3 = ExpSO3(omega);
  const TMat3 Omega = Skew(omega);
  const TMat3 Omega_sq = Omega * Omega;
  TMat3 V = TMat3::Zero();

  if (theta < std::numeric_limits<T>::epsilon()) {
    V = so3;
    /// Note: That is an accurate expansion!
  } else {
    T theta_sq = theta * theta;
    V = (TMat3::Identity() + (T(1) - cos(theta)) / (theta_sq)*Omega +
         (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
  }

  TVec3 tran = V * v;
  TMat4 result = TMat4::Identity();

  result.template block<3, 3>(0, 0) = so3;
  result.template block<3, 1>(0, 3) = tran;

  return result;
}

template <typename TMat3,
          typename = std::enable_if_t<IsFixedSizeMatrix<TMat3, 3, 3>::value>>
inline Vector3<typename TMat3::Scalar> Log(const TMat3 &R) {
  using T = typename TMat3::Scalar;
  using TVec3 = Vector3<T>;

  T EPSILON = std::numeric_limits<T>::epsilon();
  T EPSILONSQRT = std::sqrt(EPSILON);

  const T &R11 = R(0, 0), R12 = R(0, 1), R13 = R(0, 2);
  const T &R21 = R(1, 0), R22 = R(1, 1), R23 = R(1, 2);
  const T &R31 = R(2, 0), R32 = R(2, 1), R33 = R(2, 2);
  // Get trace(R)
  const T tr = R.trace();

  TVec3 omega(TVec3::Zero());
  // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
  // we do something special
  if (tr + static_cast<T>(1.0) < EPSILON) {
    if (std::abs(R33 + static_cast<T>(1.0)) > EPSILONSQRT)
      omega = (kOur_PI / sqrt(2.0 + 2.0 * R33)) * TVec3(R13, R23, 1.0 + R33);
    else if (std::abs(R22 + 1.0) > EPSILONSQRT)
      omega = (kOur_PI / sqrt(2.0 + 2.0 * R22)) * TVec3(R12, 1.0 + R22, R32);
    else
      // if(std::abs(R.r1_.x()+1.0) > 1e-5)  This is implicit
      omega = (kOur_PI / sqrt(2.0 + 2.0 * R11)) * TVec3(1.0 + R11, R21, R31);
  } else {
    T magnitude;
    const T tr_3 = tr - static_cast<T>(3.0); // always negative
    if (tr_3 < -1e-7) {
      T theta = acos((tr - 1.0) / 2.0);
      magnitude = theta / (2.0 * sin(theta));
    } else {
      // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
      // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
      magnitude = static_cast<T>(0.5) - tr_3 * tr_3 / 12;
    }
    omega = magnitude * TVec3(R32 - R23, R13 - R31, R21 - R12);
  }
  return omega;
}

template <typename TMat4,
          typename = std::enable_if_t<IsFixedSizeMatrix<TMat4, 4, 4>::value>>
inline Vector6<typename TMat4::Scalar> Log(const TMat4 &P) {
  using T = typename TMat4::Scalar;
  using TVec3 = Vector3<T>;
  using TVec6 = Vector6<T>;
  using TMat3 = Matrix3<T>;

  TVec6 drdp = TVec6::Zero();
  TMat3 R = P.template block<3, 3>(0, 0);
  TVec3 omega = Log(R);
  T theta = omega.norm();
  drdp.template head<3>() = omega;
  TMat3 Omega = Skew(omega);
  if (theta < std::numeric_limits<T>::epsilon()) {
    TMat3 V_inv =
        TMat3::Identity() - T(0.5) * Omega + T(1. / 12.) * (Omega * Omega);
    drdp.template tail<3>() = V_inv * P.template block<3, 1>(0, 3);
  } else {
    T half_theta = T(0.5) * theta;
    TMat3 V_inv = (TMat3::Identity() - T(0.5) * Omega +
                   (T(1) - theta * cos(half_theta) / (T(2) * sin(half_theta))) /
                       (theta * theta) * (Omega * Omega));
    drdp.template tail<3>() = V_inv * P.template block<3, 1>(0, 3);
  }
  return drdp;
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> Jr(const TVec3 &phi) {
  using T = typename TVec3::Scalar;
  using TMat3 = Matrix3<T>;

  T EPSILON = std::numeric_limits<T>::epsilon();

  TMat3 J = TMat3::Identity();

  T phi_norm2 = phi.squaredNorm();
  TMat3 phi_hat = Skew(phi);
  TMat3 phi_hat2 = phi_hat * phi_hat;

  if (phi_norm2 > EPSILON) {
    T phi_norm = std::sqrt(phi_norm2);
    T phi_norm3 = phi_norm2 * phi_norm;

    J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
    J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
  } else {
    // sin and cos Taylor expansion around 0
    J -= phi_hat / 2;
    J += phi_hat2 / 6;
  }

  return J;
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> JrInv(const TVec3 &phi) {
  using T = typename TVec3::Scalar;
  using TMat3 = Matrix3<T>;

  T EPSILON = std::numeric_limits<T>::epsilon();
  T EPSILONSQRT = std::sqrt(EPSILON);

  TMat3 J = TMat3::Identity();

  T phi_norm2 = phi.squaredNorm();
  TMat3 phi_hat = Skew(phi);
  TMat3 phi_hat2 = phi_hat * phi_hat;

  J += phi_hat / 2;
  if (phi_norm2 > EPSILON) {
    T phi_norm = std::sqrt(phi_norm2);

    assert(phi_norm <= kOur_PI + EPSILON &&
           "We require that the angle is in range [0, pi].");

    if (phi_norm < kOur_PI - EPSILONSQRT) {
      // regular case for range (0,pi)
      J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                           (2 * phi_norm * std::sin(phi_norm)));
    } else {
      // 0th-order Taylor expansion around pi
      J += phi_hat2 / (kOur_PI * kOur_PI);
    }
  } else {
    // Taylor expansion around 0
    J += phi_hat2 / 12;
  }
  return J;
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> Jl(const TVec3 &phi) {
  return Jr(-phi);
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> JlInv(const TVec3 &phi) {
  return JrInv(-phi);
}

template <typename TMat3,
          typename = std::enable_if_t<IsFixedSizeMatrix<TMat3, 3, 3>::value>>
inline void EnforceRot(TMat3 &R) {
  using T = typename TMat3::Scalar;
  using TVec3 = Vector3<T>;
  TVec3 v0 = R.template block<3, 1>(0, 0);
  TVec3 v1 = R.template block<3, 1>(0, 1);
  v0.normalize();
  v1 = v1 - (v1.template dot(v0)) * v0;
  v1.normalize();
  R.template block<3, 1>(0, 0) = v0;
  R.template block<3, 1>(0, 1) = v1;
  R.template block<3, 1>(0, 2) = v0.template cross(v1);
}

template <typename TMat4,
          typename = std::enable_if_t<IsFixedSizeMatrix<TMat4, 4, 4>::value>>
inline TMat4 InversePose(const TMat4 &T01) {
  TMat4 res = TMat4::Identity();
  res.template block<3, 3>(0, 0) = T01.template block<3, 3>(0, 0).transpose();
  res.template block<3, 1>(0, 3) =
      -res.template block<3, 3>(0, 0) * T01.template block<3, 1>(0, 3);
  return res;
}

// void LeftMultiply(Mat34 &M_origin, const Mat34 &dPose);

// void T_inverse(Mat34 &Tba, const Mat34 &Tab);

// Skew(n) * R
template <typename TVec3>
inline Matrix3<typename TVec3::Scalar>
LeftMultiSkew(const TVec3 &n, const Matrix3<typename TVec3::Scalar> &R) {
  Matrix3<typename TVec3::Scalar> res;
  for (int i = 0; i < 3; ++i) {
    res.col(i) = n.template cross(R.col(i));
  }
  return res;
}

// R * Skew(n)
template <typename TVec3, typename TMat3>
inline TMat3 RightMultiSkew(const TMat3 &R, const TVec3 &n) {
  TMat3 res;
  res(0, 0) = R(0, 1) * n(2) - R(0, 2) * n(1);
  res(0, 1) = R(0, 2) * n(0) - R(0, 0) * n(2);
  res(0, 2) = R(0, 0) * n(1) - R(0, 1) * n(0);
  res(1, 0) = R(1, 1) * n(2) - R(1, 2) * n(1);
  res(1, 1) = R(1, 2) * n(0) - R(1, 0) * n(2);
  res(1, 2) = R(1, 0) * n(1) - R(1, 1) * n(0);
  res(2, 0) = R(2, 1) * n(2) - R(2, 2) * n(1);
  res(2, 1) = R(2, 2) * n(0) - R(2, 0) * n(2);
  res(2, 2) = R(2, 0) * n(1) - R(2, 1) * n(0);
  return res;
}

template <typename TVec3>
inline Matrx3x2<typename TVec3::Scalar>
ProduceOtherOthogonalBasis(const TVec3 &n) {
  TVec3 N = n;
  if (N[0] < 0)
    N[0] = -N[0];
  if (N[1] < 0)
    N[1] = -N[1];
  if (N[2] < 0)
    N[2] = -N[2];

  int minIdx = 0;
  if (N[0] <= N[1]) {
    if (N[0] <= N[2])
      minIdx = 0;
    else
      minIdx = 2;
  } else {
    if (N[1] <= N[2])
      minIdx = 1;
    else
      minIdx = 2;
  }

  Matrx3x2<typename TVec3::Scalar> A(Matrx3x2<typename TVec3::Scalar>::Zero());
  switch (minIdx) {
  case 0:
    A.template block<3, 1>(0, 0) = TVec3(0, -n[2], n[1]);
    break;
  case 1:
    A.template block<3, 1>(0, 0) = TVec3(n[2], 0, -n[0]);
    break;
  case 2:
    A.template block<3, 1>(0, 0) = TVec3(n[1], -n[0], 0);
    break;
  }
  A.template block<3, 1>(0, 1) = n.template cross(A.template block<3, 1>(0, 0));

  A.template block<3, 1>(0, 0).normalize();
  A.template block<3, 1>(0, 1).normalize();

  return A;
}

template <typename TVec3, typename TMat3>
inline void Mat3RightMultiplySkewM3V3(const TMat3 &left_mat3,
                                      const TVec3 &right_skew, TMat3 &result) {
  result(0, 0) =
      left_mat3(0, 1) * right_skew(2) - left_mat3(0, 2) * right_skew(1);
  result(0, 1) =
      left_mat3(0, 2) * right_skew(0) - left_mat3(0, 0) * right_skew(2);
  result(0, 2) =
      left_mat3(0, 0) * right_skew(1) - left_mat3(0, 1) * right_skew(0);
  result(1, 0) =
      left_mat3(1, 1) * right_skew(2) - left_mat3(1, 2) * right_skew(1);
  result(1, 1) =
      left_mat3(1, 2) * right_skew(0) - left_mat3(1, 0) * right_skew(2);
  result(1, 2) =
      left_mat3(1, 0) * right_skew(1) - left_mat3(1, 1) * right_skew(0);
  result(2, 0) =
      left_mat3(2, 1) * right_skew(2) - left_mat3(2, 2) * right_skew(1);
  result(2, 1) =
      left_mat3(2, 2) * right_skew(0) - left_mat3(2, 0) * right_skew(2);
  result(2, 2) =
      left_mat3(2, 0) * right_skew(1) - left_mat3(2, 1) * right_skew(0);
}

template <typename TMat4, typename TVec3>
inline void DiffInSE3(const TMat4 &T1, const TMat4 &T2, TVec3 &dr, TVec3 &dp) {
  using T = typename TVec3::Scalar;
  using TMat3 = Matrix3<T>;
  TMat3 R =
      T1.template block<3, 3>(0, 0) * T2.template block<3, 3>(0, 0).transpose();
  TVec3 t = T1.template block<3, 1>(0, 3) - R * T2.template block<3, 1>(0, 3);
  dr = Log(R);
  dp = JrInv(-dr) * t;
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline Matrix3<typename TVec3::Scalar> GetRotationFromV1ToV2(const TVec3 &v1,
                                                             const TVec3 &v2) {
  using T = typename Matrix3<typename TVec3::Scalar>::Scalar;
  TVec3 a = v1.normalized();
  TVec3 b = v2.normalized();
  TVec3 n = a.template cross(b);
  n.normalize();
  T cos_theta = a.transpose() * b;
  T theta = 0;

  if (cos_theta >= 1) {
    theta = 0;
  } else if (cos_theta <= -1) {
    theta = kOur_PI;
  } else {
    theta = std::acos(cos_theta);
  }

  return ExpSO3(theta * n);
}

template <typename TVec3,
          typename = std::enable_if_t<IsFixedSizeVector<TVec3, 3>::value>>
inline int TriangulateWithCheckTheta(typename TVec3::Scalar &idp,
                                     const Matrix4<typename TVec3::Scalar> &T01,
                                     const TVec3 &v0, const TVec3 &v1) {
  using T = typename TVec3::Scalar;
  using TMatrix6x3 = Matrix6x3<T>;
  using TVec6 = Vector6<T>;
  using TMat3 = Matrix3<T>;

  TMatrix6x3 A = TMatrix6x3::Zero();
  TVec6 b = TVec6::Zero();
  A.template block<3, 3>(0, 0) = Skew(v0);
  A.template block<3, 3>(3, 0) =
      Skew(v1) * T01.template block<3, 3>(0, 0).transpose();
  b.template segment<3>(3) = Skew(v1) *
                             T01.template block<3, 3>(0, 0).transpose() *
                             T01.template block<3, 1>(0, 3);
  TVec3 s = A.transpose() * b;
  TMat3 AA = A.transpose() * A;
  TVec3 xyz = AA.ldlt().solve(s);

  if (xyz.z() < 0) {
    return -1;
  }
  idp = 1.0 / xyz.norm();

  TVec3 v1_0 = T01.template block<3, 3>(0, 0) * v1;
  T cos_theta = v1_0.template dot(v0);
  if (cos_theta > 0.9999) {
    return -2;
  }

  return 0;
}

inline Mat6 Adj(const Mat4 &T) {
  Mat3 R = T.topLeftCorner<3, 3>();
  Mat6 res;
  res.block(0, 0, 3, 3) = R;
  res.block(3, 3, 3, 3) = R;
  res.block(3, 0, 3, 3) = Skew(T.topRightCorner<3, 1>()) * R;
  res.block(0, 3, 3, 3) = Mat3::Zero(3, 3);
  return res;
}

inline Mat6 JrInvSE3Decoupled(const Vec6 &res) {
  Mat6 J = Mat6::Zero();
  const Vec3 &r = res.head(3);
  J.block(0, 0, 3, 3) = JrInv(r);
  J.block(3, 3, 3, 3) = ExpSO3(r);
  return J;
}

inline int Triangulate(number_t &idp, const Mat4 &T01, const Vec3 &v0,
                       const Vec3 &v1) {
  Mat63 A = Mat63::Zero();
  Vec6 b = Vec6::Zero();
  A.block<3, 3>(0, 0) = Skew(v0);
  A.block<3, 3>(3, 0) = Skew(v1) * T01.block<3, 3>(0, 0).transpose();
  b.segment<3>(3) =
      Skew(v1) * T01.block<3, 3>(0, 0).transpose() * T01.block<3, 1>(0, 3);
  Vec3 s = A.transpose() * b;
  Mat3 AA = A.transpose() * A;
  Vec3 xyz = AA.ldlt().solve(s);

  if (xyz.z() < 0) {
    return -1;
  }
  idp = 1.0 / xyz.norm();

  return 1;
}

inline number_t
MultiViewTriangulation(const std::vector<Mat4> &poses,
                       const std::vector<Vec3> &points,
                       std::vector<std::pair<number_t, int>> &err_vec,
                       VecX &errs, Vec3 &point_3d) {
  // TODO:Rewrite this for 3d point
  Eigen::MatrixXd design_matrix(poses.size() * 2, 4);
  assert(poses.size() > 0 && poses.size() == points.size() &&
         "We at least have 2 poses and number of pts and poses must equal");
  for (unsigned int i = 0; i < poses.size(); i++) {
    double p0x = points[i][0];
    double p0y = points[i][1];
    double p0z = points[i][2];
    design_matrix.row(i * 2) = p0x * poses[i].row(2) - p0z * poses[i].row(0);
    design_matrix.row(i * 2 + 1) =
        p0y * poses[i].row(2) - p0z * poses[i].row(1);
  }
  Vec4 triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);

  Eigen::MatrixXd pts(4, 1);
  pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
  errs = design_matrix * pts;

  //    std::cout << "design_matrix: " << design_matrix << std::endl;
  //    std::cout << "ERR: " << errs.sum() << ", bearing: " <<
  //    points[0].transpose() << ", pt3d: "<< point_3d.transpose() << std::endl;
  //    std::cout <<"err each: " << errs.transpose() << std::endl;
  number_t err_sum = 0;
  for (size_t i = 0; i < poses.size(); i++) {
    Vec3 rep = (poses[i].topLeftCorner<3, 3>() * point_3d +
                poses[i].topRightCorner<3, 1>())
                   .normalized();
    err_vec[i].first = 235.0 * (rep - points[i]).norm();
    err_sum += err_vec[i].first;
    //      Vec3 rep2 = rep / rep(2);
    //      Vec3 obs = points[i] / points[i](2);
    //      std::cout<<"rep: " << rep.transpose() << ", obs: " <<
    //      points[i].transpose()<<", err:
    //      "<<static_cast<number_t>(kFocalLength)*(rep - points[i]).norm()<<",
    //      err2:
    //      "<<static_cast<number_t>(kFocalLength)*(rep2 -
    //      obs).norm()<<std::endl;
  }
  // return static_cast<number_t>(kFocalLength) * errs.norm() /
  // static_cast<number_t>(errs.rows());
  return err_sum / static_cast<number_t>(poses.size());
}

} // namespace dso
