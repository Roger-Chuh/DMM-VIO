
#include <cmath>
#include <iostream>

// eigen
#include "../../camera_model/vio_def.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "polynomial.h"

//#include "util/logging.h"
//#include "util/random.h"
//#include "util/string.h"

namespace dso {
// lambda^3
//+ (m00 + m11 + m22)*lambda^2
//+ (- m01^2 - m02^2 - m12^2 + m00*m11 + m00*m22 + m11*m22)*lambda
//- m22*m01^2 + 2*m01*m02*m12 - m11*m02^2 - m00*m12^2 + m00*m11*m22

// Solves the right nullspace from QR decomposition,
// returning the size of the kernel
template <typename Derived, typename Scalar>
int solveNullspace(const Eigen::MatrixBase<Derived>& A, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& k) {
  Eigen::ColPivHouseholderQR<Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>> qr(A.transpose());
  Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> Q = qr.householderQ();

  int n = qr.dimensionOfKernel();
  k.resize(Q.rows(), n);

  k = Q.block(0, Q.cols() - n, Q.rows(), n);
  return qr.dimensionOfKernel();
}

Vec3 orthogonal(const Vec3& v) {
  int k = 0;
  if (v[1] < v[k]) k = 1;
  if (v[2] < v[k]) k = 2;
  Vec3 e = Vec3::Zero();
  e[k] = 1.;
  return v.cross(e).normalized();
}

// VecX real_roots(const VecX &real, const VecX &imag) {
//  //CHECK_EQ(real.size(), imag.size());
//
//  if (real.size() != imag.size()) {
//    printf("size dont match\n");
//    std::exit(-1);
//  }
//
//  VecX roots(real.size());
//
//  VecX::Index j = 0;
//  for (VecX::Index i = 0; i < real.size(); ++i) {
//    if (!imag(i)) {
//      roots(j) = real(i);
//      ++j;
//    }
//  }
//
//  roots.conservativeResize(j);
//  return roots;
//}

// int main(int argc, char *argv[]) {
//  // Initialize Google's logging library.
//  InitializeGlog(argv);
//  FLAGS_logtostderr = 1;
//
//  Vec3 axis(RandomReal(-1., 1.), RandomReal(-1., 1.), RandomReal(-1., 1.));
//  axis.normalize();
//
//  Vec3 o = orthogonal(axis);
//
//  Mat3 M;
//  M.setZero();
//
//  int N = 10;
//  for (int i = 0; i < N; ++i) {
//    const number_t theta = RandomReal(-EIGEN_PI, EIGEN_PI);
//    Vec3 v =
//        o * std::cos(theta) + axis.cross(o) * std::sin(theta) + axis *
//        axis.dot(o) * (1. - std::cos(theta));
//    M += v * v.transpose();
//  }
//
//  VecX coeffs(4);
//  coeffs << 1., M.trace(),
//      -std::pow(M(0, 1), 2) - std::pow(M(0, 2), 2) - std::pow(M(1, 2), 2) +
//      M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) +
//          M(1, 1) * M(2, 2),
//      -M(2, 2) * std::pow(M(0, 1), 2) + 2. * M(0, 1) * M(0, 2) * M(1, 2) -
//      M(1, 1) * std::pow(M(0, 2), 2) -
//          M(0, 0) * std::pow(M(1, 2), 2) + M(0, 0) * M(1, 1) * M(2, 2);
//
//  VecX real, imag;
//  if (!FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag)) {
//    LOG(ERROR) << "Failed to find roots\n"
//               << StringPrintf("%.16f %.16f %.16f %.16f", coeffs[0],
//               coeffs[1], coeffs[2], coeffs[3]);
//    return 1;
//  }
//
//  VecX lambdas = real_roots(real, imag);
//  if (lambdas.size() == 0) {
//    LOG(ERROR) << "No real roots found\n"
//               << StringPrintf("%.16f %.16f %.16f %.16f", coeffs[0],
//               coeffs[1], coeffs[2], coeffs[3]);
//    return 1;
//  }
//  LOG(INFO) << "Number of candidate solutions: " << lambdas.size();
//
//  bool solved = false;
//  Vec3 x;  // solution vector
//  number_t min_cost = std::numeric_limits<number_t>::max();
//  for (VecX::Index i = 0; i < lambdas.size(); ++i) {
//    const number_t lambda = lambdas[i];
//
//    MatX candidate_solution;
//    int kernel_size = solveNullspace(M + lambda * Mat3::Identity(),
//    candidate_solution); LOG(INFO) << "Kernel size: " << kernel_size;
//
//    if (kernel_size != 1) continue;
//
//    const number_t cost = candidate_solution.col(0).transpose() * M *
//    candidate_solution.col(0); if (cost < min_cost) {
//      x = candidate_solution.col(0);
//      min_cost = cost;
//      solved = true;
//    }
//  }
//
//  if (!solved) {
//    LOG(ERROR) << "Solution not found!";
//    return 1;
//  }
//
//  std::cout << "groundtruth axis: " << axis.transpose() << std::endl;
//  const number_t cost = axis.transpose() * M * axis;
//  LOG(INFO) << "Cost: " << cost;
//
//  std::cout << "estimated axis: " << x.transpose() << std::endl;
//  LOG(INFO) << "Cost: " << min_cost;
//
//  return 0;
//}
}  // namespace dso
