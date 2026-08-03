
#include <cmath>
#include <iostream>
#include <limits>

// eigen
#include "../../camera_model/vio_def.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "polynomial.h"

//#include "util/logging.h"
//#include "util/random.h"
//#include "util/string.h"

#ifndef DLSD_SOLVER_ZERO_TH
#define DLSD_SOLVER_ZERO_TH 1e-9
#endif

namespace dso {
// Solves the right nullspace from QR decomposition,
// returning the size of the kernel
MatX solveNullspace(const Mat3& A) {
  /*
  Eigen::FullPivHouseholderQR<Eigen::Matrix<typename Derived::Scalar,
  Eigen::Dynamic, Eigen::Dynamic>> qr(A.transpose()); Eigen::Matrix<typename
  Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> Q = qr.householderQ();

  int n = qr.dimensionOfKernel();
  k.resize(Q.rows(), n);

  k = Q.block(0, Q.cols() - n, Q.rows(), n);
  return qr.dimensionOfKernel();
  */

  Eigen::FullPivLU<MatX> lu(A);
  lu.setThreshold(DLSD_SOLVER_ZERO_TH);
  return lu.kernel();
}

VecX real_roots(const VecX& real, const VecX& imag) {
  // CHECK_EQ(real.size(), imag.size());
  if (real.size() != imag.size()) {
    printf("size dont match\n");
    std::exit(-1);
  }

  VecX roots(real.size());

  VecX::Index j = 0;
  for (VecX::Index i = 0; i < real.size(); ++i) {
    if (!imag(i)) {
      roots(j) = real(i);
      ++j;
    }
  }

  roots.conservativeResize(j);
  return roots;
}

bool line_fit(const std::vector<Vec3>& observations, Vec3& axis) {
  // LOG(INFO) << "Number of observations: " << observations.size();
  if (observations.size() < 3) return false;

  Mat3 M;
  M.setZero();

  for (const Vec3& v : observations) M += v * v.transpose();

  // lambda^3
  //+ (m00 + m11 + m22)*lambda^2
  //+ (- m01^2 - m02^2 - m12^2 + m00*m11 + m00*m22 + m11*m22)*lambda
  //- m22*m01^2 + 2*m01*m02*m12 - m11*m02^2 - m00*m12^2 + m00*m11*m22

  VecX coeffs(4);
  coeffs << 1., M.trace(),
      -std::pow(M(0, 1), 2) - std::pow(M(0, 2), 2) - std::pow(M(1, 2), 2) + M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) +
          M(1, 1) * M(2, 2),
      -M(2, 2) * std::pow(M(0, 1), 2) + 2. * M(0, 1) * M(0, 2) * M(1, 2) - M(1, 1) * std::pow(M(0, 2), 2) -
          M(0, 0) * std::pow(M(1, 2), 2) + M(0, 0) * M(1, 1) * M(2, 2);

  VecX real, imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag)) {
    //    LOG(ERROR) << "Failed to find roots\n"
    //               << StringPrintf("%.16f %.16f %.16f %.16f", coeffs[0],
    //               coeffs[1], coeffs[2], coeffs[3]);
    return 1;
  }

  VecX lambdas = real_roots(real, imag);
  if (lambdas.size() == 0) {
    //    LOG(ERROR) << "No real roots found\n"
    //               << StringPrintf("%.16f %.16f %.16f %.16f", coeffs[0],
    //               coeffs[1], coeffs[2], coeffs[3]);
    return 1;
  }
  // LOG(INFO) << "Number of candidate solutions: " << lambdas.size();

  bool solved;
  Vec3 x;  // solution vector
  number_t min_cost = std::numeric_limits<number_t>::max();
  for (VecX::Index i = 0; i < lambdas.size(); ++i) {
    const number_t lambda = lambdas[i];

    MatX kernel = solveNullspace(M + lambda * Mat3::Identity());
    // LOG(INFO) << "Kernel size: " << kernel.cols();

    // The size of the kernel must be 1 by construction
    if (kernel.cols() != 1) {
      //      LOG(WARNING) << "Kernel must be 1D!";
      // continue;
      solved = false;
      break;
    }
    Vec3 candidate_solution = kernel.col(0);
    candidate_solution.normalize();

    // LOG(INFO) << "Candidate solution norm: " << candidate_solution.norm();
    if (std::abs(candidate_solution.norm() - 1.) > DLSD_SOLVER_ZERO_TH) {
      // continue;
      //      LOG(WARNING) << "Candidate solution norm: " <<
      //      candidate_solution.norm();
      solved = false;
      break;
    }

    const number_t cost = candidate_solution.transpose() * M * candidate_solution;
    // LOG(INFO) << "Candidate solution cost: " << cost;
    if (cost < min_cost) {
      x = candidate_solution;
      min_cost = cost;
      solved = true;
    }
  }

  if (!solved) return false;

  // LOG(INFO) << "Cost: " << min_cost;
  //  TODO Threshold cost?

  axis = x;
  return true;
}
}  // namespace dso
