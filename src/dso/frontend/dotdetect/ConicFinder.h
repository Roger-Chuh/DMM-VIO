#pragma once

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <opencv2/core/mat.hpp>
#include <queue>
#include <vector>

#include "Conic.h"
#include "ImageProcessing.h"
#include "Label.h"

namespace dso {

namespace DotDetect {

template <typename TdI>
Mat3 FindEllipse(const int w, const int /*h*/, const TdI* dI, const IRectangle& r, double& /*residual*/) {
  // Precise ellipse estimation without contour point extraction
  // Jean-Nicolas Ouellet, Patrick Hebert

  // This normalisation is in the wrong space.
  // We'd be better off normalising image values to [0,1] range.

  // Form system Ax = b to solve
  Eigen::Matrix<double, 5, 5> A = Eigen::Matrix<double, 5, 5>::Zero();
  Eigen::Matrix<double, 5, 1> b = Eigen::Matrix<double, 5, 1>::Zero();

  //    float elementCount = 0;
  for (int v = r.y1; v <= r.y2; ++v) {
    const TdI* dIv = dI + v * w;
    for (int u = r.x1; u <= r.x2; ++u) {
      // li = (ai,bi,ci)' = (I_ui,I_vi, -dI' x_i)'
      const Eigen::Vector3d d = Eigen::Vector3d(dIv[u][0], dIv[u][1], -(dIv[u][0] * u + dIv[u][1] * v));
      //            const Eigen::Vector3d li = //H.T() * d;
      //                    Eigen::Vector3d( d[0]*H(0,0), d[1]*H(1,1),
      //                    d[0]*H(0,2) + d[1] * H(1,2) + d[2] );
      const Eigen::Vector3d& li = d;
      Eigen::Matrix<double, 5, 1> Ki;
      Ki << li[0] * li[0], li[0] * li[1], li[1] * li[1], li[0] * li[2], li[1] * li[2];
      A += Ki * Ki.transpose();
      b += -Ki * li[2] * li[2];
      //            elementCount++;
    }
  }

  const Eigen::Matrix<double, 5, 1> x = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);

  //  //compute the risidual on the system to see if the algebraic error is too
  //  large.
  //  //note: maybe there is a better error metric.
  //  const Eigen::Vector<5> error =  A*x - b;
  //  residual = error*error;
  //  //residual/=elementCount;

  Eigen::Matrix3d C_star_norm;
  C_star_norm << x[0], x[1] / 2.0, x[3] / 2.0, x[1] / 2.0, x[2], x[4] / 2.0, x[3] / 2.0, x[4] / 2.0, 1.0;

  //    const Eigen::Matrix3d C = Hinv.transpose() * C_star_norm.inverse() *
  //    Hinv;
  Eigen::Matrix3d C = C_star_norm.inverse();
  //  const Matrix3d C_star = LU<3>(C).get_inverse();
  //  return C_star/C_star[2][2];

  return C;  // C/C(2,2);
}

}  // namespace DotDetect
}  // namespace dso