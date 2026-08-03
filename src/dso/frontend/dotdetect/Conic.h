#pragma once

#include "../../camera_model//vio_def.h"
#include "Label.h"
#include "Rectangle.h"
#include <array>
#include <memory>
#include <queue>
#include <set>
#include <vector>

namespace dso::DotDetect {

struct Conic {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  IRectangle bbox;

  // C=
  //  A   B/2   D/2
  //  B/2   C    E/2
  //  D/2  E/2    F
  // A*x^2 + B*y^2 + C*xy + D*x + E*y + F = 0
  // quadratic form: x'*C*x = 0 with x = (x1,x2,1) are points on the ellipse
  Mat3 C;

  // l:=C*x is tangent line throught the point x. The dual of C is adj(C).
  // For lines through C it holds: l'*adj(C)*l = 0.
  // If C has full rank it holds up to scale: adj(C) = C^{-1}
  Mat3 Dual;

  // center (c1,c2)
  Vec2 center;

  int label = -1;
  double len = 0;
  Vec3 pos;
};

template <typename TdI>
Mat3 FindEllipse(const int w, const int /*h*/, const TdI* dI, const std::set<int>& pixels, double& /*residual*/) {
  // Precise ellipse estimation without contour point extraction
  // Jean-Nicolas Ouellet, Patrick Hebert

  // This normalisation is in the wrong space.
  // We'd be better off normalising image values to [0,1] range.

  // Form system Ax = b to solve
  Mat5 A = Mat5::Zero();
  Vec5 b = Vec5::Zero();

  //    float elementCount = 0;
  for (const int& pixel_id : pixels) {
    int col = pixel_id % w;
    int row = pixel_id / w;
    const TdI* dIv = dI + row * w;
    const Vec3 d = Vec3(dIv[col][0], dIv[col][1], -(dIv[col][0] * col + dIv[col][1] * row));
    //            const Eigen::Vector3d li = //H.T() * d;
    //                    Eigen::Vector3d( d[0]*H(0,0), d[1]*H(1,1), d[0]*H(0,2)
    //                    + d[1] * H(1,2) + d[2] );
    const Vec3& li = d;
    Vec5 Ki;
    Ki << li[0] * li[0], li[0] * li[1], li[1] * li[1], li[0] * li[2], li[1] * li[2];
    A += Ki * Ki.transpose();
    b += -Ki * li[2] * li[2];
  }

  Vec5 x = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);

  //  //compute the risidual on the system to see if the algebraic error is too
  //  large.
  //  //note: maybe there is a better error metric.
  //  const Eigen::Vector<5> error =  A*x - b;
  //  residual = error*error;
  //  //residual/=elementCount;

  Mat3 C_star_norm;
  C_star_norm << x[0], x[1] / 2.0, x[3] / 2.0, x[1] / 2.0, x[2], x[4] / 2.0, x[3] / 2.0, x[4] / 2.0, 1.0;

  //    const Eigen::Matrix3d C = Hinv.transpose() * C_star_norm.inverse() *
  //    Hinv;
  Mat3 C = C_star_norm.inverse();
  //  const Matrix3d C_star = LU<3>(C).get_inverse();
  //  return C_star/C_star[2][2];

  return C;  // C/C(2,2);
}

template <typename TdI>
void FindConics(const int w, const int h, std::vector<PixelClass>& candidates, const TdI* dI,
                std::vector<Conic>& conics, bool black_on_white) {
  const int dx[4] = {-1, 1, 0, 0};
  const int dy[4] = {0, 0, -1, 1};

  for (auto& candidate : candidates) {
    if (candidate.is_near_boarder) {
      continue;
    }
    const IRectangle& region = candidate.bbox;

    Conic conic;
    double residual = 0;
    // add inlier pixel around cluster
    candidate.cluster_center.setZero();
    candidate.ellipse_pixel_set.clear();
    std::queue<int> search_pixel_queue;
    for (const auto& pixel_id : candidate.cluster_pixel_set) {
      search_pixel_queue.emplace(pixel_id);
      candidate.ellipse_pixel_set.insert(pixel_id);
      Vec2 uv(pixel_id % w, pixel_id / w);
      candidate.cluster_center += uv;
    }
    candidate.cluster_center /= static_cast<double>(candidate.cluster_pixel_set.size());

    while (!search_pixel_queue.empty()) {
      int col = search_pixel_queue.front() % w;
      int row = search_pixel_queue.front() / w;
      for (int i = 0; i < 4; i++) {
        int nextCol = col + dx[i];
        int nextRow = row + dy[i];
        int pixel_id = nextCol + w * nextRow;
        if (nextRow >= candidate.bbox.y1 && nextRow <= candidate.bbox.y2 && nextCol >= candidate.bbox.x1 &&
            nextCol <= candidate.bbox.x2 &&
            candidate.ellipse_pixel_set.find(pixel_id) == candidate.ellipse_pixel_set.end()) {
          const TdI* dIv = dI + row * w;
          Vec2 gradient_dir(dIv[col][0], dIv[col][1]);
          Vec2 ratio_to_center;
          if (gradient_dir.norm() < 5) {
            continue;
          }
          if (black_on_white) {
            ratio_to_center = candidate.cluster_center - Vec2(col, row);
          } else {
            ratio_to_center = Vec2(col, row) - candidate.cluster_center;
          }
          gradient_dir.normalize();
          ratio_to_center.normalize();
          if (gradient_dir.dot(ratio_to_center) > 0.173648178 /* 80 degree */) {
            candidate.ellipse_pixel_set.insert(pixel_id);
            search_pixel_queue.push(pixel_id);
          }
        }
      }
      search_pixel_queue.pop();
    }

    // remove outlier pixel in cluster
    for (int pixel_id : candidate.cluster_pixel_set) {
      if (candidate.ellipse_pixel_set.find(pixel_id) == candidate.ellipse_pixel_set.end()) {
        continue;
      }
      int col = pixel_id % w;
      int row = pixel_id / w;
      const TdI* dIv = dI + row * w;
      Vec2 gradient_dir(dIv[col][0], dIv[col][1]);
      if (gradient_dir.norm() < 5) {
        candidate.ellipse_pixel_set.erase(pixel_id);
        continue;
      }
      Vec2 ratio_to_center;
      if (black_on_white) {
        ratio_to_center = candidate.cluster_center - Vec2(col, row);
      } else {
        ratio_to_center = Vec2(col, row) - candidate.cluster_center;
      }
      gradient_dir.normalize();
      ratio_to_center.normalize();
      if (gradient_dir.dot(ratio_to_center) <= 0.7 /* 45 degree */) {
        candidate.ellipse_pixel_set.erase(pixel_id);
      }
    }
    conic.C = FindEllipse(w, h, dI, candidate.ellipse_pixel_set, residual);
    //    conic.C = FindEllipse(w, h, dI, region, residual);
    conic.bbox = region;
    conic.Dual = conic.C.inverse();
    conic.Dual /= conic.Dual(2, 2);
    conic.center = Vec2(conic.Dual(0, 2), conic.Dual(1, 2));

    conic.len = std::max(region.Width(), region.Height());

    const double max_dist = (std::min(region.Width(), region.Height())) / 4.0;
    if ((conic.center - region.Center()).norm() < max_dist) conics.push_back(conic);
  }
};

}  // namespace dso::DotDetect
