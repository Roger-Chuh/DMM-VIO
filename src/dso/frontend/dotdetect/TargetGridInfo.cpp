//#define DEBUGDOT

#include "TargetGridInfo.h"

namespace dso::DotDetect {

template <typename Comparator>
bool maxRectangle(const Eigen::MatrixXi& matrix, int minSize, Rectangle& result, const Comparator& compare) {
  int m = (int)matrix.rows();
  int n = (int)matrix.cols();

  int maxArea = 0;
  Rectangle maxRect = {0, 0, 0, 0};
  bool found = false;

  // dp(i, j) stores the number of continuous zeros ending at (i, j) vertically
  Eigen::MatrixXi dp = Eigen::MatrixXi::Zero(m, n);

  // Initialize dp matrix for the first row
  for (int j = 0; j < n; ++j) {
    dp(0, j) = compare(matrix(0, j)) ? 1 : 0;
    if (dp(0, j) >= minSize) {
      maxArea = dp(0, j);
      maxRect = {0, j, 1, 1};
      found = true;
    }
  }

  // Fill the dp matrix and update maxZeroRect
  for (int i = 1; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      dp(i, j) = compare(matrix(i, j)) ? dp(i - 1, j) + 1 : 0;

      int minHeight = dp(i, j);
      for (int k = j; k >= 0 && dp(i, k) > 0; --k) {
        minHeight = std::min(minHeight, dp(i, k));
        int width = j - k + 1;
        int area = width * minHeight;
        if (area >= minSize * minSize && area > maxArea) {
          maxArea = area;
          maxRect = {i - minHeight + 1, k, minHeight, width};
          found = true;
        }
      }
    }
  }

  if (found) {
    result = maxRect;
  }

  return found;
}

bool TargetGridInfo::matchRectangle(const Eigen::MatrixXi& smallMatrix, int& sdr, int& sdc, int& idg, int& dr, int& dc,
                                    int& plateId) {
  double minDiffValue = std::numeric_limits<double>::max();
  for (const std::pair<const int, std::array<Eigen::MatrixXi, 4>>& onePlate : Plate_PG_) {
    for (int i = 0; i < onePlate.second.size(); ++i) {
      for (int idx = 0; idx + smallMatrix.rows() <= onePlate.second[i].rows(); ++idx) {
        for (int idy = 0; idy + smallMatrix.cols() <= onePlate.second[i].cols(); ++idy) {
          const Eigen::MatrixXi& diffMatrix =
              onePlate.second[i].block(idx, idy, smallMatrix.rows(), smallMatrix.cols()) - smallMatrix;
          double diffValue = (diffMatrix).cwiseAbs().sum();
          if (diffValue < minDiffValue) {
            minDiffValue = diffValue;
            idg = i;
            dr = idx - sdr;
            dc = idy - sdc;
            plateId = onePlate.first;
          }
        }
      }
    }
  }
#ifdef DEBUGDOT
  std::cerr << "minDiffValue:" << minDiffValue << " plate id:" << plateId << " idg: " << idg << std::endl;
  std::cerr << "smallMatrix:\n" << smallMatrix << std::endl;
  std::cerr << "diff mat:\n"
            << Plate_PG_.at(plateId)[idg].block(dr + sdr, dc + sdc, smallMatrix.rows(), smallMatrix.cols()) -
                   smallMatrix
            << std::endl;
#endif

  // find global min
  if (minDiffValue < 0.05 * (float)smallMatrix.cols() * (float)smallMatrix.rows()) {  // 0.05: inlier point ratio
    if (minDiffValue < 0.5) return true;
    Rectangle temp{};
    const Eigen::MatrixXi diffMatrix =
        Plate_PG_.at(plateId)[idg].block(dr + sdr, dc + sdc, smallMatrix.rows(), smallMatrix.cols()) - smallMatrix;
    return maxRectangle(diffMatrix, uniqueSize, temp, compareMatZero());
  }

  return false;
}

bool TargetGridInfo::UniqueMatch(const Eigen::MatrixXi& m, int& idg, int& dr, int& dc, int& plateId) {
  int cols = (int)m.cols();
  int rows = (int)m.rows();
  if (cols < uniqueSize || rows < uniqueSize) return false;
  Rectangle maxRec{};
  if (maxRectangle(m, uniqueSize, maxRec, compareMat01())) {
    const Eigen::MatrixXi& smallMatrix = m.block(maxRec.top, maxRec.left, maxRec.height, maxRec.width);
    return matchRectangle(smallMatrix, maxRec.top, maxRec.left, idg, dr, dc, plateId);
  }
#ifdef DEBUGDOT
  std::cerr << "match falied" << std::endl;
#endif
  return false;
}

bool TargetGridInfo::Match(std::map<Eigen::Vector2i const, Vertex*, lessCompare,
                                    Eigen::aligned_allocator<std::pair<Eigen::Vector2i const, Vertex*>>>& obs,
                           int& plateId, int& Plate_PG_idx) {
  Eigen::Vector2i omin(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
  Eigen::Vector2i omax(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());

  // find max and min
  for (auto& ob : obs) {
    omin[0] = std::min(omin[0], ob.first[0]);
    omin[1] = std::min(omin[1], ob.first[1]);
    omax[0] = std::max(omax[0], ob.first[0]);
    omax[1] = std::max(omax[1], ob.first[1]);
  }

  // Create sample matrix
  Eigen::Vector2i osize = (omax + Eigen::Vector2i(1, 1)) - omin;

  if (osize[0] > 5 && osize[1] > 5) {
    Eigen::MatrixXi m = Eigen::MatrixXi::Constant(osize(1), osize(0), -1);
    int num_valid = 0;
    for (auto& ob : obs) {
      const Eigen::Vector2i pg = ob.first - omin;
      ob.second->pg = pg;
      const int val = ob.second->value;
      m(pg(1), pg(0)) = val;
      if (val >= 0) ++num_valid;
    }
    if (num_valid < 25) return false;

#ifdef DEBUGDOT
    std::cerr << "target_m: cols:" << m.cols() << " rows:" << m.rows() << std::endl;
    std::cerr << "target_m: \n" << m << std::endl;
#endif
    // Match methods
    int dr, dc;
    bool res = UniqueMatch(m, Plate_PG_idx, dr, dc, plateId);
    if (res) {
#ifdef DEBUGDOT
      std::cerr << "dr:" << dr << " dc:" << dc << std::endl;
#endif
      Eigen::Vector2i move(dc, dr);

      for (auto& ob : obs) {
        Eigen::Vector2i originPattern = ob.second->pg + move;
        if (ob.second->value >= 0 && originPattern.y() >= 0 &&
            originPattern.y() < Plate_PG_.at(plateId)[Plate_PG_idx].rows() && originPattern.x() >= 0 &&
            originPattern.x() < Plate_PG_.at(plateId)[Plate_PG_idx].cols() && ob.second->pg(1) >= 0 &&
            ob.second->pg(0) >= 0 && ob.second->pg(1) < m.rows() && ob.second->pg(0) < m.cols() &&
            ob.second->pg(1) < Plate_PG_.at(plateId)[Plate_PG_idx].rows() &&
            ob.second->pg(0) < Plate_PG_.at(plateId)[Plate_PG_idx].cols() &&
            m(ob.second->pg(1), ob.second->pg(0)) ==
                Plate_PG_.at(plateId)[Plate_PG_idx](originPattern.y(), originPattern.x())) {
          ob.second->pg = originPattern;  // label idx
        } else {
          ob.second->pg.x() = GRID_INVALID;
          ob.second->pg.y() = GRID_INVALID;
        }
      }
      return true;
    } else {
      //      std::cerr << "not found Match matrix " << std::endl;
      return false;
    }

  } else {
    //    std::cerr << "Grid too small, ";
  }
  return false;
}
}  // namespace dso::DotDetect