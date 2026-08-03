//
// Created by zk on 24-4-19.
//

#ifndef YVR_CALIB_TARGETGRIDINFO_H
#define YVR_CALIB_TARGETGRIDINFO_H

#include "BasicStruct.h"
#include "RandomGrid.h"
#include <Eigen/Core>
namespace dso::DotDetect {
struct Rectangle {
  int top;
  int left;
  int height;
  int width;
};

class TargetGridInfo {
 private:
  enum ROTMAT { R_0 = 0, R_90, R_180, R_270 };
  std::array<Eigen::MatrixXi, 4> PG_;

 private:
  static Eigen::MatrixXi rotateMatrix(const Eigen::MatrixXi& matrix, const ROTMAT& rot) {
    Eigen::MatrixXi rotMatrix;
    switch (rot) {
      case R_0:
        rotMatrix = matrix;
        break;
      case R_90:
        rotMatrix = matrix.transpose().colwise().reverse();
        break;
      case R_180:
        rotMatrix = matrix.colwise().reverse().rowwise().reverse();
        break;
      case R_270:
        rotMatrix = matrix.colwise().reverse().transpose();
        break;
    }
    return rotMatrix;
  }

  bool UniqueMatch(const Eigen::MatrixXi& m, int& idg, int& dr, int& dc, int& plateId);
  bool matchRectangle(const Eigen::MatrixXi& smallMatrix, int& sdr, int& sdc, int& idg, int& dr, int& dc, int& plateId);

  void initGrid(uint32_t seed) {  // Create binary pattern (and rotated pattern) from seed
    PG_ = MakePatternGroup(grid_size_(1), grid_size_(0), seed);
    Eigen::MatrixXi labelMatrix = PG_[0];
    for (int row = 0; row < labelMatrix.rows(); ++row) {
      for (int col = 0; col < labelMatrix.cols(); ++col) {
        labelMatrix(row, col) = row * (int)labelMatrix.cols() + col;
      }
    }

    for (int plateId = 0; plateId < plate_num_; ++plateId) {
      std::array<Eigen::MatrixXi, 4> onePlate, subLabelMatrix;
      for (int i = 0; i < 4; ++i) {
        Eigen::MatrixXi subMatrix = PG_[0].block(grid_size_(0) * plateId, 0, grid_size_(0), grid_size_(0));
        Eigen::MatrixXi curLabel = labelMatrix.block(grid_size_(0) * plateId, 0, grid_size_(0), grid_size_(0));
        onePlate[i] = rotateMatrix(subMatrix, ROTMAT(i));
        subLabelMatrix[i] = rotateMatrix(curLabel, ROTMAT(i));
      }
      Plate_PG_.emplace(plateId, onePlate);
      Plate_PG_Label.emplace(plateId, subLabelMatrix);
    }
  }

 public:
  explicit TargetGridInfo(double grid_spacing, const Eigen::Vector2i& grid_size, uint32_t seed, int plate_num,
                          int unique_size) {
    grid_spacing_ = grid_spacing;
    grid_size_ = grid_size;
    plate_num_ = plate_num;
    uniqueSize = unique_size;
    initGrid(seed);
  }

  bool Match(std::map<Eigen::Vector2i const, Vertex*, lessCompare,
                      Eigen::aligned_allocator<std::pair<Eigen::Vector2i const, DotDetect::Vertex*>>>& obs,
             int& plateId, int& Plate_PG_idx);

 public:
  int plate_num_;
  double grid_spacing_;
  int uniqueSize = 6;
  std::unordered_map<int, std::array<Eigen::MatrixXi, 4>> Plate_PG_Label;
  std::unordered_map<int, std::array<Eigen::MatrixXi, 4>> Plate_PG_;
  Eigen::Vector2i grid_size_;
};

}  // namespace dso::DotDetect

#endif  // YVR_CALIB_TARGETGRIDINFO_H
