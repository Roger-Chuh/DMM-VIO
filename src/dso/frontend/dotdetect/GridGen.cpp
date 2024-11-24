//
// Created by zk on 24-5-10.
//
#include "GridGen.h"
#include "RandomGrid.h"
namespace dso::DotDetect {

bool isFixAreaUnique(const std::array<Eigen::MatrixXi, 4> &PG, int min_r,
                     int min_c) {
  const Eigen::MatrixXi &M = PG[0];
  //  std::cerr<<M<<std::endl;
  for (int idr_a = 0; idr_a + min_r < M.rows(); ++idr_a) {
    for (int idc_a = 0; idc_a + min_c < M.cols(); ++idc_a) {
      const Eigen::MatrixXi m_a = M.block(idr_a, idc_a, min_r, min_c);

      for (int idr_b = idr_a + 1; idr_b + min_r < M.rows(); ++idr_b) {
        for (int idc_b = idc_a + 1; idc_b + min_c < M.cols(); ++idc_b) {
          const Eigen::MatrixXi m_b = M.block(idr_b, idc_b, min_r, min_c);

          // 旋转90度
          Eigen::MatrixXi rotated_90 = m_b.transpose().colwise().reverse();

          // 旋转180度
          Eigen::MatrixXi rotated_180 =
              m_b.colwise().reverse().rowwise().reverse();

          // 旋转270度
          Eigen::MatrixXi rotated_270 = m_b.colwise().reverse().transpose();

          double abs_sum0 = (m_a - m_b).array().abs().sum();
          double abs_sum1 = (m_a - rotated_90).array().abs().sum();
          double abs_sum2 = (m_a - rotated_180).array().abs().sum();
          double abs_sum3 = (m_a - rotated_270).array().abs().sum();
          double min_sum = std::min({abs_sum0, abs_sum1, abs_sum2, abs_sum3});

          //          std::cerr<<"a:\n"<<m_a<<std::endl;
          //          std::cerr<<"b:\n"<<m_b<<"\ndelta:"<<abs_sum0<<std::endl;
          //          std::cerr<<"b_90:\n"<<rotated_90<<"\ndelta:"<<abs_sum1<<std::endl;
          //          std::cerr<<"b_180:\n"<<rotated_180<<"\ndelta:"<<abs_sum2<<std::endl;
          //          std::cerr<<"b_270:\n"<<rotated_270<<"\ndelta:"<<abs_sum3<<std::endl;
          if (min_sum < 2.5)
            return false;
        }
      }
    }
  }
  return true;
}

bool Area9Same(const Eigen::MatrixXi &mat) {
  for (int row = 0; row < mat.rows() - 3; ++row) {
    for (int col = 0; col < mat.cols() - 3; ++col) {
      if (mat.block<3, 3>(row, col).norm() == 0 ||
          mat.block<3, 3>(row, col).cwiseAbs().sum() == 9)
        return false;
    }
  }
  return true;
}

int FindBestSeed(int r, int c, int uniquesize) {
  // unique seed: 63 85 147 156 163 200 218 237 286 354 368 410 423 448 513 559
  // 577 586 595 601 621 633 666 707 714 752
  int seedMax = 10000;
  std::vector<int> areaUniqueSeed;

  for (int seed = 0; seed < seedMax; ++seed) {
    std::array<Eigen::MatrixXi, 4> pattern = MakePatternGroup(r, c, seed);
    if (!Area9Same(pattern[0]))
      continue;
    bool curres = isFixAreaUnique(pattern, uniquesize, uniquesize);
    if (curres) {
      areaUniqueSeed.emplace_back(seed);
    }
  }

  std::cerr << "useful seed:" << std::endl;
  for (int oneSeed : areaUniqueSeed) {
    std::cerr << oneSeed << " ";
  }
  std::cerr << std::endl;

  if (!areaUniqueSeed.empty())
    return areaUniqueSeed[0];
  return -1;
}

void PrintPattern(const Eigen::MatrixXi &M) {
  std::cerr << "rows: " << M.rows() << " cols:" << M.cols() << std::endl;
  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      const int v = M(r, c);
      const char b = (v == -1) ? 'x' : '0' + v;
      std::cerr << b << ' ';
    }
    std::cerr << std::endl;
  }
}
} // namespace dso::DotDetect