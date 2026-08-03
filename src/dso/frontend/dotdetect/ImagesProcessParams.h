//
// Created by zk on 24-4-24.
//

#ifndef YVR_CALIB_DOTPARAMS_H
#define YVR_CALIB_DOTPARAMS_H
namespace dso::DotDetect {

struct ParamsImageProcessing {
  explicit ParamsImageProcessing(int minDotNum, int imgWidth)
      : at_threshold(0.7),
        at_window_ratio(10),
        black_on_white(true),
        conic_min_area(25),
        conic_max_area(4E4),
        conic_symmetry(0.1),
        conic_min_aspect(0.1),
        window_size(35),
        min_dot_num(minDotNum),
        unique_size(6),
        max_line_dist_ratio(0.3),
        max_norm_triple_area(0.05),
        min_cross_area(1.5),
        max_cross_area(9.0),
        cross_radius_ratio(0.058),
        cross_line_ratio(0.036) {
    window_size = imgWidth / at_window_ratio;
    if (window_size % 2 != 1) window_size += 1;
  }
  // detect conic params
  double at_threshold;
  int at_window_ratio;
  bool black_on_white;

  double conic_min_area;
  double conic_max_area;
  double conic_symmetry;
  double conic_min_aspect;

  int window_size;
  int min_dot_num;
  int unique_size;

  // match params
  double max_line_dist_ratio;
  double max_norm_triple_area;
  double min_cross_area;
  double max_cross_area;
  double cross_radius_ratio;
  double cross_line_ratio;
  bool skip_detection = false;
};
}  // namespace dso::DotDetect

#endif  // YVR_CALIB_DOTPARAMS_H
