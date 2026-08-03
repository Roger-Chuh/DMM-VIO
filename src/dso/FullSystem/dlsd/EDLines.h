/**************************************************************************************************************
 * EDLines source codes.
 * Copyright (C) Cuneyt Akinlar & Cihan Topal
 * E-mails of the authors: cuneytakinlar@gmail.com, cihantopal@gmail.com
 *
 * Please cite the following papers if you use EDLines library:
 *
 * [1] C. Akinlar and C. Topal, “EDLines: A Real-time Line Segment Detector with
 *a False Detection Control,” Pattern Recognition Letters, 32(13), 1633-1642,
 *DOI: 10.1016/j.patrec.2011.06.001 (2011).
 *
 * [2] C. Akinlar and C. Topal, “EDLines: Realtime Line Segment Detection by
 *Edge Drawing (ED),” IEEE Int’l Conf. on Image Processing (ICIP), Sep. 2011.
 **************************************************************************************************************/

#ifndef _EDLines_
#define _EDLines_

#include "ED.h"
#include "NFA.h"

#define SS 0
#define SE 1
#define ES 2
#define EE 3

namespace dso {
// light weight struct for Start & End coordinates of the line segment
struct LS {
  cv::Point2d start;
  cv::Point2d end;

  LS(cv::Point2d _start, cv::Point2d _end) {
    start = _start;
    end = _end;
  }
};

struct LineSegment {
  number_t a, b; // y = a + bx (if invert = 0) || x = a + by (if invert = 1)
  int invert;

  number_t sx, sy; // starting x & y coordinates
  number_t ex, ey; // ending x & y coordinates

  int segmentNo;       // Edge segment that this line belongs to
  int firstPixelIndex; // Index of the first pixel within the segment of pixels
  int len;             // No of pixels making up the line segment

  LineSegment(number_t _a, number_t _b, int _invert, number_t _sx, number_t _sy,
              number_t _ex, number_t _ey, int _segmentNo, int _firstPixelIndex,
              int _len) {
    a = _a;
    b = _b;
    invert = _invert;
    sx = _sx;
    sy = _sy;
    ex = _ex;
    ey = _ey;
    segmentNo = _segmentNo;
    firstPixelIndex = _firstPixelIndex;
    len = _len;
  }
};

class EDLines : public ED {
public:
  EDLines(cv::Mat srcImage, number_t _line_error = 1.0, int _min_line_len = -1,
          number_t _max_distance_between_two_lines = 6.0,
          number_t _max_error = 1.3);
  EDLines(ED obj, number_t _line_error = 1.0, int _min_line_len = -1,
          number_t _max_distance_between_two_lines = 6.0,
          number_t _max_error = 1.3);
  EDLines() = delete;

  std::vector<LineSegment> getLines();
  int getLinesNo();
  cv::Mat getLineImage();
  cv::Mat drawOnImage();

  // EDCircle uses this one
  static void SplitSegment2Lines(number_t *x, number_t *y, int noPixels,
                                 int segmentNo, std::vector<LineSegment> &lines,
                                 int min_line_len = 6,
                                 number_t line_error = 1.0);

private:
  std::vector<LineSegment> lines;
  std::vector<LineSegment> invalidLines;
  std::vector<LS> linePoints;
  int linesNo;
  int min_line_len;
  number_t line_error;
  number_t max_distance_between_two_lines;
  number_t max_error;
  number_t prec;
  NFALUT *nfa;

  int ComputeMinLineLength();
  void SplitSegment2Lines(number_t *x, number_t *y, int noPixels,
                          int segmentNo);
  void JoinCollinearLines();

  void ValidateLineSegments();
  bool ValidateLineSegmentRect(int *x, int *y, LineSegment *ls);
  bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2,
                                int changeIndex);

  static number_t ComputeMinDistance(number_t x1, number_t y1, number_t a,
                                     number_t b, int invert);
  static void ComputeClosestPoint(number_t x1, number_t y1, number_t a,
                                  number_t b, int invert, number_t &xOut,
                                  number_t &yOut);
  static void LineFit(number_t *x, number_t *y, int count, number_t &a,
                      number_t &b, int invert);
  static void LineFit(number_t *x, number_t *y, int count, number_t &a,
                      number_t &b, number_t &e, int &invert);
  static number_t ComputeMinDistanceBetweenTwoLines(LineSegment *ls1,
                                                    LineSegment *ls2,
                                                    int *pwhich);
  static void UpdateLineParameters(LineSegment *ls);
  static void EnumerateRectPoints(number_t sx, number_t sy, number_t ex,
                                  number_t ey, int ptsx[], int ptsy[],
                                  int *pNoPoints);

  // Utility math functions
};
} // namespace dso

#endif
