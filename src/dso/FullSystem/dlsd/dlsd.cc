
#include "dlsd.h"

// stl
#include "ED.h"
#include "EDLines.h"
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../../camera_model/camera_base.h"
#include "../util/NumType.h"
#include "line_fit.h"
namespace dso {
struct GapSegment {
  // TODO segmentNo can be ommited
  int segmentNo;       // Edge segment that this gap belongs to
  int firstPixelIndex; // Index of the first pixel of this gap
  int len;             // No of pixels making up the gap segment

  GapSegment() : segmentNo(0), firstPixelIndex(0), len(0) {}

  GapSegment(int _segmentNo, int _firstPixelIndex, int _len)
      : segmentNo(_segmentNo), firstPixelIndex(_firstPixelIndex), len(_len) {}
};

struct PolygonalChain {
  int turn;      // Turn direction: 1 or -1
  int segmentNo; // Edge segment that this polygonal path belongs to

  std::vector<int> lines;
  std::vector<GapSegment> gaps;
};

//-----------------------------------------------------------------------------------------
// Computes the minimum line length using the NFA formula given width & height
// values
int ComputeMinLineLength(int width, int height) {
  // The reason we are dividing the theoretical minimum line length by 2 is
  // because we now test short line segments by a line support region rectangle
  // having width=2. This means that within a line support region rectangle for
  // a line segment of length "l" there are "2*l" many pixels. Thus, a line
  // segment of length "l" has a chance of getting validated by NFA.

  const number_t logNT =
      2.0 * (std::log10((number_t)width) + std::log10((number_t)height));
  return (int)std::round((-logNT / std::log10(0.125)) * 0.5);
} // end-ComputeMinLineLength

//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
//
void LineFit(number_t *x, number_t *y, int count, number_t &a, number_t &b,
             number_t &e, int &invert) {
  if (count < 2)
    return;

  number_t S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
  for (int i = 0; i < count; i++) {
    Sx += x[i];
    Sy += y[i];
  } // end-for

  number_t mx = Sx / count;
  number_t my = Sy / count;

  number_t dx = 0.0;
  number_t dy = 0.0;
  for (int i = 0; i < count; i++) {
    dx += (x[i] - mx) * (x[i] - mx);
    dy += (y[i] - my) * (y[i] - my);
  } // end-for

  if (dx < dy) {
    // Vertical line. Swap x & y, Sx & Sy
    invert = 1;
    number_t *t = x;
    x = y;
    y = t;

    number_t d = Sx;
    Sx = Sy;
    Sy = d;

  } else {
    invert = 0;
  } // end-else

  // Now compute Sxx & Sxy
  for (int i = 0; i < count; i++) {
    Sxx += x[i] * x[i];
    Sxy += x[i] * y[i];
  } // end-for

  number_t D = S * Sxx - Sx * Sx;
  a = (Sxx * Sy - Sx * Sxy) / D;
  b = (S * Sxy - Sx * Sy) / D;

  if (b == 0.0) {
    // Vertical or horizontal line
    number_t error = 0.0;
    for (int i = 0; i < count; i++) {
      error += fabs((a)-y[i]);
    } // end-for
    e = error / count;

  } else {
    number_t error = 0.0;
    for (int i = 0; i < count; i++) {
      // Let the line passing through (x[i], y[i]) that is perpendicular to a+bx
      // be c+dx
      number_t d = -1.0 / (b);
      number_t c = y[i] - d * x[i];
      number_t x2 = ((a)-c) / (d - (b));
      number_t y2 = (a) + (b)*x2;

      number_t dist = (x[i] - x2) * (x[i] - x2) + (y[i] - y2) * (y[i] - y2);
      error += dist;
    } // end-for

    e = std::sqrt(error / count);
  } // end-else
}

//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
// Assumes that the direction of the line is known by a previous computation
//
void LineFit(number_t *x, number_t *y, int count, number_t &a, number_t &b,
             int invert) {
  if (count < 2)
    return;

  number_t S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
  for (int i = 0; i < count; i++) {
    Sx += x[i];
    Sy += y[i];
  } // end-for

  if (invert) {
    // Vertical line. Swap x & y, Sx & Sy
    number_t *t = x;
    x = y;
    y = t;

    number_t d = Sx;
    Sx = Sy;
    Sy = d;
  } // end-if

  // Now compute Sxx & Sxy
  for (int i = 0; i < count; i++) {
    Sxx += x[i] * x[i];
    Sxy += x[i] * y[i];
  } // end-for

  number_t D = S * Sxx - Sx * Sx;
  a = (Sxx * Sy - Sx * Sxy) / D;
  b = (S * Sxy - Sx * Sy) / D;
}

number_t ComputeMinDistance(number_t x1, number_t y1, number_t a, number_t b,
                            int invert) {
  number_t x2, y2;

  if (invert == 0) {
    if (b == 0) {
      x2 = x1;
      y2 = a;

    } else {
      // Let the line passing through (x1, y1) that is perpendicular to a+bx be
      // c+dx
      number_t d = -1.0 / (b);
      number_t c = y1 - d * x1;

      x2 = (a - c) / (d - b);
      y2 = a + b * x2;
    } // end-else

  } else {
    /// invert = 1
    if (b == 0) {
      x2 = a;
      y2 = y1;

    } else {
      // Let the line passing through (x1, y1) that is perpendicular to a+by be
      // c+dy
      number_t d = -1.0 / (b);
      number_t c = x1 - d * y1;

      y2 = (a - c) / (d - b);
      x2 = a + b * y2;
    } // end-else
  }   // end-else

  return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

//---------------------------------------------------------------------------------
// Given a point (x1, y1) and a line equation y=a+bx (invert=0) OR x=a+by
// (invert=1) Computes the (x2, y2) on the line that is closest to (x1, y1)
//
void ComputeClosestPoint(number_t x1, number_t y1, number_t a, number_t b,
                         int invert, number_t &xOut, number_t &yOut) {
  number_t x2, y2;

  if (invert == 0) {
    if (b == 0) {
      x2 = x1;
      y2 = a;

    } else {
      // Let the line passing through (x1, y1) that is perpendicular to a+bx be
      // c+dx
      number_t d = -1.0 / (b);
      number_t c = y1 - d * x1;

      x2 = (a - c) / (d - b);
      y2 = a + b * x2;
    } // end-else

  } else {
    /// invert = 1
    if (b == 0) {
      x2 = a;
      y2 = y1;

    } else {
      // Let the line passing through (x1, y1) that is perpendicular to a+by be
      // c+dy
      number_t d = -1.0 / (b);
      number_t c = x1 - d * y1;

      y2 = (a - c) / (d - b);
      x2 = a + b * y2;
    } // end-else
  }   // end-else

  xOut = x2;
  yOut = y2;
}

//-----------------------------------------------------------------
// Given a full segment of pixels, splits the chain to lines
// This code is used when we use the whole segment of pixels
//
void SplitSegment2Lines(number_t *x, number_t *y, int noPixels, int segmentNo,
                        std::vector<LineSegment> &lines,
                        int min_line_len = MIN_LINE_LENGTH,
                        number_t line_error = MAX_LINE_ERROR) {
  lines.clear();

  // First pixel of the line segment within the segment of points
  int firstPixelIndex = 0;

  while (noPixels >= min_line_len) {
    // Start by fitting a line to MIN_LINE_LEN pixels
    bool valid = false;
    number_t lastA, lastB, error;
    int lastInvert;

    while (noPixels >= min_line_len) {
      LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
      if (error <= 0.5) {
        valid = true;
        break;
      }

#if 1
      noPixels -= 1; // Go slowly
      x += 1;
      y += 1;
      firstPixelIndex += 1;
#else
      noPixels -= 2; // Go faster (for speed)
      x += 2;
      y += 2;
      firstPixelIndex += 2;
#endif
    } // end-while

    if (valid == false)
      return;

    // Now try to extend this line
    int index = min_line_len;
    int len = min_line_len;

    while (index < noPixels) {
      int startIndex = index;
      int lastGoodIndex = index - 1;
      int goodPixelCount = 0;
      int badPixelCount = 0;
      while (index < noPixels) {
        number_t d =
            ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

        if (d <= line_error) {
          lastGoodIndex = index;
          goodPixelCount++;
          badPixelCount = 0;

        } else {
          badPixelCount++;
          if (badPixelCount >= 5)
            break;
        } // end-if

        index++;
      } // end-while

      if (goodPixelCount >= 2) {
        len += lastGoodIndex - startIndex + 1;
        LineFit(x, y, len, lastA, lastB, lastInvert); // faster LineFit
        index = lastGoodIndex + 1;
      } // end-if

      if (goodPixelCount < 2 || index >= noPixels) {
        // End of a line segment. Compute the end points
        number_t sx, sy, ex, ey;

        int index = 0;
        while (ComputeMinDistance(x[index], y[index], lastA, lastB,
                                  lastInvert) > line_error)
          index++;
        ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx,
                            sy);
        int noSkippedPixels = index;

        index = lastGoodIndex;
        while (ComputeMinDistance(x[index], y[index], lastA, lastB,
                                  lastInvert) > line_error)
          index--;
        ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex,
                            ey);

        // Add the line segment to lines
        lines.push_back(LineSegment(
            lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo,
            firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
        // linesNo++;
        len = index + 1;
        break;
      } // end-else
    }   // end-while

    noPixels -= len;
    x += len;
    y += len;
    firstPixelIndex += len;
  } // end-while
}

// TODO description
void GroupLines2PolygonalChains(const number_t *x, const number_t *y,
                                int noPixels, int segmentNo,
                                const std::vector<LineSegment> &lines,
                                std::vector<PolygonalChain> &chains) {
  chains.clear();

  int noLines = lines.size();
  if (noLines == 0)
    return; // no lines, no polygons...

  int firstPixelIndex = 0;

  PolygonalChain chain; // current chain
  chain.turn = 0;       // 0: not set; 1: counter clockwise; -1: clockwise;
  chain.segmentNo = segmentNo;
  chain.lines.push_back(0);
  chain.gaps.emplace_back(segmentNo, firstPixelIndex,
                          lines[0].firstPixelIndex - firstPixelIndex); // pregap

  if (noLines == 1) {
    firstPixelIndex = lines[0].firstPixelIndex + lines[0].len;
    chain.gaps.emplace_back(segmentNo, firstPixelIndex,
                            noPixels - firstPixelIndex); // postgap
    return;
  }

  // start line grouping
  GapSegment lastGap;
  for (int index = 1; index < noLines; index++) {
    const LineSegment *l1 = &lines[index - 1];
    const LineSegment *l2 = &lines[index];

    // compute the angle between the lines & their turn direction
    number_t v1x = l1->ex - l1->sx;
    number_t v1y = l1->ey - l1->sy;
    number_t v1Len = std::sqrt(v1x * v1x + v1y * v1y);

    number_t v2x = l2->ex - l2->sx;
    number_t v2y = l2->ey - l2->sy;
    number_t v2Len = std::sqrt(v2x * v2x + v2y * v2y);

    // fill gap info between l1 and l2
    firstPixelIndex = l1->firstPixelIndex + l1->len;
    lastGap.segmentNo = segmentNo;
    lastGap.firstPixelIndex = firstPixelIndex;
    lastGap.len = l2->firstPixelIndex - firstPixelIndex;
    chain.gaps.push_back(lastGap); // copy

    number_t dotProduct = (v1x * v2x + v1y * v2y) / (v1Len * v2Len);
    if (dotProduct > 1.0)
      dotProduct = 1.0;
    else if (dotProduct < -1.0)
      dotProduct = -1.0;

    number_t angle = std::acos(dotProduct);
    int sign = (v1x * v2y - v2x * v1y) >= 0 ? 1 : -1; // cross product

    // LOG(INFO) << "Angle: " << RadToDeg(angle);
    // LOG(INFO) << "Sign: " << sign;

    if (angle <= SMOOTHNESS_TH && chain.turn == sign) {
      // LOG(INFO) << "append line to current polygonal chain";

      // continue polyginal chain
      chain.lines.push_back(index);
    } else if (angle <= SMOOTHNESS_TH && chain.turn == 0) {
      // LOG(INFO) << "start new polygonal chain";

      // new polygonal chain
      chain.turn = sign;
      chain.lines.push_back(index);
    } else { // angle > SMOOTHNESS_TH || (chain.turn != sign && chain.turn != 0)
      // LOG(INFO) << "end current polygonal chain";

      // save current polygonal chain
      chains.push_back(std::move(chain));

      // re-initialize current chain
      chain.turn = 0;
      chain.segmentNo = segmentNo;
      chain.lines.clear();
      chain.lines.push_back(index);
      chain.gaps.clear();
      chain.gaps.push_back(std::move(lastGap));
    }
  }

  // postgap
  firstPixelIndex = lines[noLines - 1].firstPixelIndex + lines[noLines - 1].len;
  lastGap.segmentNo = segmentNo;
  lastGap.firstPixelIndex = firstPixelIndex;
  lastGap.len = noPixels - firstPixelIndex;
  chain.gaps.push_back(std::move(lastGap));

  // check closed contour
  number_t dx = x[0] - x[noPixels - 1];
  number_t dy = y[0] - y[noPixels - 1];
  number_t d = std::sqrt(dx * dx + dy * dy);
  // LOG(INFO) << "Distance for closed contour: " << d;

  if (d < 1.5) { // 1.5 ~ sqrt(2)
    // closed
    const LineSegment *l1 = &lines[noLines - 1];
    const LineSegment *l2 = &lines[0];

    // compute the angle between the lines & their turn direction
    number_t v1x = l1->ex - l1->sx;
    number_t v1y = l1->ey - l1->sy;
    number_t v1Len = std::sqrt(v1x * v1x + v1y * v1y);

    number_t v2x = l2->ex - l2->sx;
    number_t v2y = l2->ey - l2->sy;
    number_t v2Len = std::sqrt(v2x * v2x + v2y * v2y);

    number_t dotProduct = (v1x * v2x + v1y * v2y) / (v1Len * v2Len);
    if (dotProduct > 1.0)
      dotProduct = 1.0;
    else if (dotProduct < -1.0)
      dotProduct = -1.0;

    number_t angle = std::acos(dotProduct);
    int sign = (v1x * v2y - v2x * v1y) >= 0 ? 1 : -1; // cross product

    // LOG(INFO) << "Angle: " << RadToDeg(angle);
    // LOG(INFO) << "Sign: " << sign;

    if (angle <= SMOOTHNESS_TH && (chain.turn == sign || chain.turn == 0)) {
      // LOG(INFO) << "append line to first polygonal chain";

      // extend first polyginal chain
      chain.turn = sign;
      chains[0].turn = sign;

      chain.lines.insert(chain.lines.end(), chains[0].lines.begin(),
                         chains[0].lines.end());
      chain.gaps.insert(chain.gaps.end(), chains[0].gaps.begin(),
                        chains[0].gaps.end());
      chains[0].lines = std::move(chain.lines);
      chains[0].gaps = std::move(chain.gaps);

      chain.lines.clear();
    }
  }

  if (!chain.lines.empty())             // unsaved chain
    chains.push_back(std::move(chain)); // save current polygonal chain
}

void ComputeClosestPoint(number_t x, number_t y, MultiCamera *p_multi_camera,
                         const int &cid, const Vec3 &axis, number_t &xOut,
                         number_t &yOut) {
  Vec3 p;
  // CameraModel::ImageToWorld(params, x, y, &p.x(), &p.y());
  p_multi_camera->cid_to_cam.at(cid)->UnProject(Vec2(x, y), p);
  // p.z() = 1.;
  p /= p(2);

  Vec3 v =
      p - p.dot(axis) * axis; // direction vector projected onto fitted plane
  v /= v.z();                 // normalized image plane projection

  // CameraModel::WorldToImage(params, v.x(), v.y(), &xOut, &yOut);
  Vec2 p_out;
  p_multi_camera->cid_to_cam.at(cid)->Project(v, p_out);
  xOut = p_out(0);
  yOut = p_out(1);
}

// TODO Description
void ValidatePolygonalChain(
    const number_t *x, const number_t *y, int segmentNo,
    MultiCamera *p_multi_camera, const int &cid,
    const std::vector<PolygonalChain> &chains,
    const std::vector<LineSegment> &lines,
    std::vector<DistortedLineSegment> &distortedLineSegments,
    int min_single_line_len = MIN_SINGLE_LINE_LENGTH) {
  // Validation
  // Single lines (i.e. not assigned to a cluster) can be accepted without
  // validation Line fitting (3D) with line segment end points:
  //  1. Undistort end points
  //  2. Perform line fitting (3D)
  //  3. Project1 3D end points to the fitted plane
  //  4. Project1 "model" end points to image and accept if all end points are
  //  within a reprojection error

  for (const PolygonalChain &chain : chains) {
    int noLines = chain.lines.size();

    // filter out small single line polygonal chains
    if (noLines == 1 && lines[chain.lines[0]].len < min_single_line_len)
      continue;

    std::vector<Vec3> endpoints(2 * noLines);
    std::vector<Vec3> allLinePixels;
    for (int index = 0; index < noLines; index++) {
      const LineSegment &line = lines[chain.lines[index]];

      Vec3 s; // start point
      // CameraModel::ImageToWorld(params, line.sx, line.sy, &s.x(), &s.y());
      p_multi_camera->cid_to_cam.at(cid)->UnProject(Vec2(line.sx, line.sy), s);
      // s /= s(2);
      endpoints[2 * index] = s.normalized();

      Vec3 e; // end point
      // CameraModel::ImageToWorld(params, line.ex, line.ey, &e.x(), &e.y());
      p_multi_camera->cid_to_cam.at(cid)->UnProject(Vec2(line.ex, line.ey), e);
      // e /= e(2);
      endpoints[2 * index + 1] = e.normalized();

      if (noLines > 1) {
        for (int k = 0; k < line.len; k++) {
          Vec3 v;
          // CameraModel::ImageToWorld(params, x[line.firstPixelIndex + k],
          // y[line.firstPixelIndex + k], &v.x(), &v.y());
          p_multi_camera->cid_to_cam.at(cid)->UnProject(
              Vec2(x[line.firstPixelIndex + k], y[line.firstPixelIndex + k]),
              v);
          // v /= v(2);
          allLinePixels.push_back(v.normalized());
        }
      }
    }

    Vec3 axis;
    Vec2 p_out;
    if (noLines > 1) {
      bool success = line_fit(allLinePixels, axis);
      if (!success) {
        //        LOG(WARNING) << "Solver failed!";
        continue;
      }
    } else // Single line polygonal chains do not need validation
      axis = endpoints[0].cross(endpoints[1]).normalized();
    bool valid = true;
    if (noLines > 1) {
      for (int index = 0; index < noLines; index++) {
        number_t x, y;
        number_t dx, dy, d;
        const LineSegment &line = lines[chain.lines[index]];

        Vec3 s = endpoints[2 * index] -
                 endpoints[2 * index].dot(axis) *
                     axis; // start point projected onto fitted plane
        s /= s.z();        // normalized image plane projection
        // CameraModel::WorldToImage(params, s.x(), s.y(), &x, &y);
        p_multi_camera->cid_to_cam.at(cid)->Project(s, p_out);
        x = p_out(0);
        y = p_out(1);
        dx = line.sx - x;
        dy = line.sy - y;
        d = std::sqrt(dx * dx + dy * dy);

        if (d > REPROJECTION_TH) {
          valid = false;
          break;
        }

        Vec3 e = endpoints[2 * index + 1] -
                 endpoints[2 * index + 1].dot(axis) *
                     axis; // end point projected onto fitted plane
        e /= e.z();        // normalized image plane projection
        // CameraModel::WorldToImage(params, e.x(), e.y(), &x, &y);
        p_multi_camera->cid_to_cam.at(cid)->Project(e, p_out);
        x = p_out(0);
        y = p_out(1);
        dx = line.ex - x;
        dy = line.ey - y;
        d = std::sqrt(dx * dx + dy * dy);

        if (d > REPROJECTION_TH) {
          valid = false;
          break;
        }
      }
    }

    if (!valid)
      continue;

    DistortedLineSegment distortedLineSegment;
    distortedLineSegment.segmentNo = segmentNo;
    distortedLineSegment.axis = axis;

    // segment
    for (int line_index = 0; line_index <= noLines; line_index++) {
      // validate pregaps
      const GapSegment &gap = chain.gaps[line_index];
      bool validPixelInGap = false;
      int firstPixelIndex = 0;
      for (int k = 0; k < gap.len; k++) {
        const int index = gap.firstPixelIndex + k;

        number_t _x, _y;
        ComputeClosestPoint(x[index], y[index], p_multi_camera, cid, axis, _x,
                            _y);

        number_t dx = x[index] - _x;
        number_t dy = y[index] - _y;
        number_t d = std::sqrt(dx * dx + dy * dy);

        if (line_index == 0) { // pregap
          if (d > REPROJECTION_TH && validPixelInGap)
            distortedLineSegment.invalidPixels.insert(k - firstPixelIndex);
          else if (d <= REPROJECTION_TH) {
            validPixelInGap = true;
            firstPixelIndex = k;
            distortedLineSegment.segment.emplace_back(
                cv::Point(x[index], y[index]));
          }
        } else if (line_index == noLines) { // postgap
          if (d > REPROJECTION_TH)
            break;
          else
            distortedLineSegment.segment.emplace_back(
                cv::Point(x[index], y[index]));
        } else { // intra chain gaps
          if (d > REPROJECTION_TH)
            distortedLineSegment.invalidPixels.insert(k - firstPixelIndex);
          else
            distortedLineSegment.segment.emplace_back(
                cv::Point(x[index], y[index]));
        }
      }

      if (line_index < noLines) {
        // lines validated by default, so just add them to the distorted line
        // segment
        const LineSegment &line = lines[chain.lines[line_index]];
        for (int k = 0; k < line.len; k++)
          distortedLineSegment.segment.emplace_back(cv::Point(
              x[line.firstPixelIndex + k], y[line.firstPixelIndex + k]));
      }
    }

    distortedLineSegments.push_back(std::move(distortedLineSegment));
  }
}
#define DRAW
void dlsd(const cv::Mat &image, MultiCamera *p_multi_camera, const int &cid,
          std::vector<DistortedLineSegment> &distortedLineSegments,
          cv::Mat &edge) {
  distortedLineSegments.clear();

  int width = image.cols;
  int height = image.rows;

  // Temporary buffers used during line fitting
  number_t *x = new number_t[(width + height) * 8];
  number_t *y = new number_t[(width + height) * 8];

  ED ed =
      ED(image, SOBEL_OPERATOR, 5, 8, 1, 10, 1.0, true); // apply ED algorithm

  int segmentNo = 0;
#ifdef DRAW
  cv::Mat imageChains, imageColor;
  cv::cvtColor(image, imageColor, cv::COLOR_GRAY2RGB);
  cv::cvtColor(image, imageChains, cv::COLOR_GRAY2RGB);
#endif
  for (const std::vector<cv::Point> &segment : ed.getSortedSegments()) {
    int noPixels = segment.size();

    // CHECK_LT(segment.size(), (width + height) * 8);
    if (segment.size() >= (width + height) * 8) {
      printf("check size fail\n");
      std::exit(-1);
    }
    for (int k = 0; k < noPixels; k++) {
      x[k] = segment[k].x;
      y[k] = segment[k].y;
    }

    // SplitSegment2Lines
    std::vector<LineSegment> lines;
    SplitSegment2Lines(x, y, noPixels, segmentNo, lines, MIN_LINE_LENGTH,
                       MAX_LINE_ERROR);

#ifdef DRAW
    for (const LineSegment &line : lines) {
      cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
      for (int k = 0; k < line.len; k++)
        cv::circle(imageColor,
                   cv::Point2d(x[line.firstPixelIndex + k],
                               y[line.firstPixelIndex + k]),
                   2, color, -1);
    }
#endif

    std::vector<PolygonalChain> chains;
    GroupLines2PolygonalChains(x, y, noPixels, segmentNo, lines, chains);

#ifdef DRAW
    for (const PolygonalChain &chain : chains) {
      cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
      for (int index : chain.lines) {
        const LineSegment &line = lines[index];
        if (line.len < 30)
          continue;
        for (int k = 0; k < line.len; k++)
          cv::circle(imageChains,
                     cv::Point2d(x[line.firstPixelIndex + k],
                                 y[line.firstPixelIndex + k]),
                     2, color, -1);
        for (const GapSegment &gap : chain.gaps) {
          for (int k = 0; k < gap.len; k++)
            cv::circle(imageChains,
                       cv::Point2d(x[gap.firstPixelIndex + k],
                                   y[gap.firstPixelIndex + k]),
                       2, cv::Scalar(0, 200, 0), -1);
        }
      }
    }
#endif

    ValidatePolygonalChain(x, y, segmentNo, p_multi_camera, cid, chains, lines,
                           distortedLineSegments);

    segmentNo++;
  }
#ifdef DRAW
  cv::imshow("Lines", imageColor);
  cv::imshow("Polygonal chains", imageChains);
  cv::waitKey(0);
#endif
  delete[] x;
  delete[] y;
}
// void dlsd(const cv::Mat &image, MultiCamera *p_multi_camera,
// std::vector<DistortedLineSegment> &distortedLineSegments);
} // namespace dso
