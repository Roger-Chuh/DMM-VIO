//
// Created by root on 2024/11/16.
//

#ifndef DMVIO_VIO_DEF_H
#define DMVIO_VIO_DEF_H
#include "../util/settings.h"
#include "stdlib.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <assert.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace dso {
#define LinearAlgebraLib Eigen
using number_t = double;
// static constexpr number_t pattern_raw_def[][2] = {
//    {0, 0}, {-2, -2}, {2, -2}, {-2, 2}, {0, -4}, {0, 4}, {-4, 0}, {4, 0}};
static constexpr number_t pattern_raw_def[][2] = {
    /*{+4, 4}*/ {-0, -0},
    {-4, -4},
    {-4, -2},
    {-4, -0},
    {-4, 2},
    {-4, 4},
    {-2, -4},
    {-2, -2},
    {-2, -0},
    {-2, 2},
    {-2, 4}, // full-45-SPREAD
    {-0, -4},
    {-0, -2},
    {-0, 2},
    {-0, 4},
    {+2, -4},
    {+2, -2},
    {+2, -0},
    {+2, 2},
    {+2, 4},
    {+4, -4},
    {+4, -2},
    {+4, -0},
    {+4, 2},
    /*{-0, -0}*/ {+4, 4}};
static constexpr int PATTERN_SIZE_def =
    sizeof(pattern_raw_def) / (2 * sizeof(number_t));
using Mat2Patch = LinearAlgebraLib::Matrix<number_t, 2, PATTERN_SIZE_def>;
static const Mat2Patch pattern2_def =
    Eigen::Map<Mat2Patch>((number_t *)pattern_raw_def);
using CamId = uint8_t;
const size_t kInvalid = std::numeric_limits<size_t>::max();
const size_t kInvalid_uint64_t = std::numeric_limits<uint64_t>::max();
const uint8_t kInvalid_uint8_t = std::numeric_limits<uint8_t>::max();

enum DirectFactorRes { kInlier, kOutlier, kOOB, kWithoutSigma };

constexpr int kMaxIntrSize = 38;

// Template Matrix Define //  // for template function
template <typename T> using Matrix3 = LinearAlgebraLib::Matrix<T, 3, 3>;
template <typename T> using Matrx3x2 = LinearAlgebraLib::Matrix<T, 3, 2>;
template <typename T> using Matrix4 = LinearAlgebraLib::Matrix<T, 4, 4>;
template <typename T> using Matrix6x3 = LinearAlgebraLib::Matrix<T, 6, 3>;

template <typename T> using Vector3 = LinearAlgebraLib::Matrix<T, 3, 1>;
template <typename T> using Vector6 = LinearAlgebraLib::Matrix<T, 6, 1>;

using Mat4 = LinearAlgebraLib::Matrix<number_t, 4, 4>;
using Mat36 = LinearAlgebraLib::Matrix<number_t, 3, 6>;
using Vec2 = LinearAlgebraLib::Matrix<number_t, 2, 1>;
using Vec3 = LinearAlgebraLib::Matrix<number_t, 3, 1>;

using Vec2i = LinearAlgebraLib::Matrix<int, 2, 1>;
using Vec3i = LinearAlgebraLib::Matrix<int, 3, 1>;
using VecX_uint8 =
    LinearAlgebraLib::Matrix<uint8_t, LinearAlgebraLib::Dynamic, 1>;
using VecX_uint8_Map = LinearAlgebraLib::Map<VecX_uint8>;
typedef Eigen::Matrix<number_t, Eigen::Dynamic, 1> VecX;
using Quaternion = LinearAlgebraLib::Quaternion<number_t>;

// Check the dimensions of the input matrix
template <typename Vector, int NumDimensions,
          typename = typename std::enable_if<
              Vector::RowsAtCompileTime == NumDimensions &&
              Vector::ColsAtCompileTime == 1>::type>
struct IsFixedSizeVector : std::true_type {};

template <typename Matrix, int RowDimensions, int ColDimensions,
          typename = typename std::enable_if<
              Matrix::RowsAtCompileTime == RowDimensions &&
              Matrix::ColsAtCompileTime == ColDimensions>::type>
struct IsFixedSizeMatrix : std::true_type {};

template <typename T>
using aligned_vector = std::vector<T, LinearAlgebraLib::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, LinearAlgebraLib::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map =
    std::map<K, V, std::less<K>,
             LinearAlgebraLib::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map = std::unordered_map<
    K, V, std::hash<K>, std::equal_to<K>,
    LinearAlgebraLib::aligned_allocator<std::pair<K const, V>>>;

// Parameters Define //
constexpr number_t kOur_PI = 3.14159265358979323846;
class AlgsImage {
public:
  uint8_t *data;
  uint16_t width;
  uint16_t height;
  uint16_t stride;
  uint32_t tuningIndex = 999; // 999 as default tuning index
  uint64_t timestamp = 0;     // timestamp, not time interval
  uint32_t exposure_dt = 0;   // ns
  uint16_t gain = 0;
  bool isAlloc = false;

  AlgsImage() { ; }
  AlgsImage(uint16_t w, uint16_t h) { Reset(w, h); }

  AlgsImage(uint8_t *add, uint16_t w, uint16_t h, uint16_t s,
            uint64_t timestamp_, uint64_t data_len = 0)
      : data(add), width(w), height(h), stride(s), timestamp(timestamp_),
        isAlloc(false) {
    if (data_len) {
      assert(s * h < data_len);
    }
  }

  AlgsImage(AlgsImage &&oth) {
    width = oth.width;
    height = oth.height;
    stride = oth.stride;
    timestamp = oth.timestamp;
    data = oth.data;
    isAlloc = true;
    oth.isAlloc = false;
  }

  AlgsImage(const AlgsImage &oth) {
    width = oth.width;
    height = oth.height;
    stride = oth.stride;
    timestamp = oth.timestamp;
    data = (uint8_t *)AllocAligned(stride * height, 16);
    isAlloc = true;
    CopyFrom(oth);
  }

  AlgsImage &operator=(AlgsImage &&oth) {
    width = oth.width;
    height = oth.height;
    stride = oth.stride;
    timestamp = oth.timestamp;
    data = oth.data;
    isAlloc = true;
    oth.isAlloc = false;
    return *this;
  }

  AlgsImage &operator=(const AlgsImage &oth) {
    width = oth.width;
    height = oth.height;
    stride = oth.stride;
    timestamp = oth.timestamp;
    data = (uint8_t *)AllocAligned(stride * height, 16);
    isAlloc = true;
    CopyFrom(oth);
    return *this;
  }

  ~AlgsImage() {
    if (isAlloc) {
      FreeAligned(data);
      isAlloc = false;
    }
  }

  void Show(const std::string &title);

  void CopyTo(AlgsImage &other) const {
    if (!isAlloc) {
      return;
    }

    assert(width == other.width);
    assert(height == other.height);
    for (int i = 0; i < height; i++) {
      memcpy(other.data + other.stride * i, data + stride * i, width);
    }
  }

  void CopyTo(std::shared_ptr<AlgsImage> &other) {
    assert(width == other->width);
    assert(height == other->height);
    // assert(stride == other->stride);
    for (int i = 0; i < height; i++) {
      memcpy(other->data + other->stride * i, data + stride * i, width);
    }
    other->timestamp = timestamp;
  }

  inline size_t Size() const {
    if (!isAlloc) {
      return 0;
    }

    return stride * height;
  }

  void SetZero() {
    for (int i = 0; i < height; i++) {
      memset(data + stride * i, 0, stride);
    }
  }

  uint8_t &operator()(int row, int col) {
    assert(isAlloc);
    assert(row <= height - 1 && col <= width - 1);
    return *(data + stride * row + col);
  }

  void GetPixel(const int &row, const int &col, int &x0) {
    uint8_t *ptr = data + stride * row + col;
    x0 = *ptr;
  }

  void GetTwoPixel(const int &row, const int &col, int &x0, int &x1) {
    uint8_t *ptr = data + stride * row + col;
    x0 = static_cast<int>(*ptr);
    ptr++;
    x1 = static_cast<int>(*ptr);
  }

  //  void GetTwoPixel(const int& row, const int& col, uint8_t& x0, uint8_t& x1)
  //  {
  //    uint8_t* ptr = data + stride * row + col;
  //    x0 = *ptr;
  //    ptr++;
  //    x1 = *ptr;
  //  }

  // stride = step
  void DangerouslyCopyFrom(uint16_t oth_width, uint16_t oth_height,
                           uint16_t oth_stride, uint8_t *oth_data,
                           uint64_t time_stamp, uint32_t tuning_index,
                           uint32_t dt_exposure, uint16_t gain_exposure) {
    if (!isAlloc || oth_width != width || oth_height != height) {
      Reset(oth_width, oth_height);
    }
    timestamp = time_stamp;
    exposure_dt = dt_exposure;
    gain = gain_exposure;
    tuningIndex = tuning_index;
    for (int i = 0; i < height; i++) {
      memcpy(data + stride * i, oth_data + oth_stride * i, oth_stride);
    }
  }

  void CopyFrom(const AlgsImage &other) {
    if (!isAlloc)
      return;

    assert(width == other.width);
    assert(height == other.height);
    for (int i = 0; i < height; i++) {
      memcpy(data + stride * i, other.data + other.stride * i, stride);
    }
  }

  void SetValue(int val) {
    for (int i = 0; i < height; i++) {
      memset(data + stride * i, val, stride);
    }
  }

private:
  void Reset(uint16_t w, uint16_t h) {
    if (isAlloc) {
      FreeAligned(data);
      isAlloc = false;
    }

    width = w;
    height = h;
    stride = width % 16 ? (width / 16 + 1) * 16 : width;
    data = (uint8_t *)AllocAligned(stride * height, 16);
    timestamp = 0;
    isAlloc = true;
  }

private:
  // Default AlignedAlloc implementation will delegate to Alloc/Free after doing
  // rounding.
  static void *AllocAligned(size_t size, size_t align) {
    assert((align & (align - 1)) == 0);
    align = (align > sizeof(size_t)) ? align : sizeof(size_t);
    size_t p = (size_t)malloc(size + align);
    size_t aligned = 0;
    if (p) {
      aligned = (size_t(p) + align - 1) & ~(align - 1);
      if (aligned == p)
        aligned += align;
      *(((size_t *)aligned) - 1) = aligned - p;
    }
    return (void *)aligned;
  }

  static void FreeAligned(void *p) {
    size_t src = size_t(p) - *(((size_t *)p) - 1);
    free((void *)src);
  }
};
} // namespace dso
#endif // DMVIO_VIO_DEF_H
