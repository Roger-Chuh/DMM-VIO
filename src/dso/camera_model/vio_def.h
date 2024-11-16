//
// Created by root on 2024/11/16.
//

#ifndef DMVIO_VIO_DEF_H
#define DMVIO_VIO_DEF_H
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace dso {
#define LinearAlgebraLib Eigen
using number_t = double;
using CamId = uint8_t;
const size_t kInvalid = std::numeric_limits<size_t>::max();
const uint8_t kInvalid_uint8_t = std::numeric_limits<uint8_t>::max();

constexpr int kMaxIntrSize = 38;

// Template Matrix Define //  // for template function
template <typename T> using Matrix3 = LinearAlgebraLib::Matrix<T, 3, 3>;
template <typename T> using Matrx3x2 = LinearAlgebraLib::Matrix<T, 3, 2>;
template <typename T> using Matrix4 = LinearAlgebraLib::Matrix<T, 4, 4>;
template <typename T> using Matrix6x3 = LinearAlgebraLib::Matrix<T, 6, 3>;

template <typename T> using Vector3 = LinearAlgebraLib::Matrix<T, 3, 1>;
template <typename T> using Vector6 = LinearAlgebraLib::Matrix<T, 6, 1>;

using Vec2 = LinearAlgebraLib::Matrix<double, 2, 1>;
using Vec3 = LinearAlgebraLib::Matrix<double, 3, 1>;

using Vec2i = LinearAlgebraLib::Matrix<int, 2, 1>;
using Vec3i = LinearAlgebraLib::Matrix<int, 3, 1>;
using VecX_uint8 =
    LinearAlgebraLib::Matrix<uint8_t, LinearAlgebraLib::Dynamic, 1>;
using VecX_uint8_Map = LinearAlgebraLib::Map<VecX_uint8>;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
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
} // namespace dso
#endif // DMVIO_VIO_DEF_H
