#ifndef VEC3_H_
#define VEC3_H_

#include <math.h>
#include <algorithm>
#include <array>
#include <iostream>

#define EPS 1e-6
#define APPROX_ZERO(val) -EPS < val &&val < EPS

namespace mapped_conv {
namespace sphere {

template <typename T>
class Vec3 {
 private:
  std::array<T, 3> _XYZ;

  void assign(const T val) {
    _XYZ[0] = val;
    _XYZ[1] = val;
    _XYZ[2] = val;
  }
  void assign(const Vec3 &vec) {
    _XYZ[0] = vec[0];
    _XYZ[1] = vec[1];
    _XYZ[2] = vec[2];
  }
  void add(const T val) {
    _XYZ[0] += val;
    _XYZ[1] += val;
    _XYZ[2] += val;
  }
  void add(const Vec3 &vec) {
    _XYZ[0] += vec._XYZ[0];
    _XYZ[1] += vec._XYZ[1];
    _XYZ[2] += vec._XYZ[2];
  }
  void subtract(const T val) {
    _XYZ[0] -= val;
    _XYZ[1] -= val;
    _XYZ[2] -= val;
  }
  void subtract(const Vec3 &vec) {
    _XYZ[0] -= vec._XYZ[0];
    _XYZ[1] -= vec._XYZ[1];
    _XYZ[2] -= vec._XYZ[2];
  }
  void multiply(const T val) {
    _XYZ[0] *= val;
    _XYZ[1] *= val;
    _XYZ[2] *= val;
  }
  void multiply(const Vec3 &vec) {
    _XYZ[0] *= vec[0];
    _XYZ[1] *= vec[1];
    _XYZ[2] *= vec[2];
  }
  void divide(const T val) {
    _XYZ[0] /= val;
    _XYZ[1] /= val;
    _XYZ[2] /= val;
  }
  void divide(const Vec3 &vec) {
    _XYZ[0] /= vec[0];
    _XYZ[1] /= vec[1];
    _XYZ[2] /= vec[2];
  }

 public:
  Vec3() : _XYZ({0, 0, 0}) {}
  Vec3(const T x, const T y, const T z) : _XYZ({x, y, z}) {}
  Vec3(const T val) { this->assign(val); }
  Vec3(const Vec3 &vec) { this->assign(vec); }
  const T &operator[](const size_t &dim) const {
    // TODO: Throw exception if dim out of bounds
    return _XYZ[dim];
  }
  T &operator[](const size_t &dim) {
    // TODO: Throw exception if dim out of bounds
    return _XYZ[dim];
  }
  const bool operator==(const Vec3 &vec) {
    return APPROX_ZERO(vec[0] - _XYZ[0]) && APPROX_ZERO(vec[1] - _XYZ[1]) &&
           APPROX_ZERO(vec[2] - _XYZ[2]);
  }
  Vec3 &operator+=(const Vec3 &vec) {
    this->add(vec);
    return *this;
  }
  Vec3 &operator+=(const T rhs) {
    this->add(rhs);
    return *this;
  }
  Vec3 &operator-=(const Vec3 &vec) {
    this->subtract(vec);
    return *this;
  }
  Vec3 &operator-=(const T rhs) {
    this->subtract(rhs);
    return *this;
  }
  Vec3 &operator/=(const Vec3 &vec) {
    this->divide(vec);
    return *this;
  }
  Vec3 &operator/=(const T rhs) {
    this->divide(rhs);
    return *this;
  }
  Vec3 &operator*=(const Vec3 &vec) {
    this->multiply(vec);
    return *this;
  }
  Vec3 &operator*=(const T rhs) {
    this->multiply(rhs);
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &os, const Vec3 &vec) {
    os << "(" << vec._XYZ[0] << ", " << vec._XYZ[1] << ", " << vec._XYZ[2]
       << ")";
    return os;
  }

  friend Vec3 operator+(Vec3 lhs, const Vec3 &rhs) {
    lhs.add(rhs);
    return lhs;
  }
  friend Vec3 operator+(Vec3 lhs, const T rhs) {
    lhs.add(rhs);
    return lhs;
  }
  friend Vec3 operator-(Vec3 lhs, const Vec3 &rhs) {
    lhs.subtract(rhs);
    return lhs;
  }
  friend Vec3 operator-(Vec3 lhs, const T rhs) {
    lhs.subtract(rhs);
    return lhs;
  }
  friend Vec3 operator/(Vec3 lhs, const Vec3 &rhs) {
    lhs.divide(rhs);
    return lhs;
  }
  friend Vec3 operator/(Vec3 lhs, const T rhs) {
    lhs.divide(rhs);
    return lhs;
  }
  friend Vec3 operator*(Vec3 lhs, const Vec3 &rhs) {
    lhs.multiply(rhs);
    return lhs;
  }
  friend Vec3 operator*(Vec3 lhs, const T rhs) {
    lhs.multiply(rhs);
    return lhs;
  }

  const T sum() const { return _XYZ[0] + _XYZ[1] + _XYZ[2]; }
  T *data() { return _XYZ.data(); }
  const T norm() const { return sqrt(this->dot(*this)); }
  const T dot(const Vec3 &rhs) const {
    return _XYZ[0] * rhs[0] + _XYZ[1] * rhs[1] + _XYZ[2] * rhs[2];
  }
  const Vec3 cross(const Vec3 &rhs) const {
    return Vec3(_XYZ[1] * rhs[2] - _XYZ[2] * rhs[1],
                _XYZ[2] * rhs[0] - _XYZ[0] * rhs[2],
                _XYZ[0] * rhs[1] - _XYZ[1] * rhs[0]);
  }
  void normalize() { *this /= this->norm(); }
  const Vec3 normalized() const {
    auto vec = Vec3(*this);
    vec.normalize();
    return vec;
  }
};

}  // namespace sphere
}  // namespace mapped_conv
#endif