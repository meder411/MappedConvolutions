#ifndef QUADRIC_H_
#define QUADRIC_H_

#include <algorithm>
#include <array>

#include "vec3.h"

#define EPS 1e-6
#define APPROX_ZERO(val) -EPS < val &&val < EPS

namespace mapped_conv {
namespace sphere {

template <typename T>
class Quadric {
 private:
  std::array<T, 10> _quadric;

  void assign(const T val) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] = val; }
  }
  void assign(const Quadric &quad) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] = quad[i]; }
  }
  void add(const T val) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] += val; }
  }
  void add(const Quadric &quad) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] += quad[i]; }
  }
  void subtract(const T val) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] -= val; }
  }
  void subtract(const Quadric &quad) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] -= quad[i]; }
  }
  void multiply(const T val) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] *= val; }
  }
  void multiply(const Quadric &quad) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] *= quad[i]; }
  }
  void divide(const T val) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] /= val; }
  }
  void divide(const Quadric &quad) {
    for (size_t i = 0; i < 10; i++) { _quadric[i] /= quad[i]; }
  }

 public:
  Quadric() : _quadric({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) {}
  Quadric(const T a, const T b, const T c, const T d) {
    _quadric[0] = a * a;
    _quadric[1] = a * b;
    _quadric[2] = a * c;
    _quadric[3] = a * d;
    _quadric[4] = b * b;
    _quadric[5] = b * c;
    _quadric[6] = b * d;
    _quadric[7] = c * c;
    _quadric[8] = c * d;
    _quadric[9] = d * d;
  }
  Quadric(const T val) { this->assign(val); }
  Quadric(const Quadric &quad) { this->assign(quad); }
  const T &operator[](const size_t &dim) const {
    // TODO: Throw exception if dim out of bounds
    return _quadric[dim];
  }
  T &operator[](const size_t &dim) {
    // TODO: Throw exception if dim out of bounds
    return _quadric[dim];
  }
  const bool operator==(const Quadric &quad) {
    return APPROX_ZERO(quad[0] - _quadric[0]) &&
           APPROX_ZERO(quad[1] - _quadric[1]) &&
           APPROX_ZERO(quad[2] - _quadric[2]);
  }
  Quadric &operator+=(const Quadric &quad) {
    this->add(quad);
    return *this;
  }
  Quadric &operator+=(const T rhs) {
    this->add(rhs);
    return *this;
  }
  Quadric &operator-=(const Quadric &quad) {
    this->subtract(quad);
    return *this;
  }
  Quadric &operator-=(const T rhs) {
    this->subtract(rhs);
    return *this;
  }
  Quadric &operator/=(const Quadric &quad) {
    this->divide(quad);
    return *this;
  }
  Quadric &operator/=(const T rhs) {
    this->divide(rhs);
    return *this;
  }
  Quadric &operator*=(const Quadric &quad) {
    this->multiply(quad);
    return *this;
  }
  Quadric &operator*=(const T rhs) {
    this->multiply(rhs);
    return *this;
  }
  friend Quadric operator+(Quadric lhs, const Quadric &rhs) {
    lhs.add(rhs);
    return lhs;
  }
  friend Quadric operator+(Quadric lhs, const T rhs) {
    lhs.add(rhs);
    return lhs;
  }
  friend Quadric operator-(Quadric lhs, const Quadric &rhs) {
    lhs.subtract(rhs);
    return lhs;
  }
  friend Quadric operator-(Quadric lhs, const T rhs) {
    lhs.subtract(rhs);
    return lhs;
  }
  friend Quadric operator/(Quadric lhs, const Quadric &rhs) {
    lhs.divide(rhs);
    return lhs;
  }
  friend Quadric operator/(Quadric lhs, const T rhs) {
    lhs.divide(rhs);
    return lhs;
  }
  friend Quadric operator*(Quadric lhs, const Quadric &rhs) {
    lhs.multiply(rhs);
    return lhs;
  }
  friend Quadric operator*(Quadric lhs, const T rhs) {
    lhs.multiply(rhs);
    return lhs;
  }

  const float project(const Vec3<float> &vertex) const {
    // For convenience
    const float x  = vertex[0];
    const float y  = vertex[1];
    const float z  = vertex[2];
    const float aa = _quadric[0];
    const float ab = _quadric[1];
    const float ac = _quadric[2];
    const float ad = _quadric[3];
    const float bb = _quadric[4];
    const float bc = _quadric[5];
    const float bd = _quadric[6];
    const float cc = _quadric[7];
    const float cd = _quadric[8];
    const float dd = _quadric[9];

    const float xtc1 = x * aa + y * ab + z * ac + ad;
    const float xtc2 = x * ab + y * bb + z * bc + bd;
    const float xtc3 = x * ac + y * bc + z * cc + cd;
    const float xtc4 = x * ad + y * bd + z * cd + dd;

    return x * xtc1 + y * xtc2 + z * xtc3 + xtc4;
  }

  T *data() { return _quadric.data(); }
};

}  // namespace sphere
}  // namespace mapped_conv
#endif