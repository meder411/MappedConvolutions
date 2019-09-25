#ifndef HALF_EDGE_H_
#define HALF_EDGE_H_

#include "vec3.h"

#include <iostream>
#include <memory>
#include <vector>

namespace mapped_conv {
namespace sphere {

struct Vertex;
struct Face;

class HalfEdge {
 public:
  std::shared_ptr<Vertex> vertex;  // Vertex pointed to by half edge
  std::shared_ptr<Face> face;      // Face bordered by half edge
  size_t idx;                      // Index in storage
  bool is_boundary = true;

  HalfEdge() {}
  void set_pair(const std::shared_ptr<HalfEdge> &pair) {
    _pair = std::weak_ptr<HalfEdge>(pair);
  }
  void set_next(const std::shared_ptr<HalfEdge> &next) {
    _next = std::weak_ptr<HalfEdge>(next);
  }
  std::shared_ptr<HalfEdge> pair() {
    auto p = _pair.lock();
    if (!p) {
      std::cout << "ERROR HALF EDGE PAIR" << std::endl;
      std::exit(-1);
    }
    return p;
  }
  std::shared_ptr<HalfEdge> next() {
    auto n = _next.lock();
    if (!n) {
      std::cout << "ERROR HALF EDGE NEXT" << std::endl;
      std::exit(-1);
    }
    return n;
  }

 private:
  std::weak_ptr<HalfEdge> _pair;  // Twin half-edge
  std::weak_ptr<HalfEdge> _next;  // Next half edge around same face
};

struct Vertex {
  Vec3<float> XYZ;
  std::shared_ptr<HalfEdge> half_edge;  // An outgoing half edge
  size_t idx;                           // Index in mesh storage
  bool is_boundary = false;  // Vertex is boundary when connected to at least
                             // one boundary edge
  size_t prev_idx = -1;
  Vertex() {}
};

struct Face {
  std::shared_ptr<HalfEdge> half_edge;  // Arbitrary associated half edge
  size_t idx;                           // Index in mesh storage
  bool is_boundary =
      false;             // Face is boundary if has at least one boundary edge
  size_t prev_idx = -1;  // Index of face this one was derived from
  Face() {}
};

}  // namespace sphere
}  // namespace mapped_conv
#endif