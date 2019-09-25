#include "util/cpp/sphere.h"
#include "core/conversions.h"
#include "util/cpp/half_edge.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <utility>

#include <omp.h>

namespace mapped_conv {
namespace sphere {

Sphere::Sphere() {}

Sphere::Sphere(const Sphere &sphere) {
  // Resize the member vectors
  this->_vertices.resize(sphere._vertices.size());
  this->_faces.resize(sphere._faces.size());
  this->_half_edges.resize(sphere._half_edges.size());
  size_t i;

// Deep copy the faces
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < sphere._faces.size(); i++) {
    this->_faces[i] = std::make_shared<Face>(*(sphere._faces[i]));
  }

// Deep copy the vertices
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < sphere._vertices.size(); i++) {
    this->_vertices[i] = std::make_shared<Vertex>(*(sphere._vertices[i]));
  }

// Deep copy the half edges and link them to the new vertices and faces
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < sphere._half_edges.size(); i++) {
    // Deep copy
    this->_half_edges[i] =
        std::make_shared<HalfEdge>(*(sphere._half_edges[i]));

    // Re-linking
    const size_t v_idx           = sphere._half_edges[i]->vertex->idx;
    const size_t f_idx           = sphere._half_edges[i]->face->idx;
    this->_half_edges[i]->vertex = this->_vertices[v_idx];
    this->_half_edges[i]->face   = this->_faces[f_idx];
  }

// Now that all half edges are created, link the next-edge and paired-edges
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < this->_half_edges.size(); i++) {
    const size_t next_idx = _half_edges[i]->next()->idx;
    const size_t pair_idx = _half_edges[i]->pair()->idx;
    _half_edges[i]->set_next(_half_edges[next_idx]);
    _half_edges[i]->set_pair(_half_edges[pair_idx]);
  }

// Note: the face and vertex deep copies still point to old half edges
// Relink the faces
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < this->_faces.size(); i++) {
    const size_t he_idx        = sphere._faces[i]->half_edge->idx;
    this->_faces[i]->half_edge = this->_half_edges[he_idx];
  }

// Deep copy the vertices
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < this->_vertices.size(); i++) {
    const size_t he_idx           = sphere._vertices[i]->half_edge->idx;
    this->_vertices[i]->half_edge = this->_half_edges[he_idx];
  }
}

void link_pairs(std::shared_ptr<HalfEdge> he1, std::shared_ptr<HalfEdge> he2) {
  // Point the half edges to each other
  he1->set_pair(he2);
  he2->set_pair(he1);
  he1->is_boundary = false;
  he2->is_boundary = false;
}

void find_pairs(
    std::map<std::pair<size_t, size_t>, std::shared_ptr<HalfEdge>> &pair_map,
    std::shared_ptr<HalfEdge> he, const size_t idx_start,
    const size_t idx_end) {
  // Check for the reverse edge in a map
  auto it = pair_map.find(std::make_pair(idx_end, idx_start));
  if (it != pair_map.end()) {
    // If successful, link the pairs
    link_pairs(he, it->second);
  } else {
    // If not successful, add this half edge to the pair map
    pair_map.emplace(std::make_pair(idx_start, idx_end), he);
  }
}

Sphere::Sphere(at::Tensor pts, at::Tensor faces) {
  const size_t num_pts   = pts.size(0);
  const size_t num_faces = faces.size(0);
  this->_buildSphere(pts.data<float>(), num_pts, faces.data<int64_t>(),
                     num_faces);
}

Sphere::Sphere(const std::vector<float> &pts,
               const std::vector<int64_t> &faces) {
  const size_t num_pts   = pts.size() / 3;
  const size_t num_faces = faces.size() / 3;
  this->_buildSphere(pts.data(), num_pts, faces.data(), num_faces);
}

void Sphere::addFace(std::shared_ptr<Face> f) {
  f->idx = _faces.size();
  _faces.push_back(f);
}

void Sphere::addHalfEdge(std::shared_ptr<HalfEdge> he) {
  he->idx = _half_edges.size();
  _half_edges.push_back(he);
}

void Sphere::addVertex(std::shared_ptr<Vertex> v) {
  v->idx = _vertices.size();
  _vertices.push_back(v);
}

void Sphere::_buildSphere(const float *pts, const size_t num_pts,
                          const int64_t *faces, const size_t num_faces) {
  // Map data structure to determine half-edge pairing
  std::map<std::pair<size_t, size_t>, std::shared_ptr<HalfEdge>> pair_map;

  // Parse the vertex information
  for (size_t i = 0; i < num_pts; i++) {
    // Create a vertex
    std::shared_ptr<Vertex> v = std::make_shared<Vertex>();

    // Initialize the vertex location information
    // (but not the half edges yet)
    v->XYZ = Vec3<float>(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);

    // Add the vertex to the mesh
    this->addVertex(v);
  }

  // Parse the face information
  for (size_t i = 0; i < num_faces; i++) {
    // Create a face
    std::shared_ptr<Face> f = std::make_shared<Face>();

    // All faces have 3 half edges (vertices 0->1, 1->2, 2->0)
    std::shared_ptr<HalfEdge> he01 = std::make_shared<HalfEdge>();
    std::shared_ptr<HalfEdge> he12 = std::make_shared<HalfEdge>();
    std::shared_ptr<HalfEdge> he20 = std::make_shared<HalfEdge>();

    // Lets make the first half edge the face representative
    f->half_edge = he01;

    // Each half edge is part of this face
    he01->face = f;
    he12->face = f;
    he20->face = f;

    // Half edges point in a circle
    // (winding order already set in indices list)
    he01->set_next(he12);
    he12->set_next(he20);
    he20->set_next(he01);

    // Parse the vertex connectivity data
    size_t idx0 = faces[3 * i];
    size_t idx1 = faces[3 * i + 1];
    size_t idx2 = faces[3 * i + 2];

    // Each half edge gets assigned the vertex it points to
    he01->vertex = _vertices[idx1];
    he12->vertex = _vertices[idx2];
    he20->vertex = _vertices[idx0];

    // Each vertex points to an outgoing half-edge
    // Note: these can end up over-writing earlier assignments at times
    // but that's okay. The half-edge being pointed to by a vertex is
    // arbitrary. We just waste a bit of computation...
    _vertices[idx0]->half_edge = he01;
    _vertices[idx1]->half_edge = he12;
    _vertices[idx2]->half_edge = he20;

    // For each half edge, check for the reverse half edge that shares
    // the same two vertices
    find_pairs(pair_map, he01, idx0, idx1);
    find_pairs(pair_map, he12, idx1, idx2);
    find_pairs(pair_map, he20, idx2, idx0);

    // Add the half edges and face to the mesh
    this->addFace(f);
    this->addHalfEdge(he01);
    this->addHalfEdge(he12);
    this->addHalfEdge(he20);
  }
}

const Vec3<float> Sphere::_getFaceBarycenter(const size_t face_idx) const {
  Vec3<float> barycenter(0, 0, 0);
  for (const auto &v : this->_getFaceVertexIndices(face_idx)) {
    barycenter += _vertices[v]->XYZ;
  }

  return barycenter / 3.;
}

const std::vector<size_t> Sphere::_getFaceVertexIndices(
    const size_t face_idx) const {
  std::vector<size_t> indices;

  auto start = _faces[face_idx]->half_edge;
  auto he    = start;
  do {
    indices.push_back(he->vertex->idx);
    he = he->next();  // Next half edge in the face
  } while (he != start);

  return indices;
}

const std::vector<size_t> Sphere::_getVerticesAdjacentToVertex(
    const size_t vert_idx, const bool duplicate = false) const {
  std::vector<size_t> indices;

  auto he    = _vertices[vert_idx]->half_edge;
  auto start = he;
  do {
    indices.push_back(he->vertex->idx);
    he = he->pair()->next();
  } while (he != start);

  // Duplicate the first point if we are at one of the singular points
  if (indices.size() < 6 && duplicate) { indices.push_back(indices[0]); }

  return indices;
}

const std::vector<size_t> Sphere::_getFacesAdjacentToVertex(
    const size_t vert_idx, const bool duplicate = false) const {
  std::vector<size_t> indices;

  auto he    = _vertices[vert_idx]->half_edge;
  auto start = he;
  do {
    indices.push_back(he->face->idx);
    he = he->pair()->next();
  } while (he != start);

  // Duplicate the first point if we are at one of the singular points
  if (indices.size() < 6 && duplicate) { indices.push_back(indices[0]); }

  return indices;
}

const std::vector<size_t> Sphere::_getFacesAdjacentToFace(
    const size_t face_idx) const {
  std::vector<size_t> indices;

  auto he    = _faces[face_idx]->half_edge;
  auto start = he;
  do {
    indices.push_back(he->pair()->face->idx);
    he = he->next();
  } while (he != start);

  return indices;
}

const std::vector<size_t> Sphere::_getHalfEdgesPointingToVertex(
    const size_t vert_idx) const {
  std::vector<size_t> indices;

  // Outgoig half edge
  auto he    = _vertices[vert_idx]->half_edge;
  auto start = he;
  do {
    // Pair of the outgoiing half edge will be an incoming half edge
    indices.push_back(he->pair()->idx);

    // Move on to next outgoing half edge
    he = he->pair()->next();
  } while (he != start);

  return indices;
}

const size_t Sphere::_getVertexValence(const size_t vert_idx) const {
  size_t valence = 0;

  auto he    = _vertices[vert_idx]->half_edge;
  auto start = he;
  do {
    valence++;
    he = he->pair()->next();
  } while (he != start);

  return valence;
}

template <typename iterator_type>
const std::shared_ptr<Face> Sphere::_getNorthMostFace(
    const iterator_type &faces) const {
  auto northmost_face = *(faces.begin());
  float northmost_y   = this->_getFaceBarycenter(northmost_face->idx)[1];
  for (const auto &f : faces) {
    float y = this->_getFaceBarycenter(f->idx)[1];
    if (y > northmost_y) {
      northmost_y    = y;
      northmost_face = f;
    }
  }

  return northmost_face;
}

template <typename iterator_type>
const std::shared_ptr<Vertex> Sphere::_getNorthMostVertex(
    const iterator_type &vertices) const {
  auto northmost_vertex = *(vertices.begin());
  float northmost_y     = northmost_vertex->XYZ[1];
  for (const auto &v : vertices) {
    if (v->XYZ[1] > northmost_y) {
      northmost_y      = v->XYZ[1];
      northmost_vertex = v;
    }
  }

  return northmost_vertex;
}

const std::vector<size_t> Sphere::_getFaceConvolutionOperatorIndicesDeg1(
    const std::shared_ptr<Face> &face) const {
  // Vector of indices
  std::vector<size_t> indices;

  // The first element is the center face
  indices.push_back(face->idx);

  // Get the vertex indices of the center face
  auto vert_indices = this->_getFaceVertexIndices(face->idx);

  // Add all faces from each vertex to a set (excluding center face)
  std::set<std::shared_ptr<Face>> face_set;
  for (const auto &v_idx : vert_indices) {
    auto adj_faces_indices = this->_getFacesAdjacentToVertex(v_idx);
    for (const auto &f_idx : adj_faces_indices) {
      if (f_idx != face->idx) { face_set.insert(_faces[f_idx]); }
    }
  }

  // Find the northmost adjacent face
  auto northmost_face = this->_getNorthMostFace(face_set);

  // Get the half edge in this face that points to the vertex in the original
  // set
  auto he = northmost_face->half_edge;
  while (std::find(vert_indices.begin(), vert_indices.end(),
                   he->vertex->idx) == vert_indices.end() ||
         he->pair()->face == face) {
    he = he->next();
  }
  auto start = he;

  // Compute the string of faces according to the winding order
  do {
    // Add the index of the face associated with that half edge
    indices.push_back(he->face->idx);

    // Flip to the next face
    he = he->pair();

    // Again, get the half edge in the new face that points to the vertex in
    // the original set
    while (std::find(vert_indices.begin(), vert_indices.end(),
                     he->vertex->idx) == vert_indices.end()) {
      he = he->next();
    }
  } while (he != start);

  return indices;
}

const std::vector<size_t> Sphere::_getFaceConvolutionOperatorIndicesDeg2(
    const std::shared_ptr<Face> &face, const bool dilation) const {
  std::vector<size_t> indices;

  // First get the degree 1 indices if no dilation
  if (!dilation) {
    indices = this->_getFaceConvolutionOperatorIndicesDeg1(face);
  }
  // Otherwise, just add the center face
  else {
    indices.push_back(face->idx);
  }

  // Get the vertex indices of the center face
  auto center_vert_indices = this->_getFaceVertexIndices(face->idx);

  // Add all faces from each vertex to a set (excluding center face)
  std::set<std::shared_ptr<Face>> deg1_face_set;
  for (const auto &v_idx : center_vert_indices) {
    auto adj_faces_indices = this->_getFacesAdjacentToVertex(v_idx);
    for (const auto &f_idx : adj_faces_indices) {
      if (f_idx != face->idx) { deg1_face_set.insert(_faces[f_idx]); }
    }
  }

  // Go through all the degree 1 faces and put their vertices in a set if they
  // are not also vertices of the center face
  std::set<size_t> edge_vert_indices;
  for (const auto &f : deg1_face_set) {
    auto vert_indices = this->_getFaceVertexIndices(f->idx);
    for (const auto &idx : vert_indices) {
      if (std::find(center_vert_indices.begin(), center_vert_indices.end(),
                    idx) == center_vert_indices.end()) {
        edge_vert_indices.insert(idx);
      }
    }
  }

  // Add all faces from each edge vertex to a new set (excluding face from the
  // first set)
  std::set<std::shared_ptr<Face>> deg2_face_set;
  for (const auto &v_idx : edge_vert_indices) {
    auto adj_faces_indices = this->_getFacesAdjacentToVertex(v_idx);
    for (const auto &f_idx : adj_faces_indices) {
      if (deg1_face_set.count(_faces[f_idx]) == 0) {
        deg2_face_set.insert(_faces[f_idx]);
      }
    }
  }

  // Find the northmost adjacent face
  auto northmost_face = this->_getNorthMostFace(deg2_face_set);

  // Get the half edge in this face that points to a vertex in the original
  // set
  auto he = northmost_face->half_edge;
  while (edge_vert_indices.count(he->vertex->idx) == 0 ||
         deg1_face_set.count(he->pair()->face) > 0) {
    he = he->next();
  }
  auto start = he;

  // Compute the string of faces according to the winding order
  do {
    // Add the index of the face associated with that half edge
    indices.push_back(he->face->idx);

    // Flip to the next face
    he = he->pair();

    // Again, get the half edge in the new face that points to a vertex in the
    // original set
    while (edge_vert_indices.count(he->vertex->idx) == 0) { he = he->next(); }
  } while (he != start);

  return indices;
}

const Vec3<float> Sphere::_computeFaceNormal(const size_t face_idx) const {
  // 3 Vertices
  auto v0 = _faces[face_idx]->half_edge->vertex;
  auto v1 = _faces[face_idx]->half_edge->next()->vertex;
  auto v2 = _faces[face_idx]->half_edge->next()->next()->vertex;

  // 2 segments
  auto a = v2->XYZ - v1->XYZ;
  auto b = v0->XYZ - v1->XYZ;

  // Compute the cross product according to the winding order
  const float x = a[1] * b[2] - a[2] * b[1];
  const float y = a[2] * b[0] - a[0] * b[2];
  const float z = a[0] * b[1] - a[1] * b[0];

  auto normal = Vec3<float>(x, y, z);

  return normal.normalized();
}

const float Sphere::_computePlaneDistance(const Vec3<float> &normal,
                                          const Vec3<float> &vertex) const {
  return -(normal * vertex).sum();
}

void Sphere::_computeVertexQuadrics(std::vector<Quadric<float>> &quad) const {
  quad.resize(_vertices.size());

  size_t i;
  Quadric<float> *quad_ptr = quad.data();
#pragma omp parallel for shared(quad_ptr) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    auto adj_face_idx = _getFacesAdjacentToVertex(i);
    Quadric<float> quad;
    for (const auto &face_idx : adj_face_idx) {
      const auto normal = this->_computeFaceNormal(face_idx);
      const auto vertex = _faces[face_idx]->half_edge->vertex;
      const float d     = this->_computePlaneDistance(normal, vertex->XYZ);
      quad += Quadric<float>(normal[0], normal[1], normal[2], d);
    }

    // Store the quad
    quad_ptr[i] = quad;
  }
}

const Sphere Sphere::generateIcosphere(const size_t order) {
  const float r = (1.0 + sqrt(5.0)) / 2.0;

  std::vector<float> pts = std::vector<float>{
      -1.0, r,    0.0,  1.0, r,   0.0, -1.0, -r,   0.0,  1.0, -r,  0.0,
      0.0,  -1.0, r,    0.0, 1.0, r,   0.0,  -1.0, -r,   0.0, 1.0, -r,
      r,    0.0,  -1.0, r,   0.0, 1.0, -r,   0.0,  -1.0, -r,  0.0, 1.0};

  std::vector<int64_t> face_indices = std::vector<int64_t>{
      0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
      4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
      6, 8,  3,  8, 9,  5, 4, 9, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1};

  // Create the icosphere
  Sphere icosphere(pts, face_indices);

  // Subdivide the the degree desired
  icosphere = icosphere.loopSubdivision(order);

  // Normalize the vertices
  size_t i;
#pragma omp parallel for private(i) schedule(static)
  for (i = 0; i < icosphere._vertices.size(); i++) {
    icosphere._vertices[i]->XYZ = icosphere._vertices[i]->XYZ.normalized();
  }

  return icosphere;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// PUBLIC FUNCTIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// Returns V x 3
const at::Tensor Sphere::getVertices() const {
  at::Tensor at_vertices =
      torch::zeros({static_cast<int64_t>(_vertices.size()), 3}, at::kFloat);

  size_t i;
  float *at_vertices_ptr = at_vertices.data<float>();
#pragma omp parallel for shared(at_vertices_ptr) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    auto v = _vertices[i]->XYZ;
    std::copy(v.data(), v.data() + 3, at_vertices_ptr + 3 * i);
  }
  return at_vertices;
}

// Returns F x 3 x 3, F sets of 3 rows of points
const at::Tensor Sphere::getAllFaceVertexCoordinates() const {
  at::Tensor at_coords =
      torch::zeros({static_cast<int64_t>(_faces.size()), 3, 3}, at::kFloat);

  size_t i;
  float *at_coords_ptr = at_coords.data<float>();
#pragma omp parallel for shared(at_coords_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    const auto vert_idx = this->_getFaceVertexIndices(i);
    for (size_t j = 0; j < vert_idx.size(); j++) {
      auto coord = _vertices[vert_idx[j]]->XYZ;
      std::copy(coord.data(), coord.data() + 3, at_coords_ptr + 9 * i + 3 * j);
    }
  }
  return at_coords;
}

// Returns F x 3, F rows of 3 indices
const at::Tensor Sphere::getAllFaceVertexIndices() const {
  at::Tensor at_indices =
      torch::zeros({static_cast<int64_t>(_faces.size()), 3}, at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
#pragma omp parallel for shared(at_indices_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    auto vert_idx = this->_getFaceVertexIndices(i);
    std::copy(vert_idx.data(), vert_idx.data() + 3, at_indices_ptr + 3 * i);
  }

  return at_indices;
}

// Returns F x 3
const at::Tensor Sphere::getFaceBarycenters() const {
  at::Tensor at_barycenters =
      torch::zeros({static_cast<int64_t>(_faces.size()), 3}, at::kFloat);

  size_t i;
  float *at_barycenters_ptr = at_barycenters.data<float>();
#pragma omp parallel for shared(at_barycenters_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    auto barycenter = this->_getFaceBarycenter(i);
    std::copy(barycenter.data(), barycenter.data() + 3,
              at_barycenters_ptr + 3 * i);
  }

  return at_barycenters;
}

const float Sphere::getVertexResolution() const {
  float res = 0.;
  size_t i  = 0;
#pragma omp parallel for reduction(+ : res) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    const auto v        = _vertices[i];
    const auto adj_vert = _getVerticesAdjacentToVertex(i, false);
    float acc_norm      = 0.;
    int num_adj         = 0;
    for (const auto &adj_idx : adj_vert) {
      if (adj_idx != static_cast<size_t>(-1)) {
        acc_norm += (_vertices[adj_idx]->XYZ - v->XYZ).norm();
        num_adj++;
      }
    }
    res += acc_norm / float(num_adj);
  }
  return res / float(_vertices.size());
}

// Returns F x 3
const at::Tensor Sphere::getAdjacentFaceIndicesToFaces() const {
  at::Tensor at_indices =
      torch::zeros({static_cast<int64_t>(_faces.size()), 3}, at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
#pragma omp parallel for shared(at_indices_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    auto face_idx = this->_getFacesAdjacentToFace(i);
    std::copy(face_idx.data(), face_idx.data() + 3, at_indices_ptr + 3 * i);
  }

  return at_indices;
}

// Returns V x 6 (a value of -1 implies no adjacent face)
const at::Tensor Sphere::getAdjacentFaceIndicesToVertices() const {
  at::Tensor at_indices =
      -1 * torch::ones({static_cast<int64_t>(_vertices.size()), 6}, at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
#pragma omp parallel for shared(at_indices_ptr) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    const auto face_idx = this->_getFacesAdjacentToVertex(i, true);
    std::copy(face_idx.data(), face_idx.data() + face_idx.size(),
              at_indices_ptr + 6 * i);
  }

  return at_indices;
}

// Returns V x 6 (a value of -1 implies no adjacent vertex)
const at::Tensor Sphere::getAdjacentVertexIndicesToVertices() const {
  at::Tensor at_indices =
      -1 * torch::ones({static_cast<int64_t>(_vertices.size()), 6}, at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
#pragma omp parallel for shared(at_indices_ptr) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    const auto vert_idx = this->_getVerticesAdjacentToVertex(i, true);
    std::copy(vert_idx.data(), vert_idx.data() + vert_idx.size(),
              at_indices_ptr + 6 * i);
  }

  return at_indices;
}

// Returns F x 13 (a value of -1 implies no adjacent face)
const at::Tensor Sphere::getAllFaceConvolutionOperatorIndicesDeg1() const {
  at::Tensor at_indices =
      -1 * torch::ones({static_cast<int64_t>(_faces.size()), 13}, at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
#pragma omp parallel for shared(at_indices_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    const auto face_idx =
        this->_getFaceConvolutionOperatorIndicesDeg1(_faces[i]);
    std::copy(face_idx.data(), face_idx.data() + face_idx.size(),
              at_indices_ptr + 13 * i);
  }

  return at_indices;
}

// Returns F x 37 (a value of -1 implies no adjacent face)
const at::Tensor Sphere::getAllFaceConvolutionOperatorIndicesDeg2(
    const bool dilation) const {
  size_t ksize = 37;
  if (dilation) { ksize = 25; }
  at::Tensor at_indices =
      -1 * torch::ones({static_cast<int64_t>(_faces.size()),
                        static_cast<int64_t>(ksize)},
                       at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
  const auto faces_ptr    = _faces.data();
#pragma omp parallel for shared(at_indices_ptr, ksize) private(i) \
    schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    const auto face_idx =
        this->_getFaceConvolutionOperatorIndicesDeg2(faces_ptr[i], dilation);
    std::copy(face_idx.data(), face_idx.data() + face_idx.size(),
              at_indices_ptr + ksize * i);
  }

  return at_indices;
}

// Returns F x 0
const at::Tensor Sphere::getOriginalFaceIndices() const {
  at::Tensor at_indices =
      torch::zeros({static_cast<int64_t>(_faces.size())}, at::kLong);

  size_t i;
  int64_t *at_indices_ptr = at_indices.data<int64_t>();
  const auto faces_ptr    = _faces.data();
#pragma omp parallel for shared(at_indices_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    at_indices_ptr[i] = faces_ptr[i]->prev_idx;
  }

  return at_indices;
}

std::shared_ptr<Vertex> find_odd_vertex(
    std::map<std::pair<size_t, size_t>, std::shared_ptr<Vertex>> &odd_map,
    const size_t idx_start, const size_t idx_end) {
  // Check for the reverse edge in a map
  auto it = odd_map.find(std::make_pair(idx_start, idx_end));
  if (it != odd_map.end()) { return it->second; }
  it = odd_map.find(std::make_pair(idx_end, idx_start));
  if (it != odd_map.end()) { return it->second; }
  std::cout << "SOMETHING MESSED UP WITH ODD MAP" << std::endl;
  std::exit(-1);
}

void create_face(
    Sphere &mesh, std::shared_ptr<Vertex> v0, std::shared_ptr<Vertex> v1,
    std::shared_ptr<Vertex> v2,
    std::map<std::pair<size_t, size_t>, std::shared_ptr<HalfEdge>> &pair_map,
    const size_t prev_idx) {
  // Create a face
  std::shared_ptr<Face> f = std::make_shared<Face>();

  // Track where the face came from
  f->prev_idx = prev_idx;

  // All faces have 3 half edges (vertices 0->1, 1->2, 2->0)
  std::shared_ptr<HalfEdge> he01 = std::make_shared<HalfEdge>();
  std::shared_ptr<HalfEdge> he12 = std::make_shared<HalfEdge>();
  std::shared_ptr<HalfEdge> he20 = std::make_shared<HalfEdge>();

  // Lets make the first half edge the face representative
  f->half_edge = he01;

  // Each half edge is part of this face
  he01->face = f;
  he12->face = f;
  he20->face = f;

  // Half edges point in a circle
  // (winding order already set in indices list)
  he01->set_next(he12);
  he12->set_next(he20);
  he20->set_next(he01);

  // Make each half edge point to a vertex
  he01->vertex = v1;
  he12->vertex = v2;
  he20->vertex = v0;

  // Get the storage indices of the current vertices
  size_t idx0 = v0->idx;
  size_t idx1 = v1->idx;
  size_t idx2 = v2->idx;

  // Give each vertex an outgoing half edge
  v0->half_edge = he01;
  v1->half_edge = he12;
  v2->half_edge = he20;

  // Check for the reverse half edge has already been created
  find_pairs(pair_map, he01, idx0, idx1);
  find_pairs(pair_map, he12, idx1, idx2);
  find_pairs(pair_map, he20, idx2, idx0);

  // Add the half edges and face to the mesh
  mesh.addFace(f);
  mesh.addHalfEdge(he01);
  mesh.addHalfEdge(he12);
  mesh.addHalfEdge(he20);
}

const Sphere Sphere::_singleLoopSubdivision() const {
  Sphere new_mesh;

  // -----------------------------
  // First make all new vertices
  // -----------------------------

  // All current vertices will be in the new mesh
  // These are the even vertices
  for (const auto &v : _vertices) {
    std::shared_ptr<Vertex> v_even = std::make_shared<Vertex>();

    // Compute the valence and beta weights
    size_t valence = this->_getVertexValence(v->idx);
    float beta     = 1.0;
    if (valence > 3) {
      beta = 3.0 / (8.0 * valence);
    } else if (valence == 3) {
      beta = 3.0 / 16.0;
    } else if (valence == 2) {
      beta = 1.0 / 8.0;
    }

    // New location is weighted sum of valence vertices
    Vec3<float> weighted_sum(0, 0, 0);
    for (const auto &adj_idx : this->_getVerticesAdjacentToVertex(v->idx)) {
      weighted_sum += _vertices[adj_idx]->XYZ;
    }
    weighted_sum *= beta;
    v_even->XYZ = weighted_sum + v->XYZ * (1.0 - valence * beta);

    // Add the even vertices to the new mesh
    // Note that their storage indices will be the same as the old mesh
    new_mesh.addVertex(v_even);
  }

  // Map data structure to determine odd vertices location
  std::map<std::pair<size_t, size_t>, std::shared_ptr<Vertex>> odd_map;

  // Now go through each edge to create a new vertex on each
  // These are the odd vertices
  std::vector<bool> split(_half_edges.size(), false);
  for (const auto &he : _half_edges) {
    std::shared_ptr<Vertex> v_odd = std::make_shared<Vertex>();

    // If this half edge hasn't already been split (by its pair)
    if (!split[he->idx]) {
      // Boundary case
      if (he->is_boundary) {
        // New location is average of bounding vertices
        v_odd->XYZ = (he->vertex->XYZ + he->pair()->vertex->XYZ) * 0.5;
      } else {
        // New location is weighted sum of bounding and "opposite"
        // vertices
        const float a = 3.0 / 8.0;
        const float b = 1.0 / 8.0;
        v_odd->XYZ =
            (he->vertex->XYZ + he->pair()->vertex->XYZ) * a +
            (he->next()->vertex->XYZ + he->pair()->next()->vertex->XYZ) * b;
      }

      // Mark this half edge and its pair as split
      split[he->idx]         = true;
      split[he->pair()->idx] = true;

      // Store a mapping of (v_src, v_dest) --> v_odd
      // Note that pair's order is arbitrary so we check both later
      odd_map.emplace(std::make_pair(he->pair()->vertex->idx, he->vertex->idx),
                      v_odd);

      // Add the odd vertex to the new mesh
      new_mesh.addVertex(v_odd);
    }
  }

  // ------------------------------------
  // Now rebuild mesh with new vertices
  // ------------------------------------

  // Map data structure to determine half-edge pairing
  std::map<std::pair<size_t, size_t>, std::shared_ptr<HalfEdge>> pair_map;

  // Turn each face into 4 sub faces
  for (const auto &f : _faces) {
    // All vertices with these new faces
    auto v_a = new_mesh._vertices[f->half_edge->vertex->idx];
    auto v_b = new_mesh._vertices[f->half_edge->next()->vertex->idx];
    auto v_c = new_mesh._vertices[f->half_edge->next()->next()->vertex->idx];
    auto v_alpha = find_odd_vertex(odd_map, v_a->idx, v_b->idx);
    auto v_beta  = find_odd_vertex(odd_map, v_b->idx, v_c->idx);
    auto v_gamma = find_odd_vertex(odd_map, v_c->idx, v_a->idx);

    // Create new faces and link them up with new vertices
    create_face(new_mesh, v_a, v_alpha, v_gamma, pair_map, f->idx);
    create_face(new_mesh, v_b, v_beta, v_alpha, pair_map, f->idx);
    create_face(new_mesh, v_c, v_gamma, v_beta, pair_map, f->idx);
    create_face(new_mesh, v_alpha, v_beta, v_gamma, pair_map, f->idx);
  }

  // Return new mesh
  return new_mesh;
}

const Sphere Sphere::loopSubdivision(const size_t iterations) const {
  if (iterations == 0) { return *this; }
  Sphere mesh = this->_singleLoopSubdivision();
  for (size_t i = 1; i < iterations; i++) {
    mesh = mesh._singleLoopSubdivision();
  }
  return mesh;
}

// Scale tensor should be length V
void Sphere::scaleVertices(at::Tensor scale) {
  size_t i;
  const float *scale_tensor_ptr = scale.data<float>();
  auto vertices_ptr             = _vertices.data();
  // #pragma omp parallel for shared(vertices_ptr) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    vertices_ptr[i]->XYZ *= scale_tensor_ptr[i];
  }
}

// Returns F x 3
const at::Tensor Sphere::getFaceNormals() const {
  at::Tensor at_normals =
      torch::zeros({static_cast<int64_t>(_faces.size()), 3}, at::kFloat);

  size_t i;
  float *at_normals_ptr = at_normals.data<float>();
#pragma omp parallel for shared(at_normals_ptr) private(i) schedule(static)
  for (i = 0; i < _faces.size(); i++) {
    auto normal = this->_computeFaceNormal(i);
    std::copy(normal.data(), normal.data() + 3, at_normals_ptr + 3 * i);
  }

  return at_normals;
}

// Returns V x 3
const at::Tensor Sphere::getVertexNormals() const {
  at::Tensor at_normals =
      torch::zeros({static_cast<int64_t>(_vertices.size()), 3}, at::kFloat);

  size_t i;
  float *at_normals_ptr = at_normals.data<float>();
#pragma omp parallel for shared(at_normals_ptr) private(i) schedule(static)
  for (i = 0; i < _vertices.size(); i++) {
    const auto adj_faces_idx = _getFacesAdjacentToVertex(i);
    Vec3<float> agg_normal;
    for (const size_t idx : adj_faces_idx) {
      agg_normal += this->_computeFaceNormal(idx);
    }
    agg_normal.normalize();
    std::copy(agg_normal.data(), agg_normal.data() + 3,
              at_normals_ptr + 3 * i);
  }

  return at_normals;
}

void compute_gnomonic_projection(float *kernel, const int kernel_height,
                                 const int kernel_width, const float res_lat,
                                 const float res_lon, const float b_lat,
                                 const float b_lon) {
  for (int i = 0; i < kernel_height; i++) {
    int cur_i = i - kernel_height / 2;
    for (int j = 0; j < kernel_width; j++) {
      int cur_j     = j - kernel_width / 2;
      float cur_lon = res_lon * cur_j;
      float cur_lat = res_lat * cur_i;

      const float rho = sqrt(cur_lon * cur_lon + cur_lat * cur_lat);
      const float nu  = atan(rho);

      // If at the center of the kernel, just at the barycenter lat and lon
      if (i == kernel_height / 2 && j == kernel_width / 2) {
        kernel[i * kernel_width * 2 + j * 2]     = b_lon;
        kernel[i * kernel_width * 2 + j * 2 + 1] = b_lat;
      } else {
        // Otherwise compute the projection
        // Add the longitude projection to the kernel
        kernel[i * kernel_width * 2 + j * 2] =
            b_lon +
            atan2(cur_lon * sin(nu),
                  rho * cos(b_lat) * cos(nu) - cur_lat * sin(b_lat) * sin(nu));

        // Add the latitude projection to the kernel
        kernel[i * kernel_width * 2 + j * 2 + 1] =
            asin(cos(nu) * sin(b_lat) + cur_lat * sin(nu) * cos(b_lat) / rho);
      }
    }
  }
}

// BFS
const size_t Sphere::_searchNearbyFaces(const size_t face_idx,
                                        const Vec3<float> &pt, float *s,
                                        float *t, float *u) const {
  // Initialize BFS FIFO queue
  std::queue<size_t> faces_to_search;

  // Initialized "visited" set
  std::set<size_t> visited;
  std::set<size_t> queued;

  // Add the root face
  faces_to_search.push(face_idx);

  // While the queue is not empty, do a BFS
  size_t num_faces_searched = 0;
  while (!faces_to_search.empty() && num_faces_searched <= 1e10) {
    num_faces_searched++;

    // Pop the first element off the queue
    const size_t cur_face_idx = faces_to_search.front();
    faces_to_search.pop();

    // std::cout << "checking " << cur_face_idx << std::endl;
    // If the point is in this face, return the face index
    if (this->_isPointInFace(cur_face_idx, pt, s, t, u)) {
      // std::cout << "SEARCHED " << visited.size()+1 << " / " <<
      // this->numFaces() << " FACES" << std::endl;
      return cur_face_idx;
    }
    // Otherwise add any unvisited and un-queued neighboring faces to the
    // queue
    else {
      const auto adj_faces = this->_getFacesAdjacentToFace(cur_face_idx);
      for (const auto adj_idx : adj_faces) {
        if (visited.find(adj_idx) == visited.end() &&
            queued.find(adj_idx) == queued.end()) {
          faces_to_search.push(adj_idx);
          queued.insert(adj_idx);
        }
      }
    }

    // Mark this face as visited
    visited.insert(cur_face_idx);
  }
  return 1e10;
}

const bool Sphere::_isPointInFace(const size_t face_idx, const Vec3<float> &pt,
                                  float *s, float *t, float *u) const {
  // 3D coordinates of the triangle
  const auto vertex_idx = this->_getFaceVertexIndices(face_idx);
  const auto A          = _vertices[vertex_idx[0]]->XYZ;
  const auto B          = _vertices[vertex_idx[1]]->XYZ;
  const auto C          = _vertices[vertex_idx[2]]->XYZ;

  // Get plane of triangle
  const auto normal = this->_computeFaceNormal(face_idx);
  const float d     = -normal.dot(A);

  // Project point onto plane along ray from origin
  const auto ray    = pt.normalized();
  const float denom = normal.dot(ray);

  // Because of the winding order of faces, normals on the sphere all point
  // outward. The rays to these points should all be *parallel*. If the inner
  // product is negative, we are checking a face on the opposite side of the
  // sphere.
  if (denom < 0) { return false; }

  // If the ray and the normal are nearly perpendicular, do orthogonal
  // projection
  Vec3<float> proj_pt;
  if (denom < 1e-12) {
    const float dist = normal.dot(pt) + d;
    proj_pt          = pt - normal * dist;
  }
  // Otherwise project the point along the ray
  else {
    const float dist = A.dot(normal) / denom;
    proj_pt          = ray * dist;
  }

  // Compute barycentric coordinates
  // s --> C (vertex_idx[2])
  // t --> B (vertex_idx[1])
  // u --> A (vertex_idx[0])
  const auto BA = B - A;
  const auto CA = C - A;
  const auto N  = BA.cross(CA);
  const auto W  = proj_pt - A;
  *s            = BA.cross(W).dot(N) / N.dot(N);
  *t            = W.cross(CA).dot(N) / N.dot(N);
  *u            = 1 - *s - *t;

  // If the barycentric coordinates meet the validity criteria, this point is
  // in this face.
  return *s >= -1e-3 && *t >= -1e-3 && *u >= -1e-3;
}

const size_t maxOfThreeValues(const float s, const float t, const float u) {
  if (s > t && s > u) { return 0; }
  if (t > s && t > u) { return 1; }
  if (u > s && u > t) { return 2; }
  return 0;
}

// Returns {Oh x OW x K, Oh x Ow x K x 3, Oh x OW x K x 3}
// samples is OH x OW x K x 2
const std::vector<at::Tensor> Sphere::getPlanarConvolutionOperatorFromSamples(
    at::Tensor samples, const size_t order, const bool keepdim,
    const bool nearest) {
  // Create the the output tensor
  const size_t samples_height = samples.size(0);
  const size_t samples_width  = samples.size(1);
  const size_t kernel_size    = (samples.dim() == 3) ? 1 : samples.size(2);

  at::Tensor at_face_indices;
  at::Tensor at_vertex_indices;
  at::Tensor at_weights;
  if (kernel_size > 1 || (kernel_size == 1 && keepdim)) {
    at_face_indices   = torch::zeros({static_cast<int64_t>(samples_height),
                                    static_cast<int64_t>(samples_width),
                                    static_cast<int64_t>(kernel_size)},
                                   at::kLong);
    at_vertex_indices = torch::zeros({static_cast<int64_t>(samples_height),
                                      static_cast<int64_t>(samples_width),
                                      static_cast<int64_t>(kernel_size), 3},
                                     at::kLong);
    at_weights        = torch::zeros({static_cast<int64_t>(samples_height),
                               static_cast<int64_t>(samples_width),
                               static_cast<int64_t>(kernel_size), 3},
                              at::kFloat);
  } else {
    at_face_indices   = torch::zeros({static_cast<int64_t>(samples_height),
                                    static_cast<int64_t>(samples_width)},
                                   at::kLong);
    at_vertex_indices = torch::zeros({static_cast<int64_t>(samples_height),
                                      static_cast<int64_t>(samples_width), 3},
                                     at::kLong);
    at_weights        = torch::zeros({static_cast<int64_t>(samples_height),
                               static_cast<int64_t>(samples_width), 3},
                              at::kFloat);
  }

  const auto at_sample_coords_ptr = samples.data<float>();
  auto at_face_indices_ptr        = at_face_indices.data<int64_t>();
  auto at_vertex_indices_ptr      = at_vertex_indices.data<int64_t>();
  auto at_weights_ptr             = at_weights.data<float>();

  // Create order 0 icosphere to subdivide from
  Sphere subdiv_icosphere = generateIcosphere(0);

  // Get the face index for each sample on the 20-face, order 0 icosphere
  size_t i;
#pragma omp parallel for shared(at_face_indices_ptr, \
                                at_weights_ptr) private(i) schedule(static)
  for (i = 0; i < samples_height * samples_width; i++) {
    // Current kernel sample set
    size_t cur_kernel = i * kernel_size * 2;

    // Go through each element in the kernel and do a BFS from this face until
    // we find the containing face. For this initial matching we just search
    // the entire order 0 icosphere, which is at most 20 faces for each
    // vertex.
    for (size_t j = 0; j < kernel_size; j++) {
      // Convert to XYZ for search
      float x, y, z;
      core::SphericalToXYZ(at_sample_coords_ptr[cur_kernel + j * 2],
                           at_sample_coords_ptr[cur_kernel + j * 2 + 1], x, y,
                           z);
      auto pt3d = Vec3<float>(x, y, z);

      // Compute the barycentric coordinates as part of the search
      float s, t, u;
      const size_t face_idx =
          subdiv_icosphere._searchNearbyFaces(0, pt3d, &s, &t, &u);

      // Add the found face index
      at_face_indices_ptr[i * kernel_size + j] = face_idx;

      // Copy the vertex indices
      auto vertices = subdiv_icosphere._getFaceVertexIndices(face_idx);
      at_vertex_indices_ptr[i * kernel_size * 3 + j * 3]     = vertices[2];
      at_vertex_indices_ptr[i * kernel_size * 3 + j * 3 + 1] = vertices[1];
      at_vertex_indices_ptr[i * kernel_size * 3 + j * 3 + 2] = vertices[0];

      // Add the barycentric weights
      if (nearest) {
        const size_t maxVal = maxOfThreeValues(s, t, u);
        at_weights_ptr[i * kernel_size * 3 + j * 3]     = maxVal == 0 ? 1 : 0;
        at_weights_ptr[i * kernel_size * 3 + j * 3 + 1] = maxVal == 1 ? 1 : 0;
        at_weights_ptr[i * kernel_size * 3 + j * 3 + 2] = maxVal == 2 ? 1 : 0;
      } else {
        at_weights_ptr[i * kernel_size * 3 + j * 3]     = s;
        at_weights_ptr[i * kernel_size * 3 + j * 3 + 1] = t;
        at_weights_ptr[i * kernel_size * 3 + j * 3 + 2] = u;
      }
    }
  }

  // Now we perform a subdivision on each face, tracking the face_idx matching
  // at each phase. For each vertex, we will need to check at most 4 faces at
  // each subdivision
  for (size_t k = 0; k < order; k++) {
    // Subdivide the canonical icosphere once
    subdiv_icosphere = subdiv_icosphere._singleLoopSubdivision();

// Now all the face indices in the new one are mapped to the previous by the
// relationship (i --> 4*i + {0,1,2,3}). We leverage this to reduce the search
// space. (Except for a rare case where the B-spline interpolation changes a
// boundary slightly enough that a point on the boundary moves to a different
// triangle)
#pragma omp parallel for shared(at_face_indices_ptr, at_vertex_indices_ptr, \
                                at_weights_ptr) private(i) schedule(static)
    for (i = 0; i < samples_height * samples_width; i++) {
      // Current kernel sample set
      size_t cur_kernel = i * kernel_size * 2;

      // Iterate through kernel
      for (size_t j = 0; j < kernel_size; j++) {
        // Previous order face index
        const size_t prev_face_idx = at_face_indices_ptr[i * kernel_size + j];

        // Convert kernel sample to XYZ for search
        float x, y, z;
        core::SphericalToXYZ(at_sample_coords_ptr[cur_kernel + j * 2],
                             at_sample_coords_ptr[cur_kernel + j * 2 + 1], x,
                             y, z);

        // Find the new face coordinate by starting at the center face of the
        // subdivision. Except for some very rare border cases where the
        // B-spline interpolation moves a point from one subdivided triangle
        // to another, the new face should be one of the four subdivisions
        float s, t, u;
        const int64_t new_face_idx = subdiv_icosphere._searchNearbyFaces(
            4 * prev_face_idx + 3, Vec3<float>(x, y, z), &s, &t, &u);
        at_face_indices_ptr[i * kernel_size + j] = new_face_idx;

        // Copy the vertex indices
        auto vertices = subdiv_icosphere._getFaceVertexIndices(new_face_idx);
        at_vertex_indices_ptr[i * kernel_size * 3 + j * 3]     = vertices[2];
        at_vertex_indices_ptr[i * kernel_size * 3 + j * 3 + 1] = vertices[1];
        at_vertex_indices_ptr[i * kernel_size * 3 + j * 3 + 2] = vertices[0];

        // Copy the barycentric weights
        if (nearest) {
          const size_t maxVal = maxOfThreeValues(s, t, u);
          at_weights_ptr[i * kernel_size * 3 + j * 3] = maxVal == 0 ? 1 : 0;
          at_weights_ptr[i * kernel_size * 3 + j * 3 + 1] =
              maxVal == 1 ? 1 : 0;
          at_weights_ptr[i * kernel_size * 3 + j * 3 + 2] =
              maxVal == 2 ? 1 : 0;
        } else {
          at_weights_ptr[i * kernel_size * 3 + j * 3]     = s;
          at_weights_ptr[i * kernel_size * 3 + j * 3 + 1] = t;
          at_weights_ptr[i * kernel_size * 3 + j * 3 + 2] = u;
        }
      }
    }
  }

  return {at_face_indices, at_vertex_indices, at_weights};
}

}  // namespace sphere
}  // namespace mapped_conv