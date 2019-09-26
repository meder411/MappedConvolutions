#ifndef SPHERE_H_
#define SPHERE_H_

#include "half_edge.h"
#include "quadric.h"
#include "vec3.h"

#include <torch/extension.h>

#include <cstddef>
#include <memory>
#include <set>
#include <vector>

namespace mapped_conv {
namespace sphere {
class Sphere {
 private:
  std::vector<std::shared_ptr<HalfEdge>> _half_edges;
  std::vector<std::shared_ptr<Face>> _faces;
  std::vector<std::shared_ptr<Vertex>> _vertices;

  const Vec3<float> _getFaceBarycenter(const size_t face_idx) const;
  const std::vector<size_t> _getFaceVertexIndices(const size_t face_idx) const;
  const std::vector<size_t> _getVerticesAdjacentToVertex(
      const size_t vert_idx, const bool duplicate) const;
  const std::vector<size_t> _getFacesAdjacentToVertex(
      const size_t vert_idx, const bool duplicate) const;
  const std::vector<size_t> _getFacesAdjacentToFace(
      const size_t face_idx) const;
  const std::vector<size_t> _getHalfEdgesPointingToVertex(
      const size_t vert_idx) const;
  const size_t _getVertexValence(const size_t vert_idx) const;

  template <typename iterator_type>
  const std::shared_ptr<Face> _getNorthMostFace(
      const iterator_type &faces) const;
  template <typename iterator_type>
  const std::shared_ptr<Vertex> _getNorthMostVertex(
      const iterator_type &vertices) const;
  const std::vector<size_t> _getFaceConvolutionOperatorIndicesDeg1(
      const std::shared_ptr<Face> &face) const;
  const std::vector<size_t> _getFaceConvolutionOperatorIndicesDeg2(
      const std::shared_ptr<Face> &face, const bool dilation) const;
  const Sphere _singleLoopSubdivision() const;
  const Vec3<float> _computeFaceNormal(const size_t face_idx) const;
  const float _computePlaneDistance(const Vec3<float> &normal,
                                    const Vec3<float> &vertex) const;
  void _computeVertexQuadrics(std::vector<Quadric<float>> &quad) const;
  const bool _isPointInFace(const size_t face_idx, const Vec3<float> &pt,
                            float *s, float *t, float *u) const;
  const size_t _searchNearbyFaces(const size_t face_idx, const Vec3<float> &pt,
                                  float *s, float *t, float *u) const;

  void _buildSphere(const float *pts, const size_t num_pts,
                    const int64_t *faces, const size_t num_faces);

 public:
  typedef std::unique_ptr<Sphere> unique_ptr;

  Sphere();
  Sphere(const Sphere &sphere);
  Sphere(torch::Tensor pts, torch::Tensor faces);
  Sphere(const std::vector<float> &pts, const std::vector<int64_t> &faces);

  void addFace(std::shared_ptr<Face> f);
  void addHalfEdge(std::shared_ptr<HalfEdge> he);
  void addVertex(std::shared_ptr<Vertex> v);
  inline const size_t numVertices() const;
  inline const size_t numFaces() const;
  const float getVertexResolution() const;

  static const Sphere generateIcosphere(const size_t order);

  // Returns OH x OW x Kh*Kw
  static const std::vector<torch::Tensor>
  getPlanarConvolutionOperatorFromSamples(torch::Tensor samples,
                                          const size_t order,
                                          const bool keepdim = false,
                                          const bool nearest = false);

  // Returns V x 3
  const torch::Tensor getVertices() const;

  // Returns F x 3 x 3, F sets of 3 rows of points (each row is a point)
  const torch::Tensor getAllFaceVertexCoordinates() const;

  // Returns F x 3, F rows of 3 indices
  const torch::Tensor getAllFaceVertexIndices() const;

  // Returns F x 3
  const torch::Tensor getFaceBarycenters() const;

  // Returns F x 3
  const torch::Tensor getAdjacentFaceIndicesToFaces() const;

  // Returns <variable>
  const torch::Tensor getAdjacentFaceIndicesToVertices() const;

  // Returns variable
  const torch::Tensor getAdjacentVertexIndicesToVertices() const;

  const torch::Tensor getAllFaceConvolutionOperatorIndicesDeg1() const;

  const torch::Tensor getAllFaceConvolutionOperatorIndicesDeg2(
      const bool dilation) const;

  const Sphere loopSubdivision(const size_t iterations) const;

  const torch::Tensor getOriginalFaceIndices() const;

  void scaleVertices(torch::Tensor scale);

  const torch::Tensor getFaceNormals() const;

  const torch::Tensor getVertexNormals() const;

  const Sphere quadricEdgeCollapse(const size_t target_num_faces) const;
};

const size_t Sphere::numVertices() const {
  return _vertices.size();
}
const size_t Sphere::numFaces() const {
  return _faces.size();
}

}  // namespace sphere
}  // namespace mapped_conv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<mapped_conv::sphere::Sphere>(m, "Sphere")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def("num_vertices", &mapped_conv::sphere::Sphere::numVertices)
      .def("num_faces", &mapped_conv::sphere::Sphere::numFaces)
      .def("get_vertices", &mapped_conv::sphere::Sphere::getVertices)
      .def("get_all_face_vertex_coords",
           &mapped_conv::sphere::Sphere::getAllFaceVertexCoordinates)
      .def("get_all_face_vertex_indices",
           &mapped_conv::sphere::Sphere::getAllFaceVertexIndices)
      .def("get_face_barycenters",
           &mapped_conv::sphere::Sphere::getFaceBarycenters)
      .def("get_adjacent_face_indices_to_faces",
           &mapped_conv::sphere::Sphere::getAdjacentFaceIndicesToFaces)
      .def("get_adjacent_face_indices_to_vertices",
           &mapped_conv::sphere::Sphere::getAdjacentFaceIndicesToVertices)
      .def("get_adjacent_vertex_indices_to_vertices",
           &mapped_conv::sphere::Sphere::getAdjacentVertexIndicesToVertices)
      .def("loop_subdivision", &mapped_conv::sphere::Sphere::loopSubdivision)
      .def("get_original_face_indices",
           &mapped_conv::sphere::Sphere::getOriginalFaceIndices)
      .def("get_conv_operator_deg1",
           &mapped_conv::sphere::Sphere::
               getAllFaceConvolutionOperatorIndicesDeg1)
      .def("get_conv_operator_deg2",
           &mapped_conv::sphere::Sphere::
               getAllFaceConvolutionOperatorIndicesDeg2)
      .def("scale_vertices", &mapped_conv::sphere::Sphere::scaleVertices)
      .def("get_face_normals", &mapped_conv::sphere::Sphere::getFaceNormals)
      .def("get_vertex_normals",
           &mapped_conv::sphere::Sphere::getVertexNormals)
      .def("get_vertex_resolution",
           &mapped_conv::sphere::Sphere::getVertexResolution)
      .def("get_planar_conv_operator_from_samples",
           &mapped_conv::sphere::Sphere::
               getPlanarConvolutionOperatorFromSamples)
      .def("generate_icosphere",
           &mapped_conv::sphere::Sphere::generateIcosphere);
}

#endif