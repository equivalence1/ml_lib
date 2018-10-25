#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <nntree/dataset.h>
#include "least_squares.h"
#include "convolution.h"

// TODO(equivalence1) make it a collection of separate files

namespace py = pybind11;

namespace nntree {
namespace pymodule {

template<typename T>
struct PyBufferInfo: public core::buffer_info<T> {
  explicit PyBufferInfo(py::buffer_info &&buff) {
    this->ptr = (T*)buff.ptr;
    this->size = buff.size;
    this->ndim = buff.ndim;
    this->shape = buff.shape;
    this->strides = buff.strides;
  }
};

// We have to store x and y arrays in DataSet
// otherwise there will be memory leaks
// TODO(equivalence1) for now all types are floats
template<typename IN_T = float, typename OUT_T = float>
class DataSet: public core::DataSet<IN_T, OUT_T> {
public:
  DataSet(py::array_t<IN_T> x, py::array_t<OUT_T> y)
      : core::DataSet<IN_T, OUT_T>(PyBufferInfo<IN_T>(x.request()), PyBufferInfo<OUT_T>(y.request()))
      , x_(std::move(x))
      , y_(std::move(y)) {}

  // Just a test function to check that we correctly accept data from python
  py::array_t<float> TestPrint(ssize_t size) {
    auto X = x_.request(false);
    printf("%zu %zu\n", X.size / X.itemsize, X.itemsize);
    auto res = core::convolution((float *)X.ptr, size, 1, 1); // least_squares((float *)X.ptr, (float *)y_.request(false).ptr, X.size / X.itemsize, X.itemsize);
//    size = std::min(size, buff.size / buff.itemsize);
//    for (ssize_t i = 0; i < size; i++) {
//        printf("%f ", ((float*)buff.ptr)[i]);
//    }
    py::array_t<float> result = py::array_t<float>(res.size());
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;
    for(size_t i = 0; i < res.size(); ++i) {
      ptr[i] = res[i];
    }
    //result.resize({size, 1});
    return result;
  }

private:
  py::array_t<IN_T> x_;
  py::array_t<OUT_T> y_;
};

py::array_t<float> least_squares(DataSet<float, float> ds) {
  auto buff = core::LeastSquares(ds);
  py::array_t<float> res = py::array_t<float>((size_t)buff.size);
  auto res_buff = res.request();
  auto res_buff_ptr = (float*)res_buff.ptr;
  for (int i = 0; i < res.size(); i++) {
    res_buff_ptr[i] = buff.ptr[i];
  }
  res.resize(buff.shape);
  return res;
}

PYBIND11_MODULE(nntreepy, m) {
  m.doc() = "nntreepy module to work with intel mkl through python";
  py::class_<DataSet<float, float>> dataset(m, "DataSet");
  dataset.def(py::init<py::array_t<float>, py::array_t<float>> ());
  dataset.def("test_print", &DataSet<float, float>::TestPrint);
  m.def("least_squares", &least_squares);
}

}
}
