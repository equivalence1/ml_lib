#include "dataset.h"

DataSet::DataSet(py::array_t<float> arr) {
    data_ = std::move(arr);
}

DataSet::~DataSet() = default;

PYBIND11_MODULE(example, m) {
    m.doc() = "nn_intro module to work with intel mkl through python";
    py::class_<DataSet>(m, "DataSet")
            .def(py::init<py::array_t<float>>());
}
