#include "dataset.h"

#include <stdio.h>

DataSet::DataSet(py::array_t<float> arr) {
    data_ = std::move(arr);
}

// Just a test function to check that we correctly accept data from python
void DataSet::TestPrint(ssize_t size) {
    auto buff = data_.request();
    size = std::min(size, buff.size / buff.itemsize);
    for (ssize_t i = 0; i < size; i++) {
        printf("%f ", ((float*)buff.ptr)[i]);
    }
}

DataSet::~DataSet() = default;

PYBIND11_MODULE(example, m) {
    m.doc() = "nn_intro module to work with intel mkl through python";
    py::class_<DataSet> dataset(m, "DataSet");
    dataset.def(py::init<py::array_t<float>>());
    dataset.def("test_print", &DataSet::TestPrint);
}
