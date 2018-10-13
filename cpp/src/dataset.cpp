#include "dataset.h"
#include "least_squares.h"

#include <stdio.h>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-pass-by-value"
#pragma ide diagnostic ignored "performance-unnecessary-value-param"
DataSet::DataSet(py::array_t<float> x, py::array_t<float> y): mx_x_(x), y_(y) {}
#pragma clang diagnostic pop

// Just a test function to check that we correctly accept data from python
void DataSet::TestPrint(ssize_t size) {
    auto X = mx_x_.request(false);
    printf("%zu %zu\n", X.size / X.itemsize, X.itemsize);
    auto res = least_squares((float *)X.ptr, (float *)y_.request(false).ptr, X.size / X.itemsize, X.itemsize);
    printf("%f\n", *res);
//    size = std::min(size, buff.size / buff.itemsize);
//    for (ssize_t i = 0; i < size; i++) {
//        printf("%f ", ((float*)buff.ptr)[i]);
//    }
//
    
}

DataSet::~DataSet() = default;

PYBIND11_MODULE(example, m) {
    m.doc() = "nn_intro module to work with intel mkl through python";
    py::class_<DataSet> dataset(m, "DataSet");
    dataset.def(py::init<py::array_t<float>, py::array_t<float>>());
    dataset.def("test_print", &DataSet::TestPrint);
}
