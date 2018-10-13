#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class DataSet {
public:
    explicit DataSet(py::array_t<float> X, py::array_t<float> y);
    void TestPrint(ssize_t size);
    virtual ~DataSet();
private:
    py::array_t<float> mx_x_;
    py::array_t<float> y_;
};
