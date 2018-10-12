#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class DataSet {
public:
    explicit DataSet(py::array_t<float> arr);
    ~DataSet();
private:
    py::array_t<float> data_;
};
