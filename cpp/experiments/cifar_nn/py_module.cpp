#include "simple_net.h"
#include "tensor_pair_dataset.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PyTensorPairDataset: public TensorPairDataset {
public:
    PyTensorPairDataset(py::array_t<float> x, py::array_t<long> y):
            PyTensorPairDataset(x.request(), y.request()) {
        py_x_ = std::move(x);
        py_y_ = std::move(y);
    }

    ~PyTensorPairDataset() override {
        py_x_.release();
        py_y_.release();
    }

private:
    PyTensorPairDataset(py::buffer_info x, py::buffer_info y):
            TensorPairDataset() {
        std::vector<int64_t> xStrides(x.strides.size());
        std::transform(x.strides.begin(), x.strides.end(), xStrides.begin(),
                       [&](auto el){return el / x.itemsize;});

        std::vector<int64_t> yStrides(y.strides.size());
        std::transform(y.strides.begin(), y.strides.end(), yStrides.begin(),
                       [&](auto el){return el / y.itemsize;});

        x_ = torch::from_blob(x.ptr,
                              x.shape,
                              xStrides,
                              torch::kFloat32);
        y_ = torch::from_blob(y.ptr,
                              y.shape,
                              yStrides,
                              torch::kLong);
    }

    py::array_t<float> py_x_;
    py::array_t<long> py_y_;
};

#include <iostream>

class PySimpleNet: public SimpleNet {
public:
    PySimpleNet(): SimpleNet() {}

    py::array_t<float> forward_np(py::array_t<float> x) {
        auto x_buff = x.request();

        std::vector<int64_t> xStrides(x_buff.strides.size());
        std::transform(x_buff.strides.begin(), x_buff.strides.end(), xStrides.begin(),
                       [&](auto el){return el / x_buff.itemsize;});

        auto torch_x = torch::from_blob(x_buff.ptr,
                                        x_buff.shape,
                                        xStrides,
                                        torch::kFloat32);

        auto res = this->forward(torch_x);

        std::vector<int64_t> sizes(res.sizes().begin(), res.sizes().end());
        std::vector<int64_t> strides(res.strides().begin(), res.strides().end());
        std::transform(strides.begin(), strides.end(), strides.begin(),
                [](int64_t stride){return stride * sizeof(float);});

        std::cout << "sizes: " << sizes.size() << std::endl;
        for (auto el: sizes)
            std::cout << el << " ";
        std::cout << std::endl;

        std::cout << "strides: " << strides.size() << std::endl;
        for (auto el: strides)
            std::cout << el << " ";
        std::cout << std::endl;

        py::array_t<float> a(sizes, strides, (const float*)res.data_ptr());

        std::cout << "returning" << std::endl;
        return a;
    }
};

void py_train_model(PySimpleNet *model, PyTensorPairDataset *ds, int epochs = 10) {
    train_model(model, ds, epochs);
}

PYBIND11_MODULE(cifar_nn_py, m) {
    m.doc() = "experiments";

    py::class_<PyTensorPairDataset> dataset(m, "Dataset");
    dataset.def(py::init<py::array_t<double>, py::array_t<double>>());
//
//            m.def("least_squares", &least_squares);
//
    py::class_<PySimpleNet, std::shared_ptr<PySimpleNet>> simple_net(m, "SimpleNet");
    simple_net.def(py::init<>());
    simple_net.def("forward", &PySimpleNet::forward_np);

    m.def("train", py_train_model, py::arg("model"), py::arg("ds"), py::arg("epochs") = 10);
}
