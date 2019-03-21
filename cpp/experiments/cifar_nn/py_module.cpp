#include "lenet.h"
#include "tensor_pair_dataset.h"
#include "linear_train.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

torch::Tensor npToTorch(py::buffer_info& x_buff, torch::ScalarType t) {
    std::vector<int64_t> xStrides(x_buff.strides.size());
    std::transform(x_buff.strides.begin(), x_buff.strides.end(), xStrides.begin(),
                   [&](auto el){return el / x_buff.itemsize;});

    std::vector<int64_t> shape;
    for (auto  val : x_buff.shape) {
        shape.push_back(val);
    }
    return torch::from_blob(x_buff.ptr,
                            shape,
                            xStrides,
                            t);
}

py::array_t<float> torchToNp(torch::Tensor& x) {
    std::vector<int64_t> sizes(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
    std::transform(strides.begin(), strides.end(), strides.begin(),
                   [](int64_t stride){return stride * sizeof(float);});

    py::array_t<float> a(sizes, strides, (const float*)x.data_ptr());
    return a;
}

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
        x_ = npToTorch(x, torch::kFloat32);
        y_ = npToTorch(y, torch::kLong);
    }

    py::array_t<float> py_x_;
    py::array_t<long> py_y_;
};

//class PyModel : public Model {
//public:
//    py::array_t<float> forward_np(py::array_t<float> x) {
//        auto x_buff = x.request();
//        auto torch_x = npToTorch(x_buff);
//        auto res = this->forward(torch_x);
//        return torchToNp(res);
//    }
//};

class PyLeNet : public LeNet {
public:
    PyLeNet() : LeNet() {}

    py::array_t<float> forward_np(py::array_t<float> x) {
        py::buffer_info x_buff = x.request();
        torch::Tensor torch_x = npToTorch(x_buff, torch::kFloat32);
        torch::Tensor res = this->forward(torch_x);
        return torchToNp(res);
    }
};

class PyWrapperModel : public Model {
public:
    explicit PyWrapperModel(ModelPtr model) : model_(std::move(model)) {}

    torch::Tensor forward(torch::Tensor x) override {
        return model_->forward(x);
    }

    py::array_t<float> forward_np(py::array_t<float> x) {
        py::buffer_info x_buff = x.request();
        torch::Tensor torch_x = npToTorch(x_buff, torch::kFloat32);
        torch::Tensor res = this->forward(torch_x);
        return torchToNp(res);
    }

private:
    ModelPtr model_;
};

class PyLeNetLinearTrainer : public LeNetLinearTrainer {
public:
    PyLeNetLinearTrainer(uint32_t it_global,
            uint32_t it_repr,
            uint32_t it_decision) : LeNetLinearTrainer(it_global, it_repr, it_decision) {}

    ModelPtr getTrainedModel_py(PyTensorPairDataset* ds) {
        return std::make_shared<PyWrapperModel>(this->getTrainedModel(*ds));
    }
};

void py_train_model(std::shared_ptr<PyLeNet> model, PyTensorPairDataset* ds, int epochs = 10) {
    DefaultSGDOptimizer optim(epochs);
    auto loss = std::make_shared<CrossEntropyLoss>();
    optim.train(*ds, loss, model);
}

PYBIND11_MODULE(cifar_nn_py, m) {
    m.doc() = "experiments";

    // Dataset
    py::class_<PyTensorPairDataset> dataset(m, "PyDataset");
    dataset.def(py::init<py::array_t<float>, py::array_t<long>>());

    // Models
//    py::class_<PyModel, std::shared_ptr<PyModel>> model(m, "PyModel");
//    model.def("forward", &PyModel::forward_np);

    py::class_<PyLeNet, std::shared_ptr<PyLeNet>> lenet(m, "PyLeNet");
    lenet.def(py::init<>());
    lenet.def("forward", &PyLeNet::forward_np);

    py::class_<PyWrapperModel, std::shared_ptr<PyWrapperModel>> wrapper_model(m, "PyWrapperModel");
    wrapper_model.def("forward", &PyWrapperModel::forward_np);

    // Training
    py::class_<PyLeNetLinearTrainer> lenet_linear_trainer(m, "PyLeNetLinearTrainer");
    lenet_linear_trainer.def(py::init<uint32_t, uint32_t, uint32_t>());
    lenet_linear_trainer.def("get_trained_model", &PyLeNetLinearTrainer::getTrainedModel_py);

    m.def("train", py_train_model, py::arg("model"), py::arg("ds"), py::arg("epochs") = 10);
}
