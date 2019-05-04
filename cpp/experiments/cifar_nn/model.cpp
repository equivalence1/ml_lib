#include "model.h"
using namespace experiments;

LinearCifarClassifier::LinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor LinearCifarClassifier::forward(torch::Tensor x) {
    return fc1_->forward(x.view({x.size(0), -1}));
}

torch::Tensor experiments::ConvModel::forward(torch::Tensor x) {
    x = conv()->forward(x);
    return classifier()->forward(x);
}

void experiments::ConvModel::train(bool on) {
    conv()->train(on);
    classifier()->train(on);
}

torch::Tensor experiments::Classifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    auto result = classifier_->forward(x);
    if (baseline_)  {
        result += baseline_->forward(x);
    }
    return result;
}
