#pragma once

#include "tensor_pair_dataset.h"

#include <torch/torch.h>
#include <memory>

namespace experiments {

class Model : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

using ModelPtr = std::shared_ptr<Model>;

class ConvModel : public Model {
public:
    virtual ModelPtr conv() = 0;

    virtual ModelPtr classifier() = 0;
};

}

class CompositionalModel : public experiments::Model {
public:
    CompositionalModel(experiments::ModelPtr first, experiments::ModelPtr second) {
        first_ = register_module("first_", std::move(first));
        second_ = register_module("second_", std::move(second));
    }

    torch::Tensor forward(torch::Tensor x) override {
        return second_->forward(first_->forward(x));
    }

private:
    experiments::ModelPtr first_;
    experiments::ModelPtr second_;
};
