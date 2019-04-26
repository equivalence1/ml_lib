#pragma once

#include "tensor_pair_dataset.h"

#include <torch/torch.h>
#include <memory>

namespace experiments {

class Model : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    //WTF torch, this should be default behaviour
    void train(bool on = true) override {
        for (auto& param : parameters()) {
            param.set_requires_grad(on);
        }
        torch::nn::Module::train(on);
    }

};

using ModelPtr = std::shared_ptr<Model>;

class ConvModel : public Model {
public:
    virtual ModelPtr conv() = 0;

    virtual ModelPtr classifier() = 0;

    void train(bool on = true) override {
        conv()->train(on);
        classifier()->train(on);
    }
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

    virtual void train(bool on = true) {
        first_->train(on);
        second_->train(on);
    }


private:
    experiments::ModelPtr first_;
    experiments::ModelPtr second_;
};
