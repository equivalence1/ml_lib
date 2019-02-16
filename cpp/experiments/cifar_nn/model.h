#pragma once

#include "tensor_pair_dataset.h"

#include <torch/torch.h>
#include <memory>

class Model: public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

using ModelPtr = std::shared_ptr<Model>;

class CompositionalModel : public Model {
public:
    CompositionalModel(ModelPtr first, ModelPtr second)
            : first_(std::move(first))
            , second_(std::move(second)) {

    }

    CompositionalModel(const CompositionalModel& model) = default;

    torch::Tensor forward(torch::Tensor x) override {
        return second_->forward(first_->forward(x));
    }

private:
    ModelPtr first_;
    ModelPtr second_;
};
