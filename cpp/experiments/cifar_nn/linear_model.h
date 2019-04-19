#pragma once

#include "model.h"
#include "linear_function.h"

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <cstdint>

class LinearModel : public experiments::Model {
public:
    LinearModel(uint32_t in, uint32_t out) : experiments::Model() {
//        weights_ = register_parameter("weights", torch::zeros({out, in}, torch::kFloat32));
//        biases_ = register_parameter("biases", torch::zeros({out}, torch::kFloat32));
//        torch::nn::init::xavier_uniform_(weights_);
        fc1_ = register_module("fc1_", torch::nn::Linear(16 * 5 * 5, 10));
    }

    torch::Tensor forward(torch::Tensor x) override {
//        LinearFunction f;
//        return f.apply({x, weights_, biases_})[0];
//        return torch::mm(weights_, x.t()).t() + biases_;
        x = x.view({x.size(0), -1});
        return fc1_->forward(x);
    }

private:
//    torch::Tensor weights_;
//    torch::Tensor biases_;
    torch::nn::Linear fc1_{nullptr};
};
