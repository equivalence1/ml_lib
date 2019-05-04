#pragma once


#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>

#include <cstdint>

class LayerNorm : public torch::nn::Module  {
public:
    LayerNorm(uint32_t dim)  {
        alpha_ = register_parameter("weights", torch::ones({dim}, torch::kFloat32));
        beta_ = register_parameter("biases", torch::zeros({dim}, torch::kFloat32));
    }

    torch::Tensor forward(torch::Tensor x)  {
        x = x.view({x.size(0), -1});
        auto mean = x.mean({0}, true);
        auto sd = x.std({0}, true);
        return alpha_ * (x - mean) / (sd + 1e-5) + beta_;
    }

private:
    torch::Tensor alpha_;
    torch::Tensor beta_;

};

using LayerNormPtr = std::shared_ptr<LayerNorm>;
