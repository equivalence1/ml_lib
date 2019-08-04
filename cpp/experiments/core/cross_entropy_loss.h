#pragma once

#include "loss.h"
#include "model.h"

#include <torch/torch.h>

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss() = default;

    torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
        auto cTargets = experiments::correctDevice(targets, outputs.device());
        return torch::nll_loss(torch::log_softmax(outputs, 1), cTargets);
    }
};
