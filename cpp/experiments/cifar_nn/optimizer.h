#pragma once

#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>
#include <memory>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void train(const TensorPairDataset& ds, const Loss& loss, ModelPtr model) const = 0;
    virtual void train(const torch::Tensor& ds, const torch::Tensor& target, const Loss& loss, ModelPtr model) const = 0;
};

using OptimizerPtr = std::shared_ptr<Optimizer>;
