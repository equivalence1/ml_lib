#pragma once

#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>
#include <memory>

class Initializer {
public:
    virtual void init(const TensorPairDataset& ds, LossPtr loss, experiments::ModelPtr* representation, experiments::ModelPtr* decisionFunc) = 0;

    virtual ~Initializer() = default;
};

using InitializerPtr = std::shared_ptr<Initializer>;

class NoopInitializer : public Initializer {
public:
    NoopInitializer() = default;

    void init(const TensorPairDataset& ds, LossPtr loss, experiments::ModelPtr* representation, experiments::ModelPtr* decisionFunc) override {
    }
};
