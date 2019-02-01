#pragma once

#include "tensor_pair_dataset.h"

#include <torch/torch.h>

#include <memory>

class Model: public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

void tain_model(Model *model, TensorPairDataset *d, int epochs = 10);
