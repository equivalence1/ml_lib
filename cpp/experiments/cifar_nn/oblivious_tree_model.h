#pragma once

#include "model.h"
#include "oblivious_tree_function.h"

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <cstdint>

class ObliviousTreeModel : public experiments::Model {
public:
    ObliviousTreeModel(ModelPtr mod) : f(mod) {}

    torch::Tensor forward(torch::Tensor x) override {
        auto rs = f.apply({x})[0];
        return rs;
    }

    ObliviousTreeFunction f;
};