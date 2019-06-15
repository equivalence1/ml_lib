#pragma once

#include "experiments/core/tensor_pair_dataset.h"

#include <torch/torch.h>

#include <utility>

namespace experiments::svhn {

std::pair<TensorPairDataset, TensorPairDataset> read_dataset(
        int trainLimit = -1,
        int testLimit = -1);

}
