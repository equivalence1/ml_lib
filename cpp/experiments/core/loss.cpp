#include "loss.h"
#include "model.h"

torch::Tensor ZeroLoss::value(const torch::Tensor &outputs, const torch::Tensor &targets) const {
    auto res = torch::zeros({1}, torch::kFloat32);
    res = experiments::correctDevice(res, outputs.device());
    return res;
}