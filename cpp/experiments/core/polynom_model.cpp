#include "polynom_model.h"
#include <core/torch_helpers.h>

torch::Tensor PolynomModel::forward(torch::Tensor samples) {
    samples = experiments::correctDevice(samples, *this);
    VERIFY(polynom_, "set polynom first");
    const int batchSize = samples.size(0);
    auto yDim = TorchHelpers::totalSize(samples) / batchSize;
    samples = samples.reshape({batchSize, yDim}).contiguous();
    if (this->device().is_cpu()) {
      auto polynomForward = PolynomForward(polynom_);
      return polynomForward.apply({samples})[0];
    } else {
      if (polynomCuda_ == nullptr) {
        polynomCuda_ = std::make_shared<PolynomCuda>(polynom_);
      }
      auto polynomForward = PolynomForwardCuda(polynomCuda_);
      return polynomForward.apply({samples})[0];
    }
}
