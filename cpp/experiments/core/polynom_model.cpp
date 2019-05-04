#include "polynom_model.h"
#include <core/torch_helpers.h>

torch::Tensor PolynomModel::forward(torch::Tensor samples){
    VERIFY(polynom_, "set polynom first");
    const int batchSize = samples.size(0);
    auto yDim = TorchHelpers::totalSize(samples) / batchSize;
    samples = samples.reshape({batchSize, yDim}).contiguous();
    auto samplesDevice = samples.device();
    if (samplesDevice.is_cpu()) {
      auto polynomForward = PolynomForward(polynom_);
      return polynomForward.apply({samples.to(torch::kCPU)})[0].to(samplesDevice);
    } else {
      VERIFY(samplesDevice.is_cuda(), "error: we work only on CPU or GPU: " << samplesDevice);
      if (polynomCuda_ == nullptr) {
        polynomCuda_ = std::make_shared<PolynomCuda>(polynom_);
      }
      auto polynomForward = PolynomForwardCuda(polynomCuda_);
      return polynomForward.apply({samples})[0];
    }
}
