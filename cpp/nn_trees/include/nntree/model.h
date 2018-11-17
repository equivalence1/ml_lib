#pragma once

#include "tensor.h"

#include <functional>

namespace nntree {
namespace core {

;
class Model {
public:
  virtual void Apply(Tensor<double>& x, Tensor<double>& res) const = 0;
//  virtual std::vector<double> Backward(std::function<Tensor<double>* (const Model*)>) const = 0;
  virtual void AdjustParams(const std::vector<double>&) = 0;
};

}
}
