#pragma once

#include "buffer_info.h"

#include <functional>

namespace nntree {
namespace core {

class Model {
public:
  virtual buffer_info<double> Apply(buffer_info<double>) const = 0;
  virtual std::vector<double> Backward(std::function<double (const Model*)>) const = 0;
  virtual void AdjustParams(const std::vector<double>&) = 0;
};

}
}
