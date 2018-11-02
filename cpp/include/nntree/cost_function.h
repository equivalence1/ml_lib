#pragma once

#include "model.h"
#include "dataset.h"

#include <functional>

namespace nntree {
namespace core {

class CostFunction {
public:
  virtual double Apply(DataSet<double, double>&, const Model&) const = 0;
  virtual std::function<buffer_info<double> (const Model*)> Backward(DataSet<double, double>&) const = 0;
};

}
}
