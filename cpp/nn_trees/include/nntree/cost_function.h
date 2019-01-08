#pragma once

#include <functional>

#include "model.h"
#include "dataset.h"
#include "tensor.h"

namespace nntree {
    namespace core {

        class CostFunction {
        public:
            virtual double Apply(DataSet<double, double>&, const Model&) const = 0;
//  virtual std::function<Tensor<double> (const Model*)> Backward(DataSet<double, double>&) const = 0;
        };

    }
}
