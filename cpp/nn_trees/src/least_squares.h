#pragma once

#include "nntree/dataset.h"
#include "nntree/tensor.h"

namespace nntree {
namespace core {

void LeastSquares(Tensor<double>& X, Tensor<double>& y, Tensor<double>& res);
void LeastSquares(DataSet<double, double>& ds, Tensor<double>& res);

}
}
