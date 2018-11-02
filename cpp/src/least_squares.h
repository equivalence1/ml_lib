#pragma once

#include <nntree/dataset.h>
#include <nntree/buffer_info.h>

namespace nntree {
namespace core {

double* LeastSquares(double *X, double *y, int rows, int colsX, int colsY);
struct buffer_info<double> LeastSquares(core::DataSet<double, double> &ds);

}
}
