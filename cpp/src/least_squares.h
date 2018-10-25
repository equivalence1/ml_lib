#pragma once

#include <nntree/dataset.h>
#include <nntree/buffer_info.h>

namespace nntree {
namespace core {

float* LeastSquares(float *X, float *y, int rows, int colsX, int colsY);
struct buffer_info<float> LeastSquares(core::DataSet<float, float> &ds);

}
}
