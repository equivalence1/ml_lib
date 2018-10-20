#pragma once

#include <nntree/dataset.h>
#include <nntree/buffer_info.h>

namespace nntree {
namespace core {

float* least_squares(float* X, float* y, int rows, int colsX, int colsY);
struct buffer_info<float> least_squares(core::DataSet<float, float> &ds);

}
}
