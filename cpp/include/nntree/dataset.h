#pragma once

#include <memory>
#include <string>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdint>

#include "nd_array.h"
#include "buffer_info.h"

namespace nntree {
namespace core {

/**
 * @class DataSet
 * @brief represents data set.
 *
 * Basically, it's just an input data array, plus
 * output data array, plus some API.
 */
template<typename IN_T, typename OUT_T>
class DataSet {
public:
  explicit DataSet(buffer_info<IN_T> x, buffer_info<OUT_T> y): x_(x), y_(y) {}

  buffer_info<IN_T>& GetInput() {
    return x_;
  }

  buffer_info<OUT_T>& GetOutput() {
    return y_;
  }

  // super ugly, leads to leaks, but for now allows to check SGD
  DataSet RandomBatch(size_t batch_size) {
    batch_size = std::min(batch_size, (size_t)x_.size);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, (int)x_.size - 1);

    buffer_info<IN_T> batch_x;
    buffer_info<OUT_T> batch_y;

    std::vector<IN_T> vx;
    std::vector<OUT_T> vy;

    size_t x_row_sz = 1;
    size_t y_row_sz = 1;

    std::for_each(x_.shape.begin() + 1, x_.shape.end(), [&](int64_t sz){x_row_sz *= sz;});
    std::for_each(y_.shape.begin() + 1, y_.shape.end(), [&](int64_t sz){y_row_sz *= sz;});

    for (size_t i = 0; i < batch_size; ++i) {
      auto id = uni(rng);
      auto x_ptr = x_.ptr + id * x_row_sz;
      auto y_ptr = y_.ptr + id * y_row_sz;

      for (size_t j = 0; j < x_row_sz; ++j)
        vx.push_back(x_ptr[j]);
      for (size_t j = 0; j < y_row_sz; ++j)
        vy.push_back(y_ptr[j]);
    }

    batch_x.ndim = x_.ndim;
    batch_x.size = batch_size * x_row_sz;
    batch_x.shape = std::vector<int64_t>(x_.shape.begin() + 1, x_.shape.end());
    batch_x.shape.insert(batch_x.shape.begin(), batch_size);
    batch_x.ptr = new IN_T[batch_x.size];
    std::memcpy(batch_x.ptr, vx.data(), vx.size());

    batch_y.ndim = y_.ndim;
    batch_y.size = batch_size * y_row_sz;
    batch_y.shape = std::vector<int64_t>(y_.shape.begin() + 1, y_.shape.end());
    batch_y.shape.insert(batch_y.shape.begin(), batch_size);
    batch_y.ptr = new OUT_T[batch_y.size];
    std::memcpy(batch_y.ptr, vy.data(), vy.size());

    return DataSet(batch_x, batch_y);
  }

  virtual ~DataSet() = default;

private:
  buffer_info<IN_T> x_;
  buffer_info<OUT_T> y_;
};

}
}
