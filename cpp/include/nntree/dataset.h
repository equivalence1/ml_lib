#pragma once

#include <memory>
#include <string>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cassert>

#include "tensor.h"
#include "cpu_tensor.h"

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
  explicit DataSet(Tensor<IN_T>* x, Tensor<OUT_T>* y): x_(x), y_(y) {
    assert(x_->Nrows() == y_->Nrows());
  }

  Tensor<IN_T>& GetInput() {
    return *x_;
  }

  Tensor<OUT_T>& GetOutput() {
    return *y_;
  }

  // super ugly, leads to leaks, but for now allows to check SGD
  DataSet RandomBatch(uint64_t batch_size) {
    batch_size = std::min(batch_size, (size_t)x_->Nrows());

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, (int)x_->Nrows() - 1);

    std::vector<uint64_t> batch_x_shape(x_->Shape().begin() + 1, x_->Shape().end());
    batch_x_shape.insert(batch_x_shape.begin(), batch_size);
    std::vector<uint64_t> batch_y_shape(y_->Shape().begin() + 1, y_->Shape().end());
    batch_y_shape.insert(batch_y_shape.begin(), batch_size);

    auto batch_x = new CpuTensor<IN_T>(batch_x_shape);
    auto batch_y = new CpuTensor<IN_T>(batch_y_shape);

    for (uint64_t i = 0; i < batch_size; ++i) {
      auto id = (uint64_t)uni(rng);

      CpuTensor<IN_T> row_x;
      x_->GetRow(id, row_x);
      batch_x->SetRow(i, row_x);

      CpuTensor<OUT_T> row_y;
      y_->GetRow(id, row_y);
      batch_y->SetRow(i, row_y);
    }

    return DataSet(batch_x, batch_y);
  }

  virtual ~DataSet() = default;

private:
  Tensor<IN_T>* x_;
  Tensor<OUT_T>* y_;
};

}
}
