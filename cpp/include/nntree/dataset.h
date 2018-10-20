#pragma once

#include <memory>
#include <string>

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

  virtual ~DataSet() = default;

private:
  buffer_info<IN_T> x_;
  buffer_info<OUT_T> y_;
};

}
}
