#pragma once

#include <sys/types.h>
#include <vector>
#include <numeric>
#include <functional>
#include <string>

#include "buffer_info.h"

namespace nntree {
namespace core {

// TODO(equivalence1) + need discussion

///**
// * @class NdArray
// * @brief wrapper around buffer_info struct, which serves as a general interface
// * representing multidimensional arrays.
// */
//template<typename T>
//class NdArray {
//private:
//  buffer_info<T> buff_;
//public:
//  NdArray(buffer_info<T> &&buff): buff_(buff) {};
//
//  /**
//   * @return pointer to the underlying memory
//   */
//  void* Data();
//
//  /**
//   * @return number of dimensions of the array
//   */
//  ssize_t Ndim() const;
//
//  /**
//   * @return sizes of all dimensions
//   */
//  std::vector<ssize_t> Shape() const;
//
//  /**
//   * @return strides of the array
//   */
//  std::vector<ssize_t> Strides() const;
//
//  /**
//   * @return size of a single item in the array
//   */
//  ssize_t Itemsize() const;
//
//  /**
//   * @return total number of elements
//   */
//  ssize_t Size() const {
//    auto shape = Shape();
//    return std::accumulate(shape.begin(), shape.end(), (ssize_t)1, std::multiplies<>());
//  }
//
//  virtual ~NdArray() = default;
//};

}
}
