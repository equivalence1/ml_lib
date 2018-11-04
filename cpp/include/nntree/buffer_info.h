#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace nntree {
namespace core {

template<typename T>
struct buffer_info {
public:
  T *ptr = nullptr;             // Pointer to the underlying storage
  int64_t size = 0;             // Total number of entries
  int64_t ndim = 0;             // Number of dimensions
  std::vector<int64_t> shape;   // Shape of the tensor (1 entry per dimension)
  std::vector<int64_t> strides; // Number of entries between adjacent entries (for each per dimension)

  const int64_t itemsize = sizeof(T);

  buffer_info() = default;

  /**
   * Note that copy constructor for buffer_info only makes a shallow copy.
   */
  buffer_info(const buffer_info&) = default;

  // TODO implement operators/move semantics
};

}
}
