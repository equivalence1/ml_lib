#pragma once

#include <sys/types.h>
#include <string>
#include <vector>

namespace nntree {
namespace core {

template<typename T>
struct buffer_info {
public:
  T *ptr = nullptr;             // Pointer to the underlying storage
  ssize_t size = 0;             // Total number of entries
  ssize_t ndim = 0;             // Number of dimensions
  std::vector<ssize_t> shape;   // Shape of the tensor (1 entry per dimension)
  std::vector<ssize_t> strides; // Number of entries between adjacent entries (for each per dimension)

  const ssize_t itemsize = sizeof(T);

  buffer_info() = default;

  /**
   * Note that copy constructor for buffer_info only makes a shallow copy.
   */
  buffer_info(const buffer_info&) = default;

  // TODO implement operators/move semantics
};

}
}
