#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>

namespace nntree {
namespace core {

/**
 * Tensor is essentially a multidimensional array.
 * Analogues to numpy ndarray or pytorch tensor.
 *
 * @tparam T data type of the individual elements of this tensor
 */
template<typename T>
class Tensor {
public:
  virtual ~Tensor() = default;

  /**
   * Initialize Tensor object with a given memory and shape/strides info.
   *
   * Depending on the implementation, this method might not copy given memory,
   * tensor will simply point to the given memory location. E.g. this is the
   * default behaviour for CpuTensor implementation. However, keep in mind that
   * implementations like GpuTensor might have to make a copy (e.g. in order to
   * move it to a different device).
   *
   * Owner parameter specifies whether this Tensor object should manage
   * the memory pointed by ptr parameter. This means that if owner is set
   * to true, then the memory pointed by ptr will be freed on this object
   * destruction. Otherwise, when this object is deleted the memory won't
   * be freed.
   *
   * @param ptr pointer to the memory
   * @param shape shape of the Tensor to create
   * @param strides strides of the data pointed by ptr argument
   * @param owner if set to true then this Tensor controls the memory pointed by ptr
   */
  virtual void FromMem(T* ptr,
                       const std::vector<uint64_t>& shape,
                       const std::vector<uint64_t>& strides,
                       bool owner) = 0;

  /**
   * Get value by an absolute id. In this case we interpret Tensor object as
   * a vector of values. We enumerate values first along Shape()[0] axis,
   * then along Shape()[1] axis, ...
   *
   * E.g. if we have 2-dimensional tensor like this:
   *               [[0, 1, 2],
   *                [3, 4, 5]]
   * Then GevVal(0) will return 3. GetVal(4) is 4.
   *
   * @param id absolute id of the element to get
   * @return value on the specified place
   */
  virtual T GetVal(uint64_t id) const = 0;

  /**
   * Get value by specifying its ids along each of the axes.
   *
   * E.g. if we have 2-dimensional tensor like this:
   *               [[0, 1, 2],
   *                [3, 4, 5]]
   * Then GevVal({0, 2}) will return 2. GetVal({1, 1}) is 4.
   *
   * @param ids ids of the value along each of the axes
   * @return value on the specified place
   */
  virtual T GetVal(std::initializer_list<uint64_t> ids) const = 0;

  /**
   * Set a value using an absolute id, interpreting
   * Tensor object as a vector of value.
   *
   * @param id absolute id of the element to set
   * @param val value to set
   * @return this tensor object reference
   */
  virtual Tensor<T>& SetVal(uint64_t id, T val) = 0;

  /**
   * Same as SetVal above, but specifying ids of the element along
   * each of the axes
   *
   * @param ids ids of the value along each of the axes
   * @param val value to set
   * @return this tensor object reference
   */
  virtual Tensor<T>& SetVal(const std::initializer_list<uint64_t>& ids, T val) = 0;

  // TODO(equivalence1) not sure how this function should behave if Tensor is on GPU
  /**
   * Pointer to the raw data of this tensor
   *
   * @return pointer to the raw data of this tensor
   */
  virtual T* Data() = 0;

  /**
   * Shape of this tensor. Same as numpy arrays' shape.
   *
   * E.g. if we have 3-dimensional tensor like this:
   *               [[[0], [1], [2]],
   *                [[3], [4], [5]]]
   * Then its shape is (2, 3, 1).
   *
   * @return shape of this tensor
   */
  virtual std::vector<uint64_t> Shape() const = 0;

  // TODO(equivalence1) not sure how this function should behave if Tensor is on GPU
  /**
   * Strides of this tensor's data. Same as numpy arrays' strides.
   *
   * TODO
   * @return strides of elements
   */
  virtual std::vector<uint64_t> Strides() const = 0;

  /**
   * Size of this tensor in terms of total number of elements it contains.
   *
   * @return size of this tensor in terms of total number of elements it contains
   */
  uint64_t Size() const {
    auto shape = Shape();
    return std::accumulate(
        shape.begin(),
        shape.end(),
        (uint64_t)1,
        [](uint64_t a, uint64_t b) {return a * b;});
  }

  /**
   * Copy (deep) this tensor's content into the given tensor.
   *
   * @param t tensor to copy this tensor's content to
   */
  virtual void Copy(Tensor<T>& t) const = 0;

  /**
   * Get a copy of this tensor's row. It's not specified whether
   * this copy will be shallow or deep and depends on the implementation.
   * E.g. it's logical for CpuTensor to return a shallow copy and this
   * might not be the case for GpuTensor.
   *
   * @param id id of the row
   * @param t tensor to (shallow) copy row's content to
   */
  virtual void GetRow(uint64_t id, Tensor<T>& t) const = 0;

  /**
   * Set a row of this vector to a specific tensor value.
   *
   * @param id id of the row
   * @param t tensor which content will be copied to the specified row
   * @return this tensor object reference
   */
  virtual Tensor<T>& SetRow(uint64_t id, Tensor<T>& t) = 0;

  /**
   * Shorthand of Shape()[0]. In case of Ndim == 2, and thus this Tensor is
   * in fact a matrix, this value actually equals to the number of row
   * of this matrix.
   *
   * If Ndim == 0 this method will fail.
   *
   * @return shape of this tensor along its first axis
   */
  virtual uint64_t Nrows() const = 0;

  /**
   * Shorthand of Shape()[1]. In case of Ndim == 2, and thus this Tensor is
   * in fact a matrix, this value actually equals to the number of columns
   * of this matrix.
   *
   * If Ndim < 2 this method will fail.
   *
   * @return shape of this tensor along its second axis
   */
  virtual uint64_t Ncols() const = 0;

  /**
   * Number of dimensions of this tensor.
   *
   * @return number of dimensions of this tensor
   */
  virtual uint64_t Ndim() const {
    return Shape().size();
  };
};

}
}
