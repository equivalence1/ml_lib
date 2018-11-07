#pragma once

#include <string>
#include <cassert>

#include "tensor.h"

namespace nntree {
namespace core {

template<unsigned int D, typename T>
class CpuTensor: public Tensor<D, T> {
public:
  CpuTensor(): owner_(false), ptr_(nullptr) {};

  CpuTensor(const std::vector<uint64_t>& shape) {
    PResize(shape);
  }

  CpuTensor(T* ptr,
            const std::vector<uint64_t>& shape,
            const std::vector<uint64_t>& strides,
            bool owner = true) {
    PFromMem(ptr, shape, strides, owner);
  }

  virtual ~CpuTensor() {
    this->Clean();
  };

  void FromMem(T* ptr,
               std::vector<uint64_t>& shape,
               std::vector<uint64_t>& strides,
               bool owner) override {
    PFromMem(ptr, shape, strides, owner);
  }

  T GetVal(uint64_t id) const override {
    return *GetItemPtr(id);
  }

  T GetVal(std::initializer_list<uint64_t> ids) const override {
    std::vector<uint64_t> ids_v(ids);
    return *GetItemPtr(ids_v);
  }

  Tensor<D, T>& SetVal(uint64_t id, T val) override {
    auto ptr = GetItemPtr(id);
    *ptr = val;
    return this;
  }

  Tensor<D, T>& SetVal(std::initializer_list<uint64_t> ids, T val) override {
    auto ptr = GetItemPtr(ids);
    *ptr = val;
    return this;
  }

  T* Data() override {
    return ptr_;
  }

  std::vector<uint64_t> Shape() const override {
    return shape_;
  }

  std::vector<uint64_t> Strides() const override {
    return strides_;
  }

  void Copy(Tensor<D, T>& t) const override {
    auto ct = dynamic_cast<CpuTensor<D, T>&>(t);

    auto size = this->Size();
    auto data = new T[size];
    // TODO if array is contiguous we can optimize it with memcpy
    for (uint64_t i = 0; i < size; i++) {
      auto item_ptr = GetItemPtr(i);
      data[i] = *item_ptr;
    }

    ct.owner_ = true;
    ct.ptr_ = data;
    ct.shape_ = shape_;
    ct.strides_.resize(0);
    uint64_t stride = sizeof(T);
    for (auto it = shape_.rbegin(); it != shape_.rend(); ++it) {
      ct.strides_.push_back(stride);
      stride *= *it;
    }
  }

  void GetRow(uint64_t id, Tensor<D - 1, T>& t) const override {
    assert(id < Nrows());
    auto ct = dynamic_cast<CpuTensor<D, T>&>(t);

    ct.owner_ = false;
    ct.ptr_ = ptr_ + (strides_[0] / sizeof(T)) * id;
    ct.strides_ = std::vector<uint64_t>(strides_.begin() + 1, strides_.end());
    ct.shape_ = std::vector<uint64_t>(shape_.begin() + 1, shape_.end());
  }

  Tensor<D, T>& SetRow(uint64_t id, Tensor<D - 1, T>& t) override {
    assert(id < Nrows());
    auto ct = dynamic_cast<CpuTensor<D, T>&>(t);

    for (int i = 1; i < shape_.size(); i++)
      assert(shape_[i] == ct.shape_[i - 1]);

    auto skip = this->Size() / shape_[0];
    for (uint64_t i = 0; i < t.Size(); i++) {
      auto val = *t.GetVal(i);
      this->SetVal(i + skip, val);
    }

    return this;
  }

  uint64_t Nrows() const override {
    return shape_[0];
  }

  uint64_t Ncols() const override {
    assert(D > 1);
    return shape_[1];
  }

protected:
  T* ptr_;
  std::vector<uint64_t> shape_;
  std::vector<uint64_t> strides_;

  bool owner_;

private:
  T* GetItemPtr(uint64_t id) {
    std::vector<uint64_t> ids;

    for (unsigned int i = 0; i < D; i++) {
      id /= shape_[i];
      ids.push_back(id);
    }

    return GetItemPtr(ids);
  }

  T* GetItemPtr(std::vector<uint64_t>& ids) {
    auto ptr = (char*)ptr_;

    assert(D == ids.size());

    for (unsigned int i = 0; i < D; i++)
      ptr += strides_[i] * ids[i];

    return (T*)ptr;
  }

  void Clean() {
    if (owner_ && ptr_ != nullptr)
      delete ptr_;
  }

  void PFromMem(T* ptr,
                std::vector<uint64_t>& shape,
                std::vector<uint64_t>& strides,
                bool owner) override {
    this->Clean();

    owner_ = owner;
    ptr_ = ptr;
    assert(shape.size() == D);
    shape_ = shape;
    assert(strides.size() == D);
    strides_ = strides;
  }

  void PResize(std::vector<uint64_t>& shape) {
    this->Clean();

    owner_ = true;
    shape_ = shape;
    auto size = this->Size();
    ptr_ = new T[size];

    uint64_t stride = sizeof(T) * size;
    for (auto it = shape_.rbegin(); it != shape_.rend(); ++it) {
      stride /= *it;
      strides_.push_back(stride);
    }
  }
};

}
}
