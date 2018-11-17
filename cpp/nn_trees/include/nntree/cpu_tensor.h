#pragma once

#include <string>
#include <cassert>

#include "tensor.h"

namespace nntree {
namespace core {

template<typename T>
class CpuTensor : public Tensor<T> {
public:
    CpuTensor()
        : owner_(false), ptr_(nullptr) {};

    CpuTensor(const std::vector<uint64_t>& shape)
    : CpuTensor() {
        PResize(shape);
    }

    CpuTensor(T* ptr,
              const std::vector<uint64_t>& shape,
              const std::vector<uint64_t>& strides,
              bool owner = true) : CpuTensor() {
        PFromMem(ptr, shape, strides, owner);
    }

    virtual ~CpuTensor() {
        this->Clean();
    };

    void FromMem(T* ptr,
                 const std::vector<uint64_t>& shape,
                 const std::vector<uint64_t>& strides,
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

    Tensor<T>& SetVal(uint64_t id, T val) override {
        auto ptr = GetItemPtr(id);
        *ptr = val;
        return *this;
    }

    Tensor<T>& SetVal(const std::initializer_list<uint64_t>& ids, T val) override {
        auto ptr = GetItemPtr(ids);
        *ptr = val;
        return *this;
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

    void Copy(Tensor<T>& t) const override {
        auto ct = dynamic_cast<CpuTensor<T>&>(t);

        auto size = this->Size();
        auto data = new T[size];
        // TODO if array is contiguous we can optimize it with memcpy
        for (uint64_t i = 0; i < size; i++) {
            auto item_ptr = GetItemPtr(i);
            data[i] = *item_ptr;
        }

        std::vector<uint64_t> strides;
        uint64_t stride = sizeof(T);
        for (auto it = shape_.rbegin(); it != shape_.rend(); ++it) {
            strides.push_back(stride);
            stride *= *it;
        }

        ct.FromMem(data, shape_, strides, true);
    }

    void GetRow(uint64_t id, Tensor<T>& t) const override {
        auto ptr = ptr_ + (strides_[0] / sizeof(T)) * id;
        auto shape = std::vector<uint64_t>(shape_.begin() + 1, shape_.end());
        auto strides = std::vector<uint64_t>(strides_.begin() + 1, strides_.end());

        t.FromMem(ptr, shape, strides, false);
    }

    Tensor<T>& SetRow(uint64_t id, Tensor<T>& t) override {
        assert(id < Nrows());
        assert(this->Ndim() == t.Ndim());

        auto ct = dynamic_cast<CpuTensor<T>&>(t);

        for (unsigned int i = 1; i < shape_.size(); i++)
            assert(shape_[i] == ct.shape_[i - 1]);

        auto skip = this->Size() / shape_[0];
        for (uint64_t i = 0; i < t.Size(); i++) {
            auto val = t.GetVal(i);
            this->SetVal(i + skip, val);
        }

        return *this;
    }

    uint64_t Nrows() const override {
        return shape_[0];
    }

    uint64_t Ncols() const override {
        assert(this->Ndim() > 1);
        return shape_[1];
    }

protected:
    bool owner_;
    T* ptr_;
    std::vector<uint64_t> shape_;
    std::vector<uint64_t> strides_;

private:
    T* GetItemPtr(uint64_t id) const {
        std::vector<uint64_t> ids;
        auto ndim = this->Ndim();
        auto subsize = this->Size();

        for (unsigned int i = 0; i < ndim; i++) {
            subsize /= shape_[i];
            ids.push_back(id / subsize);
            id %= subsize;
        }

        return GetItemPtr(ids);
    }

    T* GetItemPtr(const std::vector<uint64_t>& ids) const {
        auto ptr = (char*) ptr_;
        auto ndim = this->Ndim();

        assert(ndim == ids.size());

        for (unsigned int i = 0; i < ndim; i++)
            ptr += strides_[i] * ids[i];

        return (T*) ptr;
    }

    void Clean() {
        if (owner_ && ptr_ != nullptr)
            delete ptr_;
    }

    void PFromMem(T* ptr,
                  const std::vector<uint64_t>& shape,
                  const std::vector<uint64_t>& strides,
                  bool owner) {
        this->Clean();

        owner_ = owner;
        ptr_ = ptr;
        shape_ = shape;
        assert(strides.size() == shape.size());
        strides_ = strides;
    }

    void PResize(const std::vector<uint64_t>& shape) {
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
