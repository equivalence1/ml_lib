#pragma once

#include <cstdint>
#include <variant>
#include <memory>

class ArrayVec;
using ArrayVecPtr = std::shared_ptr<ArrayVec>;

class VecRef;
using VecRefPtr = std::shared_ptr<VecRef>;

class SingleElemVec;
using SingleElemVecPtr = std::shared_ptr<SingleElemVec>;

class Vec {
public:

    Vec(Vec& other) = default;

    Vec(const Vec& other) = default;

    Vec(Vec&& other) = default;

    void set(int64_t index, double value);

    double get(int64_t index) const;

    double operator()(int64_t index) const {
        return get(index);
    }

    int64_t dim() const;

//    Vec slice(int64_t from, int64_t to);
//    Vec slice(int64_t from, int64_t to) const;

protected:
    template <class T>
    using Ptr = std::shared_ptr<T>;

    template <class T>
    explicit Vec(Ptr<T>&& data)
        : data_(std::move(data)) {

    }

private:
    std::variant<ArrayVecPtr,
                 VecRefPtr,
                 SingleElemVecPtr> data_;

    friend class VecFactory;
};
