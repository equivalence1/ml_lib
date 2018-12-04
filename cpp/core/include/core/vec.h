#pragma once

#include "object.h"
#include <cstdint>
#include <variant>
#include <memory>
#include <utility>
#include <functional>


class ConstVecRef;

class ConstVec  {
public:

    ConstVec(ConstVec& other) = default;
    ConstVec(const ConstVec& other) = default;


    double get(int64_t index) const;

    double operator()(int64_t index) const {
        return get(index);
    }

    int64_t dim() const;

    const AnyVec* anyVec() const {
        return ptr_.get();
    }
//    ConstVecRef slice(int64_t from, int64_t to) const;

protected:

    template <class T>
    explicit ConstVec(std::shared_ptr<const T>&& ptr)
        : ptr_(std::static_pointer_cast<const AnyVec>(ptr)) {

    }

    std::shared_ptr<const AnyVec> ptr_;
};

class ConstVecRef   {
public:

    ConstVecRef(const ConstVec& vec)
    : ptr_(const_cast<ConstVec*>(&vec)) {

    }

    ConstVecRef(ConstVecRef& other) = default;
    ConstVecRef(ConstVecRef&& other) = default;
    ConstVecRef(const ConstVecRef& other) = default;

    double get(int64_t index) const {
        return ptr_->get(index);
    }

    double operator()(int64_t index) const {
        return get(index);
    }

    operator ConstVec&() const {
        return *ptr_;
    }

    int64_t dim() const {
        return ptr_->dim();
    }

    const AnyVec* anyVec() const {
        return ptr_->anyVec();//ptr_.get();
    }

protected:

    ConstVec* ptr() const {
        return ptr_;
    }
private:
    ConstVec* ptr_;
};

class VecRef;


class Vec : public ConstVec {
public:
    explicit Vec(int64_t dim);

    Vec(Vec&& other) = default;
    Vec(Vec& other) = default;
    Vec(const Vec& other) = default;


    void set(int64_t index, double value);


//    Vec slice(int64_t from, int64_t to);

    AnyVec* anyVec() {
        return const_cast<AnyVec*>(ptr_.get());
    }

    operator ConstVecRef() const {
        const ConstVec& vec = *this;
        return ConstVecRef(vec);
    }

protected:
    template <class T>
    explicit Vec(std::shared_ptr<T>&& ptr)
        : ConstVec(std::static_pointer_cast<const AnyVec>(ptr)) {

    }


    friend class VecFactory;
};



class VecRef : public ConstVecRef {
public:
    VecRef(Vec& vec)
    : ConstVecRef(vec) {

    }

    VecRef(VecRef& other) = default;
    VecRef(const VecRef& other) = default;
    VecRef(VecRef&& other) = default;

    void set(int64_t index, double value) {
       asVecRef().set(index, value);
    }

//    Vec slice(int64_t from, int64_t to);

    AnyVec* anyVec() {
        return asVecRef().anyVec();
    }

    operator Vec&() {
        return asVecRef();
    }

private:

    Vec& asVecRef() {
        ConstVec* parent = ptr();
        return *reinterpret_cast<Vec*>(parent);
    }
};
