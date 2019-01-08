#pragma once

#include "object.h"

class AnyIndexTransformation : public Object {
public:
    virtual int64_t forward(int64_t newIndex) const = 0;
    virtual int64_t backward(int64_t oldIndex) const = 0;
    virtual int64_t newDim() const = 0;
};

class IndexTransformation : public AnyIndexTransformation {
public:
    IndexTransformation(const IndexTransformation& other) = default;

    int64_t forward(int64_t newIndex) const final {
        return trans_ ? trans_->forward(newIndex) : newIndex;
    }

    int64_t backward(int64_t oldIndex) const final {
        return trans_ ? trans_->backward(oldIndex) : oldIndex;
    }

    int64_t newDim() const final {
        return dim_;
    }

    static IndexTransformation identity(int64_t dim) {
        return IndexTransformation(dim);
    }

protected:
    IndexTransformation(std::shared_ptr<AnyIndexTransformation>&& trans)
        : trans_(std::move(trans))
          , dim_(trans_->newDim()) {

    }

    IndexTransformation(int64_t dim)
        : dim_(dim) {

    }
private:

    template <class Impl>
    friend
    class IndexTransformationStub;
private:
    std::shared_ptr<AnyIndexTransformation> trans_;
    int64_t dim_;
};

template <class Impl>
class IndexTransformationStub : public AnyIndexTransformation {
public:
    operator IndexTransformation() const {
        auto indexTrans = std::make_shared<Impl>(*static_cast<const Impl*>(this));
        return IndexTransformation(std::static_pointer_cast<AnyIndexTransformation>(indexTrans));
    }
};

class SliceIndexTransformation : public IndexTransformationStub<SliceIndexTransformation> {
public:
    SliceIndexTransformation(const SliceIndexTransformation& other) = default;

    SliceIndexTransformation(int64_t offset, int64_t size)
        : offset_(offset)
          , size_(size) {

    }

    int64_t forward(int64_t newIndex) const final {
        return newIndex + offset_;
    }

    int64_t backward(int64_t oldIndex) const final {
        if (oldIndex < offset_ || oldIndex >= offset_ + size_) {
            return -1;
        }
        return oldIndex - offset_;
    }

    int64_t newDim() const final {
        return size_;
    }

private:
    int64_t offset_ = 0;
    int64_t size_ = 0;
};

IndexTransformation Combine(IndexTransformation left, IndexTransformation right);
