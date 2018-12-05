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
    int64_t forward(int64_t newIndex) const final {
        return trans_->forward(newIndex);
    }

    int64_t backward(int64_t oldIndex) const final {
        return trans_->backward(oldIndex);
    }

    int64_t newDim() const final {
        return trans_->newDim();
    }
protected:
    IndexTransformation(ObjectPtr<AnyIndexTransformation>&& trans)
        : trans_(std::move(trans)) {

    }
private:
    template <class Impl>
    friend class IndexTransformationStub;
private:
    ObjectPtr<AnyIndexTransformation> trans_;
};

template <class Impl>
class IndexTransformationStub : public AnyIndexTransformation {
public:
    operator IndexTransformation() const {
        auto ptr = std::make_shared<Impl>(*static_cast<const Impl*>(this));
        return IndexTransformation(std::static_pointer_cast<AnyIndexTransformation>(ptr));
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



//IndexTransformation Combine(IndexTransformation left, IndexTransformation right);
