#pragma once

#include <core/trans.h>

class ComposeTrans: public Trans {
public:
    ComposeTrans(ObjectPtr<Trans> left, ObjectPtr<Trans> right):
            left_(std::move(left)), right_(std::move(right)) {

    }

    int64_t xdim() const override;
    int64_t ydim() const override;

    const Trans& trans(const Vec& x, Vec& to) const override;

    ObjectPtr<Trans> gradient() const override;

private:
    ObjectPtr<Trans> left_;
    ObjectPtr<Trans> right_;
};
