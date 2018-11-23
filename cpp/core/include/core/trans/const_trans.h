#pragma once

#include <core/vec.h>
#include <core/trans.h>

#include <utility>

class ConstTrans: public Trans {
public:
    ConstTrans(Vec params): params_(std::move(params)) {

    }

    int64_t xdim() const override;

    int64_t ydim() const override;

    const Trans& trans(const Vec& x, Vec& to) const override;

    ObjectPtr<Trans> gradient() const override;

private:
    Vec params_;
};
