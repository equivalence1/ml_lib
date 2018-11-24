#pragma once

#include <core/trans.h>

class ExpTrans: public Trans {
public:
    ExpTrans(double exp, int64_t dim): exp_(exp), dim_(dim) {

    }

    int64_t xdim() const override;
    int64_t ydim() const override;

    const Trans& trans(const Vec& x, Vec& to) const override;

    ObjectPtr<Trans> gradient() const override;

private:
    double exp_;
    int64_t dim_; // TODO Transes with nonfixed dims
};
