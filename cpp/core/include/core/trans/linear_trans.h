#pragma once

#include <core/trans.h>

class LinearTrans: public Trans {
public:
    LinearTrans(Vec param): param_(param) {

    }

    int64_t xdim() const override;
    int64_t ydim() const override;

    const Trans& trans(const Vec& x, Vec& to) const override;

    ObjectPtr<Trans> gradient() const override;

private:
    Vec param_;
};
