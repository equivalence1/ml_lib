#pragma once

#include "trans.h"

class Func : public Trans {
public:
    int64_t ydim() const final {
        return 1;
    }

    virtual double value(const Vec& x) const = 0;

    const Trans& trans(const Vec& x, Vec& to) const override {
        to.set(0, value(x));
        return *this;
    }

    double operator()(const Vec& x) const {
        return value(x);
    }

    ObjectPtr<Trans> gradient() const override = 0;
};
