#pragma once

#include <core/func.h>
#include <trans/fill.h>

class Linear : public FuncC1Stub<Linear> {
public:
    Linear(const Vec& param, double bias)
        : FuncC1Stub<Linear>(param.dim())
          , param_(param)
          , bias_(bias) {

    }

    DoubleRef valueTo(const Vec& x, DoubleRef to) const;

    std::unique_ptr<Trans> gradient() const;
private:
    Vec param_;
    double bias_;

};
