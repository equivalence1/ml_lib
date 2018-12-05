#pragma once

#include <core/func.h>
#include <core/trans/fill.h>

class Linear : public FuncC1Stub<Linear> {
public:
    Linear(ConstVecRef param, double bias)
    :FuncC1Stub<Linear>(param.dim())
    , param_(param)
    , bias_(bias){

    }

    DoubleRef valueTo(ConstVecRef x, DoubleRef to) const;

    Trans gradient() const;
private:
    Vec param_;
    double bias_;

};
