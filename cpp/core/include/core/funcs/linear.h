#pragma once

#include <core/func.h>

class Linear : public Func {
public:
    Linear(const Vec& param)
    : param_(param) {

    }
    int64_t xdim() const override;
    double value(const Vec& x) const override;
private:
    Vec param_;

};
