#pragma once

#include "vec.h"
#include "vec_factory.h"
#include "func.h"

class Optimizer {
public:
    virtual Vec optimize(const FuncC1& f, Vec x0) const = 0;

    virtual ~Optimizer() = default;

};
