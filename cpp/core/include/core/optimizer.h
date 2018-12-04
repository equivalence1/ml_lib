#pragma once

#include "vec.h"
#include "vec_factory.h"
#include "func.h"


class Optimizer {

    virtual VecRef optimize(FuncC1 f, VecRef x0) const = 0;

};
