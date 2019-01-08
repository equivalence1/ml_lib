#pragma once

#include "context.h"
#include "vec.h"

class VecFactory {
public:
    static Vec create(ComputeDevice device, int64_t dim);

    static Vec clone(const Vec& other);

    static Vec uninitializedCopy(const Vec& other);

};
