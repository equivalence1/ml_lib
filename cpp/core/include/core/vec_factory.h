#pragma once

#include "vec.h"

enum class VecType {
    Cpu,
    Gpu
};

class VecFactory {
public:
    static Vec create(VecType type, int64_t dim);
    static Vec createRef(float* ptr, int64_t dim);
};
