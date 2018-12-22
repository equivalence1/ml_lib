#include <core/vec_tools/fill.h>

#include <cmath>
#include <cassert>
#include <iostream>




Vec VecTools::fill(double alpha, Vec x) {
    x.data().fill_(alpha);
    return x;
}

Vec VecTools::makeSequence(double from, double step, Vec x) {
    const float to = static_cast<float>(from + step * (x.dim() - 1));
    at::range_out(x.data(), (float)from, to, (float)step);
    return x;
}
