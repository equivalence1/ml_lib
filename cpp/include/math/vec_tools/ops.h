#pragma once

#include "vec.h"
#include <span>

namespace VecTools {

    Vec& scale(Vec& x, double alpha);

    Vec& add(Vec& x, const Vec& y);

    Vec sum(const Vec& x, const Vec& y);

    Vec& subtract(Vec& x, const Vec& y);

    Vec sub(const Vec& x, const Vec& y);

    Vec& adjust(const Vec& x, double alpha);

    Vec& fill(Vec& x, double alpha);

    Vec& exp(Vec& x);

    Vec& log(Vec& x);

    Vec& abs(Vec& x);
}
