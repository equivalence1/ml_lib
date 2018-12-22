#include <core/vec_tools/stats.h>

#include <core/scalar.h>
#include <core/vec.h>

#include <cmath>

namespace VecTools {

    double sum2(const Vec& x) {
        return sum(x ^ 2);
    }

    double sum(const Vec& x) {
        return Scalar(torch::sum(x));
    }

}
