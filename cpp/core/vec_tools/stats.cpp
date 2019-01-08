#include <core/vec_tools/stats.h>

namespace VecTools {

    Scalar sum2(const Vec& x) {
        return sum(x ^ 2);
    }

    Scalar sum(const Vec& x) {
        return Scalar(torch::sum(x));
    }

}
