#include <core/vec_tools/stats.h>

#include <core/vec.h>

#include <cmath>

namespace VecTools {

    double sum2(ConstVecRef x) {
        double res = 0;
        for (auto i = 0; i < x.dim(); i++) {
            res += std::pow(x(i), 2);
        }
        return res;
    }

    double sum(ConstVecRef x) {
        double res = 0;
        for (auto i = 0; i < x.dim(); i++) {
            res += x(i);
        }
        return res;
    }

}
