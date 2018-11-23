#include <core/vec_tools/distance.h>

#include <core/vec_tools/stats.h>

#include <cmath>
#include <cassert>

namespace VecTools {

    double distanceP(double p, const Vec& left, const Vec& right) {
        assert(left.dim() == right.dim());
        double res = 0;
        for (auto i = 0; i < left.dim(); i++) {
            res += std::pow(std::abs(left(i) - right(i)), p);
        }
        return std::pow(res, 1/p);
    }

    double distanceL2(const Vec& left, const Vec& right) {
        return distanceP(2, left, right);
    }

    double distanceL1(const Vec& left, const Vec& right) {
        return distanceP(1, left, right);
    }

    double normP(double p, const Vec& v) {
        double res = 0;
        for (auto i = 0; i < v.dim(); i++) {
            res += std::pow(std::abs(v(i)), p);
        }
        return std::pow(res, 1/p);
    }

    double norm(const Vec& v) {
        return normP(2, v);
    }

}
