#include <core/vec_tools/transform.h>

#include <core/vec.h>
#include <core/vec_factory.h>

namespace VecTools {

    Vec copyTo(const Vec& from, Vec to) {
        to.data().copy_(from);
        return to;
    }

    Vec copy(const Vec& other) {
        return VecFactory::clone(other);
    }

    Vec subtract(Vec x, const Vec& y) {
        x -= y;
        return x;
    }

    Vec pow(Scalar p, const Vec& from, Vec to) {
        assert(from.dim() == to.dim());
        at::pow_out(to, from, p);
        return to;
    }

    Vec mul(const Vec& x, Vec y) {
        assert(x.dim() == y.dim());
        y *= x;
        return y;
    }

    Vec mul(Scalar alpha, Vec x) {
        x.data() *= (float)alpha;
        return x;
    }

    Vec pow(Scalar p, Vec x) {
        x.data().pow(p);
        return x;
    }

    Vec sign(const Vec& x, Vec to) {
        assert(x.dim() == to.dim());
        at::sign_out(to, x);
        return to;
    }

    Vec abs(Vec x) {
        at::abs_(x.data());
        return x;
    }

    Vec absCopy(const Vec& x) {
        return abs(copy(x));
    }

}
