#include <core/vec_tools/transform.h>

#include <core/vec.h>
#include <core/vec_factory.h>

namespace VecTools {

    Vec copyTo(const Vec& from, Vec to) {
        for (auto i = 0; i < from.dim(); i++) {
            to.set(i, from(i));
        }
        return to;
    }

    Vec copy(const Vec& other) {
        Vec x(other.dim());
        copyTo(other, x);
        return x;
    }


}
