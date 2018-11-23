#include <core/vec_tools/transform.h>

#include <core/vec.h>
#include <core/vec_factory.h>

namespace VecTools {

    Vec copy(const Vec& other) {
        Vec x = VecFactory::create(VecType::Cpu, other.dim());
        copyTo(other, x);
        return x;
    }

    void copyTo(const Vec& from, Vec& to) {
        for (auto i = 0; i < from.dim(); i++) {
            to.set(i, from(i));
        }
    }

}
