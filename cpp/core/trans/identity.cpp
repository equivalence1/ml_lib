#include <core/trans/identity.h>
#include <core/vec_tools/transform.h>
#include <core/vec_tools/fill.h>

Vec IdentityMap::trans(const Vec& x, Vec to) const {
    VecTools::copyTo(x, to);
    return to;
}


Vec IdentityMap::gradientRowTo(const Vec&, Vec to, int64_t row) const {
    VecTools::fill(0.0, to);
    to.set(row, 1.0);
    return to;
}
