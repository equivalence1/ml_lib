#include <core/trans/identity.h>
#include <core/vec_tools/transform.h>

Vec IdentityMap::trans(const Vec& x, Vec to) const {
    VecTools::copyTo(x, to);
    return to;
}

Trans IdentityMap::gradient() const {
    return FillConst(1.0, xdim(), ydim());
}
