#include <core/trans/identity.h>
#include <core/vec_tools/transform.h>

VecRef IdentityMap::trans(ConstVecRef x, VecRef to) const {
    VecTools::copyTo(x, to);
    return to;
}

FillConst IdentityMap::gradient() const {
    return FillConst(1.0, xdim(), ydim());
}
