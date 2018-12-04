#include <core/trans/add_vec.h>

#include <core/vec_tools/fill.h>
#include <core/vec_tools/transform.h>
#include <core/vec_factory.h>
#include <core/trans/fill.h>

#include <memory>


VecRef AddVecTrans::trans(ConstVecRef x, VecRef to) const {
    assert(x.dim() == to.dim());
    VecTools::copyTo(x, to);
    VecTools::subtract(to, b_);
    return to;
}

IdentityMap AddVecTrans::gradient() const {
    return IdentityMap(xdim());
}
