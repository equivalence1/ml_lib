#include <core/trans/fill.h>
#include <core/vec_tools/fill.h>
#include <memory>
#include <cassert>
#include <core/vec_tools/transform.h>

VecRef FillConst::trans(ConstVecRef, VecRef to) const {
    VecTools::fill(value_, to);
    return to;
}

Trans FillConst::gradient() const {
    return FillConst(0, xdim(), ydim());
}

VecRef FillVec::trans(ConstVecRef, VecRef to) const {
    VecTools::copyTo(params_, to);
    return to;
}

Trans FillVec::gradient() const {
    return FillConst(0, xdim(), ydim());
}
