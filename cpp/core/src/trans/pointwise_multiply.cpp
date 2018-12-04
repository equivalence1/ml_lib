#include <core/trans/pointwise_multiply.h>

#include <core/vec_tools/fill.h>
#include <core/vec_tools/transform.h>

#include <cassert>



VecRef PointwiseMultiply::trans(ConstVecRef x, VecRef to) const {
    VecTools::copyTo(x, to);
    VecTools::mul(param_, to);
    return to;
}
