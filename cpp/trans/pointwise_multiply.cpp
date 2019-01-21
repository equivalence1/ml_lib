#include <trans/pointwise_multiply.h>

#include <vec_tools/fill.h>
#include <vec_tools/transform.h>

Vec PointwiseMultiply::trans(const Vec& x, Vec to) const {
    VecTools::copyTo(x, to);
    VecTools::mul(param_, to);
    return to;
}
