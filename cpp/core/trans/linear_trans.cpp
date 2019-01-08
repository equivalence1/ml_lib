#include <core/trans/linear_trans.h>
#include <core/mx_tools/ops.h>
#include <core/trans/fill.h>
#include <core/vec_tools/transform.h>

Vec LinearTrans::trans(const Vec& x, Vec to) const {
    MxTools::multiply(mx_, x, to);
    return to;
}
Trans LinearTrans::gradient() const {
    return FillVec(mx_, xdim());
}
Vec LinearTrans::gradientRowTo(const Vec&, Vec to, int64_t row) const {
    VecTools::copyTo(mx_.row(row), to);
    return to;
}
