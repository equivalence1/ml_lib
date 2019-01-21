#include <trans/linear_trans.h>
#include <mx_tools/ops.h>
#include <trans/fill.h>
#include <vec_tools/transform.h>

Vec LinearTrans::trans(const Vec& x, Vec to) const {
    MxTools::multiply(mx_, x, to);
    return to;
}


std::unique_ptr<Trans> LinearTrans::gradient() const {
    return FillVec(mx_, xdim());
}
Vec LinearTrans::gradientRowTo(const Vec&, Vec to, int64_t row) const {
    VecTools::copyTo(mx_.row(row), to);
    return to;
}
