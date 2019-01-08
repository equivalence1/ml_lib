#include <core/trans/add_vec.h>

#include <core/vec_tools/fill.h>
#include <core/vec_tools/transform.h>

Vec AddVecTrans::trans(const Vec& x, Vec to) const {
    assert(x.dim() == to.dim());
    VecTools::copyTo(x, to);
    VecTools::subtract(to, b_);
    return to;
}

Trans AddVecTrans::gradient() const {
    return Detail::GradientAsTransStub<AddVecTrans>(*this);
}

Vec AddVecTrans::gradientRowTo(const Vec& x, Vec to, int64_t row) const {
    VecTools::fill(0, to);
    to.set(row, x.get(row));
    return to;
}
