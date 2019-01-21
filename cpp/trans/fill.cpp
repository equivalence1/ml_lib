#include <trans/fill.h>
#include <vec_tools/fill.h>
#include <vec_tools/transform.h>

Vec FillConst::trans(const Vec&, Vec to) const {
    VecTools::fill(value_, to);
    return to;
}

std::unique_ptr<Trans> FillConst::gradient() const {
    return FillConst(0, xdim(), ydim());
}

Vec FillVec::trans(const Vec&, Vec to) const {
    VecTools::copyTo(params_, to);
    return to;
}

std::unique_ptr<Trans> FillVec::gradient() const {
    return FillConst(0, xdim(), ydim());
}

Vec FillVec::gradientRowTo(const Vec&, Vec to, int64_t) const {
    VecTools::fill(0, to);
    return to;
}
