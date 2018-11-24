#pragma once

#include <core/trans.h>
#include <core/vec.h>

class OffsetTrans: public Trans {
public:
    OffsetTrans(Vec b): b_(std::move(b)) {

    }

    int64_t xdim() const override;
    int64_t ydim() const override;

    const Trans& trans(const Vec& x, Vec& to) const override;

    ObjectPtr<Trans> gradient() const override;

private:
    Vec b_;
};
