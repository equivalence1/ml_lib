#pragma once

#include <core/vec.h>
#include <core/func.h>

#include <functional>

/**
 * || x - b ||_p^p
 */
class PDistFunc: public Func {
public:
    PDistFunc(double p,
             Vec b): p_(p), b_(std::move(b)) {

    }

    int64_t xdim() const override;

    double value(const Vec& x) const override;

    ObjectPtr<Trans> gradient() const override;

private:
    double p_;
    Vec b_;
};
