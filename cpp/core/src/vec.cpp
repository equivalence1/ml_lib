#include "fixed_size_vec.h"
#include "array_vec.h"
#include "vec_ref.h"
#include <core/vec.h>

namespace {
    struct VecGetter {

        VecGetter(int64_t i)
            : idx_(i) {

        }

        template <class Impl>
        double operator()(const Impl& impl) const {
            return impl->get(idx_);
        }

        int64_t idx_;
    };

    struct VecSetter {

        VecSetter(int64_t i, double value)
            : idx_(i)
              , value_(value) {

        }

        template <class Impl>
        void operator()(Impl& impl) const {
            impl->set(idx_, value_);
        }

        int64_t idx_;
        double value_;
    };

    struct VecDim {
        template <class Impl>
        int64_t operator()(const Impl& impl) const {
            return impl->dim();
        }
    };
}


void Vec::set(int64_t index, double value) {
    std::visit(VecSetter(index, value), data_);
}
double Vec::get(int64_t index) const {
    return std::visit(VecGetter(index), data_);
}

int64_t Vec::dim() const {
    return std::visit(VecDim(), data_);
}
