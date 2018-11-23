#include "vec_impls.h"

#include <core/vec.h>

namespace {

    template <class T>
    struct GetImpl {
        static double Get(const T& vec, int64_t idx) {
            return vec->get(idx);
        }
    };

    #if defined(CUDA)
    template<>
    struct GetImpl<CudaVecPtr> {
        static double Get(const CudaVecPtr& impl, int64_t idx) {
            //don't do this in production, just for test
            float val;
            Cuda::CopyMemory(impl->data() + idx, &val, 1);
            return val;
        }
    };
    #endif

    struct VecGetter {

        VecGetter(int64_t i)
            : idx_(i) {

        }


        template <class Ptr>
        double operator()(const Ptr& impl) const {
            return GetImpl<Ptr>::Get(impl, idx_);
        }



        int64_t idx_;
    };

    template <class T>
    struct SetImpl {
        static void Set(T& vec, int64_t idx, double value) {
            return vec->set(idx, value);
        }
    };

    #if defined(CUDA)
    template <>
    struct SetImpl<CudaVecPtr> {

        static void Set(CudaVecPtr& vec, int64_t idx, double value) {
            float val = value;
            Cuda::CopyMemory(&val, vec->data() + idx, 1);;
        }
    };
    #endif

    struct VecSetter {

        VecSetter(int64_t i, double value)
            : idx_(i)
              , value_(value) {

        }

        template <class T>
        void operator()(T& impl) const {
            SetImpl<T>::Set(impl, idx_, value_);
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
