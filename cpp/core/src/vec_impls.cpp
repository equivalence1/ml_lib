#include "vec_impls.h"

#include "fixed_size_vec.h"
#include "array_vec.h"
#include "non_owning_vec.h"
#include "cuda_vec.h"

#include <optional>
#include <util/exception.h>
#include <iostream>

using namespace Impl;

VecVariant Impl::DynamicDispatch(AnyVec* vec) {
    if (auto ptr = dynamic_cast<ArrayVec*>(vec)) {
        return ptr;
    } else if (auto ptr = dynamic_cast<SingleElemVec*>(vec)) {
        return ptr;
    } else if (auto ptr = dynamic_cast<NonOwningVec*>(vec)) {
        return ptr;
    } else {
        #if defined(CUDA)
        if (auto ptr = dynamic_cast<CudaVec*>(vec)) {
            return ptr;
        }
        #endif
        throw Exception() << "unknown vec type";
    }
}
ConstVecVariant Impl::DynamicDispatch(const AnyVec* vec) {
    if (auto ptr = dynamic_cast<const ArrayVec*>(vec)) {
        return ptr;
    } else if (auto ptr = dynamic_cast<const SingleElemVec*>(vec)) {
        return ptr;
    } else if (auto ptr = dynamic_cast<const NonOwningVec*>(vec)) {
        return ptr;
    } else {
        #if defined(CUDA)
        if (auto ptr = dynamic_cast<const CudaVec*>(vec)) {
            return ptr;
        }
        #endif
        throw Exception() << "unknown vec type";
    }
}
