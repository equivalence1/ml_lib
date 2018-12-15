#include "vec_impls.h"

#include "fixed_size_vec.h"
#include "array_vec.h"
#include "non_owning_vec.h"
#include "cuda_vec.h"

#include <optional>
#include <util/exception.h>

using namespace Impl;


VecVariant Impl::DynamicDispatch(AnyVec* vec) {
    const auto typeId = vec->id();
    if (typeId == AnyVec::typeIndex<ArrayVec>()) {
        return reinterpret_cast<ArrayVec*>(vec);
    } else if (typeId == AnyVec::typeIndex<SingleElemVec>()) {
        return reinterpret_cast<SingleElemVec*>(vec);
    } else if (typeId == AnyVec::typeIndex<NonOwningVec>()) {
        return reinterpret_cast<NonOwningVec*>(vec);
    } else if (typeId == AnyVec::typeIndex<SubVec>()) {
        return reinterpret_cast<SubVec*>(vec);
    } else {
        #if defined(CUDA)
        if (typeId ==  AnyVec::typeIndex<CudaVec>()) {
            return reinterpret_cast<CudaVec*>(vec);
        }
        #endif
        throw Exception() << "unknown vec type " << typeid(vec).name();
    }
}
ConstVecVariant Impl::DynamicDispatch(const AnyVec* vec) {
    VecVariant variant = DynamicDispatch(const_cast<AnyVec*>(vec));
    return std::visit([](auto&& ptr) -> ConstVecVariant {
        return ConstVecVariant(ptr);
    }, variant);
}
