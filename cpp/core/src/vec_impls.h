#pragma once

#include "fixed_size_vec.h"
#include "array_vec.h"
#include "non_owning_vec.h"
#include "placeholder_vec.h"
#include "cuda_vec.h"
#include "vec_impls.h"

#include <core/object.h>
#include <core/vec.h>

#include <variant>

namespace Impl {
//    class ArrayVec;
//    class NonOwningVec;
//    class SingleElemVec;
//    #if defined(CUDA)
//    class CudaVec;
//    #endif

//    using ArrayVecPtr = std::shared_ptr<ArrayVec>;
//    using ConstArrayVecPtr = std::shared_ptr<const ArrayVec>;
//
//    using NonOwningVecPtr = std::shared_ptr<NonOwningVec>;
//    using ConstNonOwningVecPtr = std::shared_ptr<const NonOwningVec>;
//
//    using SingleElemVecPtr = std::shared_ptr<SingleElemVec>;
//    using ConstSingleElemVecPtr = std::shared_ptr<const SingleElemVec>;
//



    using VecVariant =  std::variant<ArrayVec*,
                                     NonOwningVec*,
                                     #if defined(CUDA)
                                     CudaVec*,
                                     #endif
                                     SingleElemVec*>;


    using ConstVecVariant =  std::variant<const ArrayVec*,
                                          const NonOwningVec*,
                                          #if defined(CUDA)
                                          const CudaVec*,
                                          #endif
                                          const SingleElemVec*>;

    VecVariant DynamicDispatch(AnyVec* vec);

    ConstVecVariant DynamicDispatch(const AnyVec* vec);

}
