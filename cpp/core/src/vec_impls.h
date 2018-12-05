#pragma once

#include "fixed_size_vec.h"
#include "array_vec.h"
#include "non_owning_vec.h"
#include "placeholder_vec.h"
#include "cuda_vec.h"
#include "subvec.h"
#include "vec_impls.h"

#include <core/object.h>
#include <core/vec.h>

#include <variant>

namespace Impl {

    using VecVariant =  std::variant<ArrayVec*,
                                     NonOwningVec*,
                                     #if defined(CUDA)
                                     CudaVec*,
                                     #endif
                                     SingleElemVec*,
                                     SubVec*
                                     >;


    using ConstVecVariant =  std::variant<const ArrayVec*,
                                          const NonOwningVec*,
                                          #if defined(CUDA)
                                          const CudaVec*,
                                          #endif
                                          const SingleElemVec*,
                                          const SubVec*>;

    VecVariant DynamicDispatch(AnyVec* vec);

    ConstVecVariant DynamicDispatch(const AnyVec* vec);

}
