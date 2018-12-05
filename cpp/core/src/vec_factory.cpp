#include "fixed_size_vec.h"
#include "array_vec.h"
#include "non_owning_vec.h"
#include "cuda_vec.h"

#include <core/vec_factory.h>
#include <core/vec.h>
#include <util/exception.h>

using namespace Impl;

Vec VecFactory::create(VecType type, int64_t dim) {
    switch (type) {
        case VecType::Cpu: {
            if (dim == 1) {
                return Vec(std::make_shared<SingleElemVec>());
            } else {
                return Vec(std::make_shared<ArrayVec>(dim));
            }
        }
        case VecType::Gpu: {
            #if defined(CUDA)
            return Vec(std::make_shared<CudaVec>(dim));
            #else
            throw Exception() << "No cuda support";
            #endif
        }
    }
}

Vec VecFactory::createRef(float* ptr, int64_t dim) {
    return Vec(std::make_shared<NonOwningVec>(ptr, dim));
}