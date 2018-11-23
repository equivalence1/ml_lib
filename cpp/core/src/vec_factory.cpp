#include "fixed_size_vec.h"
#include "array_vec.h"
#include "vec_ref.h"

#include <core/vec_factory.h>
#include <core/vec.h>

Vec VecFactory::create(VecType type, int64_t dim) {
    if (type == VecType::Cpu) {
        if (dim == 1) {
            return Vec(std::make_shared<SingleElemVec>());
        } else {
            return Vec(std::make_shared<ArrayVec>(dim));
        }
    } else {
        throw std::exception();
    }
}

Vec VecFactory::createRef(float* ptr, int64_t dim) {
    return Vec(std::make_shared<VecRef>(ptr, dim));
}
