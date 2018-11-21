#include "array_vec.h"
#include <core/vec_factory.h>
#include <core/vec.h>

Vec VecFactory::create(VecType type, int64_t dim) {
    if (type == VecType::Cpu) {
        return Vec(std::shared_ptr<Object>(new ArrayVec(dim)));
    } else {
        throw std::exception();
    }
}
