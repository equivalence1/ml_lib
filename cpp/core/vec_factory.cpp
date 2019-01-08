#include "vec_factory.h"
#include "vec.h"
#include <util/exception.h>


Vec VecFactory::create(ComputeDevice device, int64_t dim) {
    return Vec(dim, device);
}

Vec VecFactory::clone(const Vec& other) {
    return Vec(other.data().clone());
}

Vec VecFactory::uninitializedCopy(const Vec& other) {
    return Vec(torch::tensor(other.data().sizes(), other.data().options()));
}
