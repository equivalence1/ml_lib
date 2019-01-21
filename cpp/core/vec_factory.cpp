#include "vec_factory.h"
#include "torch_helpers.h"

Vec VecFactory::create(ComputeDevice device, int64_t dim) {
    return Vec(dim, device);
}

Vec VecFactory::clone(const Vec& other) {
    return Vec(other.data().clone());
}

Vec VecFactory::uninitializedCopy(const Vec& other) {
    return Vec(torch::tensor(other.data().sizes(), other.data().options()));
}

Vec VecFactory::toDevice(const Vec& vec, const ComputeDevice& device) {
    return Vec(vec.vec_.to(TorchHelpers::torchDevice(device)));
}


