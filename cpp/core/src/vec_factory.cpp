#include <core/vec_factory.h>
#include <core/vec.h>
#include <util/exception.h>


Vec VecFactory::create(VecType type, int64_t dim) {
    switch (type) {
        case VecType::Cpu: {
            return Vec(dim);
        }
        case VecType::Gpu: {
            #if defined(CUDA)
            throw Exception() << "Implement me";
//            return Vec(std::make_shared<CudaVec>(dim));
            #else
            throw Exception() << "No cuda support";
            #endif
        }
    }
}
Vec VecFactory::clone(const Vec& other) {
    return Vec(other.data().clone());
}
Vec VecFactory::uninitializedCopy(const Vec& other) {
    return Vec(torch::tensor(other.data().sizes(), other.data().options()));
}
