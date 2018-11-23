#pragma once

#include <memory>

class ArrayVec;
using ArrayVecPtr = std::shared_ptr<ArrayVec>;

class VecRef;
using VecRefPtr = std::shared_ptr<VecRef>;

class SingleElemVec;
using SingleElemVecPtr = std::shared_ptr<SingleElemVec>;

#if defined(CUDA)
class CudaVec;
using CudaVecPtr = std::shared_ptr<CudaVec>;
#endif
