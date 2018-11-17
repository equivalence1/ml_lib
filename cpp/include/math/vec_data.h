#pragma once

#include <data.h>
#include <cstdint>
#include <memory>

class VecData  {
public:
    using VecDataPtr = std::shared_ptr<VecData>;

    virtual ~VecData() = default;

    virtual double get(int64_t idx) const = 0;

    virtual void set(int64_t idx, double value) = 0;

    virtual int64_t dim() = 0;

    virtual VecDataPtr slice(int64_t from, int64_t to);

    virtual VecDataPtr instance(int64_t size, bool fillZero);
};


using VecDataPtr = VecData::VecDataPtr;
