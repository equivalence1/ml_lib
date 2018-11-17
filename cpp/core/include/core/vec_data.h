#pragma once

#include <data.h>
#include <cstdint>
#include <memory>

class VecData {
public:
    using VecDataPtr = std::shared_ptr<VecData>;

    virtual ~VecData() = default;

    virtual double get(int64_t idx) const = 0;

    virtual void set(int64_t idx, double value) = 0;

    virtual int64_t dim() = 0;

    virtual VecDataPtr slice(int64_t from, int64_t to) = 0;

    virtual VecDataPtr instance(int64_t size, bool fillZero) = 0;
};

using VecDataPtr = VecData::VecDataPtr;

namespace Impl {



    template<class T = double>
    class ArrayVecData : public VecData {
    public:

    protected:
        ArrayVecData(ui64 size);
        ArrayVecData(ArrayVecData data, ui64 offset, ui64 size);
    private:
        std::shared_ptr<T> data_;
        ui64 size_;
        ui64 offset_;
    };



}
