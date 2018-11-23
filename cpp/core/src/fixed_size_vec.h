#pragma once

#include <cstdint>
#include <array>
#include <cassert>

class SingleElemVec{
public:
    static const int64_t Size = 1;

    SingleElemVec() {
        data_.fill(0);
    }
    SingleElemVec(SingleElemVec&& other) = default;
    SingleElemVec(const SingleElemVec& other) = default;

    void set(int64_t idx, double val) {
        assert(idx < (int64_t)Size);
        data_[idx] = val;
    }

    double get(int64_t idx)  const {
        assert(idx < (int64_t)Size);
        return data_[idx];
    }

    constexpr int64_t dim() const {
        return static_cast<int64_t>(Size);
    }
private:
    std::array<double, Size> data_;
};
