#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <core/vec.h>

namespace Impl {

    //TODO: uuid for placeholders so we could make slices
    class PlaceholderVec : public AnyVec {
    public:

        explicit PlaceholderVec(int64_t size)
            : offset_(0)
            , size_(size) {
            std::terminate();
        }

        PlaceholderVec(PlaceholderVec&& other) = default;
        PlaceholderVec(const  PlaceholderVec& other) = default;


        int64_t size() const {
            return static_cast<int64_t>(size_ - offset_);
        }
    private:
        int64_t size_;
        int64_t offset_;
    };

}
