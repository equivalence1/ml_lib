#include "array_vec.h"
#include <core/vec.h>

void Vec::set(int64_t index, double value) {
    if (dynamic_cast<ArrayVec*>(data()) != nullptr) {
        dynamic_cast<ArrayVec*>(data())->data()[index] = static_cast<float>(value);
    } else {
        assert(false);
    }
}
double Vec::get(int64_t index) const {
    if (dynamic_cast<const ArrayVec*>(data()) != nullptr) {
        return dynamic_cast<const ArrayVec*>(data())->data()[index];
    } else {
        assert(false);
    }
    return 0;
}

int64_t Vec::dim() const {
    if (dynamic_cast<const ArrayVec*>(data()) != nullptr) {
        return dynamic_cast<const ArrayVec*>(data())->dim();
    } else {
        assert(false);
    }
    return 0;
}
