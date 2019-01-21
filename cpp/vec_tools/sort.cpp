#include "sort.h"

Vec VecTools::sort(const Vec& vec) {
    auto& data = vec.data();
    auto [sorted, indices] = data.sort(0);
    return Vec(sorted);
}
