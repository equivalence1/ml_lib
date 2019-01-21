#pragma once

#include <cstdint>
#include <functional>

void parallelFor(int64_t from, int64_t to, std::function<void(int64_t)> func);


