#pragma once
/*
 This file is copy from CatBoost code
 CatBoost code is available under APACHE2 License, license text is available at https://github.com/catboost
 */

#include <utility>
#include <vector>
#include <cstdint>
#include <cstring>
// NOTE: These functions provide CityHash 1.0 implementation whose results are *different* from
// the mainline version of CityHash.

#include <cstdint>
#include <cstring>

using uint128 = std::pair<uint64_t, uint64_t>;

constexpr uint64_t Uint128Low64(const uint128& x) {
    return x.first;
}

constexpr uint64_t Uint128High64(const uint128& x) {
    return x.second;
}
// Hash functions for a byte array.
// http://en.wikipedia.org/wiki/CityHash

uint64_t CityHash64(const char* buf, uint64_t len) noexcept;

uint64_t CityHash64WithSeed(const char* buf, uint64_t len, uint64_t seed) noexcept;

uint64_t CityHash64WithSeeds(const char* buf, uint64_t len, uint64_t seed0, uint64_t seed1) noexcept;


uint128 CityHash128(const char* s, size_t len) noexcept;
uint128 CityHash128WithSeed(const char* s, size_t len, uint128 seed) noexcept;

// Hash 128 input bits down to 64 bits of output.
// This is intended to be a reasonably good hash function.
inline uint64_t Hash128to64(const uint128& x) {
    // Murmur-inspired hashing.
    const uint64_t kMul = 0x9ddfea08eb382d69ULL;
    uint64_t a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
    a ^= (a >> 47);
    uint64_t b = (Uint128High64(x) ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
}


template <class T>
inline uint64_t VecCityHash(const std::vector<T>& data) {
    return CityHash64(reinterpret_cast<const char*>(data.data()), sizeof(T) * data.size());
}
