#pragma once

#include <cstdint>
#include <vector>
#include <util/array_ref.h>





template <class AdditiveStat,
    class I,
    int64_t BundleSize = 4,
    int64_t N = 4>
void buidHistograms(
    ConstArrayRef<AdditiveStat> statistics,
    ConstArrayRef<I> binLoadIndices,
    ConstArrayRef<int32_t> binOffsets,
    ConstArrayRef<uint8_t> data,
    ArrayRef<AdditiveStat> dst) {
    std::array<int32_t, BundleSize * N> localBins;
    std::array<AdditiveStat, N> localStat;

    const auto size = static_cast<const int64_t>(binLoadIndices.size());

    for (int64_t i = 0; i < (size / N) * N; i += N) {
        for (int64_t k = 0; k < N; ++k) {
            const int64_t loadIdx = binLoadIndices[i + k];
            for (int64_t b = 0; b < BundleSize; ++b) {
                localBins[b * N + k] = data[loadIdx * BundleSize + b];
            }
        }
        for (int64_t k = 0; k < N; ++k) {
            localStat[k] = statistics[i + k];
        }

        for (int64_t b = 0; b < BundleSize; ++b) {
            for (int64_t k = 0; k < N; ++k) {
                dst[binOffsets[b] + localBins[b * N + k]] += localStat[k];
            }
        }
    }

    for (int64_t i = ((size / N) * N); i < size; ++i) {
        for (int64_t b = 0; b < BundleSize; ++b) {
            const int64_t loadIdx = binLoadIndices[i];
            dst[binOffsets[b] + data[loadIdx * BundleSize + b]] += statistics[i];
        }
    }
}


template <class AdditiveStat,
        class I,
        int64_t N = 4>
void buidHistograms(
    int32_t bundleSize,
    ConstArrayRef<AdditiveStat> statistics,
    ConstArrayRef<I> binLoadIndices,
    ConstArrayRef<int32_t> binOffsets,
    ConstArrayRef<uint8_t> data,
    ArrayRef<AdditiveStat> dst) {

    assert(bundleSize <= 8);
    #define DIISPATCH(sz)\
    buidHistograms<AdditiveStat, I, sz, N>(statistics, binLoadIndices, binOffsets, data, dst);

    if (bundleSize == 8) {
        DIISPATCH(8);
    } else if (bundleSize == 7) {
        DIISPATCH(7);
    } else if (bundleSize == 6) {
        DIISPATCH(6);
    } else if (bundleSize == 5) {
        DIISPATCH(5);
    } else if (bundleSize == 4) {
        DIISPATCH(4);
    } else if (bundleSize == 3) {
        DIISPATCH(3);
    } else if (bundleSize == 2) {
        DIISPATCH(2);
    } else if (bundleSize == 1) {
        DIISPATCH(1);
    } else {
        assert(false);
    }

    #undef DISPATCH
}
