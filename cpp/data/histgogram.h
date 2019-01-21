#pragma once

#include <cstdint>
#include <vector>
#include <util/array_ref.h>



template <class AdditiveStat,
          class I,
          int64_t BundleSize = 4,
          int64_t N = 4>
void BuidHistograms(ConstArrayRef<AdditiveStat> statistics,
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
            dst[binOffsets[b] + localBins[loadIdx * BundleSize + b]] += statistics[i];
        }
    }
}


