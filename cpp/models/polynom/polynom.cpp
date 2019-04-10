#include <util/exception.h>
#include "polynom.h"


struct PathBit {
    int Bits = 0;
    int Sign = 1;
};

inline std::vector<PathBit> LeafToPolynoms(const int path, int maxDepth) {
    std::vector<PathBit> pathBits = {{}};
    for (int depth = 0; depth <  maxDepth; ++depth) {
        const int mask = 1 << depth;
        const bool isOne = path & mask;

        if (isOne) {
            for (auto& bit : pathBits) {
                bit.Bits |= 1 << depth;
            }
        } else {
            uint64_t currentPaths = pathBits.size();
            for (uint64_t i = 0; i < currentPaths; ++i) {
                auto bit = pathBits[i];
                bit.Bits |= 1 << depth;
                bit.Sign *= -1;
                pathBits.push_back(bit);
            }
        }
    }
    return pathBits;
}

template <class Vec>
inline void SortUnique(Vec& vec) {
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

void PolynomBuilder::AddTree(const TSymmetricTree& tree)  {


    const int maxDepth = static_cast<int>(tree.Conditions.size());
    const int leaves = 1 << maxDepth;

    std::vector<double> weights(leaves);
    std::vector<double> values(leaves * tree.OutputDim());

    const int outputDim = tree.OutputDim();
    for (int path = 0; path < leaves; ++path) {
        auto polynoms = LeafToPolynoms(path, maxDepth);

        for (const auto& polynom : polynoms) {
            int idx = polynom.Bits;
            for (int dim = 0; dim < outputDim; ++dim) {
                values[idx * outputDim + dim] += polynom.Sign * tree.Leaves[path * outputDim + dim];
            }
        }
    }
    for (int path = 0; path < leaves; ++path) {
        for (int i = 0; i < leaves; ++i) {
            if ((i & path) == path) {
                weights[path] += tree.Weights[i];
            }
        }
    }


    for (int i = 0; i < leaves; ++i) {
        if (weights[i] == 0) {
            continue;
        }
        PolynomStructure polynomStructure;
        for (int depth = 0; depth < maxDepth; ++depth) {
            int mask = 1 << depth;
            if (i & mask) {
                BinarySplit split;
                split.Feature = tree.Features[depth];
                split.Condition = tree.Conditions[depth];
                polynomStructure.Splits.push_back(split);
            }
        }
        SortUnique(polynomStructure.Splits);
        auto& dst = EnsemblePolynoms[polynomStructure];
        if (dst.Weight < 0) {
            dst.Weight = weights[i];
        } else {
            VERIFY(dst.Weight == weights[i], "Weight for fixed path should be equal for all polynoms");
        }
        dst.Value.resize(tree.OutputDim());
        for (uint32_t dim = 0; dim < tree.OutputDim(); ++dim) {
            dst.Value[dim] += values[i * tree.OutputDim() + dim];
        }
    }
}




