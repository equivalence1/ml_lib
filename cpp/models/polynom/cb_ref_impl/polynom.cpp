#include "polynom.h"
#include "flat_tree_builder.h"

using namespace NCatboostCuda;

template <class T>
inline T Sign(T val) {
    return  val < 0 ? -1 : val > 0 ? 1 : 0;
}

struct TPathBit {
    int Bits = 0;
    int Sign = 1;
};



inline TVector<TPathBit> LeafToPolynoms(const int path, int maxDepth) {
    TVector<TPathBit> pathBits = {{}};
    for (int depth = 0; depth < maxDepth; ++depth) {
        const int mask = 1 << depth;
        const bool isOne = path & mask;

        if (isOne) {
            for (auto& bit : pathBits) {
                bit.Bits |= 1 << depth;
            }
        } else {
            ui64 currentPaths = pathBits.size();
            for (ui64 i = 0; i < currentPaths; ++i) {
                auto bit = pathBits[i];
                bit.Bits |= 1 << depth;
                bit.Sign *= -1;
                pathBits.push_back(bit);
            }
        }
    }
    return pathBits;
}

void NCatboostCuda::TPolynomBuilder::AddTree(const NCatboostCuda::TObliviousTreeModel& tree) {

    const auto& structure = tree.GetStructure();

    const int maxDepth = static_cast<int>(structure.GetDepth());
    const int leaves = 1 << maxDepth;

    TVector<double> values(leaves * tree.OutputDim());


    for (int path = 0; path < leaves; ++path) {
        auto polynoms = LeafToPolynoms(path, maxDepth);

        for (const auto& polynom : polynoms) {
            int idx = polynom.Bits;
            for (ui32 dim = 0; dim < tree.OutputDim(); ++dim) {
                values[idx * tree.OutputDim() + dim] += polynom.Sign * tree.GetValues()[path * tree.OutputDim() + dim];
            }
        }
    }
//



    THashMap<TPolynomStructure, TStat> ensemblePolynoms;

    for (int i = 0; i < leaves; ++i) {
        TPolynomStructure polynomStructure;

        THashMap<int, int> maxBins;
        TVector<TBinarySplit> splits;
        for (int depth = 0; depth < maxDepth; ++depth) {
            int mask = 1 << depth;
            if (i & mask) {
                auto& split = structure.Splits[depth];
                maxBins[split.FeatureId] = Max<int>(split.BinIdx,   maxBins[split.FeatureId]);
                splits.push_back(split);
            }
        }
        THashMap<int, int> oheSplits;
        bool degenerate = false;

        for (const auto split: splits) {
            if (split.SplitType == EBinSplitType::TakeBin) {
                oheSplits[split.FeatureId] ++;
                if (oheSplits[split.FeatureId] > 1) {
                    degenerate = true;
                }
                polynomStructure.Splits.push_back(split);
            } else {
                TBinarySplit fixedSplit = split;
                fixedSplit.BinIdx = maxBins[split.FeatureId];
                polynomStructure.Splits.push_back(fixedSplit);
            }
        }
        if (degenerate) {
            continue;
        }
        SortUnique(polynomStructure.Splits);

        int weightMask = 0;
        for (const auto& split : polynomStructure.Splits) {
            for (int depth = 0; depth < maxDepth; ++depth) {
                 const auto& treeSplit = structure.Splits[depth];
                 if (treeSplit == split) {
                     weightMask |= 1 << depth;
                 }
            }
        }
        double polynomWeight = 0;
        for (int leaf = 0; leaf < leaves; ++leaf) {
            if ((leaf & weightMask) == weightMask) {
                polynomWeight += tree.GetWeights()[leaf];
            }
        }


        auto& dst = ensemblePolynoms[polynomStructure];

        if (dst.Weight < 0) {
            dst.Weight = polynomWeight;
        } else {
            CB_ENSURE(dst.Weight ==  polynomWeight, "error: monom weight depends on dataset only: " << polynomWeight << " ≠ " << dst.Weight);
        }


        if (dst.Value.size() < tree.OutputDim()) {
            dst.Value.resize(tree.OutputDim());
        }
        for (ui32 dim = 0; dim < tree.OutputDim(); ++dim) {
            dst.Value[dim] += values[i * tree.OutputDim() + dim];
        }
    }

    for (const auto& [polynomStructure, stat] : ensemblePolynoms) {
        auto& dst = EnsemblePolynoms[polynomStructure];
        if (dst.Weight < 0) {
            dst.Weight = stat.Weight;
        } else {
            CB_ENSURE(dst.Weight == stat.Weight, "error: monom weight depends on dataset only: " << stat.Weight << " ≠ " << dst.Weight);
        }
        if (dst.Value.size() < tree.OutputDim()) {
            dst.Value.resize(tree.OutputDim());
        }
        for (ui32 k = 0; k < tree.OutputDim(); ++k) {
            dst.Value[k] += stat.Value[k];
        }
    }
}

TNonSymmetricTree BiasPolynom(TConstArrayRef<double> bias, double w) {
    TFlatTreeBuilder builder(bias.size());

    TVector<float> floatValues(bias.begin(),
                               bias.end());
    TBinarySplit fakeSplit;
    TLeafPath path;
    path.Splits = {fakeSplit};

    path.Directions = {ESplitValue::Zero};
    builder.Add(path, floatValues, w);
    path.Directions = {ESplitValue::One};
    builder.Add(path, floatValues, w);
    return builder.BuildTree();
}

TNonSymmetricTree FromPolynom(const TPolynom& polynom, double totalWeight, TArrayRef<double> bias) {
    CB_ENSURE(polynom.Path.GetDepth() > 0);
    TFlatTreeBuilder builder(polynom.Value.size());

    TVector<float> rightValues(polynom.Value.begin(),
                               polynom.Value.end());

    TVector<float> biasValues(rightValues.size());


    for (ui32 i = 0; i < rightValues.size(); ++i) {
        float expected = rightValues[i] * polynom.Weight / totalWeight;
        biasValues[i] -= expected;
        rightValues[i] -= expected;
        bias[i] += expected;
    }

    {
        TLeafPath path;
        path.Splits = polynom.Path.Splits;
        path.Directions.resize(path.Splits.size(), ESplitValue::One);
        builder.Add(path, rightValues, polynom.Weight);
        for (ui32 d = polynom.Path.GetDepth(); d > 0; --d) {
            path.Directions.resize(d);
            path.Splits.resize(d);
            path.Directions[d - 1] = ESplitValue::Zero;
            builder.Add(path, biasValues, 0);
        }
    }
    return builder.BuildTree();
}


NCatboostCuda::TAdditiveModel<NCatboostCuda::TNonSymmetricTree> NCatboostCuda::TPolynomBuilder::Build() {
    TVector<TPolynom> polynoms;
    TPolynom bias;

    for (const auto& [polynomStructure, stat] : EnsemblePolynoms) {
        TPolynom polynom;
        polynom.Path = polynomStructure;
        polynom.Weight = stat.Weight;
        polynom.Value = stat.Value;
        polynom.Weight += 0.5;
        if (polynom.Path.GetDepth() == 0) {
            bias = polynom;
        } else {
            polynoms.push_back(std::move(polynom));
        }
    }

    Sort(polynoms.begin(), polynoms.end(), [&](const TPolynom& left, const TPolynom& right) -> bool {
        return left.Weight * left.Norm() >  right.Weight * right.Norm();
    });
    TOFStream outWeights("polynom.cw2");
    const double totalWeight = bias.Weight;

    outWeights << bias.Weight * bias.Norm() / totalWeight << Endl;

    for (ui32 i = 0; i < polynoms.size(); ++i) {
        const auto& polynom = polynoms[i];
        outWeights << polynom.Weight * polynom.Norm() / totalWeight << Endl;
    }

    TAdditiveModel<TNonSymmetricTree> trees;

    trees.AddWeakModel(BiasPolynom(bias.Value, totalWeight));

    for (const auto& polynom : polynoms) {
        trees.AddWeakModel(FromPolynom(polynom, totalWeight, bias.Value));
    }
    trees.WeakModels[0] = BiasPolynom(bias.Value, totalWeight);
    return trees;
}

void TNonSymmetricToSymmetricConverter::AddTree(const TNonSymmetricTree& tree) {
    auto visitor = [&](const TLeafPath& path, TConstArrayRef<float> leaf, double) {
        CB_ENSURE(leaf.size());

        int bits = 0;
        for (ui32 depth = 0; depth < path.GetDepth(); ++depth) {
            if (path.Directions[depth] == ESplitValue::One) {
                bits |= 1 << depth;
            }
        }
        auto monoms = LeafToPolynoms(bits, path.GetDepth());

        for (const auto& monom : monoms) {
            const int activeFeatures = monom.Bits;
            THashMap<TBinarySplit, ui32> splits;

            for (ui32 i = 0; i < path.GetDepth(); ++i) {
                if (activeFeatures & (1 << i)) {
                    auto srcSplit = path.Splits[i];

                    if (srcSplit.SplitType == EBinSplitType::TakeGreater) {
                        auto baseSplit = srcSplit;
                        baseSplit.BinIdx = 0;
                        splits[baseSplit] = std::max<ui32>(splits[baseSplit], srcSplit.BinIdx);
                    } else {
                        splits[srcSplit] = srcSplit.BinIdx;
                    }
                }
            }

            TPolynomStructure structure;
            for (const auto& [baseSplit, binIdx] : splits) {
                auto split = baseSplit;
                split.BinIdx = binIdx;
                structure.Splits.push_back(split);
            }
            SortUnique(structure.Splits);
            auto& dst = EnsemblePolynoms[structure];
            dst.Value.resize(leaf.size());
            for (ui32 i = 0; i < leaf.size(); ++i) {
                dst.Value[i] += monom.Sign * leaf[i];
            }
        }
    };
    tree.VisitLeavesAndWeights(visitor);
}


inline void AddMonomToTree(const TPolynom& polynom, const TObliviousTreeStructure& structure, TVector<float>* dstPtr) {
    auto& leaves = *dstPtr;
    const auto outDim = polynom.Value.size();
    const auto& polynomSplits = polynom.Path.Splits;
    const auto& otSplits = structure.Splits;


    TVector<int> bitsToFill;
    int baseLeaf = 0;
    {
        {
            ui32 polynomCursor = 0;
            ui32 otCursor = 0;
            while (otCursor < otSplits.size()) {
                if (polynomCursor < polynomSplits.size() && polynomSplits[polynomCursor] == otSplits[otCursor]) {
                    baseLeaf |= 1 << otCursor;
                    ++polynomCursor;
                } else {
                    bitsToFill.push_back(otCursor);
                }
                ++otCursor;
            }
        }

        const int iterCount = 1 << bitsToFill.size();
        for (int i = 0; i < iterCount; ++i) {
            int leaf = baseLeaf;

            for (ui32 j = 0; j < bitsToFill.size(); ++j) {
                if (i & (1 << j)) {
                    leaf |= 1 << bitsToFill[j];
                }
            }
            for (ui32 dim = 0; dim < outDim; ++dim) {
                leaves[leaf * outDim + dim] += polynom.Value[dim];
            }
        }
    }
}

TAdditiveModel<TObliviousTreeModel> TNonSymmetricToSymmetricConverter::Build() {
    TVector<TPolynom> polynoms;
    for (const auto& [polynomStructure, stat] : EnsemblePolynoms) {
        TPolynom polynom;
        polynom.Path = polynomStructure;
        polynom.Weight = stat.Weight;
        polynom.Value = stat.Value;
        polynoms.push_back(std::move(polynom));
    }

    Sort(polynoms.begin(), polynoms.end(), [&](const TPolynom& left, const TPolynom& right) -> bool {
        return left.Path.GetDepth() > right.Path.GetDepth();
    });
    CATBOOST_INFO_LOG << "Polynom size: " << polynoms.size() << Endl;

    TVector<TObliviousTreeStructure> structures;
    TVector<TVector<float>> leaves;
    int outDim = polynoms.back().Value.size();

    for (const auto& polynom : polynoms) {
        bool addNew = true;
        for (ui32 i = 0; i < structures.size(); ++i) {
            if (NCB::IsSubset(polynom.Path.Splits, structures[i].Splits)) {
                AddMonomToTree(polynom, structures[i], &leaves[i]);
                addNew = false;
                break;
            }
        }
        if (addNew) {
            TObliviousTreeStructure newStructure;
            newStructure.Splits = polynom.Path.Splits;
            if (polynom.Path.GetDepth() == 0) {
                TBinarySplit fakeSplit;
                newStructure.Splits = {fakeSplit};
            }
            structures.push_back(newStructure);
            leaves.push_back(TVector<float>(outDim * newStructure.LeavesCount()));
            AddMonomToTree(polynom, structures.back(), &leaves.back());
        }
    }
    TAdditiveModel<TObliviousTreeModel> result;
    for (ui32 i = 0; i < structures.size(); ++i) {
        result.AddWeakModel(TObliviousTreeModel(std::move(structures[i]), std::move(leaves[i]), outDim));
    }
    CATBOOST_INFO_LOG << "Generated symmetric tree size: " << result.Size() << Endl;

    return result;
}
