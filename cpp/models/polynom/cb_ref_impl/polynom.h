#pragma once

#include "oblivious_model.h"
#include "non_symmetric_tree.h"
#include "additive_model.h"
#include <catboost/cuda/data/leaf_path.h>
#include <util/generic/hash.h>

namespace NCatboostCuda {

    struct TPolynomStructure {
        TVector<TBinarySplit> Splits;

        ui32 GetDepth() const {
            return Splits.size();
        }

        void AddSplit(const TBinarySplit& split) {
            Splits.push_back(split);
        }

        bool operator==(const TPolynomStructure& rhs) const {
            return std::tie(Splits) == std::tie(rhs.Splits);
        }

        bool operator!=(const TPolynomStructure& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return VecCityHash(Splits);
        }

        bool IsSorted() const {
            for (ui32 i = 1; i < Splits.size(); ++i) {
                if (Splits[i] <= Splits[i - 1]) {
                    return false;
                }
            }
            return true;
        }

        bool HasDuplicates() const {
            for (ui32 i = 1; i < Splits.size(); ++i) {
                if (Splits[i] == Splits[i - 1]) {
                    return true;
                }
            }
            return false;
        }

        Y_SAVELOAD_DEFINE(Splits);
    };
}

template <>
struct THash<NCatboostCuda::TPolynomStructure> {
    inline size_t operator()(const NCatboostCuda::TPolynomStructure& value) const {
        return value.GetHash();
    }
};


namespace NCatboostCuda {


    struct TPolynom {
        TVector<double> Value;
        double Weight = 0;
        TPolynomStructure Path;

        double Norm() const {
            double total = 0;
            for (auto val : Value) {
                total += val * val;
            }
            return total;
        }
    };




    struct TStat {
        TVector<double> Value;
        double Weight = -1;
    };

    class TPolynomBuilder {
    public:

        void AddTree(const TObliviousTreeModel& tree);

        TAdditiveModel<TNonSymmetricTree> Build();
    private:
        THashMap<TPolynomStructure, TStat> EnsemblePolynoms;
    };


    class TNonSymmetricToSymmetricConverter {
    public:

        void AddTree(const TNonSymmetricTree& tree);

        TAdditiveModel<TObliviousTreeModel> Build();
    private:
        THashMap<TPolynomStructure, TStat> EnsemblePolynoms;
    };
}
