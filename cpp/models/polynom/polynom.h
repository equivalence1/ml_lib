#pragma once

#include <util/city.h>
#include <catboost_wrapper.h>
#include <unordered_map>

struct BinarySplit {
    int Feature = 0;
    float Condition = 0;
    bool operator==(const BinarySplit& rhs) const {
        return std::tie(Feature, Condition) == std::tie(rhs.Feature, rhs.Condition);
    }
    bool operator!=(const BinarySplit& rhs) const {
        return !(rhs == *this);
    }

    bool operator<(const BinarySplit& rhs) const {
        return std::tie(Feature, Condition) < std::tie(rhs.Feature, rhs.Condition);
    }
    bool operator>(const BinarySplit& rhs) const {
        return rhs < *this;
    }
    bool operator<=(const BinarySplit& rhs) const {
        return !(rhs < *this);
    }
    bool operator>=(const BinarySplit& rhs) const {
        return !(*this < rhs);
    }
};


struct PolynomStructure {
    std::vector<BinarySplit> Splits;

    uint32_t GetDepth() const {
        return Splits.size();
    }

    void AddSplit(const BinarySplit& split) {
        Splits.push_back(split);
    }

    bool operator==(const PolynomStructure& rhs) const {
        return std::tie(Splits) == std::tie(rhs.Splits);
    }

    bool operator!=(const PolynomStructure& rhs) const {
        return !(rhs == *this);
    }

    uint64_t GetHash() const {
        return VecCityHash(Splits);
    }

    bool IsSorted() const {
        for (uint32_t i = 1; i < Splits.size(); ++i) {
            if (Splits[i] <= Splits[i - 1]) {
                return false;
            }
        }
        return true;
    }

    bool HasDuplicates() const {
        for (uint32_t i = 1; i < Splits.size(); ++i) {
            if (Splits[i] == Splits[i - 1]) {
                return true;
            }
        }
        return false;
    }

};

template <>
struct std::hash<PolynomStructure> {
    inline size_t operator()(const PolynomStructure& value) const {
        return value.GetHash();
    }
};


struct TPolynom {
    std::vector<double> Value;
    double Weight = 0;
    PolynomStructure Path;

    double Norm() const {
        double total = 0;
        for (auto val : Value) {
            total += val * val;
        }
        return total;
    }
};


struct TStat {
    std::vector<double> Value;
    double Weight = -1;
};

class PolynomBuilder {
public:

    void AddTree(const TSymmetricTree& tree);
    void AddEnsemble(const TEnsemble& ensemble) {
        for (const auto& tree : ensemble.Trees) {
            AddTree(tree);
        }
    }

    std::unordered_map<PolynomStructure, TStat> Build() {
        return EnsemblePolynoms;
    }
private:
    std::unordered_map<PolynomStructure, TStat> EnsemblePolynoms;
};

