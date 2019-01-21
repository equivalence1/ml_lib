#pragma once

#include "bin_optimized_model.h"
#include <data/grid.h>
#include <core/vec_factory.h>
#include <core/func.h>


class ObliviousTree final : public BinOptimizedModelStub<ObliviousTree> {
public:

    ObliviousTree(GridPtr grid,
                  std::vector<BinaryFeature> binFeatures,
                  Vec leaves
                  )
      : BinOptimizedModelStub<ObliviousTree>(grid->origFeaturesCount(), 1)
      , grid_(std::move(grid))
      , splits_(std::move(binFeatures))
      , leaves_(VecFactory::clone(leaves)) {

    }


    ObliviousTree(const ObliviousTree& other)
    : BinOptimizedModelStub<ObliviousTree>(other)
    , grid_(other.grid_)
    , splits_(other.splits_)
    , leaves_(other.leaves_) {

    }

    const Grid& grid() const {
        return *grid_;
    }

    Vec trans(const Vec& x, Vec to) const override;

    void applyToBds(const BinarizedDataSet& ds, Mx to) const override;

    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const;




private:
    GridPtr grid_;
    std::vector<BinaryFeature> splits_;
    Vec leaves_;
};
