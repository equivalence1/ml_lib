#pragma once

#include "optimizer.h"
#include <models/model.h>
#include <data/grid.h>

class GreedyObliviousTree : public Optimizer {
public:

    GreedyObliviousTree(GridPtr grid, int32_t maxDepth)
    : grid_(grid)
    , maxDepth_(maxDepth) {

    }

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;


private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
};
