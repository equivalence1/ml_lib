#include <core/vec.h>
#include <core/vec_factory.h>
#include <core/matrix.h>

#include <data/dataset.h>
#include <data/grid_builder.h>

#include <targets/l2.h>

#include <methods/greedy_linear_oblivious_trees.h>

#include <iostream>
#include <vector>

DataSet simpleDs() {
    Vec dsDataVec = VecFactory::fromVector({
        0.1,   0,     0, 1.1,
        2,     0.1,   0, 1.2,
        3,     -17,   0, 1.3,
        4,     1,     0, 1.4,
        5,     .2,    0, 1.5,
        6,     .1337, 0, 1.6,
        8,     2.17,  0, 1.7,
    });
    Vec target = VecFactory::fromVector({
        1,
        3.5,
        3.9,
        0,
        -6,
        -6.8,
        -9.1,
     });

    return DataSet(Mx(dsDataVec, 7, 4), target);
}

void testHistSimple() {
    std::cout << "\n\n======= Hist Simple\n" << std::endl;

    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});
    std::set<int> usedFeatures({});

    BinarizationConfig config;
    config.bordersCount_ = 32;
    GridPtr grid = buildGrid(ds, config);
    BinarizedDataSetPtr bds = binarize(ds, grid, 4);

    Histogram h(grid);

    std::vector<std::vector<int>> a;
    a.resize(grid->nzFeatures().size());

    for (int f = 0; f < grid->nzFeatures().size(); ++f) {
        a[f].resize(ds.samplesCount(), 0);
        bds->visitFeature(f, [&](int64_t i, uint8_t bin) {
            a[f][i] = bin;
        });
    }

    for (int i = 0; i < ds.samplesCount(); ++i) {
        for (int f = 0; f < grid->nzFeatures().size(); ++f) {
            std::cout << std::setw(3) << a[f][i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "h.build" << std::endl;
    h.build(ds, usedFeatures, indices);

    double bestSplitScore = 1e9;
    int bestSplitFId = -1;
    int bestSplitCond = -1;

    int nFeatures = grid->nzFeatures().size();
    for (int fId = 0; fId < nFeatures; ++fId) {
        int condCount = grid->conditionsCount(fId);
        for (int cond = 0; cond < condCount; ++cond) {
            auto sScore = h.splitScore(fId, cond);
            std::cout << "fId: " << fId << ", cond: " << cond << ", score: (" << sScore.first
                    << ", " << sScore.second << ")\n";

            auto score = sScore.first + sScore.second;
            if (score < bestSplitScore) {
                std::cout << "new best score: " << score << ", fId: " << fId << ", cond: " << cond << std::endl;
                bestSplitScore = score;
                bestSplitFId = fId;
                bestSplitCond = cond;
            }
        }
        std::cout << std::endl;
    }

    double border = grid->borders(bestSplitFId).at(bestSplitCond);
//    bestSplitFId = grid->nzFeatures().at(bestSplitFId).origFeatureId_;
    std::cout << "best split feature: " << bestSplitFId << ", border: "
            << std::setprecision(2) << border << ", cond id: " << bestSplitCond << std::endl;
}

void testLeafSimple() {
    std::cout << "\n\n======= Leaf Simple\n" << std::endl;

    auto ds = simpleDs();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    GridPtr grid = buildGrid(ds, config);

    GreedyLinearObliviousTree g(grid, 2);
    L2 target(ds);

    g.fit(ds, target);

    for (int i = 0; i < ds.samplesCount(); i++) {
        std::cout << i << ": " << g.value(ds.sample(i)) << std::endl;
    }
}

int main() {
    testHistSimple();
    testLeafSimple();
}
