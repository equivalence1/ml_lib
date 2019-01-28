#pragma once

#include <models/model.h>
#include <targets/target.h>
#include <data/dataset.h>

class Optimizer {
public:
    virtual ~Optimizer() {

    }

    virtual ModelPtr fit(const DataSet& dataSet, const Target& target) = 0;
};


