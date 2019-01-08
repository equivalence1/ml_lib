#pragma once

#include "model.h"
#include "cost_function.h"

namespace nntree {
    namespace core {

        class Optimizer {
        public:
            Optimizer();
            void Step();

        private:
            DataSet<double, double>* ds_;
            CostFunction* loss_;
            Model* model_;
        };

    }
}
