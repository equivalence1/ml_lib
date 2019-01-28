#include "ensemble.h"

void Ensemble::appendTo(const Vec& x, Vec to) const  {
    for (const auto& modelPtr : models_) {
        modelPtr->appendTo(x, to);
    }
    if (scale_ != 1.0) {
        to *= scale_;
    }
}
