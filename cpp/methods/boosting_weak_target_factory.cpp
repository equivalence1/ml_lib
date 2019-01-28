#include "boosting_weak_target_factory.h"

SharedPtr<Target> GradientBoostingWeakTargetFactory::create(
    const DataSet& ds,
    const Target& target,
    const Mx& startPoint) {
    const Vec cursor = startPoint;
    Vec der(cursor.dim());
    target.gradientTo(cursor, der);
    return std::static_pointer_cast<Target>(std::make_shared<L2>(ds, der));
}
