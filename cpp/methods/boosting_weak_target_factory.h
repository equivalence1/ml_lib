#pragma once

#include <core/object.h>
#include <data/dataset.h>
#include <targets/target.h>
#include <core/matrix.h>
#include <targets/l2.h>

class GradientBoostingWeakTargetFactory : public EmpiricalTargetFactory {
public:
    virtual SharedPtr<Target> create(const DataSet& ds,
                                     const Target& target,
                                     const Mx& startPoint)  override;
private:
//    bool UseNewtonForC2 = false;
};



//class GradientBoostingBootstrappedWeakTargetFactory : public Object {
//public:
//    virtual SharedPtr<Target> create(const DataSet& ds,
//                                     const Target& target,
//                                     const Mx& startPoint) override  {
//        const Vec cursor = startPoint;
//        Vec der(cursor.dim());
//        target.gradientTo(cursor, der);
//        return std::make_shared<L2>(der);
//    }
//};
