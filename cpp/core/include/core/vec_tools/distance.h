#pragma once

#include <core/vec.h>

namespace VecTools {

    double distanceP(double p, ConstVecRef left, ConstVecRef right);

    double distanceL2(ConstVecRef left, ConstVecRef right);

    double distanceL1(ConstVecRef left, ConstVecRef right);

    double normP(double p, ConstVecRef v);

    double norm(ConstVecRef v);

}
