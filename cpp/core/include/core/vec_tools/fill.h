#pragma once

#include <core/vec.h>

namespace VecTools {

    double dotProduct(ConstVecRef left, ConstVecRef right);

    VecRef fill(double alpha, VecRef x);

    VecRef makeSequence(double from, double step, VecRef x);
//    VecRefscale(VecRefx, double alpha);
//
//    VecRefadd(VecRefx, ConstVecRef y);
//
//    Vec sum(ConstVecRef x, ConstVecRef y);

    VecRef subtract(VecRef x, ConstVecRef y);

//    Vec sub(ConstVecRef x, ConstVecRef y);
//
//    VecRefadjust(ConstVecRef x, double alpha);
//
//
//
    VecRef pow(double p, ConstVecRef from, VecRef to);
    VecRef pow(double p, VecRef x);

    VecRef sign(ConstVecRef x, VecRef to);
    VecRef abs(VecRef to);

    VecRef mul(ConstVecRef x, VecRef y);
    VecRef mul(double alpha, VecRef x);

//
//    VecReflog(VecRefx);
//
//    VecRefabs(VecRefx);

}
