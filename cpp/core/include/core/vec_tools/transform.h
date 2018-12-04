#pragma once

#include <core/vec.h>

namespace VecTools {

    Vec copy(ConstVecRef other);

    VecRef copyTo(ConstVecRef from, VecRef to);

}
