#pragma once

#include "vec_ops.h"
#include <vector>
#include <memory>
#include <cassert>
#include <core/vec.h>
#include <core/index_transformation.h>

namespace Impl {

    class SubVec : public AnyVec {
    public:

        explicit SubVec(Vec vec, IndexTransformation trans)
            : data_(vec)
            , indexTrans_(trans) {

        }

        SubVec(SubVec&& other) = default;
        SubVec(const SubVec& other) = default;

        void set(int64_t idx, double val) {
            data_.set(indexTrans_.forward(idx), val);
        }

        double get(int64_t idx) const {
            return data_.get(indexTrans_.forward(idx));
        }

        int64_t dim() const {
            return indexTrans_.newDim();
        }

    private:
        Vec data_;
        IndexTransformation indexTrans_;
    };



}
