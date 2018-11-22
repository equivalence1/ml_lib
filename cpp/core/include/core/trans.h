#pragma once

#include "object.h"
#include "batch.h"
#include "vec.h"

#include <memory>

class Trans : public Object {
public:
    virtual int64_t xdim() const = 0;
    virtual int64_t ydim() const = 0;

    virtual const Trans* trans(const Vec& x, Vec& to) const = 0;

    virtual const Trans* trans(const Batch<Vec>& x, Batch<Vec>& to) const {
        for (int64_t i = 0; i < x.size(); ++i) {
            trans(x[i], to[i]);
        }
        return this;
    }
};
