#pragma once

#include <core/buffer.h>
#include <core/object.h>




//this one should be template
template <class Stat>
class StatBasedLoss : public Object {
public:
    using AdditiveStat = Stat;

    virtual void makeStats(Buffer<Stat>* stats, Buffer<int32_t>* indices) const = 0;

    virtual double score(const Stat& comb) const = 0;

    virtual double bestIncrement(const Stat& comb) const = 0;

};
