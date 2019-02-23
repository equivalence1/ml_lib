#include "matrix.h"

Scalar Mx::get(int64_t x, int64_t y) const {
    return Vec::get(seqIndex(x, y));
}

Mx& Mx::set(int64_t x, int64_t y, Scalar val) {
    Vec::set(seqIndex(x, y), val);
    return *this;
}

int64_t Mx::seqIndex(int64_t x, int64_t y) const {
    assert(x < cols_);
    assert(y < rows_);
    return y * cols_ + x;
}
Mx& Mx::operator+=(const Mx& other) {
    Vec::operator+=(other);
    return *this;
}
Mx& Mx::operator-=(const Mx& other) {
    Vec::operator-=(other);
    return *this;
}

Mx& Mx::operator+=(Scalar value) {
    Vec::operator+=(value);
    return *this;
}
Mx& Mx::operator-=(Scalar value) {
    Vec::operator-=(value);
    return *this;
}

Mx& Mx::operator*=(Scalar value) {
    Vec::operator*=(value);
    return *this;
}
Mx& Mx::operator/=(Scalar value) {
    Vec::operator/=(value);
    return *this;
}
Mx& Mx::operator^=(Scalar q) {
    Vec::operator^=(q);
    return *this;
}

Vec Mx::row(int64_t idx) {
    return Vec::slice(idx * xdim(), xdim());
}
Vec Mx::row(int64_t idx) const {
    return Vec::slice(idx * xdim(), xdim());
}

//Vec Mx::col(int64_t idx) {
//    return Vec(0);
//}
//Vec Mx::col(int64_t idx) const {
//    return Vec(0);
//}
