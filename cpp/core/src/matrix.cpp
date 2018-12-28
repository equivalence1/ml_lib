#include <core/matrix.h>


Scalar Mx::get(int64_t x, int64_t y) const {
    return vec_.get(seqIndex(x, y));
}

Mx& Mx::set(int64_t x, int64_t y, Scalar val) {
    vec_.set(seqIndex(x, y), val);
    return *this;
}

int64_t Mx::seqIndex(int64_t x, int64_t y) const {
    return y * cols_ + x;;
}
Mx& Mx::operator+=(const Mx& other) {
    this->vec_ += other.vec_;
    return *this;
}
Mx& Mx::operator-=(const Mx& other) {
    this->vec_ -= other.vec_;
    return *this;
}

Mx& Mx::operator+=(Scalar value) {
    this->vec_ += value;
    return *this;
}
Mx& Mx::operator-=(Scalar value) {
    this->vec_ -= value;
    return *this;
}

Mx& Mx::operator*=(Scalar value) {
    this->vec_ *= value;
    return *this;
}
Mx& Mx::operator/=(Scalar value) {
    this->vec_ /= value;
    return *this;
}
Mx& Mx::operator^=(Scalar q) {
    this->vec_ ^= q;
    return *this;
}
