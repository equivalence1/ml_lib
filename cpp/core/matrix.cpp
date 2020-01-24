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

Mx operator*(const Mx& A, const Mx& B) {
    assert(A.xdim() == B.ydim());

    auto dataA = A.data().view({A.ydim(), A.xdim()});
    auto dataB = B.data().view({B.ydim(), B.xdim()});
    auto dataRes = torch::mm(dataA, dataB).contiguous().view({-1});
    //auto dataRes = torch::mm(dataA.to(torch::kFloat64), dataB.to(torch::kFloat64)).contiguous().view({-1});
    auto tmpResVec = Vec(dataRes);

    return Mx(tmpResVec, A.ydim(), B.xdim());
}

Mx operator*(const Mx& A, Scalar s) {
    Mx B(Vec(A.copy()), A.ydim(), A.xdim());
    B *= s;
    return B;
}

Mx operator-(const Mx& A, const Mx& B) {
    Mx res(Vec(A.copy()), A.ydim(), A.xdim());
    res -= B;
    return res;
}

Mx operator+(const Mx& A, const Mx& B) {
    Mx res(Vec(A.copy()), A.ydim(), A.xdim());
    res += B;
    return res;
}

Vec Mx::row(int64_t idx) {
    return Vec::slice(idx * xdim(), xdim());
}
Vec Mx::row(int64_t idx) const {
    return Vec::slice(idx * xdim(), xdim());
}

void Mx::addColumn(const Vec& column) {
    assert(column.size() == rows_);
    auto data = this->data_.view({rows_, cols_});
    auto columnData = column.data().view({rows_, 1});
    this->data_ = torch::cat({columnData, data}, 1).contiguous().view({-1});
    cols_ += 1;
}

//Vec Mx::col(int64_t idx) {
//    return Vec(0);
//}
//Vec Mx::col(int64_t idx) const {
//    return Vec(0);
//}
