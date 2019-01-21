#include "vec.h"
#include "vec_factory.h"
#include "torch_helpers.h"



void Vec::set(int64_t index, double value) {
    vec_.accessor<float, 1>()[index] = value;
}

double Vec::get(int64_t index) const {
    return vec_.accessor<float, 1>()[index];
}

int64_t Vec::dim() const {
    return TorchHelpers::totalSize(vec_);
}

//
////TODO: should be placeholder
Vec::Vec(int64_t dim)
    : vec_(torch::zeros({dim}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice()))) {

}

Vec::Vec(int64_t dim, const ComputeDevice& device)
    : vec_(torch::zeros({dim}, TorchHelpers::tensorOptionsOnDevice(device))) {
}

Vec Vec::slice(int64_t from, int64_t size) {
    assert(vec_.dim() == 1);
    return Vec(vec_.slice(0, from, from + size));
}

Vec Vec::slice(int64_t from, int64_t size) const {
    return Vec(vec_.slice(0, from, from + size));
}

Vec& Vec::operator+=(const Vec& other) {
    vec_ += other;
    return *this;
}
Vec& Vec::operator-=(const Vec& other) {
    vec_ -= other;
    return *this;
}
Vec& Vec::operator*=(const Vec& other) {
    vec_ *= other;
    return *this;
}
Vec& Vec::operator/=(const Vec& other) {
    vec_ /= other;
    return *this;
}
Vec& Vec::operator+=(Scalar value) {
    vec_ += value;
    return *this;
}
Vec& Vec::operator-=(Scalar value) {
    vec_ -= value;
    return *this;
}

Vec& Vec::operator*=(Scalar value) {
    vec_ *= value;
    return *this;
}
Vec& Vec::operator/=(Scalar value) {
    vec_ /= value;
    return *this;
}
Vec& Vec::operator^=(const Vec& other) {
    vec_.pow_(other);
    return *this;
}
Vec& Vec::operator^=(Scalar q) {
    vec_.pow_(q);
    return *this;
}

ComputeDevice Vec::device() const {
    return TorchHelpers::getDevice(vec_);
}

Vec operator+(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::add_out(result, left, right);
    return result;
}
Vec operator-(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::sub_out(result, left, right);
    return result;
}
Vec operator*(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::mul_out(result, left, right);
    return result;
}
Vec operator/(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::div_out(result, left, right);
    return result;
}
Vec operator^(const Vec& left, Scalar q) {
    auto result = VecFactory::uninitializedCopy(left);
    at::pow_out(result, left, q);
    return result;
}

Vec operator+(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result += right;
    return result;
}

Vec operator-(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result -= right;
    return result;
}
Vec operator*(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result *= right;
    return result;
}

Vec operator/(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result /= right;
    return result;
}
Vec operator^(const Vec& left, const Vec& right) {
    auto result = VecFactory::clone(left);
    result ^= right;
    return result;
}
