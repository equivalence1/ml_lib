#include "vec.h"
#include "vec_factory.h"

inline int64_t TotalSize(const torch::Tensor& tensor) {
    int64_t size = 0;
    for (auto dimSize : tensor.sizes()) {
        size += dimSize;
    }
    return size;
}

void Vec::set(int64_t index, double value) {
    assert(!immutable_);
    vec_.accessor<float, 1>()[index] = value;
}

double Vec::get(int64_t index) const {
    return vec_.accessor<float, 1>()[index];
}

int64_t Vec::dim() const {
    return TotalSize(vec_);
}

static inline torch::TensorOptions tensorOptionsOnDevice(const ComputeDevice device) {
    torch::TensorOptions baseOptions = [&]() {
        switch (device.deviceType()) {
            case ComputeType::Cpu: {
                return torch::device(torch::DeviceType::CPU);
            }
            case ComputeType::Gpu: {
                return torch::device(torch::DeviceType::CUDA);
            }
        }
    }();
    baseOptions = baseOptions.requires_grad(false);
    baseOptions = baseOptions.dtype(torch::ScalarType::Float);
    return baseOptions;
}

//
////TODO: should be placeholder
Vec::Vec(int64_t dim)
    : vec_(torch::zeros({dim}, tensorOptionsOnDevice(CurrentDevice())))
      , immutable_(false) {

}

Vec::Vec(int64_t dim, const ComputeDevice& device)
    : vec_(torch::zeros({dim}, tensorOptionsOnDevice(device)))
      , immutable_(false) {
}

Vec Vec::slice(int64_t from, int64_t size) {
    assert(vec_.dim() == 1);
    return Vec(vec_.slice(0, from, from + size), false);
}

Vec Vec::slice(int64_t from, int64_t size) const {
    return Vec(vec_.slice(0, from, from + size), true);
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
