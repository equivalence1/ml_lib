#pragma once

template <class T>
class DataHolder {
public:
    using DataPtr = std::shared_ptr<T>;

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

protected:
    DataHolder(DataPtr ptr)
        : data_(ptr) {

    };
private:
    DataPtr data_;

};
