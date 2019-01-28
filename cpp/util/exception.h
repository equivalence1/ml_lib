#pragma once

#include <exception>
#include <string>
#include <sstream>

class Exception : public std::exception {
public:
    Exception() = default;

    const char* what() const noexcept override {
        return buf_.data();
    }

    template <class T>
    Exception& operator<<(const T& t) {
        std::stringstream s;
        s << buf_;
        s << t;
        buf_ = s.str();
        return *this;
    }

private:
    mutable std::string buf_;
};


//TODO: source line for exceptions


#define VERIFY(Condition, Message)\
    if (!(Condition))  {\
        throw Exception() << __FILE__ << ":" << __LINE__ << ". Verify failed: " << Message;\
    }
