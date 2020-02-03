#pragma once
#include <memory>

//the most straightforward way...
namespace Private {

    template <int N, class T>
    class SingletonImpl {
    public:
        T* operator->() {
            return instance();
        }

        const T* operator->() const {
            return instance();
        }

        T& operator*() {
            return *instance();
        }

        const T& operator*() const {
            return *instance();
        }
    private:

        T* instance() const {
            static T instance_;
            return &instance_;
        }

        template <class TC>
        friend TC& Instance();
    };

    template <int N, class T>
    inline T& Instance() {
        return *Private::SingletonImpl<N, T>();
    }

    template <int N, class T>
    class TlsSingletonImpl {
    public:
        T* operator->() {
            return instance();
        }

        const T* operator->() const {
            return instance();
        }

        T& operator*() {
            return *instance();
        }

        const T& operator*() const {
            return *instance();
        }
    private:

        T* instance() const {
            thread_local T instance_;
            return &instance_;
        }

        template <class TC>
        friend TC& TlsInstance();
    };

    template <int N, class T>
    inline T& TlsInstance() {
        return *Private::TlsSingletonImpl<N, T>();
    }
}

template <int N, class T>
inline T& Singleton() {
    return Private::Instance<N, T>();
};

template <int N, class T>
inline T& TlsSingleton() {
    return Private::TlsInstance<T>();
};
