#pragma once
#include <mutex>
/*
 * Based on implementation catboost/util/system/guard.h  and catboost/util/generic/scope.h
 *  under APACHE2 License, license text is available at https://github.com/catboost
 */

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)
#define UNIQUE_ID(x) CONCAT(x, __COUNTER__)

namespace {
    template <class T>
    struct LockWrapper {
        std::lock_guard<T> guard_;

        explicit LockWrapper(T& x)
            : guard_(x) {

        }
        operator bool() const {
            return true;
        }

        operator std::lock_guard<T>&() {
            return guard_;
        }

        operator const std::lock_guard<T>&() const {
            return guard_;
        }
    };

}

template <class T>
static inline LockWrapper<T> make_guard(T& x) {
    return LockWrapper<T>(x);
}

/*
 * with_guard(lock) {
 *   ...
 * }
 */
#define with_guard(lock)                                     \
    if (auto UNIQUE_ID(var) = make_guard(lock)) {                 \
        goto CONCAT(THIS_IS_GUARD, __LINE__);                \
    } else                                                   \
        CONCAT(THIS_IS_GUARD, __LINE__)                      \
            :

namespace Private {
    template <typename F>
    class ScopeGuard {
    public:
        ScopeGuard(const F& function)
            : Function_{function} {
        }

        ScopeGuard(F&& function)
            : Function_{std::move(function)} {
        }

        ScopeGuard(ScopeGuard&&) = default;
        ScopeGuard(const ScopeGuard&) = default;

        ~ScopeGuard() {
            Function_();
        }

    private:
        F Function_;
    };

    struct MakeGuardHelper {
        template <class F>
        ScopeGuard<F> operator|(F&& function) const {
            return std::forward<F>(function);
        }
    };
}

#define scope_exit(...) const auto UNIQUE_ID(scopeGuard)  = ::Private::MakeGuardHelper{} | [__VA_ARGS__]() mutable -> void
#define defer scope_exit(&)
