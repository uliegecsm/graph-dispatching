#ifndef GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
#define GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP

#include "kokkos-utils/concepts/View.hpp"

#include "tests/utils/Counter.hpp"

namespace tests {

//! Increment @ref data.
template <Kokkos::utils::concepts::ViewOfRank<0> ViewType, bool MayThrow = true>
struct ThenFunctor {
    typename ViewType::non_const_type data;

    KOKKOS_FUNCTION
    void operator()() const noexcept(!MayThrow) {
        ++data();
    }
};

//! Does nothing.
template <bool MayThrowOnCall = true, bool MayThrowOnCopy = true, bool MayThrowOnMove = true>
struct ThenNoOp {
    ThenNoOp() = default;

    ThenNoOp(const ThenNoOp&) noexcept(!MayThrowOnCopy) { // NOLINT(modernize-use-equals-default)
    }
    ThenNoOp(ThenNoOp&&) noexcept(!MayThrowOnMove) {
    }
    ThenNoOp& operator=(const ThenNoOp&) noexcept(!MayThrowOnCopy) { // NOLINT(modernize-use-equals-default)
        return *this;
    }
    ThenNoOp& operator=(ThenNoOp&&) noexcept(!MayThrowOnMove) {
        return *this;
    }

    ~ThenNoOp() = default;

    KOKKOS_FUNCTION
    void operator()() const noexcept(!MayThrowOnCall) {
    }
};

template <bool MayThrowOnCall = true, bool MayThrowOnCopy = true, bool MayThrowOnMove = true>
struct ThenNoOpWithCounter
    : public ThenNoOp<MayThrowOnCall, MayThrowOnCopy, MayThrowOnMove>
    , public utils::Counter {
    using ThenNoOp<MayThrowOnCall, MayThrowOnCopy, MayThrowOnMove>::operator();
};

//! Add @ref value to @ref data.
template <Kokkos::utils::concepts::ViewOfRank<1> ViewType, bool MayThrow = true>
struct AddValueOffset {
    typename ViewType::non_const_type data;
    typename ViewType::value_type value;
    typename ViewType::size_type offset = 0;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const noexcept(!MayThrow) {
        data(offset + index) += value;
    }
};

} // namespace tests

#endif // GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
