#ifndef GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
#define GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP

#include "kokkos-utils/concepts/View.hpp"

namespace tests {

//! Increment @ref data.
template <Kokkos::utils::concepts::ViewOfRank<0> ViewType, bool MayThrow = true>
struct ThenFunctor {
    typename ViewType::non_const_type data;

    KOKKOS_FUNCTION
    void operator()() const noexcept(MayThrow == false) {
        ++data();
    }
};

//! Does nothing.
template <bool MayThrow = true>
struct ThenNoOp {
    KOKKOS_FUNCTION
    void operator()() const noexcept(MayThrow == false) {
    }
};

//! Add @ref value to @ref data.
template <Kokkos::utils::concepts::ViewOfRank<1> ViewType, bool MayThrow = true>
struct AddValueOffset {
    typename ViewType::non_const_type data;
    typename ViewType::value_type value;
    typename ViewType::size_type offset = 0;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const noexcept(MayThrow == false) {
        data(offset + index) += value;
    }
};

} // namespace tests

#endif // GRAPH_DISPATCHING_TESTS_FUNCTORS_HPP
