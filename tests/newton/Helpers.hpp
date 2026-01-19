#ifndef GRAPH_DISPATCHING_TESTS_NEWTON_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_NEWTON_HELPERS_HPP

#include "kokkos-utils/concepts/View.hpp"

namespace tests::newton {

//! Functor that performs element-wise in-place subtraction of @ref src from @ref dst for rank-1 views.
template <Kokkos::utils::concepts::ViewOfRank<1> DstType, Kokkos::utils::concepts::ViewOfRank<1> SrcType>
struct Subtract {
    typename DstType::non_const_type dst;
    typename SrcType::const_type src;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index) const {
        dst(index) -= src(index);
    }
};

} // namespace tests::newton

#endif // GRAPH_DISPATCHING_TESTS_NEWTON_HELPERS_HPP
