#ifndef GRAPH_DISPATCHING_TESTS_ARBORX_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_ARBORX_HELPERS_HPP

#include "ArborX.hpp"

namespace tests::ArborX
{

//! Create 2D boxes.
template <
    typename MemorySpace,
    typename Exec,
    typename ExecHost
>
auto create_boxes(const Exec& exec, const ExecHost& exec_h)
{
    using box_t      = ::ArborX::Box<2>;
    using box_view_t = Kokkos::View<const box_t[4], MemorySpace>;

    const typename box_view_t::non_const_type boxes_nc(Kokkos::view_alloc("boxes", Kokkos::WithoutInitializing, exec));

    const auto boxes_h = Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec_h, Kokkos::HostSpace{}), boxes_nc);

    boxes_h[0] = {{0, 0}, {1, 1}};
    boxes_h[1] = {{1, 0}, {2, 1}};
    boxes_h[2] = {{0, 1}, {1, 2}};
    boxes_h[3] = {{1, 1}, {2, 2}};

    Kokkos::deep_copy(exec, boxes_nc, boxes_h);

    return box_view_t{boxes_nc};
}

} // namespace tests::ArborX

#endif // GRAPH_DISPATCHING_TESTS_ARBORX_HELPERS_HPP
