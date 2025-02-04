#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CHAINHANDLER_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CHAINHANDLER_HPP

#include "Kokkos_Core.hpp"

namespace Kokkos::Experimental::graph::details
{

//! Helper class to avoid exposing @c Kokkos::Graph itself to the user.
template <typename Exec>
struct ChainHandler
{
    using execution_space = Exec;

    using graph_t = Kokkos::Experimental::Graph<Exec>;
    using root_t  = decltype(Kokkos::Impl::GraphAccess::create_root_ref(std::declval<graph_t&>()));

    graph_t graph;
    root_t  root;

    ChainHandler(const Exec& exec)
        : graph(Kokkos::Impl::GraphAccess::construct_graph(exec)),
          root (Kokkos::Impl::GraphAccess::create_root_ref(graph))
    {}

    template <typename... Args>
    decltype(auto) then_parallel_for(Args&&... args) const {
        return root.then_parallel_for(std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) then_parallel_reduce(Args&&... args) const {
        return root.then_parallel_reduce(std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) then(Args&&... args) const {
        return root.then(std::forward<Args>(args)...);
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CHAINHANDLER_HPP
