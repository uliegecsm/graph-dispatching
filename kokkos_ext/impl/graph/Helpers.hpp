#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP

#include "Kokkos_Core.hpp"

namespace Kokkos::Experimental::details::graph {

//! If @c opstate is queryable for node, use it. Otherwise, return the root node of @c graph.
template <typename OpstateType, Kokkos::ExecutionSpace Exec>
auto get_predecessor(const OpstateType& opstate, const Kokkos::Experimental::Graph<Exec>& graph) {
    if constexpr (requires { opstate.get_node(); }) {
        return *opstate.get_node();
    } else {
        return graph.root_node();
    }
}

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP
