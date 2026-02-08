#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP

#include "Kokkos_Core.hpp"

#include "kokkos_ext/impl/graph/get_node.hpp"

namespace Kokkos::Experimental::details::graph {

template <typename T>
concept has_node = requires(const T& obj) {
    requires stdexec::__is_instance_of<
        std::remove_cvref_t<decltype(*obj.get_node())>,
        Kokkos::Experimental::GraphNodeRef
    >;
};

/**
 * If @c opstate is queryable for node, use it.
 * Otherwise, if @c env is queryable for @ref Kokkos::Experimental::details::graph::get_node_t, use it.
 * Otherwise, return the root node of @c graph.
 */
template <typename OpstateType, typename Env, Kokkos::ExecutionSpace Exec>
auto get_predecessor(const OpstateType& opstate, const Env& env, const Kokkos::Experimental::Graph<Exec>& graph) {
    if constexpr (has_node<OpstateType>) {
        return *opstate.get_node();
    } else if constexpr (stdexec::__queryable_with<Env, get_node_t>) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_WARNING << "Returning the node from the environment.";
#endif
        return env.query(get_node);
    } else {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_WARNING << "Could not find a predecessor, returning the root node.";
#endif
        return graph.root_node();
    }
}

//! Determine if the node needs to be added.
template <Kokkos::ExecutionSpace ExecutionSpace, typename OpstateType>
bool proceed(const State<ExecutionSpace>& state, const OpstateType& opstate) {
    if (state.is_instantiated)
        return false;
    if constexpr (requires { opstate.error; })
        return opstate.error == nullptr;
    return true;
}

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP
