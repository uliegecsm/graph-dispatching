#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_HELPERS_HPP

#include "Kokkos_Core.hpp"

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
#    include "plog/Log.h"
#endif

#include "kokkos_ext/impl/graph/get_node.hpp"

namespace Kokkos::Experimental::details::graph {

/**
 * If @c opstate is queryable for node, use it.
 * Otherwise, if @c env is queryable for @ref Kokkos::Experimental::details::graph::get_node_t, use it.
 * Otherwise, return the root node of @c graph.
 */
template <typename OpstateType, typename Env, Kokkos::ExecutionSpace Exec>
decltype(auto)
    get_predecessor(const OpstateType& opstate, const Env& env, const Kokkos::Experimental::Graph<Exec>& graph) {
    if constexpr (stdexec::__queryable_with<OpstateType, get_node_t>) {
        return opstate.query(get_node);
    } else if constexpr (stdexec::__queryable_with<Env, get_node_t>) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_WARNING << "Returning the node from the environment of type " << Kokkos::Impl::TypeInfo<Env>::name()
                     << '.';
#endif
        return env.query(get_node).get();
    } else {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_WARNING << "Could not find a predecessor, returning the root node.";
#endif
        return graph.root_node();
    }
}

template <typename OpstateType, typename Env, Kokkos::ExecutionSpace Exec>
using get_predecessor_t = decltype(get_predecessor(
    std::declval<const OpstateType&>(),
    std::declval<const Env&>(),
    std::declval<const Kokkos::Experimental::Graph<Exec>&>()));

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
