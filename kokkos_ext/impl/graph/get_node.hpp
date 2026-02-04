#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_GET_NODE_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_GET_NODE_HPP

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::graph {
struct get_node_t
    : public ::stdexec::__query<get_node_t>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<get_node_t>::operator();
};

inline constexpr get_node_t get_node{};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_NODE(_rcvr_type_, _rcvr_obj_, _node_type_, _node_obj_)                       \
    [[nodiscard]]                                                                                                      \
    constexpr auto get_env() const noexcept -> stdexec::__join_env_t<                                                  \
        stdexec::prop<Kokkos::Experimental::details::graph::get_node_t, _node_type_>,                                  \
        stdexec::__fwd_env_t<stdexec::env_of_t<_rcvr_type_>>                                                           \
    > {                                                                                                                \
        return stdexec::__env::__join(                                                                                 \
            stdexec::prop{Kokkos::Experimental::details::graph::get_node, _node_obj_},                                 \
            stdexec::__fwd_env(stdexec::get_env(_rcvr_obj_)));                                                         \
    }

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_GET_NODE_HPP
