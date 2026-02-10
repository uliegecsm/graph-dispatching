#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_GET_NODE_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_GET_NODE_HPP

#include "stdexec/execution.hpp"

#include "Kokkos_Graph_fwd.hpp"

namespace Kokkos::Experimental::details::graph {
struct get_node_t
    : public ::stdexec::__query<get_node_t>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<get_node_t>::operator();
};

inline constexpr get_node_t get_node{};

/**
 * @brief Wrap a @c Kokkos graph node to make it cheap to copy/move in new environments.
 *
 * @warning It does not extend the lifetime of the node (it does not own it).
 *
 * Inspired by https://github.com/NVIDIA/cccl/blob/cc7c08209ed4b3ef4c80dc17fa1b8507e9d1e51f/libcudacxx/include/cuda/__stream/stream_ref.h.
 *
 * @note The @c Kokkos graph node is not cheap to copy/move because it has reference counted members, see
 *       https://github.com/kokkos/kokkos/blob/1707507d7682bf77f0ed02a2fd51d6f707514c09/core/src/Kokkos_GraphNode.hpp#L96-L107.
 */
template <stdexec::__is_instance_of<Kokkos::Experimental::GraphNodeRef> NodeType>
struct NodeRef {
    NodeType const * m_node_ptr;

    explicit constexpr NodeRef(const NodeType& node) noexcept
        : m_node_ptr(&node) {
    }

    const NodeType& get() const noexcept {
        return *m_node_ptr;
    }

    friend constexpr auto operator<=>(const NodeRef&, const NodeRef&) noexcept = default;

    [[nodiscard]]
    constexpr const NodeRef& query(const get_node_t&) const noexcept {
        return *this;
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_NODE_TYPE(_rcvr_type_, _node_type_)                                          \
    stdexec::__join_env_t<                                                                                             \
        stdexec::prop<                                                                                                 \
            Kokkos::Experimental::details::graph::get_node_t,                                                          \
            Kokkos::Experimental::details::graph::NodeRef<_node_type_>                                                 \
        >,                                                                                                             \
        stdexec::__fwd_env_t<stdexec::env_of_t<_rcvr_type_>>                                                           \
    >

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_NODE(_rcvr_type_, _rcvr_obj_, _node_type_, _node_obj_)                       \
    [[nodiscard]]                                                                                                      \
    constexpr auto get_env() const noexcept -> GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_NODE_TYPE(_rcvr_type_, _node_type_) { \
        return stdexec::__env::__join(                                                                                 \
            stdexec::prop{                                                                                             \
                Kokkos::Experimental::details::graph::get_node,                                                        \
                Kokkos::Experimental::details::graph::NodeRef{_node_obj_}},                                            \
            stdexec::__fwd_env(stdexec::get_env(_rcvr_obj_)));                                                         \
    }

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_GET_NODE_HPP
