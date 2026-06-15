#ifndef GRAPH_DISPATCHING_TESTS_UTILS_GRAPH_HPP
#define GRAPH_DISPATCHING_TESTS_UTILS_GRAPH_HPP

#include "Kokkos_Graph.hpp"

namespace tests::utils {

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)                                                          \
    || (defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT))
#    define KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(_exec_)
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(_exec_)                                                                \
        MATCHER_FOR_BEGIN_FENCE(_exec_, "Kokkos::DefaultGraph::submit: fencing before launching graph nodes"),
#endif

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(_exec_predecessor_)                                                           \
    MATCHER_FOR_BEGIN_FENCE(_exec_predecessor_, "Kokkos::DefaultGraphNode::execute_node: sync with predecessors")

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(_exec_)                                                                 \
    MATCHER_FOR_BEGIN_FENCE(_exec_, "Kokkos::DefaultGraph::submit: fencing before ending graph submit")

//! Inspired by https://github.com/kokkos/kokkos/blob/02eba5e5a94173a6d580638eb92a7357e2f9a7f8/core/unit_test/TestGraph.hpp#L1142-L1158.
template <Kokkos::ExecutionSpace>
struct GraphIsDefaulted : std::true_type { };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)                                                          \
    || (defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT))
template <>
struct GraphIsDefaulted<Kokkos::DefaultExecutionSpace> : std::false_type { };
#endif

template <Kokkos::ExecutionSpace Exec>
constexpr bool is_graph_defaulted = GraphIsDefaulted<Exec>::value;

//! Get the @c then node type.
template <Kokkos::ExecutionSpace Exec, typename Functor, typename Predecessor>
using then_node_t = Kokkos::Experimental::GraphNodeRef<
    Exec,
    Kokkos::Impl::GraphNodeThenImpl<Exec, Kokkos::Experimental::ThenPolicy<>, Functor>,
    Predecessor
>;

//! Type of the @c Kokkos graph root node.
template <Kokkos::ExecutionSpace Exec>
using root_node_t = Kokkos::Experimental::Graph<Exec>::root_t;

template <Kokkos::ExecutionSpace Exec>
struct AggregateNode {
    using type = Kokkos::Impl::GraphNodeAggregateDefaultImpl<Exec>;
};

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct AggregateNode<Kokkos::Cuda> {
    using type = Kokkos::Impl::CudaGraphNodeAggregate;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
struct AggregateNode<Kokkos::HIP> {
    using type = Kokkos::Impl::HIPGraphNodeAggregate;
};
#endif

#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_SYCL_GRAPH_SUPPORT)
template <>
struct AggregateNode<Kokkos::SYCL> {
    using type = Kokkos::Impl::SYCLGraphNodeAggregate;
};
#endif

//! Type of the @c Kokkos aggregate node.
template <Kokkos::ExecutionSpace Exec>
using aggregate_node_t = Kokkos::Experimental::GraphNodeRef<Exec, typename AggregateNode<Exec>::type>;

} // namespace tests::utils

#endif // GRAPH_DISPATCHING_TESTS_UTILS_GRAPH_HPP
