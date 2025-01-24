#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP

#include <concepts>

#include "kokkos_ext/impl/ChainHandler.hpp"

namespace Kokkos::Experimental
{

namespace details
{
/**
 * @brief Scheduler for a @c Kokkos::Graph
 *
 * @warning It is a puppet and does not verify the @c std::execution::scheduler concept.
 */
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct GraphScheduler
{
    graph::details::ChainHandler<Exec> chain;

    auto& schedule() { return chain; }
};

} // namespace details

//! Graph context using a @c Kokkos::Graph space under the hood.
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct GraphContext
{
    graph::details::ChainHandler<Exec> chain;

    template <typename T>
    GraphContext(T&& exec) : chain(std::forward<T>(exec)) {}

    auto get_scheduler() { return details::GraphScheduler{.chain = chain}; }
};

template <typename Exec>
GraphContext(Exec&&) -> GraphContext<std::remove_cvref_t<Exec>>;

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
