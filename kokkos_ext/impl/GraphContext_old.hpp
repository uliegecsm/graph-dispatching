#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP

#include <concepts>

#include "kokkos_ext/impl/ChainHandler.hpp"

namespace Kokkos::Experimental
{

namespace graph::details
{
/**
 * @brief Scheduler for a @c Kokkos::Graph
 *
 * @warning It is a puppet and does not verify the @c std::execution::scheduler concept.
 */
template <Kokkos::ExecutionSpace Exec>
struct GraphScheduler
{
    ChainHandler<Exec> chain;
    Exec exec;

    auto& schedule() { return chain; }
};

template <typename Exec>
struct Pool
{
    std::vector<Exec> execs;
};

} // namespace graph::details

//! Graph context using a @c Kokkos::Graph under the hood.
template <Kokkos::ExecutionSpace Exec>
struct GraphContext
{
    graph::details::Pool<Exec> pool;

    graph::details::ChainHandler<Exec> chain;

    template <typename... Args> requires (std::same_as<std::remove_cvref_t<Args>, Exec> && ...)
    explicit GraphContext(Args&&... args) : pool{.execs = {std::forward<Args>(args)...}}, chain(pool.execs.at(0)) {}

    // return the scheduler from the right device
    auto get_scheduler(const unsigned short int devID = 0) { return graph::details::GraphScheduler{.chain = chain, .exec = pool.execs.at(devID)}; }
};

template <typename... Exec>
GraphContext(Exec&&...) -> GraphContext<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Exec...>>>>;

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
