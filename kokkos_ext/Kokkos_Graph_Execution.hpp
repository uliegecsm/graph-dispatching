#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_KOKKOS_GRAPH_EXECUTION_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_KOKKOS_GRAPH_EXECUTION_HPP

#include <concepts>
#include <functional>

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos_ext/impl/PartialAlgorithm_ParallelFor.hpp"
#include "kokkos_ext/impl/PartialAlgorithm_ParallelReduce.hpp"

/**
 * @file
 *
 * This file contains helpers for enriching the @c Kokkos API to allow *à la*
 * P2300 creation of @c Kokkos::Graph.
 *
 * This file should disappear at some point, because @c Kokkos would provide
 * its functionalities.
 *
 * The functionalities can be found in @ref Kokkos_Graph_Execution.hpp.
 */

namespace Kokkos::Experimental
{

namespace graph
{

namespace details
{

template <typename Sender, typename PA>
constexpr decltype(auto) operator|(Sender&& input, PA&& partial) {
    return std::invoke(std::forward<PA>(partial), std::forward<Sender>(input));
}

//! Pipable @c parallel whatever.
template <typename Tag, typename... Args>
constexpr decltype(auto) parallel(Args&&... args)
{
    return PartialAlgorithm<Tag, Args...>(std::forward<Args>(args)...);
}

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
};

} // namespace details

/**
 * @brief Start the chain.
 *
 * @warning If this is used to create the underlying @c Kokkos::Graph, it should be
 *          called only once per graph.
 *
 * @todo It is not clear yet if the @p exec should be passed, but for now it is used
 *       by @ref details::ChainHandler::graph.
 */
template <typename Exec>
constexpr decltype(auto) create_graph(const Exec& exec) {
    return details::ChainHandler(exec);
}

//! For @c Kokkos::Graph, @c split is a no-op.
constexpr decltype(auto) split() {
    return std::identity{};
}

//! Pipable @c Kokkos parallel-for, with partially-specified algorithm.
template <typename Policy, typename Functor>
constexpr decltype(auto) parallel_for(Policy&& policy, Functor&& functor)
{
    return details::parallel<Kokkos::ParallelForTag>(
        std::forward<Policy>(policy),
        std::forward<Functor>(functor)
    );
}

//! Pipable @c Kokkos parallel-reduce, with partially-specified algorithm.
template <typename Policy, typename Functor, typename Reducer>
constexpr decltype(auto) parallel_reduce(Policy&& policy, Functor&& functor, Reducer&& reducer)
{
    return details::parallel<Kokkos::ParallelReduceTag>(
        std::forward<Policy>(policy),
        std::forward<Functor>(functor),
        std::forward<Reducer>(reducer)
    );
}

/**
 * @brief Submit a graph-based sender.
 *
 * @warning Non-blocking.
 */
template <typename Exec, typename Sender>
constexpr void submit(const Exec& exec, Sender&& sender) {
    Kokkos::Impl::GraphAccess::get_graph_weak_ptr(sender).lock()->submit(exec);
}

} // namespace graph

/**
 * @brief Fallback for @ref graph::create_graph when we don't want the underlying graph but simply regular code.
 *
 * @todo The usage of this function is unclear and might be misleading. Indeed, the @p exec
 *       is not passed to the following senders, as one would expect with @c stdexec::just.
 *       We might want to simply get rid of this.
 */
template <typename Exec>
constexpr decltype(auto) just(Exec&& exec) {
    return std::forward<Exec>(exec);
}

/**
 * Fallback for @ref graph::submit when we don't want the underlying graph but simply regular code.
 *
 * @todo Decide what should be done here. Probably nothing is fine.
 */
template <typename... Args>
void submit(Args&&...) {}

/**
 * @overload
 *
 * This is needed to support a chain of senders that are not graph-like senders.
 * It is expected that in such a case, the objects passed to this function are
 * execution space instances.
 *
 * To support eager execution, we arbitrarily fence all but the first one.
 */
template <typename Exec, typename... Args>
requires (! graph::details::is_graph_sender<std::remove_reference_t<Args>> && ...)
decltype(auto) when_all(Exec&& exec, Args&&... args)
{
    (args.fence() && ...);
    return std::forward<Exec>(exec);
}

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_KOKKOS_GRAPH_EXECUTION_HPP
