#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_KOKKOS_GRAPH_EXECUTION_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_KOKKOS_GRAPH_EXECUTION_HPP

#include <concepts>
#include <functional>

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

namespace Kokkos::Experimental::graph
{

namespace details
{

/**
 * @brief Helper for piping support.
 *
 * Before the @c operator| is called, we have a *partial* algorithm because we
 * don't know the *parent* yet.
 */
template <typename Tag, typename Policy, typename Functor>
struct PartialAlgorithm
{
    Tag tag;
    Policy policy;
    Functor functor;

    template <typename Sender> requires (std::same_as<Tag, Kokkos::ParallelForTag>)
    decltype(auto) operator()(Sender&& input)
    {
        return std::forward<Sender>(input).then_parallel_for(
            std::move(policy),
            std::move(functor)
        );
    }

    //! Helper for piping support.
    template <typename Sender, typename PA>
    friend constexpr decltype(auto) operator|(Sender&& input, PA&& partial);
};

template <typename Sender, typename PA>
constexpr decltype(auto) operator|(Sender&& input, PA&& partial) {
    return std::invoke(std::forward<PA>(partial), std::forward<Sender>(input));
}

//! Pipable @c parallel whatever.
template <typename Tag, typename Policy, typename Functor>
constexpr decltype(auto) parallel(const Tag, Policy&& policy, Functor&& functor)
{
    return PartialAlgorithm(Tag{}, std::forward<Policy>(policy), std::forward<Functor>(functor));
}

//! Helper class to avoid exposing @c Kokkos::Graph itself to the user.
template <typename Exec>
struct ChainHandler
{
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
constexpr decltype(auto) just(const Exec& exec) {
    return details::ChainHandler(exec);
}

//! For @c Kokkos::Graph, @c split is a no-op.
constexpr decltype(auto) split() {
    return std::identity{};
}

/**
 * @name Parallel-for redirections.
 */
///@{
//! Regular @c Kokkos parallel-for with fully-specified algorithm.
template <typename Sender, typename Policy, typename Functor>
constexpr decltype(auto) parallel_for(Sender&& input, Policy&& policy, Functor&& functor)
{
    return std::forward<Sender>(input).then_parallel_for(
        std::forward<Policy>(policy),
        std::forward<Functor>(functor)
    );
}

//! Pipable @c Kokkos parallel-for, with partially-specified algorithm.
template <typename Policy, typename Functor>
constexpr decltype(auto) parallel_for(Policy&& policy, Functor&& functor)
{
    return details::parallel(
        Kokkos::ParallelForTag{},
        std::forward<Policy>(policy),
        std::forward<Functor>(functor)
    );
}
///@}

/**
 * @brief Submit a graph-based sender.
 *
 * @warning Non-blocking.
 */
template <typename Exec, typename Sender>
constexpr void submit(const Exec& exec, Sender&& sender) {
    Kokkos::Impl::GraphAccess::get_graph_weak_ptr(sender).lock()->submit(exec);
}

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_KOKKOS_GRAPH_EXECUTION_HPP
