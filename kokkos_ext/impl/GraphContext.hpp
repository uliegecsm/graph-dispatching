#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP

#include <concepts>

#include "kokkos_ext/impl/ChainHandler.hpp"
#include "kokkos_ext/impl/Concepts.hpp"
#include "kokkos_ext/impl/Utils.hpp"

namespace Kokkos::Experimental
{

namespace graph::details
{
template <typename Exec>
struct ContextState
{
    using graph_t = Kokkos::Experimental::Graph<Exec>;
    using root_t  = decltype(Kokkos::Impl::GraphAccess::create_root_ref(std::declval<graph_t&>()));

    graph_t graph; // points to the underlying whole graph
    root_t root;
    Exec exec; // points to the device on which nodes must be added
};

/**
 * @brief Scheduler for a @c Kokkos::Graph
 *
 * @warning It is a puppet and does not verify the @c std::execution::scheduler concept.
 */
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct GraphScheduler
{
    using context_state_t = ContextState<Exec>;
    using graph_t = typename context_state_t::graph_t;
    using root_t = typename context_state_t::root_t;

    using Env = context_state_t; //! For now, our environment only contains the context state.

    template <receiver Receiver>
    struct OperationState
    {
        Receiver rcvr;

        void start() { std::move(rcvr).set_value(); }
    };

    struct Sender
    {
        Env env;

        template <typename T>
        explicit Sender(T&& ctx) : env{std::forward<T>(ctx)} {}

        template <typename Receiver> requires receiver<std::remove_cvref_t<Receiver>>
        operation_state auto connect(Receiver&& rcvr) {
            return OperationState<std::remove_cvref_t<Receiver>>{std::forward<Receiver>(rcvr)};
        }

        auto& get_env() const { return env; };

        decltype(auto) get_completion_scheduler() const {
            return GraphScheduler(env);
        }
    };

    GraphScheduler(graph_t graph, root_t root, Exec exec)
        : m_context_state{.graph = std::move(graph), .root = std::move(root), .exec = std::move(exec)} {}

    explicit GraphScheduler(context_state_t ctx) : m_context_state(std::move(ctx)) {}

    sender auto schedule() const { return Sender{m_context_state}; }

    template <typename Policy, typename Functor>
    void parallel_for(Policy&& policy, Functor&& functor)
    {
        //! @todo this is shitty
        return m_context_state.root.then_parallel_for(
            graph::details::update_policy(std::forward<Policy>(policy), m_context_state.exec),
            std::forward<Functor>(functor)
        );
    }

    context_state_t m_context_state;
};

template <typename Exec>
struct Pool
{
    std::vector<Exec> execs;
};

} // namespace graph::details

//! Graph context using a @c Kokkos::Graph under the hood.
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct GraphContext
{
    graph::details::Pool<Exec> pool;

    using graph_t = Kokkos::Experimental::Graph<Exec>;
    using root_t  = decltype(Kokkos::Impl::GraphAccess::create_root_ref(std::declval<graph_t&>()));
    graph_t m_graph;
    root_t m_root;

    template <typename... Args> requires (std::same_as<std::remove_cvref_t<Args>, Exec> && ...)
    GraphContext(Args&&... args)
        : pool{.execs = {std::forward<Args>(args)...}},
          m_graph(Kokkos::Impl::GraphAccess::construct_graph(pool.execs.at(0))),
          m_root(Kokkos::Impl::GraphAccess::create_root_ref(m_graph))
    {}

    // return the scheduler from the right device
    scheduler auto get_scheduler(const unsigned short int devID = 0) {
        return graph::details::GraphScheduler<Exec>(m_graph, m_root, pool.execs.at(devID));
    }
};

template <typename... Exec>
GraphContext(Exec&&...) -> GraphContext<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Exec...>>>>;

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPHCONTEXT_HPP
