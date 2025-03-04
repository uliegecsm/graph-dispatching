#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP

#include "Kokkos_Core.hpp"

#include "kokkos_ext/impl/Concepts.hpp"
#include "kokkos_ext/impl/Utils.hpp"

namespace Kokkos::Experimental
{

namespace details::execution_space
{
//! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ContextState
{
    Exec exec;
};

/**
 * @brief Scheduler for a @c Kokkos execution space.
 *
 * @warning It is a puppet and does not verify the @c std::execution::scheduler concept.
 *
 * References:
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#example-schedulers-inline
 */
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceScheduler
{
    using context_state_t = ContextState<Exec>;

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

        scheduler auto get_completion_scheduler() const {
            return ExecutionSpaceScheduler(env.exec);
        }
    };

    template <typename T>
    explicit ExecutionSpaceScheduler(T&& exec) : m_context_state{std::forward<T>(exec)} {}

    sender auto schedule() const { return Sender{m_context_state}; }

    template <typename Policy, typename Functor>
    void parallel_for(Policy&& policy, Functor&& functor) const
    {
        Kokkos::parallel_for(
            graph::details::update_policy(std::forward<Policy>(policy), m_context_state.exec),
            std::forward<Functor>(functor)
        );
    }

    context_state_t m_context_state;
};

//! Deduction guide for @ref ExecutionSpaceScheduler.
template <typename Exec>
ExecutionSpaceScheduler(Exec&&) -> ExecutionSpaceScheduler<std::remove_cvref_t<Exec>>;

} // namespace details::execution_space

/**
 * @brief Execution context using a @c Kokkos execution space under the hood.
 *
 * For instance, if @p Exec is @c Kokkos::Cuda, the following holds true:
 *  1. The execution context will be the @c Cuda stream stored by the @c Kokkos::Cuda instance @ref exec.
 *  2. The execution resource is the GPU the stream is attached to.
 */
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceContext
{
    Exec exec;

    scheduler auto get_scheduler() { return details::execution_space::ExecutionSpaceScheduler{exec}; }
};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
