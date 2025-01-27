#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP

#include <concepts>

#include "Kokkos_Core.hpp"

namespace Kokkos::Experimental
{

//! Subset of @c std::execution::sender.
template <typename T>
concept Sender = true;

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
 */
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceScheduler
{
    using context_state_t = ContextState<Exec>;

    using Env = context_state_t; //! For now, our environment only contains the context state.

    template <typename Receiver>
    struct OperationState_
    {
        Receiver rcvr;

        template <typename T>
        OperationState_(T&& rcvr_) : rcvr(std::forward<T>(rcvr_)) {}

        void start() { rcvr.set_value(); }
    };

    struct Sender_
    {
        Env env;

        template <typename T>
        Sender_(T&& ctx) : env{std::forward<T>(ctx)} {}

        //! @todo Constraint with a receiver concept.
        template <typename Receiver>
        OperationState_<std::remove_cvref_t<Receiver>> connect(Receiver&& rcvr) {
            return {std::forward<Receiver>(rcvr)};
        }

        auto& get_env() const { return env; };
    };

    template <typename T>
    explicit ExecutionSpaceScheduler(T&& exec) : m_context_state{std::forward<T>(exec)} {}

    Sender auto schedule() const { return Sender_{m_context_state}; }

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

    auto get_scheduler() { return details::execution_space::ExecutionSpaceScheduler{exec}; }
};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
