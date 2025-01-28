#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP

#include "kokkos_ext/impl/Concepts.hpp"
#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"
#include "kokkos_ext/impl/PartialAlgorithm.hpp"
#include "kokkos_ext/impl/Utils.hpp"

namespace Kokkos::Experimental::graph::details
{

/**
 * @brief To be done soon.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L652-L663
 */
template <sender Sender, receiver Receiver>
struct OperationState
{
    using op_state_t = decltype(std::declval<Sender&&>().connect(std::declval<Receiver>()));

    op_state_t op_state;

    template <typename SenderType, typename ReceiverType>
    OperationState(SenderType&& sndr, ReceiverType&& rcvr)
        : op_state(std::forward<SenderType>(sndr).connect(std::forward<ReceiverType>(rcvr)))
    {}

    void start() { op_state.start(); }
};

/**
 * @brief Sender that works with @ref ParallelForSender.
 *
 * Since @ref ParallelForSender is eagerly executing, there is nothing to do.
 */
template <receiver Receiver>
struct ParallelForReceiver
{
    Receiver rcvr;

    void set_value() && { std::move(rcvr).set_value(); }
};

/**
 * @brief Eager execution sender.
 *
 * Pros and cons of eager senders are listed at
 * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-lazy-algorithms-detached.
 */
template <sender Sender, typename Policy, typename Functor>
struct ParallelForSender
{
    Sender sndr;

    //! Eager execution of the functor.
    template <sender SenderType, typename PolicyType, typename FunctorType>
    ParallelForSender(SenderType&& sender, PolicyType&& policy, FunctorType&& functor)
        : sndr(std::forward<SenderType>(sender))
    {
        Kokkos::parallel_for(
            update_policy(std::forward<PolicyType>(policy), get_env().exec),
            std::forward<FunctorType>(functor)
        );
    }

    template <typename Receiver> requires receiver<std::remove_cvref_t<Receiver>>
    operation_state auto connect(Receiver&& rcvr) &&
    {
        using recv_t = ParallelForReceiver<std::remove_cvref_t<Receiver>>;

        return OperationState<Sender, recv_t>(
            std::move(this->sndr),
            recv_t{.rcvr = std::forward<Receiver>(rcvr)}
        );
    }

    auto& get_env() const { return sndr.get_env(); }
};

//! Specialization for @c Kokkos parallel-for.
template <typename Policy, typename Functor>
struct PartialAlgorithm<Kokkos::ParallelForTag, std::string, Policy, Functor>
{
    std::string label;
    Policy policy;
    Functor functor;

    //! To avoid unwanted copies, this is only available when in a movable state.
    template <typename Sender> requires sender<std::remove_cvref_t<Sender>>
    decltype(auto) operator()(Sender&& sndr) &&
    {
        using sender_t = ParallelForSender<std::remove_cvref_t<Sender>, Policy, Functor>;
        return sender_t(
            std::forward<Sender>(sndr),
            std::move(this->policy),
            std::move(this->functor)
        );
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
