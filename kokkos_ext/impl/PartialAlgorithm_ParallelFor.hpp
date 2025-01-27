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
 * @todo Constraint template arguments.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L652-L663
 */
template <typename Sender, typename Receiver>
struct OperationState
{
    using op_state_t = decltype(std::declval<Sender>().connect(std::declval<Receiver>()));

    op_state_t op_state;

    template <typename SenderType, typename ReceiverType>
    OperationState(SenderType&& sndr, ReceiverType&& rcvr)
        : op_state(std::forward<SenderType>(sndr).connect(std::forward<Receiver>(rcvr)))
    {}

    //! @todo Should me move @ref op_state ?
    void start() { op_state.start(); }
};

/**
 * @brief Sender that works with @ref ParallelForSender.
 *
 * Since @ref ParallelForSender is eagerly executing, there is nothing to do.
 *
 * @todo Short description, references and better type constraints.
 */
template <typename Receiver>
struct ParallelForReceiver
{
    Receiver rcvr;

    void set_value() const { rcvr.set_value(); }
};

/**
 * @brief Eager execution sender.
 *
 * @todo Reference, and better type constraitns.
 */
template <typename Sender, typename Policy, typename Functor>
struct ParallelForSender
{
    Sender sender;

    //! Eager execution of the functor. @todo Pros and cons ?
    template <typename SenderType, typename PolicyType, typename FunctorType>
    ParallelForSender(SenderType&& sender_, PolicyType&& policy, FunctorType&& functor)
        : sender(std::forward<SenderType>(sender_))
    {
        Kokkos::parallel_for(
            update_policy(std::forward<PolicyType>(policy), sender.get_env().exec),
            std::forward<FunctorType>(functor)
        );
    }

    //! @todo Constraint the input and output types.
    template <typename Receiver>
    auto connect(Receiver&& rcvr)
    {
        using recv_t = ParallelForReceiver<std::remove_cvref_t<Receiver>>;

        return OperationState<Sender, recv_t>(
            this->sender,
            recv_t{.rcvr = std::forward<Receiver>(rcvr)}
        );
    }

    auto& get_env() const { return sender.get_env(); }
};

//! Specialization for @c Kokkos parallel-for.
template <typename Policy, typename Functor>
struct PartialAlgorithm<Kokkos::ParallelForTag, std::string, Policy, Functor>
{
    std::string label;
    Policy policy;
    Functor functor;

    //! To avoid spurious copies, this is only available when in a movable state.
    template <typename Sender>
    decltype(auto) operator()(Sender&& sender) &&
    {
        using sender_t = ParallelForSender<std::remove_cvref_t<Sender>, Policy, Functor>;
        return sender_t(
            std::forward<Sender>(sender),
            std::move(this->policy),
            std::move(this->functor)
        );
    }
};

// TO BE MOVED TO APPROPRIATE INCLUDE FILE
template <typename Exec>
struct SyncWaitReceiver
{
    Exec exec;

    void set_value() const { exec.fence(); }
};

//! @todo Short description.
struct SyncWait
{
    template <typename Sender>
    void operator()(Sender&& sndr) const
    {
        std::forward<Sender>(sndr).connect(
            SyncWaitReceiver{sndr.get_env().exec}
        ).start();
    }
};

} // namespace Kokkos::Experimental::graph::details

namespace Kokkos::Experimental::graph
{
//! @todo Constraint with a sender concept.
template <typename Sender>
decltype(auto) sync_wait(Sender&& sender) {
    return details::SyncWait{}.operator()(std::forward<Sender>(sender));
}
}

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
