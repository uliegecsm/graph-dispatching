#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP

#include "kokkos_ext/impl/Concepts.hpp"
#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"
#include "kokkos_ext/impl/PartialAlgorithm.hpp"
#include "kokkos_ext/impl/Utils.hpp"

namespace Kokkos::Experimental::graph::details
{



template <typename Sender,
          typename Receiver>
struct OperationState
{
    using op_state_t = decltype(std::declval<Sender>().sender.connect(std::declval<Receiver>()));

    op_state_t op_state;

    // check type (forward rvalue etc)
    template <typename ReceiverType> //requires std::same_as<std::remove_cvref_t<ReceiverType>, Receiver>
    OperationState(Sender sndr_, ReceiverType rcvr_)
        : op_state(sndr_.sender.connect(rcvr_))
    {}

    void start() { op_state.start(); }
};

template <typename Receiver,
          typename Policy,
          typename Functor>
struct ParallelForReceiver
{
    Policy policy;
    Functor functor;
    Receiver rcvr;

    void set_value()
    {
        Kokkos::parallel_for(
            // update_policy(policy, context_state.get_instance()),
            policy,
            functor
        );

        rcvr.set_value();
    }

    // todo check type
    template <typename ReceiverType>// requires std::same_as<std::remove_cvref_t<ReceiverType>, Receiver>
    explicit ParallelForReceiver(Policy policy_,
                                 Functor functor_,
                                 ReceiverType && rcvr_)
        : policy(policy_),
          functor(std::move(functor_)),
          rcvr(std::forward<ReceiverType>(rcvr_))
    {}
};

template <typename Sender,
          typename Policy,
          typename Functor>
struct ParallelForSender
{
    Sender sender;
    Policy policy;
    Functor functor;

    // todo check type
    template <typename Receiver> //requires std::same_as<std::remove_cvref_t<SenderType>, Sender>
    auto connect(Receiver rcvr)
    {
        using sndr_t = ParallelForSender  <Sender, Policy, Functor>;
        using recv_t = ParallelForReceiver<Receiver, Policy, Functor>;

        return OperationState<sndr_t, recv_t>(
            *this,
            recv_t(this->policy, this->functor, rcvr)
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

    template <typename Sender>
    decltype(auto) operator()(Sender&& sender) &&
    {
        using sender_t = ParallelForSender<std::remove_cvref_t<Sender>, Policy, Functor>;
        return sender_t{
            .sender  = std::forward<Sender>(sender),
            .policy  = std::move(this->policy),
            .functor = std::move(this->functor)
        };
    }
};

// TO BE MOVED TO APPROPRIATE INCLUDE FILE
template <typename Exec>
struct SyncWaitReceiver
{
    Exec exec;

    void set_value() { exec.fence(); }
};

struct SyncWait
{
    template <typename Sender>
    void operator()(Sender &&sndr) const
    {
        auto exec = sndr.get_env().exec;

        auto op_state = std::forward<Sender>(sndr).connect(SyncWaitReceiver{exec});

        op_state.start();
    }
};

} // namespace Kokkos::Experimental::graph::details

namespace Kokkos::Experimental::graph
{
template <typename Sender>
decltype(auto) sync_wait(Sender&& sender)
{
    return details::SyncWait{}.operator()(std::forward<Sender>(sender));
}
}

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
