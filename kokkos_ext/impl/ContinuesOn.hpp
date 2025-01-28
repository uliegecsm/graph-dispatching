#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONTINUESON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONTINUESON_HPP

#include "Kokkos_Core_fwd.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"

namespace Kokkos::Experimental::graph::details
{
template <sender Sender, receiver Receiver>
struct OperationStateScheduler
{
    using op_state_t = decltype(std::declval<Sender&&>().connect(std::declval<Receiver>()));

    op_state_t op_state;

    template <typename SenderType, typename ReceiverType>
    OperationStateScheduler(SenderType&& sndr, ReceiverType&& rcvr)
        : op_state(std::forward<SenderType>(sndr).connect(std::forward<ReceiverType>(rcvr)))
    {}

    void start() { op_state.start(); }
};

template <receiver Receiver>
struct ContinuesOnReceiver
{
    Receiver rcvr;

    void set_value() && { std::move(rcvr).set_value(); }
};

template <sender Sender, scheduler Scheduler>
struct ContinuesOnSender
{
    Sender sndr;
    Scheduler sch;

    template <typename Receiver>
    operation_state auto connect(Receiver&& rcvr) &&
    {
        return OperationStateScheduler<Sender, std::remove_cvref_t<Receiver>>(
            std::move(sndr),
            std::forward<Receiver>(rcvr)
        );
    }

    auto& get_env() const { return sch.m_context_state; }
};

template <typename Scheduler> requires scheduler<std::remove_cvref_t<Scheduler>>
struct ContinuesOnPartial
{
    Scheduler sch;

    //! For the piping operator.
    template <typename Sender>
    sender auto operator()(Sender&& sndr) &&
    {
        return ContinuesOnSender{.sndr = std::forward<Sender>(sndr), .sch = std::move(this->sch)};
    }
};
} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_CONTINUESON_HPP
