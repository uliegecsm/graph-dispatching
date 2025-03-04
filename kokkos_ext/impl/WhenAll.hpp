#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_WHENALL_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_WHENALL_HPP

#include "Kokkos_Core_fwd.hpp"

namespace Kokkos::Experimental::graph::details
{
//! @todo Put these things in a namespace to make it less ugly.
template <typename Receiver, typename... Senders>
struct OperationStateWhenAll;

template <typename Receiver, typename Sender>
struct OperationStateWhenAll<Receiver, Sender>
{
    using op_state_t = decltype(std::declval<Sender&&>().connect(std::declval<Receiver>()));

    op_state_t op_state;

    template <typename ReceiverType, typename SenderTypes>
    OperationStateWhenAll(ReceiverType&& rcvr, SenderTypes&& sndrs)
        : op_state(std::get<std::tuple_size_v<SenderTypes> - 1>(std::forward<SenderTypes>(sndrs)).connect(rcvr))
    {}

    void start() { op_state.start(); }
};

template <typename Receiver, typename Sender, typename... Senders>
struct OperationStateWhenAll<Receiver, Sender, Senders...> : public OperationStateWhenAll<Receiver, Senders...>
{
    using base_t = OperationStateWhenAll<Receiver, Senders...>;

    using op_state_t = decltype(std::declval<Sender&&>().connect(std::declval<Receiver>()));

    op_state_t op_state;

    template <typename ReceiverType, typename SenderTypes> requires ((sizeof...(Senders) + 1) == std::tuple_size_v<SenderTypes>)
    OperationStateWhenAll(ReceiverType&& rcvr, SenderTypes&& sndrs)
        : base_t(std::forward<ReceiverType>(rcvr), std::forward<SenderTypes>(sndrs)),
          op_state(std::get<std::tuple_size_v<SenderTypes> - 1 - sizeof...(Senders)>(std::forward<SenderTypes>(sndrs)).connect(std::forward<ReceiverType>(rcvr)))
    {}

    template <typename ReceiverType, typename SenderTypes> requires ((sizeof...(Senders) + 1) < std::tuple_size_v<SenderTypes>)
    OperationStateWhenAll(ReceiverType&& rcvr, SenderTypes&& sndrs)
        : base_t(std::forward<ReceiverType>(rcvr), std::forward<SenderTypes>(sndrs)),
          op_state(std::get<std::tuple_size_v<SenderTypes> - 1 - sizeof...(Senders)>(std::forward<SenderTypes>(sndrs)).connect(rcvr))
    {}

    void start() { base_t::start(); op_state.start(); }
};

template <receiver Receiver>
struct WhenAllReceiver
{
    Receiver rcvr;

    void set_value() && { std::move(rcvr).set_value(); }
};

//! @todo Short description.
template <typename... Senders>
struct WhenAllSender
{
    std::tuple<Senders...> sndrs;

    template <typename Receiver> requires receiver<std::remove_cvref_t<Receiver>>
    operation_state auto connect(Receiver&& rcvr) &&
    {
        using recv_t = WhenAllReceiver<std::remove_cvref_t<Receiver>>;

        return OperationStateWhenAll<recv_t, Senders...>(
            recv_t{.rcvr = std::forward<Receiver>(rcvr)},
            std::move(this->sndrs)
        );
    }

    auto& get_env() const { Kokkos::abort("would not work"); return sndrs; }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_WHENALL_HPP
