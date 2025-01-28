#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SPLIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SPLIT_HPP

#include "Kokkos_Core_fwd.hpp"

namespace Kokkos::Experimental::graph::details
{

template <typename State, receiver Receiver>
struct OperationStateSplit
{
    Receiver rcvr;
    State state;

    void start() { state->start(); state->add_continuation(std::move(rcvr)); }
};

template <typename Receiver>
struct SplitReceiver
{
    Receiver rcvr;

    void set_value() && { std::move(rcvr).set_value(); }
};

// Not sure this implenetation is OK, and not sure why they all use atomics ...
template <sender Sender>
struct SharedState
{
    Sender sndr;

    std::atomic<bool> start_called = false;
    std::atomic<unsigned> counter = 0;

    std::mutex continuation;

    template <typename T> requires std::same_as<std::remove_cvref_t<T>, Sender>
    SharedState(T&& sndr_) : sndr(std::forward<T>(sndr_)) {};

    ~SharedState() { if(!start_called) Kokkos::abort("It was never started."); }

    void start() & noexcept { start_called = true; }

    template <typename Receiver>
    void add_continuation(Receiver&& rcvr)
    {
        const std::lock_guard<std::mutex> guard(continuation);
        printf("%s\n", __PRETTY_FUNCTION__);
        std::forward<Receiver>(rcvr).set_value();
    }
};

template <sender Sender>
struct SplitSender
{
    using state_t = SharedState<Sender>;

    std::shared_ptr<state_t> state;

    template <typename T> requires std::same_as<std::remove_cvref_t<T>, Sender>
    SplitSender(T&& sndr_) : state(std::make_shared<state_t>(std::forward<T>(sndr_))) {}

    //! Multi-shot, so we don't require it to be in a move-from state.
    template <typename Receiver>
    operation_state auto connect(Receiver&& rcvr)
    {
        using rcvr_t = SplitReceiver<std::remove_cvref_t<Receiver>>;
        return OperationStateSplit<std::shared_ptr<state_t>, rcvr_t>{rcvr_t{std::forward<Receiver>(rcvr)}, state};
    }

    auto& get_env() const { return state->sndr.get_env(); }

    decltype(auto) get_completion_scheduler() const { return state->sndr.get_completion_scheduler(); }
};

struct SplitPartial
{
    template <typename Sender>
    sender auto operator()(Sender&& sndr) &&
    {
        return SplitSender<std::remove_cvref_t<Sender>>{sndr};
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SPLIT_HPP
