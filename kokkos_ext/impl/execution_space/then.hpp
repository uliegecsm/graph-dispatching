#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/execution_space/receiver.hpp"

namespace Kokkos::Experimental::details::execution_space {

//! Inspired by https://github.com/kokkos/kokkos/blob/69273c3a4e7b6adeb95066341ca201d62fe1e698/core/src/impl/Kokkos_GraphNodeThenImpl.hpp#L28.
template <typename Functor>
struct ThenWrapper {
    Functor functor;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T) const {
        functor();
    }
};

template <typename... Fused>
struct ThenFusedWrapper {
    std::tuple<Fused...> fused;

    template <size_t... Is>
    KOKKOS_FUNCTION void call_impl(std::index_sequence<Is...>) const {
        (std::get<Is>(fused)(), ...);
    }
    
    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T) const {
        call_impl(std::make_index_sequence<sizeof...(Fused)>{});
    }
};

template <stdexec::sender Sndr, stdexec::receiver Rcvr, typename Functor, stdexec::__is_instance_of<Scheduler> Schd>
struct ThenReceiver;

template <stdexec::sender Sndr, stdexec::receiver InnerRcvr, typename Functor, stdexec::__is_instance_of<Scheduler> Schd>
struct ThenOpState {
    using rcvr_t = ThenReceiver<Sndr, InnerRcvr, Functor, Schd>;
    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;
    using functor_t = Functor;

    Schd schd;
    InnerRcvr inner_rcvr;
    inner_opstate_t inner_opstate;
    Functor functor;

    bool do_dispatch = true;

    template <stdexec::sender Sender, stdexec::receiver Rcvr, typename Func>
    ThenOpState(Schd&& schd_, Sender&& sndr, Rcvr&& rcvr, Func&& func)
        : schd(std::move(schd_))
        , inner_rcvr(std::forward<Rcvr>(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sender>(sndr), rcvr_t{this}))
        , functor(std::forward<Func>(func)) {
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<typename Schd::properties_t>::name();
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<inner_opstate_t>::name();
        if constexpr (stdexec::__is_instance_of_<inner_opstate_t, ThenOpState>) {
            PLOG_DEBUG << "Inner opstate derives from ThenOpState.";
            // add check on the scheduler properties, mode must be "allow fusion"
            inner_opstate.do_dispatch = false;
        }
    }

    //! The operation state solve the problem with the resource ? It should in principle but not for our kokkos customization.
    void dispatch() noexcept {
        if (do_dispatch) {
            if constexpr (stdexec::__is_instance_of_<inner_opstate_t, ThenOpState>) {
                Kokkos::parallel_for(
                    std::format("{}: then", Kokkos::Impl::TypeInfo<decltype(this->schd.exec)>::name()),
                    Kokkos::RangePolicy(std::move(this->schd).exec, 0, 1),
                    // ThenWrapper{std::move(functor)}
                    ThenFusedWrapper<typename inner_opstate_t::functor_t, Functor>{{std::move(inner_opstate.functor), std::move(functor)}}
                );
            } else {
                Kokkos::parallel_for(
                    std::format("{}: then", Kokkos::Impl::TypeInfo<decltype(this->schd.exec)>::name()),
                    Kokkos::RangePolicy(std::move(this->schd).exec, 0, 1),
                    ThenWrapper{std::move(functor)});
            }
        }
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }

    template <typename Tag, typename... Args>
    void propagate_completion_signal(Tag, Args&&... args) && noexcept {
        Tag()(std::move(inner_rcvr), std::forward<Args>(args)...);
    }

    auto get_env() const noexcept -> stdexec::env_of_t<InnerRcvr> {
        return ::stdexec::get_env(inner_rcvr);
    }
};

/**
 * @brief Receiver for @c then.
 *
 * @note It must be nothrow moveable, see @cite P3383R3.
 */
template <stdexec::sender Sndr, stdexec::receiver Rcvr, typename Functor, stdexec::__is_instance_of<Scheduler> Schd>
struct ThenReceiver {
    using receiver_concept = stdexec::receiver_t;
    using opstate_t = ThenOpState<Sndr, Rcvr, Functor, Schd>;

    opstate_t* opstate;

    void set_value() && noexcept {
        opstate->dispatch();
        std::move(*opstate).propagate_completion_signal(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        std::move(*opstate).propagate_completion_signal(::stdexec::set_error, std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        std::move(*opstate).propagate_completion_signal(::stdexec::set_stopped);
    }

    auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return opstate->get_env();
    }
};

/**
 * @brief Sender for @c then.
 *
 * @todo We should decide what to do with a "throwing situation". There are 2 reasons for @c ThenReceiver::set_value to throw:
 *          1. The functor itself has a call operator that may throw.
 *          2. @c Kokkos itself throws before launching the functor (for some reason).
 */
template <stdexec::sender Sndr, typename Functor, typename Schd>
struct ThenSender {
    using sender_concept = stdexec::sender_t;

    //! @c Kokkos may throw while launching the kernel.
    using with_error_invoke_t =
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(ThenSender)

    //! See also https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/then.cuh#L52.
    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ThenOpState<Sndr, Rcvr, Functor, Schd>(std::move(schd), std::move(sndr), std::forward<Rcvr>(rcvr), std::move(functor));
    }

    Sndr sndr;
    Functor functor;
    Schd schd;

    auto get_env() const noexcept -> ::stdexec::env_of_t<Sndr> {
        return stdexec::get_env(sndr);
    }
};

template <typename Env>
struct transform_sender_for<stdexec::then_t, Env> {
    template <typename Functor, typename Sndr>
    requires execution_space_completing_sender<Sndr, Env>
    auto operator()(stdexec::then_t, Functor&& functor, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        return ThenSender{
            .sndr = std::forward<Sndr>(sndr), .functor = std::forward<Functor>(functor), .schd = std::move(schd)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP
