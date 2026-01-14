#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

namespace Kokkos::Experimental::details::execution_space
{

//! Receiver for @c continues_on.
template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Schd schd;
    Rcvr rcvr;

    void set_value() && noexcept {
        ::stdexec::set_value(std::move(rcvr));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        ::stdexec::set_stopped(std::move(rcvr));
    }

    decltype(auto) get_env() const noexcept { return SchedulerEnv{schd.exec}; }
};

//! Sender for @c continues_on.
template <::stdexec::scheduler Schd, ::stdexec::sender Sndr>
struct ContinuesOnSender
{
    using sender_concept = ::stdexec::sender_t;

    //! The transition may throw or forwards the error channel.
    using with_error_invoke_t = ::stdexec::completion_signatures<
        ::stdexec::set_value_t(),
        ::stdexec::set_error_t(std::exception_ptr)
    >;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <typename... Env>
    [[nodiscard]] constexpr auto
    get_completion_signatures(Env&&...) -> _completion_signatures<ContinuesOnSender, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        using recv_t = ContinuesOnReceiver<Schd, std::remove_cvref_t<Rcvr>>;

        return ::stdexec::connect(
            std::move(sndr),
            recv_t{.schd = std::move(schd), .rcvr = std::forward<Rcvr>(rcvr)}
        );
    }

    decltype(auto) get_env() const noexcept { return SchedulerEnv{schd.exec}; }

    Schd schd;
    Sndr sndr;
};

template <typename Env>
struct transform_sender_for<stdexec::continues_on_t, Env>
{
    template <stdexec::scheduler Schd, ::stdexec::sender Sndr>
        requires stdexec::__is_instance_of<Schd, Scheduler>
    auto operator()(stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) && noexcept {
        return ContinuesOnSender{.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
