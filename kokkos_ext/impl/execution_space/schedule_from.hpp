#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCHEDULE_FROM_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCHEDULE_FROM_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

namespace Kokkos::Experimental::details::execution_space
{

//! Receiver for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ScheduleFromReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Schd schd;
    Rcvr rcvr;
    bool skip;

    void set_value() && noexcept {
        if (!skip) schd.exec.fence(std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<decltype(schd.exec)>::name()));
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

//! Sender for @c schedule_from.
template <stdexec::scheduler Schd, ::stdexec::sender Sndr>
struct ScheduleFromSender
{
    using sender_concept = ::stdexec::sender_t;

    //! The transition may throw or forwards the error channel.
    using with_error_invoke_t = ::stdexec::completion_signatures<
        ::stdexec::set_value_t(),
        ::stdexec::set_error_t(std::exception_ptr)
    >;

    template <typename Self, typename... Env_>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env_...>,
        with_error_invoke_t
    >;

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <typename... Env_>
    [[nodiscard]] constexpr auto
    get_completion_signatures(Env_&&...) -> _completion_signatures<ScheduleFromSender, Env_...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        using recv_t = ScheduleFromReceiver<Schd, std::remove_cvref_t<Rcvr>>;

        return ::stdexec::connect(
            std::move(sndr),
            recv_t{.schd = std::move(schd), .rcvr = std::forward<Rcvr>(rcvr), .skip = skip}
        );
    }

    auto get_env() const noexcept -> ::stdexec::env_of_t<Sndr> {
        return ::stdexec::get_env(sndr);
    }

    Schd schd;
    Sndr sndr;
    bool skip;
};

template <typename Env>
struct transform_sender_for<stdexec::schedule_from_t, Env>
{
    template <::stdexec::sender Sndr>
    auto operator()(stdexec::schedule_from_t, ::stdexec::__ignore, Sndr&& sndr) && noexcept
    {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        static_assert(stdexec::__is_instance_of<decltype(schd), Scheduler>);

        const bool skip = [&](){
            if constexpr (stdexec::__is_instance_of<std::remove_cvref_t<Env>, SchedulerEnv>) {
                if constexpr (std::same_as<
                    typename std::remove_cvref_t<decltype(env_.exec)>,
                    typename std::remove_cvref_t<decltype(schd.exec)>
                >) {
                    return schd.exec == env_.exec;
                }
            }
            return false;
        }();
        return ScheduleFromSender{
            .schd = std::move(schd),
            .sndr = std::forward<Sndr>(sndr),
            .skip = skip
        };
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCHEDULE_FROM_HPP
