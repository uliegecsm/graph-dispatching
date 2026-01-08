#ifndef GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
#define GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP

#include "stdexec/execution.hpp"

namespace tests::stdexec
{

template <class Sndr, class Tag>
concept has_completion_scheduler_for = ::stdexec::queryable<Sndr> && std::invocable<
    ::stdexec::get_completion_scheduler_t<Tag>,
    const ::stdexec::env_of_t<Sndr>&
>;

template <class Sndr, class... Signatures>
concept has_completion_signatures = ::stdexec::queryable<Sndr> && std::same_as<
    std::invoke_result_t<::stdexec::get_completion_signatures_t, Sndr>,
    ::stdexec::completion_signatures<Signatures...>
>;

namespace impl
{
template <::stdexec::scheduler Scheduler, typename Tag, ::stdexec::sender Sndr>
struct CheckSchedulerSender
{
    using sender_concept = ::stdexec::sender_t;

    template <typename Self, typename... Env>
    using completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>
    >;

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <class... Env>
    auto get_completion_signatures(Env&&...) -> completion_signatures<CheckSchedulerSender, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <::stdexec::receiver Rcvr>
    constexpr ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        /// First, try to get the completion scheduler from the sender environment.
        if constexpr (requires { ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(sndr), ::stdexec::get_env(rcvr)); })
        {
            using scheduler_t = decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(sndr), ::stdexec::get_env(rcvr)));

            static_assert(
                std::same_as<std::remove_cvref_t<scheduler_t>, Scheduler>,
                "Scheduler type mismatch: completion scheduler doesn't match expected type."
            );
        }
        /// Fallback on the receiver environment.
        else if constexpr (requires { ::stdexec::get_scheduler(::stdexec::get_env(rcvr)); })
        {
            using scheduler_t = decltype(::stdexec::get_scheduler(::stdexec::get_env(rcvr)));

            static_assert(
                std::same_as<std::remove_cvref_t<scheduler_t>, Scheduler>,
                "Scheduler type mismatch: receiver scheduler doesn't match expected type."
            );
        }
        else {
            static_assert(std::same_as<
                decltype(::stdexec::get_completion_signatures(sndr, ::stdexec::get_env(rcvr))),
                int
            >, "No scheduler found.");
        }

        return ::stdexec::connect(std::move(sndr), std::forward<Rcvr>(rcvr));
    }

    Sndr sndr;

    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(sndr); }
};

template <::stdexec::scheduler Scheduler, typename Tag>
struct CheckScheduler
{
    template <::stdexec::sender Sndr>
    constexpr auto operator()(Sndr&& sndr) const noexcept -> CheckSchedulerSender<Scheduler, Tag, Sndr> {
        return {.sndr = std::forward<Sndr>(sndr)};
    }

    constexpr auto operator()() const noexcept {
        return ::stdexec::__closure{*this};
    }
};
} // namespace impl

//! Sender adaptor to check the type of scheduler that is currently used.
template <::stdexec::scheduler Scheduler, typename Tag = ::stdexec::set_value_t>
constexpr auto check_scheduler() {
    return impl::CheckScheduler<Scheduler, Tag>{}();
};

} // namespace tests::stdexec

#endif // GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
