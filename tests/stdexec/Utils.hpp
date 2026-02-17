#ifndef GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
#define GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/env.hpp"

namespace tests::stdexec {

template <class Sndr, class Tag>
concept has_completion_scheduler_for =
    ::stdexec::__queryable<Sndr>
    && std::invocable<::stdexec::get_completion_scheduler_t<Tag>, const ::stdexec::env_of_t<Sndr>&>;

template <typename Sndr, typename Signatures, typename... Env>
concept has_completion_signatures =
    ::stdexec::__mset_eq<Signatures, ::stdexec::__completion_signatures_of_t<Sndr, Env...>>;

namespace impl {

template <::stdexec::scheduler Scheduler, typename Tag, ::stdexec::sender Sndr>
struct CheckSchedulerSender {
    using sender_concept = ::stdexec::sender_t;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_KEEP(CheckSchedulerSender)

    template <::stdexec::receiver Rcvr>
    constexpr ::stdexec::operation_state auto
        connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        /// First, try to get the completion scheduler from the sender environment.
        if constexpr (requires {
                          ::stdexec::get_completion_scheduler<Tag>(::stdexec::get_env(sndr), ::stdexec::get_env(rcvr));
                      }) {
            using scheduler_t =
                decltype(::stdexec::get_completion_scheduler<Tag>(::stdexec::get_env(sndr), ::stdexec::get_env(rcvr)));

            static_assert(
                std::same_as<std::remove_cvref_t<scheduler_t>, Scheduler>,
                "Scheduler type mismatch: completion scheduler doesn't match expected type.");
        }
        /// Fallback on the receiver environment.
        else if constexpr (requires { ::stdexec::get_scheduler(::stdexec::get_env(rcvr)); }) {
            using scheduler_t = decltype(::stdexec::get_scheduler(::stdexec::get_env(rcvr)));

            static_assert(
                std::same_as<std::remove_cvref_t<scheduler_t>, Scheduler>,
                "Scheduler type mismatch: receiver scheduler doesn't match expected type.");
        } else {
            static_assert(
                std::same_as<decltype(::stdexec::get_completion_signatures(sndr, ::stdexec::get_env(rcvr))), int>,
                "No scheduler found.");
        }

        return ::stdexec::connect(std::move(sndr), std::forward<Rcvr>(rcvr));
    }

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Sndr, sndr)
};

template <::stdexec::scheduler Scheduler, typename Tag>
struct CheckScheduler {
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

//! A receiver that can handle all completions and does nothing with them.
struct SinkReceiver {
    using receiver_concept = ::stdexec::receiver_t;

    void set_value(auto&&...) noexcept {
    }
    void set_error(auto&&) noexcept {
    }
    void set_stopped() noexcept {
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> ::stdexec::env<> {
        return {};
    }
};

//! Receiver for a value, inspired by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/test/test_common/receivers.hpp#L95.
template <typename ValueType, typename Env = ::stdexec::env<>>
struct ValueReceiver {
    using receiver_concept = ::stdexec::receiver_t;

    ValueType* value;
    Env env_{};

    constexpr void set_value(ValueType value_) noexcept {
        *value = std::move(value_);
    }
    void set_error(std::exception_ptr) noexcept { // NOLINT(performance-unnecessary-value-param)
    }
    void set_stopped() noexcept {
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> const Env& {
        return env_;
    }
};

//! Default scheduler type when none provided.
using default_scheduler_t = ::stdexec::run_loop::scheduler;

} // namespace tests::stdexec

#endif // GRAPH_DISPATCHING_TESTS_STDEXEC_UTILS_HPP
