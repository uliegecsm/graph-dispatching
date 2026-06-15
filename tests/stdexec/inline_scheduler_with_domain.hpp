#ifndef GRAPH_DISPATCHING_TESTS_STDEXEC_INLINE_SCHEDULER_WITH_DOMAIN_HPP
#define GRAPH_DISPATCHING_TESTS_STDEXEC_INLINE_SCHEDULER_WITH_DOMAIN_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos_ext/impl/completion_signatures.hpp"

namespace iswd {

//! Specialize me to customize an algorithm for the @ref iswd::inline_scheduler.
template <typename Tag>
struct transform_sender_for;

//! Domain for @ref iswd::inline_scheduler.
struct domain : public stdexec::default_domain {
    template <stdexec::sender Sndr, typename Env>
    requires stdexec::__applicable<transform_sender_for<stdexec::tag_of_t<Sndr>>, Sndr, const Env&>
    static auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env) {
        return stdexec::__apply(transform_sender_for<stdexec::tag_of_t<Sndr>>{}, std::forward<Sndr>(sndr), env);
    }
};

//! Inline scheduler completing on @ref iswd::domain.
struct inline_scheduler {
    struct sender_t {
        using sender_concept = ::stdexec::sender_t;

        template <::stdexec::receiver Rcvr>
        struct operation_state {
            using operation_state_concept = ::stdexec::operation_state_t;

            Rcvr rcvr;

            void start() & noexcept {
                ::stdexec::set_value(std::move(rcvr));
            }
        };

        using completion_signatures = ::stdexec::completion_signatures<::stdexec::set_value_t()>;

        template <::stdexec::receiver_of<completion_signatures> Rcvr>
        auto connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) -> operation_state<Rcvr> {
            return {.rcvr = std::move(rcvr)};
        }

        struct attrs {
            constexpr auto query(::stdexec::get_completion_scheduler_t<::stdexec::set_value_t>) const noexcept
                -> inline_scheduler {
                return {};
            }
        };

        constexpr auto get_env() const noexcept -> attrs {
            return {};
        }
    };

    [[nodiscard]]
    constexpr auto query(::stdexec::get_completion_domain_t<::stdexec::set_value_t>) const noexcept -> domain {
        return {};
    }

    [[nodiscard]]
    constexpr auto
        query(::stdexec::get_completion_scheduler_t<::stdexec::set_value_t>) const noexcept -> inline_scheduler {
        return {};
    }

    auto schedule() const noexcept -> sender_t {
        return {};
    }

    friend auto operator<=>(const inline_scheduler&, const inline_scheduler&) noexcept = default;
};

} // namespace iswd

#endif // GRAPH_DISPATCHING_TESTS_STDEXEC_INLINE_SCHEDULER_WITH_DOMAIN_HPP
