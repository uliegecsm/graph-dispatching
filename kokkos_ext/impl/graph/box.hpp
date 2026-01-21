#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_BOX_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_BOX_HPP

#include "stdexec/execution.hpp"

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
#    include "plog/Log.h"
#endif

#include "kokkos_ext/impl/completion_signatures.hpp"

namespace Kokkos::Experimental::details::graph {

/**
 * @brief Wrapper for a sender that is cheap to copy.
 *
 * It "buries" the graph (topology and data) sender in a "box", much like @c stdexec::__shared::__box.
 * It thus makes a heap allocation.
 *
 * Wrapping allows a stable sender "shape" while reducing the cost of copying the graph sender (which now is just
 * costing the copy of a @c std::shared_ptr).
 *
 * Formally, it does the following:
 *
 *  1. A sender @c graph_sndr describes the graph topology and data.
 *     It is passed to a wrapper sender @c box_sndr that:
 *      a. holds @c graph_sndr inside a shared heap object (a "box")
 *      b. is itself cheap to copy
 *  2. Algorithms such as @c exec::repeat_effect_until operate on @c box_sndr, and each triggered
 *     reconnection reconnects to @c box_sndr.
 *  3. Only the firstly created operation state will:
 *      a. connects to @c graph_sndr
 *      b. lowers it to a @c Kokkos graph
 *      c. instantiate, submit the graph
 *  4. Subsequent operation states will:
 *      a. reuse the instantiated graph and just submit it
 *      b. do not reconnect @c graph_sndr
 *
 * @warning It still does not work well when there are sender upstream of the burying box, because on subsequent runs, they won't be connected
 *          either, so they are not executed.
 */
template <stdexec::sender Sndr, stdexec::__sender_adaptor_closure Closure>
struct BurySender {
    using sender_concept = stdexec::sender_t;

    struct State {
        Sndr sndr;
        Closure closure;
        std::atomic<bool> consumed{false};

        explicit State(Sndr&& sndr_, Closure&& closure_)
            : sndr(std::move(sndr_))
            , closure(std::move(closure_)) {
        }
    };

    std::shared_ptr<State> m_state;

    explicit BurySender(Sndr&& sndr, Closure&& closure)
        : m_state(std::make_shared<State>(std::move(sndr), std::move(closure))) {
    }

    BurySender(const BurySender&) = default;
    BurySender(BurySender&&) = default;
    BurySender& operator=(const BurySender&) = default;
    BurySender& operator=(BurySender&&) = default;
    ~BurySender() = default;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(BurySender)

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect_sndr(Rcvr&& rcvr) const noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return stdexec::connect(m_state->sndr, std::forward<Rcvr>(rcvr));
    }

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr&& rcvr) const noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return stdexec::connect(m_state->sndr | m_state->closure, std::forward<Rcvr>(rcvr));
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Sndr>> {
        return stdexec::__fwd_env(stdexec::get_env(m_state->sndr));
    }
};

//! This one is recreated each time. But it's veeery cheap.
template <stdexec::scheduler Schd, stdexec::__is_instance_of<BurySender> Sndr>
struct BoxSender {
    using sender_concept = stdexec::sender_t;

    Schd schd;
    Sndr sndr;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(BoxSender)

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return BoxOpState<std::remove_cvref_t<Rcvr>>{std::move(schd), std::move(sndr), std::forward<Rcvr>(rcvr)};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Sndr>> {
        return stdexec::__fwd_env(stdexec::get_env(sndr));
    }

    template <stdexec::receiver Rcvr>
    struct BoxOpState {
        Schd schd;
        Sndr sndr;
        Rcvr rcvr;

        //! @todo This is not really thread-safe, and being thread-safe is probably not required.
        void start() & noexcept {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
            PLOG_DEBUG << "schd.state_ptr: " << schd.state_ptr;
            PLOG_DEBUG << "sndr.m_state: " << sndr.m_state.get();
#endif
            if (sndr.m_state->consumed.load(std::memory_order_acquire)) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
                PLOG_DEBUG << "Directly submitting the graph";
#endif
                auto build = sndr.connect_sndr(BoxOpReceiver<true>{this});
                stdexec::start(build);
            } else {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
                PLOG_DEBUG << "Connecting the graph sender.";
#endif
                auto build = stdexec::connect(sndr, BoxOpReceiver<false>{this});
                stdexec::start(build);
                sndr.m_state->consumed.store(true, std::memory_order_release);
            }

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
            PLOG_DEBUG << "Calling set_value of next receiver.";
#endif
            stdexec::set_value(std::move(rcvr));
        }

        template <bool submit>
        struct BoxOpReceiver {
            using receiver_concept = stdexec::receiver_t;

            BoxOpState* state;

            void set_value() && noexcept {
                if constexpr (submit)
                    state->schd.state_ptr->submit();
            }

            template <class Error>
            void set_error(Error&&) && noexcept {
            }

            void set_stopped() && noexcept {
            }

            auto get_env() const noexcept -> stdexec::env<> {
                return {};
            }
        };
    };
};

struct Box {
    template <stdexec::sender Sndr, stdexec::__sender_adaptor_closure Closure>
    auto operator()(Sndr&& sndr, Closure&& closure) const noexcept {
        return stdexec::__make_sexpr<Box>(
            {},
            BurySender<std::remove_cvref_t<Sndr>, Closure>(std::forward<Sndr>(sndr), std::forward<Closure>(closure)));
    }

    template <stdexec::__sender_adaptor_closure Closure>
    auto operator()(Closure&& closure) const noexcept {
        return stdexec::__closure{*this, std::forward<Closure>(closure)};
    }
};

template <typename Env>
struct transform_sender_for<Box, Env> {
    template <::stdexec::sender Sndr>
    auto operator()(Box, ::stdexec::__ignore, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        static_assert(stdexec::__is_instance_of<decltype(schd), Scheduler>);
        using Schd = std::remove_cvref_t<decltype(schd)>;

        return BoxSender<Schd, Sndr>(std::move(schd), std::forward<Sndr>(sndr));
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

namespace Kokkos::Experimental {

inline constexpr Kokkos::Experimental::details::graph::Box box{};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_BOX_HPP
