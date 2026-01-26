#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_SCHEDULE_FROM_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_SCHEDULE_FROM_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"

namespace Kokkos::Experimental::details::graph {


//! Operation state for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::sender Sndr, stdexec::receiver InnerRcvr>
struct ScheduleFromOpState {
    using operation_state_concept = stdexec::operation_state_t;
    //! Receiver for @c schedule_from.
    struct ScheduleFromReceiver {
        using receiver_concept = stdexec::receiver_t;

        ScheduleFromOpState* opstate;

        void set_value() && noexcept {
            if (!opstate->skip)
                opstate->schd.ctx_ptr->m_exec.fence(
                    std::format(
                        "{}: schedule_from", Kokkos::Impl::TypeInfo<decltype(opstate->schd.ctx_ptr->m_exec)>::name()));
            ::stdexec::set_value(std::move(opstate->inner_rcvr));
        }

        template <class Error>
        void set_error(Error&& err) && noexcept {
            ::stdexec::set_error(std::move(opstate->inner_rcvr), std::forward<Error>(err));
        }

        void set_stopped() && noexcept {
            ::stdexec::set_stopped(std::move(opstate->inner_rcvr));
        }

        auto get_env() const noexcept -> SchedulerEnv<typename Schd::execution_space> {
            return SchedulerEnv{opstate->schd.ctx_ptr};
        }
    };

    // using rcvr_t = ScheduleFromReceiver<Schd, Sndr, InnerRcvr>;
    using inner_opstate_t = stdexec::connect_result_t<Sndr, ScheduleFromReceiver>;

    Schd schd;
    InnerRcvr inner_rcvr;
    bool skip;
    inner_opstate_t inner_opstate;

    ScheduleFromOpState(Schd&& schd_, Sndr&& sndr, bool skip_, InnerRcvr&& rcvr_)
        : schd(std::move(schd_))
        , inner_rcvr(std::move(rcvr_))
        , skip(skip_)
        , inner_opstate(stdexec::connect(std::move(sndr), ScheduleFromReceiver{this})) {
    }

    void start() & noexcept {
        PLOG_INFO << "schedule_from start";
        ::stdexec::start(inner_opstate);
    }
};

//! Sender for @c schedule_from.
template <stdexec::scheduler Schd, ::stdexec::sender Sndr>
struct ScheduleFromSender {
    using sender_concept = ::stdexec::sender_t;

    //! The transition may throw or forwards the error channel.
    using with_error_invoke_t =
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env_>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env_...>,
        with_error_invoke_t
    >;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(ScheduleFromSender)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ScheduleFromOpState<Schd, Sndr, std::remove_cvref_t<Rcvr>>(
            std::move(schd), std::move(sndr), skip, std::forward<Rcvr>(rcvr));
    }

    auto get_env() const noexcept -> ::stdexec::env_of_t<Sndr> {
        return ::stdexec::get_env(sndr);
    }

    Schd schd;
    Sndr sndr;
    bool skip;
};

template <typename Env>
struct transform_sender_for<stdexec::schedule_from_t, Env> {
    template <::stdexec::sender Sndr>
    auto operator()(stdexec::schedule_from_t, ::stdexec::__ignore, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        static_assert(stdexec::__is_instance_of<decltype(schd), Scheduler>);

        const bool skip = [&]() {
            if constexpr (stdexec::__is_instance_of<std::remove_cvref_t<Env>, SchedulerEnv>) {
                if constexpr (std::same_as<
                                  typename std::remove_cvref_t<decltype(env_.state_ptr)>,
                                  typename std::remove_cvref_t<decltype(schd.state_ptr)>
                              >) {
                    return schd.state_ptr == env_.state_ptr;
                }
            }
            return false;
        }();
        return ScheduleFromSender{.schd = std::move(schd), .sndr = std::forward<Sndr>(sndr), .skip = skip};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_SCHEDULE_FROM_HPP
