#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_CONTINUES_ON_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"

namespace Kokkos::Experimental::details::graph {

//! Receiver for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr, stdexec::receiver InnerRcvr>
struct ContinuesOnOpState {
    struct ContinuesOnReceiver {
        using receiver_concept = stdexec::receiver_t;

        ContinuesOnOpState* opstate;

        void set_value() && noexcept {
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

    using inner_opstate_t = stdexec::connect_result_t<Sndr, ContinuesOnReceiver>;

    Schd schd;
    InnerRcvr inner_rcvr;
    inner_opstate_t inner_opstate;

    ContinuesOnOpState(Schd&& schd_, Sndr&& sndr, InnerRcvr&& rcvr_)
        : schd(std::move(schd_))
        , inner_rcvr(std::move(rcvr_))
        , inner_opstate(stdexec::connect(std::move(sndr), ContinuesOnReceiver{this})) {
    }

    void start() & noexcept {
        PLOG_INFO << "continues_on start";
        ::stdexec::start(inner_opstate);
    }
};

//! Sender for @c continues_on.
template <::stdexec::scheduler Schd, ::stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = ::stdexec::sender_t;

    //! The transition may throw or forwards the error channel.
    using with_error_invoke_t =
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(ContinuesOnSender)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ContinuesOnOpState<Schd, Sndr, Rcvr>(std::move(schd), std::move(sndr), std::forward<Rcvr>(rcvr));
    }

    decltype(auto) get_env() const noexcept {
        return SchedulerEnv{schd.ctx_ptr};
    }

    Schd schd;
    Sndr sndr;
};

template <typename Env>
struct transform_sender_for<stdexec::continues_on_t, Env> {
    template <stdexec::scheduler Schd, ::stdexec::sender Sndr>
    requires stdexec::__is_instance_of<Schd, Scheduler>
    auto operator()(stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) && noexcept {
        return ContinuesOnSender{.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_CONTINUES_ON_HPP
