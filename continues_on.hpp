#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/env.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"

namespace Kokkos::Experimental::details::execution_space {

//! Receiver for @c continues_on.
template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnReceiver {
    using receiver_concept = stdexec::receiver_t;

    Schd schd;
    Rcvr rcvr;

    void set_value() && noexcept {
        stdexec::set_value(std::move(rcvr));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(rcvr));
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_EXEC(typename Schd::execution_space, schd.state->exec, Rcvr, rcvr)
};

//! Sender for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_t;

    Schd schd;
    Sndr sndr;

    //! The transition may throw or forwards the error channel.
    using with_error_invoke_t =
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env>
    using _completion_signatures = stdexec::transform_completion_signatures<
        stdexec::completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(ContinuesOnSender)

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        using recv_t = ContinuesOnReceiver<Schd, std::remove_cvref_t<Rcvr>>;

        return stdexec::connect(std::move(sndr), recv_t{.schd = std::move(schd), .rcvr = std::forward<Rcvr>(rcvr)});
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Sndr, sndr)
};

template <typename Env>
struct transform_sender_for<stdexec::continues_on_t, Env> {
    template <stdexec::__is_instance_of<Scheduler> Schd, stdexec::sender Sndr>
    auto operator()(stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) && noexcept {
        return ContinuesOnSender{.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
