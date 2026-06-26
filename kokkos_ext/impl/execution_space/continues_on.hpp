#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/env.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"

namespace Kokkos::Experimental::details::execution_space {

//! Receiver for @c continues_on.
template <stdexec::receiver Rcvr>
struct ContinuesOnReceiver {
    using receiver_concept = stdexec::receiver_t;

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

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Rcvr, rcvr)
};

//! Sender for @c continues_on.
template <stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_t;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <typename Rcvr>
    using rcvr_t = ContinuesOnReceiver<Rcvr>;

    template <stdexec::receiver Rcvr>
    auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<rcvr_t<Rcvr>, Rcvr&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t<Rcvr>>)
        -> stdexec::connect_result_t<Sndr, rcvr_t<Rcvr>> {
        return stdexec::connect(std::forward<Sndr>(sndr), rcvr_t<Rcvr>{.rcvr = std::move(rcvr)});
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Sndr, sndr)

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct transform_sender_for<stdexec::continues_on_t> {
    template <typename Env, stdexec::__is_instance_of<Scheduler> Schd, stdexec::sender Sndr>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&&, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Sndr>, Sndr&&>) {
        return ContinuesOnSender<Sndr>{.sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
