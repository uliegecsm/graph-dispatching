#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_SCHEDULE_FROM_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_SCHEDULE_FROM_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/env.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"

namespace Kokkos::Experimental::details::graph {

//! Receiver for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ScheduleFromReceiver {
    using receiver_concept = stdexec::receiver_t;

    Schd schd;
    Rcvr rcvr;
    bool skip;

    void set_value() && noexcept {
        if (!skip)
            schd.state_ptr->wait(
                std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<decltype(schd.state_ptr->exec)>::name()));
        ::stdexec::set_value(std::move(rcvr));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        ::stdexec::set_stopped(std::move(rcvr));
    }

    //! Make others aware of which execution space instance it may synchronize.
    GRAPH_DISPATCHING_KOKKOS_EXT_UPSERT_EXEC(typename Schd::execution_space, schd.state_ptr->exec, Rcvr, rcvr)
};

//! Sender for @c schedule_from.
template <stdexec::scheduler Schd, ::stdexec::sender Sndr>
struct ScheduleFromSender {
    using sender_concept = ::stdexec::sender_t;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        using recv_t = ScheduleFromReceiver<Schd, Rcvr>;

        return ::stdexec::connect(
            std::move(sndr), recv_t{.schd = std::move(schd), .rcvr = std::move(rcvr), .skip = skip});
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Sndr, sndr)

    Schd schd;
    Sndr sndr;
    bool skip;
};

template <>
struct transform_sender_for<stdexec::schedule_from_t> {
    template <typename Env, graph_completing_sender<Env> Sndr>
    auto operator()(const Env& env, stdexec::schedule_from_t, ::stdexec::__ignore, Sndr&& sndr) const noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        const bool skip = [&]() {
            if constexpr (stdexec::__queryable_with<Env, Kokkos::Experimental::details::execution_space::get_exec_t>) {
                if constexpr (std::same_as<
                                  std::remove_cvref_t<decltype(Kokkos::Experimental::details::execution_space::get_exec(
                                                                   env)
                                                                   .get())>,
                                  std::remove_cvref_t<decltype(schd.state_ptr->exec)>
                              >) {
                    return schd.state_ptr->exec == Kokkos::Experimental::details::execution_space::get_exec(env).get();
                }
            }
            return false;
        }();
        return ScheduleFromSender{.schd = std::move(schd), .sndr = std::forward<Sndr>(sndr), .skip = skip};
    }
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_SCHEDULE_FROM_HPP
