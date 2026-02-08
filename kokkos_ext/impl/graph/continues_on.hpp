#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_CONTINUES_ON_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"
#include "kokkos_ext/impl/graph/Helpers.hpp"

namespace Kokkos::Experimental::details::graph {

//! Operation state for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr, stdexec::receiver InnerRcvr>
struct ContinuesOnOpState {
    using operation_state_concept = stdexec::operation_state_t;

    using env_t = GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_EXEC_TYPE(typename Schd::execution_space, InnerRcvr);

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

        auto get_env() const noexcept -> env_t {
            return opstate->get_env();
        }
    };

    using inner_opstate_t = stdexec::connect_result_t<Sndr, ContinuesOnReceiver>;

    Schd schd;
    InnerRcvr inner_rcvr;
    inner_opstate_t inner_opstate;

    ContinuesOnOpState(Schd&& schd_, Sndr&& sndr, InnerRcvr&& inner_rcvr_)
        : schd(std::move(schd_))
        , inner_rcvr(std::move(inner_rcvr_))
        , inner_opstate(stdexec::connect(std::move(sndr), ContinuesOnReceiver{this})) {
    }

    /**
     * In case of a @c when_all that is followed by a @c continues_on, both in our domain, @c stdexec will insert an intermediate
     * operation state related to a @c schedule_from since @c when_all does not have a completion scheduler.
     * Yet, we want to optimize the underlying graph by ensuring that we don't create 2 graphs (since no intermediate work was inserted).
     *
     * According to https://cplusplus.github.io/sender-receiver/execution.html#environments-and-attributes, operation states
     * could feature forwarding queries.
     *
     * But for now, we have to manually "go through" one intermediate operation state layer to introspect more inner operation states.
     *
     * See also @ref tests::kokkos_ext::WhenAllTest_join_topology_Test.
     */
    decltype(auto) query(get_node_t) const noexcept {
        if constexpr (requires {
                          requires stdexec::__is_instance_of<inner_opstate_t, stdexec::__opstate>;
                          requires std::same_as<typename inner_opstate_t::__tag_t, stdexec::schedule_from_t>;
                          requires stdexec::__is_instance_of<typename inner_opstate_t::__child_ops_t, stdexec::__tuple>;
                          requires stdexec::__tuple_size_v<typename inner_opstate_t::__child_ops_t> == 1;
                          requires stdexec::__queryable_with<
                              stdexec::__tuple_element_t<0, typename inner_opstate_t::__child_ops_t>,
                              get_node_t
                          >;
                      }) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
            PLOG_INFO << "The 'get_node' query goes through the 'schedule_from'.";
#endif
            return stdexec::__get<0>(inner_opstate.__child_ops_).query(get_node);
        } else {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
            PLOG_INFO << "The 'get_node' query uses 'get_predecessor' that will return a node of type "
                      << Kokkos::Impl::TypeInfo<
                             get_predecessor_t<inner_opstate_t, env_t, typename Schd::execution_space>
                         >::name()
                      << '.';
#endif
            return get_predecessor(inner_opstate, this->get_env(), schd.state_ptr->get_graph());
        }
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_JOIN_EXEC(typename Schd::execution_space, schd.state_ptr->exec, InnerRcvr, inner_rcvr)
};

//! Sender for @c continues_on.
template <::stdexec::scheduler Schd, ::stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = ::stdexec::sender_t;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ContinuesOnOpState<Schd, Sndr, std::remove_cvref_t<Rcvr>>(
            std::move(schd), std::move(sndr), std::forward<Rcvr>(rcvr));
    }

    decltype(auto) get_env() const noexcept {
        return SchedulerEnv{schd.state_ptr};
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
