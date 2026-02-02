#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_THEN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_THEN_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/env.hpp"
#include "kokkos_ext/impl/graph/Helpers.hpp"

namespace Kokkos::Experimental::details::graph {

//! Build a @c then node after the node returned by @ref get_predecessor.
template <Kokkos::ExecutionSpace Exec, typename OpstateType, typename Functor>
auto build_then_node(State<Exec>& state, const OpstateType& opstate, Functor&& functor) {
    return get_predecessor(opstate, state.get_graph())
        .then(
            std::format("{}: then", Kokkos::Impl::TypeInfo<Exec>::name()), state.exec, std::forward<Functor>(functor));
}

template <stdexec::scheduler Schd, stdexec::sender Sndr, stdexec::receiver InnerRcvr, typename Functor>
struct ThenOpState {
    using operation_state_concept = stdexec::operation_state_t;

    using env_t = stdexec::__fwd_env_t<stdexec::env_of_t<InnerRcvr>>;

    /**
     * @brief Receiver for @c then.
     *
     * @note It must be nothrow moveable, see @cite P3383R3.
     */
    struct ThenReceiver {
        using receiver_concept = stdexec::receiver_t;

        ThenOpState* opstate;

        void set_value() && noexcept {
            opstate->schd.state_ptr->submit();
            std::move(*opstate).propagate_completion_signal(stdexec::set_value);
        }

        template <typename Error>
        void set_error(Error&& err) && noexcept {
            std::move(*opstate).propagate_completion_signal(::stdexec::set_error, std::forward<Error>(err));
        }

        void set_stopped() && noexcept {
            std::move(*opstate).propagate_completion_signal(::stdexec::set_stopped);
        }

        auto get_env() const noexcept -> env_t {
            return opstate->get_env();
        }
    };

    using inner_opstate_t = stdexec::connect_result_t<Sndr, ThenReceiver>;

    //! In the long term, the user could be able to opt for type erased nodes.
    using node_t = decltype(build_then_node(
        *std::declval<Schd&>().state_ptr,
        std::declval<const inner_opstate_t&>(),
        std::declval<Functor>()));

    Schd schd;
    InnerRcvr inner_rcvr;
    inner_opstate_t inner_opstate;
    Functor functor;
    std::optional<node_t> node = std::nullopt;
    std::exception_ptr error = nullptr;

    template <stdexec::scheduler Scheduler, stdexec::sender Sender, stdexec::receiver Rcvr, typename Func>
    ThenOpState(Scheduler&& scheduler, Sender&& sndr, Rcvr&& rcvr, Func&& func)
        : schd(std::forward<Scheduler>(scheduler))
        , inner_rcvr(std::forward<Rcvr>(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sender>(sndr), ThenReceiver{this}))
        , functor(std::forward<Func>(func)) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << "Adding a then node to graph in state " << schd.state_ptr << " of type "
                   << Kokkos::Impl::TypeInfo<node_t>::name();
#endif
        this->create_node();
    }

    //! Create the node only if the predecessor has one.
    void create_node() {
        if (proceed(*schd.state_ptr, inner_opstate)) {
            try {
                this->node.emplace(build_then_node(*schd.state_ptr, inner_opstate, std::move(functor)));
            } catch (...) {
                this->error = std::current_exception();
            }
        }
    }

    decltype(auto) get_node() const {
        return node;
    }

    void start() & noexcept {
        if (error)
            stdexec::set_error(std::move(inner_rcvr), error);
        stdexec::start(inner_opstate);
    }

    template <typename Tag, typename... Args>
    void propagate_completion_signal(Tag, Args&&... args) && noexcept {
        Tag()(std::move(inner_rcvr), std::forward<Args>(args)...);
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(InnerRcvr, inner_rcvr)
};

//! Sender for @c then.
template <stdexec::sender Sndr, typename Functor, typename Schd>
struct ThenSender {
    using sender_concept = stdexec::sender_t;

    //! @c Kokkos may throw while launching the kernel.
    using with_error_invoke_t =
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(ThenSender)

    //! See also https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/then.cuh#L52.
    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return ThenOpState<Schd, Sndr, std::remove_cvref_t<Rcvr>, Functor>(
            std::move(schd), std::move(sndr), std::forward<Rcvr>(rcvr), std::move(functor));
    }

    Sndr sndr;
    Functor functor;
    Schd schd;

    auto get_env() const noexcept -> stdexec::env_of_t<Sndr> {
        return stdexec::get_env(sndr);
    }
};

template <typename Env>
struct transform_sender_for<stdexec::then_t, Env> {
    template <typename Functor, typename Sndr>
    requires graph_completing_sender<Sndr, Env>
    auto operator()(stdexec::then_t, Functor&& functor, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        return ThenSender{
            .sndr = std::forward<Sndr>(sndr), .functor = std::forward<Functor>(functor), .schd = std::move(schd)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_THEN_HPP
