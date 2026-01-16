#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_BULK_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_BULK_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/GraphContext_fwd.hpp"
#include "kokkos_ext/impl/bulk.hpp"
#include "kokkos_ext/impl/graph/Helpers.hpp"

namespace Kokkos::Experimental::details::graph {

template <
    stdexec::__is_instance_of<Scheduler> Schd,
    stdexec::sender Sndr,
    stdexec::receiver Rcvr,
    stdexec::__is_instance_of<stdexec::__bulk::__data> Data
>
requires Kokkos::Experimental::details::impl::parallel_policy<Data>
struct BulkReceiver;

//! Build a @c parallel_for node after the node returned by @ref get_predecessor.
template <Kokkos::ExecutionSpace Exec, typename OpstateType, typename Data>
auto build_parallel_for_node(State<Exec>& state, const OpstateType& opstate, Data&& data) {
    auto [policy, shape, functor] = std::forward<Data>(data);
    return get_predecessor(opstate, state.get_graph())
        .then_parallel_for(
            std::format("{}: bulk", Kokkos::Impl::TypeInfo<Exec>::name()),
            Kokkos::RangePolicy(state.exec, 0, std::move(shape)),
            std::move(functor));
}

template <
    stdexec::scheduler Schd,
    stdexec::sender Sndr,
    stdexec::receiver InnerRcvr,
    stdexec::__is_instance_of<stdexec::__bulk::__data> Data
>
requires Kokkos::Experimental::details::impl::parallel_policy<Data>
struct BulkOpState {
    using rcvr_t = BulkReceiver<Schd, Sndr, InnerRcvr, Data>;
    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    //! In the long term, the user could be able to opt for type erased nodes.
    using node_t = decltype(build_parallel_for_node(
        *std::declval<Schd&>().state_ptr,
        std::declval<const inner_opstate_t&>(),
        std::declval<Data&&>()));

    Schd schd;
    InnerRcvr inner_rcvr;
    inner_opstate_t inner_opstate;
    Data data;
    std::optional<node_t> node = std::nullopt;
    std::exception_ptr error = nullptr;

    template <stdexec::scheduler Scheduler, stdexec::sender Sender, stdexec::receiver Rcvr>
    BulkOpState(Scheduler&& scheduler, Sender&& sndr, Rcvr&& rcvr, Data&& data_)
        : schd(std::forward<Scheduler>(scheduler))
        , inner_rcvr(std::forward<Rcvr>(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sender>(sndr), rcvr_t{this}))
        , data(std::move(data_)) {
        this->create_node();
    }

    //! Create the node only if the predecessor has one.
    void create_node() {
        const auto proceed = [&]() {
            if constexpr (requires { inner_opstate.error; })
                return inner_opstate.error == nullptr;
            return true;
        }();
        if (proceed) {
            try {
                this->node.emplace(build_parallel_for_node(*schd.state_ptr, inner_opstate, std::move(data)));
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

    auto get_env() const noexcept -> stdexec::env_of_t<InnerRcvr> {
        return ::stdexec::get_env(inner_rcvr);
    }
};

/**
 * @brief Receiver for @c bulk.
 *
 * @note It must be nothrow moveable, see @cite P3383R3.
 */
template <
    stdexec::__is_instance_of<Scheduler> Schd,
    stdexec::sender Sndr,
    stdexec::receiver Rcvr,
    stdexec::__is_instance_of<stdexec::__bulk::__data> Data
>
requires Kokkos::Experimental::details::impl::parallel_policy<Data>
struct BulkReceiver {
    using receiver_concept = stdexec::receiver_t;

    using opstate_t = BulkOpState<Schd, Sndr, Rcvr, Data>;

    opstate_t* opstate;

    void set_value() && noexcept {
        std::move(*opstate).propagate_completion_signal(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        std::move(*opstate).propagate_completion_signal(::stdexec::set_error, std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        std::move(*opstate).propagate_completion_signal(::stdexec::set_stopped);
    }

    auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return opstate->get_env();
    }
};

//! Sender for @c bulk.
template <stdexec::sender Sndr, stdexec::__is_instance_of<stdexec::__bulk::__data> Data, typename Schd>
struct BulkSender {
    using sender_concept = stdexec::sender_t;

    //! @c Kokkos may throw while launching the kernel.
    using with_error_invoke_t =
        ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env>
    using _completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <typename... Env>
    [[nodiscard]]
    constexpr auto get_completion_signatures(Env&&...) -> _completion_signatures<BulkSender, Env...> {
        return {};
    }

    //! See also https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/then.cuh#L52.
    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        return BulkOpState<Schd, Sndr, std::remove_cvref_t<Rcvr>, Data>(
            std::move(schd), std::move(sndr), std::forward<Rcvr>(rcvr), std::move(data));
    }

    Sndr sndr;
    Data data;
    Schd schd;

    auto get_env() const noexcept -> stdexec::env_of_t<Sndr> {
        return stdexec::get_env(sndr);
    }
};

template <typename Env>
struct transform_sender_for<stdexec::bulk_t, Env> {
    template <typename Data, typename Sndr>
    requires graph_completing_sender<Sndr, Env>
    auto operator()(stdexec::bulk_t, Data&& data, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        return BulkSender{.sndr = std::forward<Sndr>(sndr), .data = std::forward<Data>(data), .schd = std::move(schd)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::graph

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_GRAPH_BULK_HPP
