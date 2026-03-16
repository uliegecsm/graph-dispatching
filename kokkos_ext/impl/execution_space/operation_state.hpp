#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_OPERATION_STATE_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_OPERATION_STATE_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos_ext/impl/execution_space/domain.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"

namespace Kokkos::Experimental::details::execution_space {

template <typename Clsr>
concept Closure = requires(const Clsr& clsr) {
    typename Clsr::execution_space;

    { clsr.execute() } -> std::same_as<void>;

    { clsr.get_policy() };
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<decltype(clsr.get_policy())>>;

    requires std::same_as<std::remove_cvref_t<decltype(clsr.get_policy().space())>, typename Clsr::execution_space>;
};

template <stdexec::receiver Rcvr, Closure... Clsrs>
requires(sizeof...(Clsrs) > 0)
struct OpStateBase : public stdexec::__immovable {
    using execution_space = typename stdexec::__tuple_element_t<0, stdexec::__tuple<Clsrs...>>::execution_space;

    using receiver_t = Rcvr;
    using closures_t = stdexec::__tuple<Clsrs...>;

    receiver_t rcvr;
    closures_t clsrs;

    constexpr explicit OpStateBase(Rcvr rcvr_, Clsrs... clsrs_)
        noexcept(std::is_nothrow_move_constructible_v<Rcvr> && (std::is_nothrow_move_constructible_v<Clsrs> && ...))
        : rcvr(std::move(rcvr_))
        , clsrs(std::move(clsrs_)...) {
    }

    void propagate_completion_signal(stdexec::set_value_t) noexcept {
        try {
            stdexec::__apply([](auto&... clsr) { (clsr.execute(), ...); }, clsrs);
        } catch (...) {
            this->propagate_completion_signal(stdexec::set_error, std::current_exception());
            return;
        }

        /**
         * Sync behavior is a design decision in progress.
         *
         * We would like ``structured concurrency`` whereby ``child operations`` complete before their ``parents``.
         *
         * The question is what we consider a ``child operation`` and where we want to place the boundary
         * of the work enqueued on an execution space.
         *
         * Currently, we consider that the ``child operation`` is the whole chain of work enqueued on the execution space.
         * We fold this whole chain of work into a single operation state holding a tuple of closures. The operation state's
         * downstream receiver is the synchronization boundary where we fence after enqueueing the work. It is either a
         * @c stdexec::schedule_from receiver (fences if needed before transferring control to the subsequent
         * @c stdexec::continues_on) or a @c stdexec::sync_wait receiver (fences before returning to the caller). There is
         * thus a fence as we exit the composed operation. This approach is a ``compositional`` approach in which the functionalities
         * of the operation state and those of the downstream receiver together ensure the structured concurrency.
         *
         * @note The default implementation of @c stdexec::when_all, does not terminate branches by a @c stdexec::schedule_from,
         * thus implying that there may not be a fence. It is to handle such a scenario that we fence eagerly
         * when the downstream receiver environment is not queryable for @ref Kokkos::Experimental::details::execution_space::get_exec_t.
         *
         * @todo Think of whether we should adopt a more ``local`` approach by moving the synchronization boundary
         * from the downstream receiver into @c propagate_completion_signal(set_value) itself. This approach would
         * entail fencing after the closures are enqueued, before calling @c set_value on the downstream receiver,
         * unconditionally. This approach would rely only on functionalities of the operation state. It would relieve the downstream
         * receiver from the responsibility to fence, and customization of @c stdexec::schedule_from and @c stdexec::sync_wait
         * for the execution space domain may become unnecessary. However, this approach may introduce unnecessary fences.
         *
         * @todo Explore event-based synchronization for cases in which the successor is still on the device,
         * but on a different execution space. The objective would be to avoid occupying the current host thread.
         */
        if constexpr (!stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>) {
            this->query(get_exec)
                .get()
                .fence(std::format("{}: continuation", Kokkos::Impl::TypeInfo<execution_space>::name()));
        }
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void propagate_completion_signal(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(error));
    }

    void propagate_completion_signal(stdexec::set_stopped_t) noexcept {
        stdexec::set_stopped(std::move(rcvr));
    }

    [[nodiscard]]
    constexpr auto query(get_exec_t) const noexcept -> ExecutionSpaceRef<execution_space> {
        return ExecutionSpaceRef<execution_space>{stdexec::__get<0>(clsrs).get_policy().space()};
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Rcvr, rcvr)
};

template <typename ParentOp>
struct OpStateReceiver {
    using receiver_concept = stdexec::receiver_t;

    ParentOp* parent_op;

    void set_value() && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_error, std::forward<Error>(error));
    }

    void set_stopped() && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_stopped);
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_UPSERT_EXEC(
        typename ParentOp::execution_space,
        parent_op->query(get_exec).get(),
        typename ParentOp::receiver_t,
        parent_op->rcvr)
};

template <typename...>
struct OpState;

template <stdexec::sender Sndr, stdexec::receiver Rcvr, Closure... Clsrs>
requires(!requires { typename stdexec::transform_sender_result_t<Sndr, stdexec::env_of_t<Rcvr>>::closure_t; })
struct OpState<Sndr, Rcvr, Clsrs...> : public OpStateBase<Rcvr, Clsrs...> {
    using operation_state_concept = stdexec::operation_state_t;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, OpStateReceiver<OpStateBase<Rcvr, Clsrs...>>>;

    static constexpr bool opstate_base_is_nothrow_constructible =
        std::is_nothrow_constructible_v<OpStateBase<Rcvr, Clsrs...>, Rcvr&&, Clsrs&&...>;

    static constexpr bool inner_opstate_is_nothrow_constructible =
        stdexec::__nothrow_connectable<Sndr&&, OpStateReceiver<OpStateBase<Rcvr, Clsrs...>>>;

    inner_opstate_t inner_opstate;

    constexpr explicit OpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr_,
        Clsrs... clsrs_) noexcept(opstate_base_is_nothrow_constructible && inner_opstate_is_nothrow_constructible)
        : OpStateBase<Rcvr, Clsrs...>(std::move(rcvr_), std::move(clsrs_)...)
        , inner_opstate(
              stdexec::connect(std::forward<Sndr>(sndr), OpStateReceiver<OpStateBase<Rcvr, Clsrs...>>{this})) {
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

/**
 * Specialisation of @ref OpState that recursively folds a sequence of senders enqueueing work on the same execution space
 * into a single operation state.
 */
template <stdexec::sender Sndr, stdexec::receiver Rcvr, typename... Clsrs>
requires requires { typename stdexec::transform_sender_result_t<Sndr, stdexec::env_of_t<Rcvr>>::closure_t; }
struct OpState<Sndr, Rcvr, Clsrs...>
    : OpState<
          stdexec::__child_of<Sndr>,
          Rcvr,
          typename stdexec::transform_sender_result_t<Sndr, stdexec::env_of_t<Rcvr>>::closure_t,
          Clsrs...
      > {
    using child_of_sndr_t = stdexec::__child_of<Sndr>;
    using clsr_of_sndr_t = typename stdexec::transform_sender_result_t<Sndr, stdexec::env_of_t<Rcvr>>::closure_t;

    using opstate_base_t = OpState<child_of_sndr_t, Rcvr, clsr_of_sndr_t, Clsrs...>;

    static constexpr bool sndr_has_nothrow_transform_sender = stdexec::__detail::__has_nothrow_transform_sender<
        Kokkos::Experimental::details::execution_space::Domain,
        stdexec::set_value_t,
        Sndr&&,
        stdexec::env_of_t<Rcvr>
    >;

    static constexpr bool opstate_base_is_nothrow_constructible =
        std::is_nothrow_constructible_v<opstate_base_t, child_of_sndr_t&&, Rcvr&&, clsr_of_sndr_t&&, Clsrs&&...>;

    OpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr_,  // NOLINT(performance-unnecessary-value-param)
        Clsrs... clsrs) noexcept(sndr_has_nothrow_transform_sender && opstate_base_is_nothrow_constructible)
        : OpState(
              Kokkos::Experimental::details::execution_space::Domain{}
                  .transform_sender(stdexec::set_value, std::forward<Sndr>(sndr), stdexec::get_env(rcvr_)),
              rcvr_,
              std::move(clsrs)...) {
    }

   private:
    template <stdexec::sender TrnsfrmdSndr>
    OpState(
        TrnsfrmdSndr&& trnsfrmd_sndr, // NOLINT(cppcoreguidelines-missing-std-forward)
        Rcvr rcvr_,
        Clsrs... clsrs) noexcept(opstate_base_is_nothrow_constructible)
        : opstate_base_t(
              stdexec::__forward_like<TrnsfrmdSndr>(trnsfrmd_sndr.sndr),
              std::move(rcvr_),
              stdexec::__forward_like<TrnsfrmdSndr>(trnsfrmd_sndr.clsr),
              std::move(clsrs)...) {
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_OPERATION_STATE_HPP
