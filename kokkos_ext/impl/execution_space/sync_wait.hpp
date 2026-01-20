#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"
#include "kokkos_ext/impl/sync_wait.hpp"

namespace Kokkos::Experimental::details::execution_space {

//! Receiver for @c sync_wait.
template <Kokkos::ExecutionSpace Exec>
struct SyncWaitReceiver {
    using receiver_concept = stdexec::receiver_t;

    State<Exec>* state;
    Kokkos::Experimental::details::impl::State* runloop_state;

    void set_value() && noexcept {
        state->exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<Exec>::name()));
        runloop_state->loop.finish();
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        runloop_state->error = std::forward<Error>(err);
        state->exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<Exec>::name()));
        runloop_state->loop.finish();
    }

    void set_stopped() noexcept {
        state->exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<Exec>::name()));
        runloop_state->loop.finish();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> Kokkos::Experimental::details::impl::env {
        return {runloop_state->loop.get_scheduler()};
    }
};

struct SyncWait {
    /**
     * According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait,
     * it has to return an engaged optional (on the value channel).
     *
     * @todo Make the @c noexcept specifier depend on the completion signatures of @p sndr.
     */
    template <stdexec::sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept(false) -> std::optional<std::tuple<>> {
        Kokkos::Experimental::details::impl::State runloop_state;

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));

        auto op_state = stdexec::connect(
            std::forward<Sndr>(sndr),
            SyncWaitReceiver{.state = std::move(schd.state), .runloop_state = &runloop_state});

        stdexec::start(op_state);

        runloop_state.loop.run();

        if (runloop_state.error)
            std::rethrow_exception(std::move(runloop_state.error));

        return std::tuple{};
    }
};

/**
 * @brief Customize @c sync_wait.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/e8a6a7b25fbc2463e1dfe0ee20973b1fe622bfcf/include/nvexec/stream_context.cuh#L247-L251
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-dispatch
 *
 * @todo Make the @c noexcept specifier depend on the completion signatures of @p sndr.
 */
template <>
struct apply_sender_for<stdexec::sync_wait_t> {
    template <execution_space_completing_sender Sndr>
    auto operator()(Sndr&& sndr) && noexcept(false) {
        return SyncWait{}(std::forward<Sndr>(sndr));
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
