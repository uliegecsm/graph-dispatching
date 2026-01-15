#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"

namespace Kokkos::Experimental::details::execution_space
{
namespace impl
{
struct env
{
    ::stdexec::run_loop::scheduler schd;

    [[nodiscard]]
    auto query(::stdexec::get_scheduler_t) const noexcept -> ::stdexec::run_loop::scheduler {
        return schd;
    }

    [[nodiscard]]
    auto query(::stdexec::get_delegation_scheduler_t) const noexcept -> ::stdexec::run_loop::scheduler {
        return schd;
    }
};
} // namespace impl

//! Receiver for @c sync_wait.
template <Kokkos::ExecutionSpace Exec>
struct SyncWaitReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Exec exec;
    std::shared_ptr<std::exception_ptr> error;
    std::shared_ptr<stdexec::run_loop> loop;

    void set_value() && noexcept
    {
        exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(exec)>::name()));
        loop->finish();
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        *error = std::forward<Error>(err);
        exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(exec)>::name()));
        loop->finish();
    }

    void set_stopped() noexcept {
        exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(exec)>::name()));
        loop->finish();
    }

    [[nodiscard]]
    auto get_env() const noexcept -> impl::env {
        return {loop->get_scheduler()};
    }
};

struct SyncWait
{
    /**
     * According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait,
     * it has to return an engaged optional (on the value channel).
     *
     * @todo Make the @c noexcept specifier depend on the completion signatures of @p sndr.
     */
    template <stdexec::sender Sndr, typename Env>
    auto operator()(Sndr&& sndr, [[maybe_unused]] Env&& env) const noexcept(false) -> std::optional<std::tuple<>>
    {
        auto error = std::make_shared<std::exception_ptr>();

        auto loop = std::make_shared<stdexec::run_loop>();

        auto exec = get_exec(stdexec::get_env(sndr));

        auto op_state = stdexec::connect(
            std::forward<Sndr>(sndr),
            SyncWaitReceiver{
                .exec = exec,
                .error = error,
                .loop = loop,
            }
        );

        stdexec::start(op_state);

        if (*error) std::rethrow_exception(std::move(*error));

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
struct apply_sender_for<stdexec::sync_wait_t>
{
    template <execution_space_completing_sender Sndr>
    auto operator()(Sndr&& sndr) && noexcept(false)
    {
        return SyncWait{}(std::forward<Sndr>(sndr), stdexec::env{});
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
