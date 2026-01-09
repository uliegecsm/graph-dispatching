#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

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
template <stdexec::scheduler Schd> requires stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>
struct SyncWaitReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Schd schd;
    std::shared_ptr<std::exception_ptr> error;
    std::shared_ptr<stdexec::run_loop> loop;

    void set_value() && noexcept
    {
        schd.env.exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));
        loop->finish();
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        *error = std::forward<Error>(err);
        schd.env.exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));
        loop->finish();
    }

    void set_stopped() noexcept {
        schd.env.exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));
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
    template <stdexec::scheduler Schd, stdexec::sender Sndr>
    auto operator()(Schd&& schd, Sndr&& sndr) const noexcept(false) -> std::optional<std::tuple<>>
    {
        auto error = std::make_shared<std::exception_ptr>();

        auto loop = std::make_shared<stdexec::run_loop>();

        auto op_state = stdexec::connect(
            std::forward<Sndr>(sndr),
            SyncWaitReceiver{
                .schd = std::forward<Schd>(schd),
                .error = error,
                .loop = loop,
            }
        );

        stdexec::start(op_state);

        loop->run();

        if (*error) std::rethrow_exception(std::move(*error));

        return std::tuple{};
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
