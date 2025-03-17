#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP

#include <concepts>

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::execution_space
{
//! Receiver for @c sync_wait. @todo Better constrain the scheduler type.
template <stdexec::scheduler Schd>
struct SyncWaitReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Schd schd;

    std::shared_ptr<std::exception_ptr> error;

    void set_value() && noexcept
    {
        const auto& exec = schd.env.exec;

        exec.fence(std::format("{}: sync_wait", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        *error = std::forward<Error>(err);
    }

    decltype(auto) get_env() const noexcept { return schd.env; }
};

struct SyncWait
{
    /**
     * According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait,
     * it has to return an engaged optional (on the value channel).
     *
     * @todo Better constrain the scheduler type.
     * @todo Make the @c noexcept specifier depend on the completion signatures of @p sndr.
     */
    template <stdexec::scheduler Schd, stdexec::sender Sndr>
    auto operator()(Schd&& schd, Sndr&& sndr) const noexcept(false) -> std::optional<std::tuple<>>
    {
        auto error = std::make_shared<std::exception_ptr>();

        auto op_state = stdexec::connect(
            std::forward<Sndr>(sndr),
            SyncWaitReceiver{.schd = std::forward<Schd>(schd), .error = error}
        );

        stdexec::start(op_state);

        if (*error) std::rethrow_exception(std::move(*error));

        return std::tuple{};
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SYNC_WAIT_HPP
