#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

namespace Kokkos::Experimental::details::execution_space
{

/**
 * @brief Receiver for @c bulk.
 *
 * @note Only integral shape is supported for now.
 */
template <stdexec::receiver Rcvr, typename Policy, std::integral Shape, typename Functor, stdexec::scheduler Schd> requires (
    stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>
    && std::same_as<Policy, ::stdexec::__bulk::__policy_wrapper<::stdexec::parallel_policy>>
)
struct BulkReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Rcvr rcvr;
    Policy policy;
    Shape shape;
    Functor functor;
    Schd schd;

    void set_value() && noexcept
    {
        try {
            Kokkos::parallel_for(
                std::format("{}: bulk", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()),
                Kokkos::RangePolicy(std::move(schd).env.exec, 0, shape),
                std::move(functor)
            );
        } catch(...) {
            stdexec::set_error(std::move(rcvr), std::current_exception());
        }
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(rcvr); }
};

/**
 * @brief Sender for @c bulk.
 *
 * @todo We should decide what to do with a "throwing situation". There are 2 reasons for @c BulkReceiver::set_value to throw:
 *          1. The functor itself has a call operator that may throw.
 *          2. @c Kokkos itself throws before launching the functor (for some reason).
 */
template <stdexec::sender Sndr, typename Policy, typename Shape, typename Functor, stdexec::scheduler Schd>
struct BulkSender
{
    using sender_concept = stdexec::sender_t;

    //! @c Kokkos may throw while launching the kernel.
    using with_error_invoke_t = ::stdexec::completion_signatures<
        ::stdexec::set_value_t(),
        ::stdexec::set_error_t(std::exception_ptr)
    >;

    template <typename Self, typename... Env>
    using completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    template <typename Self, typename... Env>
    static auto get_completion_signatures(Self&&, Env&&...) -> completion_signatures<Self, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        using recv_t = BulkReceiver<std::remove_cvref_t<Rcvr>, Policy, Shape, Functor, Schd>;

        return ::stdexec::connect(
            std::move(sndr),
            recv_t{.rcvr = std::forward<Rcvr>(rcvr), .policy = std::move(policy), .shape = std::move(shape), .functor = std::move(functor), .schd = std::move(schd)}
        );
    }

    Sndr sndr;
    Policy policy;
    Shape shape;
    Functor functor;
    Schd schd;

    decltype(auto) get_env() const noexcept { return stdexec::get_env(sndr); }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
