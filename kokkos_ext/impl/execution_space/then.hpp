#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

namespace Kokkos::Experimental::details::execution_space
{

/**
 * @brief Receiver for @c then.
 *
 * @note It must be nothrow moveable, see @cite P3383R3.
 */
template <stdexec::receiver Rcvr, typename Functor, stdexec::scheduler Schd> requires stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>
struct ThenReceiver
{
    using receiver_concept = stdexec::receiver_t;

    //! Inspired by https://github.com/kokkos/kokkos/blob/69273c3a4e7b6adeb95066341ca201d62fe1e698/core/src/impl/Kokkos_GraphNodeThenImpl.hpp#L28.
    struct ThenWrapper
    {
        Functor functor;

        template <std::integral T>
        KOKKOS_FUNCTION void operator()(const T) const { functor(); }
    };

    Rcvr rcvr;
    ThenWrapper wrapper;
    Schd schd;

    template <typename Rcv, typename Fun, typename Sch>
    ThenReceiver(Rcv&& rcv, Fun&& fun, Sch&& sch)
        : rcvr(std::forward<Rcv>(rcv)),
          wrapper{std::forward<Fun>(fun)},
          schd(std::forward<Sch>(sch)) {}

    ThenReceiver(ThenReceiver&&) noexcept = default;

    void set_value() && noexcept
    {
        std::cout << "then: set_value" << "(" << Kokkos::Tools::Experimental::device_id(schd.env.exec) << ")" << std::endl;
        try {
            Kokkos::parallel_for(
                std::format("{}: then", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()),
                Kokkos::RangePolicy(std::move(schd).env.exec, 0, 1),
                /// We cannot write:
                /// @code
                /// ThenWrapper{.functor = std::move(functor)}
                /// @endcode
                /// because, as @c Kokkos::parallel_for spawns a possibly asynchronous
                /// kernel, we must keep any resource alive.
                /// @todo Using @c async_scope from https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3149r3.html#executionasync_scope
                ///       would move the responsibility of lifetime bookkeeping to the user.
                wrapper
            );
        } catch(...) {
            stdexec::set_error(std::move(rcvr), std::current_exception());
        }
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        std::cout << "then: set_error" << "(" << Kokkos::Tools::Experimental::device_id(schd.env.exec) << ")" << std::endl;
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        std::cout << "then: set_stopped" << "(" << Kokkos::Tools::Experimental::device_id(schd.env.exec) << ")" << std::endl;
        ::stdexec::set_stopped(std::move(rcvr));
    }

    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(rcvr); }
};

/**
 * @brief Sender for @c then.
 *
 * @todo We should decide what to do with a "throwing situation". There are 2 reasons for @c ThenReceiver::set_value to throw:
 *          1. The functor itself has a call operator that may throw.
 *          2. @c Kokkos itself throws before launching the functor (for some reason).
 */
template <stdexec::sender Sndr, typename Functor, typename Schd>
struct ThenSender
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

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <typename... Env>
    auto get_completion_signatures(Env&&...) -> completion_signatures<ThenSender, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    //! See also https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/then.cuh#L52.
    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        using recv_t = ThenReceiver<std::remove_cvref_t<Rcvr>, Functor, Schd>;

        return ::stdexec::connect(
            std::move(sndr),
            recv_t{std::forward<Rcvr>(rcvr),std::move(functor), std::move(schd)}
        );
    }

    Sndr sndr;
    Functor functor;
    Schd schd;

    decltype(auto) get_env() const noexcept { return stdexec::get_env(sndr); }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP
