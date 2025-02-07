#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP

#include <concepts>

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::execution_space
{
//! Receiver for @c continues_on. @todo Better constrain the scheduler type.
template <stdexec::receiver Rcvr, stdexec::scheduler Schd>
struct ContinuesOnReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Rcvr rcvr;
    Schd schd;

    //! If set to @c true, @c set_value will fence.
    bool fencing_required;

    //! For now, we don't support any argument.
    void set_value() && noexcept
    {
        if(fencing_required) schd.env.exec.fence(std::format("{}: continues_on", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));

        stdexec::set_value(std::move(rcvr));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(rcvr); }
};

template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender
{
    using sender_concept = stdexec::sender_t;

    //! It should "forward" the values on the value channel.
    template <typename... Args>
    using set_value_t = ::stdexec::completion_signatures<::stdexec::set_value_t(Args...)>;

    //! It should "forward" the errors on the error channel.
    template <typename... Args>
    using set_error_t = ::stdexec::completion_signatures<::stdexec::set_error_t(Args...)>;

    //! Inspired by https://github.com/NVIDIA/stdexec/blob/46f8c6368dc419260e19f585de35ca3c1bb47ee0/include/nvexec/stream/schedule_from.cuh#L224-L232.
    template <class Self, typename... Env>
    using completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        stdexec::completion_signatures</* We probably miss something here. */>,
        set_value_t,
        set_error_t
    >;

    template <class Self, class... Env>
    static auto get_completion_signatures(Self&&, Env&&...) -> completion_signatures<Self, Env...> { return {}; }

    //! See also https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/then.cuh#L52.
    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        using recv_t = ContinuesOnReceiver<std::remove_cvref_t<Rcvr>, Schd>;

        return ::stdexec::connect(
            std::move(sndr),
            recv_t{.rcvr = std::forward<Rcvr>(rcvr), .schd = std::move(schd), .fencing_required = fencing_required}
        );
    }

    Schd schd;
    Sndr sndr;
    bool fencing_required;

    decltype(auto) get_env() const noexcept { return stdexec::get_env(sndr); }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
