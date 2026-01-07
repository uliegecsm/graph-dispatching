#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP

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

    void set_value() && noexcept
    {
        std::cout << "ContinuesOnReceiver: set_value(" << Kokkos::Impl::TypeInfo<Schd>::name() << ")" << std::endl;
        // if(fencing_required) schd.env.exec.fence(std::format("{}: continues_on", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));

        stdexec::set_value(std::move(rcvr));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        std::cout << "ContinuesOnReceiver: set_error(" << Kokkos::Impl::TypeInfo<Schd>::name() << ")" << std::endl;
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    decltype(auto) get_env() const noexcept { return stdexec::get_env(rcvr); }//schd.env; }
};

//! Sender for @c continues_on. @todo Better constrain the scheduler type.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender
{
    using sender_concept = stdexec::sender_t;

    using with_error_invoke_t = ::stdexec::completion_signatures<
        ::stdexec::set_value_t(),
        ::stdexec::set_error_t(std::exception_ptr)
    >;

    //! Inspired by https://github.com/NVIDIA/stdexec/blob/46f8c6368dc419260e19f585de35ca3c1bb47ee0/include/nvexec/stream/schedule_from.cuh#L224-L232.
    template <class Self, typename... Env>
    using completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    template <class... Env>
    auto get_completion_signatures(Env&&...) -> completion_signatures<ContinuesOnSender, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

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
    bool fencing_required = false;

    decltype(auto) get_env() const noexcept { return stdexec::get_env(sndr); }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
