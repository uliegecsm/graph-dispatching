#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCHEDULE_FROM_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_SCHEDULE_FROM_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

namespace Kokkos::Experimental::details::execution_space
{

//! Receiver for @c continues_on. @todo Better constrain the scheduler type.
template <stdexec::receiver Rcvr, stdexec::scheduler Schd> //requires stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>
struct ScheduleFromReceiver
{
    using receiver_concept = stdexec::receiver_t;

    Rcvr rcvr;
    Schd schd;
    

    //! If set to @c true, @c set_value will fence.
    bool fencing_required;

    size_t id;

    void set_value() && noexcept
    {
        std::cout << "ScheduleFromReceiver: set_value(" << Kokkos::Impl::TypeInfo<Schd>::name() << ")" << std::endl;
        if constexpr (stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>) {
            if(fencing_required) schd.env.exec.fence(std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<decltype(schd.env.exec)>::name()));
        }

        stdexec::set_value(std::move(rcvr));
    }

    template <class Error>
    void set_error(Error&& err) && noexcept {
        std::cout << "ScheduleFromReceiver: set_error(" << Kokkos::Impl::TypeInfo<Schd>::name() << ")" << std::endl;
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(rcvr); }
};

//! Sender for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ScheduleFromSender
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

    template <class Self, class... Env>
    static auto get_completion_signatures(Self&&, Env&&...) -> completion_signatures<Self, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <::stdexec::receiver Rcvr>
    ::stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        

        if constexpr (::stdexec::__has_completion_scheduler<Sndr, stdexec::set_value_t>)
        {
            auto completion_schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));

            const bool fencing_required_ = [&]() {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(schd)>, std::remove_cvref_t<decltype(completion_schd)>>)
                    return schd != completion_schd;
                else
                    return true;
            }();

            using recv_t = ScheduleFromReceiver<std::remove_cvref_t<Rcvr>, decltype(completion_schd)>;

            return ::stdexec::connect(
                std::move(sndr),
                recv_t{.rcvr = std::forward<Rcvr>(rcvr), .schd = std::move(completion_schd), .fencing_required = fencing_required_, .id = id}
            );
        } else {
            using recv_t = ScheduleFromReceiver<std::remove_cvref_t<Rcvr>, Schd>;

            return ::stdexec::connect(
                std::move(sndr),
                recv_t{.rcvr = std::forward<Rcvr>(rcvr), .schd = std::move(schd), .fencing_required = false, .id = id}
            );
        }
    }

    Schd schd;
    Sndr sndr;
    bool fencing_required = false;
    size_t id = 0;

    // THIS IS KEY
    decltype(auto) get_env() const noexcept {
        if constexpr (stdexec::__is_instance_of_<Schd, ExecutionSpaceScheduler>) {
            return schd.env;
        } else {
            return stdexec::get_env(schd);
        }
    }
};
} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_CONTINUES_ON_HPP
