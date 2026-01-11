#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/bulk.hpp"
#include "kokkos_ext/impl/completion_signatures.hpp"
#include "kokkos_ext/impl/execution_space/receiver.hpp"

namespace Kokkos::Experimental::details::execution_space {

//! See https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__bulk.hpp#L100.
template <typename Data>
concept parallel_policy = requires(const Data& data) {
    { data.__pol_ } -> std::same_as<const stdexec::__bulk::__policy_wrapper<stdexec::parallel_policy>&>;
};

/**
 * @brief Receiver for @c bulk.
 *
 * @note Only integral shape is supported for now.
 *
 * @note It must be nothrow moveable, see @cite P3383R3.
 */
template <
    stdexec::receiver Rcvr,
    stdexec::__is_instance_of<stdexec::__bulk::__data> Data,
    stdexec::__is_instance_of<Scheduler> Schd
>
requires Kokkos::Experimental::details::impl::parallel_policy<Data>
struct BulkReceiver : public Receiver<Schd, Rcvr> {
    using base_t = Receiver<Schd, Rcvr>;

    Data data;

    template <typename Rcv, typename Sch>
    BulkReceiver(Rcv&& rcv, Data&& data_, Sch&& sch)
        : base_t{std::forward<Sch>(sch), std::forward<Rcv>(rcv)}
        , data(std::move(data_)) {
    }

    BulkReceiver(const BulkReceiver&) = delete;
    BulkReceiver& operator=(const BulkReceiver&) = delete;
    BulkReceiver(BulkReceiver&&) noexcept = default;
    BulkReceiver& operator=(BulkReceiver&&) noexcept = default;
    ~BulkReceiver() = default;

    void set_value() && noexcept {
        try {
            auto [policy, shape, functor] = std::move(data);
            Kokkos::parallel_for(
                std::format("{}: bulk", Kokkos::Impl::TypeInfo<typename Schd::execution_space>::name()),
                Kokkos::RangePolicy(this->schd.state->exec, 0, std::move(shape)),
                std::move(functor));
        } catch (...) {
            std::move(*this).propagate_completion_signal(stdexec::set_error, std::current_exception());
        }
        std::move(*this).propagate_completion_signal(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        std::move(*this).propagate_completion_signal(::stdexec::set_error, std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        std::move(*this).propagate_completion_signal(::stdexec::set_stopped);
    }
};

/**
 * @brief Sender for @c bulk.
 *
 * @todo We should decide what to do with a "throwing situation". There are 2 reasons for @c BulkReceiver::set_value to throw:
 *          1. The functor itself has a call operator that may throw.
 *          2. @c Kokkos itself throws before launching the functor (for some reason).
 */
template <stdexec::sender Sndr, stdexec::__is_instance_of<stdexec::__bulk::__data> Data, stdexec::scheduler Schd>
struct BulkSender {
    using sender_concept = stdexec::sender_t;

    //! @c Kokkos may throw while launching the kernel.
    using with_error_invoke_t =
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>;

    template <typename Self, typename... Env>
    using _completion_signatures = stdexec::transform_completion_signatures<
        stdexec::completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t
    >;

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPLETION_SIGNATURES(BulkSender)

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        using recv_t = BulkReceiver<std::remove_cvref_t<Rcvr>, Data, Schd>;

        return stdexec::connect(std::move(sndr), recv_t{std::forward<Rcvr>(rcvr), std::move(data), std::move(schd)});
    }

    Sndr sndr;
    Data data;
    Schd schd;

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Sndr>> {
        return stdexec::__fwd_env(stdexec::get_env(sndr));
    }
};

template <typename Env>
struct transform_sender_for<stdexec::bulk_t, Env> {
    template <typename Data, execution_space_completing_sender<Env> Sndr>
    auto operator()(stdexec::bulk_t, Data&& data, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        return BulkSender{.sndr = std::forward<Sndr>(sndr), .data = std::forward<Data>(data), .schd = std::move(schd)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
