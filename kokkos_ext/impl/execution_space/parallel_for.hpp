#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_PARALLEL_FOR_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_PARALLEL_FOR_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/execution_space/operation_state.hpp"
#include "kokkos_ext/impl/parallel_for.hpp"

namespace Kokkos::Experimental::details::execution_space {

template <typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForClosure {
    using policy_t = ExecPolicy;
    using execution_space = typename policy_t::execution_space;

    Kokkos::Experimental::ParallelForData<Functor, policy_t> data;

    void execute() const & {
        Kokkos::parallel_for(data.label, data.policy, data.functor);
    }

    const policy_t& get_policy() const & noexcept {
        return data.policy;
    }
};

template <stdexec::sender Sndr, typename Functor, Kokkos::ExecutionPolicy ExecPolicy>
struct ParallelForSender {
    using sender_concept = stdexec::sender_t;

    using closure_t = ParallelForClosure<Functor, ExecPolicy>;
    using execution_space = typename closure_t::execution_space;

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_ADD(ParallelForSender, stdexec::set_error_t(std::exception_ptr))

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<OpState<Sndr, Rcvr, closure_t>, Sndr&&, Rcvr&&, closure_t&&>)
        -> OpState<Sndr, Rcvr, closure_t> {
        return OpState<Sndr, Rcvr, closure_t>(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));
    }

    GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(Sndr, sndr)
};

template <>
struct transform_sender_for<Kokkos::Experimental::parallel_for_t> {
    template <typename Data>
    using policy_t = typename std::remove_cvref_t<Data>::policy_t;

    template <typename Data>
    using functor_t = typename std::remove_cvref_t<Data>::functor_t;

    template <typename Data>
    using closure_t = ParallelForClosure<functor_t<Data>, policy_t<Data>>;

    template <typename Data, typename Sndr>
    using sndr_t = ParallelForSender<Sndr, functor_t<Data>, policy_t<Data>>;

    template <typename Env, typename Data, execution_space_completing_sender<Env> Sndr>
    auto operator()(const Env& env, Kokkos::Experimental::parallel_for_t, Data&& data, Sndr&& sndr) const noexcept(
        stdexec::__nothrow_decay_copyable<Data&&>
        && std::is_nothrow_constructible_v<sndr_t<Data, Sndr>, closure_t<Data>&&, Sndr&&>) {
        auto [label, functor, policy] = std::forward<Data>(data);

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return sndr_t<Data, Sndr>{
            {{std::move(label), std::move(functor), impl_policy_update(std::move(policy), schd.state->exec)}},
            std::forward<Sndr>(sndr)};
    }

   private:
    /**
     * @note Marked @c noexcept because @c Kokkos policy update does not throw,
     * despite not having a @c noexcept specification.
     */
    template <Kokkos::ExecutionPolicy ExecPolicy>
    static auto impl_policy_update(ExecPolicy&& policy, const auto& exec) noexcept {
        return std::remove_cvref_t<ExecPolicy>(Kokkos::Impl::PolicyUpdate{}, std::forward<ExecPolicy>(policy), exec);
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_PARALLEL_FOR_HPP
