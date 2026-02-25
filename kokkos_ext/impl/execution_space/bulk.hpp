#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/bulk.hpp"
#include "kokkos_ext/impl/execution_space/parallel_for.hpp"

namespace Kokkos::Experimental::details::execution_space {

template <>
struct transform_sender_for<stdexec::bulk_t> {
    template <typename Sndr, typename Env>
    using schd_t = stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, const Env&>;

    template <typename Sndr, typename Env>
    using execution_space = typename schd_t<Sndr, Env>::execution_space;

    template <typename Sndr, typename Env>
    using policy_t = Kokkos::RangePolicy<execution_space<Sndr, Env>>;

    template <typename Data>
    using functor_t = stdexec::__copy_cvref_t<Data, std::remove_cvref_t<decltype(std::declval<Data>().__fun_)>>;

    template <typename Sndr, typename Data, typename Env>
    using closure_t = ParallelForClosure<functor_t<Data>, policy_t<Sndr, Env>>;

    template <typename Sndr, typename Data, typename Env>
    using sndr_t = ParallelForSender<Sndr, functor_t<Data>, policy_t<Sndr, Env>>;

    template <
        typename Env,
        Kokkos::Experimental::details::impl::has_parallel_policy Data,
        execution_space_completing_sender<Env> Sndr
    >
    auto operator()(const Env& env, stdexec::bulk_t, Data&& data, Sndr&& sndr) const noexcept(
        stdexec::__nothrow_decay_copyable<Data&&>
        && std::is_nothrow_constructible_v<sndr_t<Sndr, Data, Env>, closure_t<Sndr, Data, Env>&&, Sndr&&>) {
        auto& [parallel_policy, shape, functor] = data;

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        std::string label(std::format("{}: bulk", Kokkos::Impl::TypeInfo<execution_space<Sndr, Env>>::name()));

        return sndr_t<Sndr, Data, Env>{
            {{std::move(label),
              stdexec::__forward_like<Data>(functor),
              impl_policy_construct<policy_t<Sndr, Env>>(schd.state->exec, shape)}},
            std::forward<Sndr>(sndr)};
    }

   private:
    /**
     * @note Marked @c noexcept because @c Kokkos policy construction does not throw,
     * despite not having a @c noexcept specification.
     */
    template <Kokkos::ExecutionPolicy ExecPolicy>
    static auto impl_policy_construct(const auto& exec, auto shape) noexcept {
        return ExecPolicy(exec, 0, shape);
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_BULK_HPP
