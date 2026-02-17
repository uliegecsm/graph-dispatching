#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"
#include "kokkos_ext/impl/execution_space/parallel_for.hpp"

namespace Kokkos::Experimental::details::execution_space {

//! Inspired by https://github.com/kokkos/kokkos/blob/69273c3a4e7b6adeb95066341ca201d62fe1e698/core/src/impl/Kokkos_GraphNodeThenImpl.hpp#L28.
template <typename Functor>
struct ThenWrapper {
    Functor functor;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T) const {
        functor();
    }
};

template <typename Env>
struct transform_sender_for<stdexec::then_t, Env> {
    template <typename Functor, execution_space_completing_sender<Env> Sndr>
    auto operator()(stdexec::then_t, Functor&& functor, Sndr&& sndr) && noexcept {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
        auto exec = schd.state->exec;

        using execution_space = decltype(exec);

        Kokkos::RangePolicy<execution_space, Kokkos::LaunchBounds<1>> policy(std::move(exec), 0, 1);
        std::string label(std::format("{}: then", Kokkos::Impl::TypeInfo<execution_space>::name()));

        return ParallelForSender<Sndr, ThenWrapper<Functor>, decltype(policy)>{
            {{std::move(label), ThenWrapper{std::forward<Functor>(functor)}, std::move(policy)}},
            std::forward<Sndr>(sndr)};
    }

    const Env& env_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_THEN_HPP
