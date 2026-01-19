#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Experimental {
enum class Mode : std::uint8_t {
    NORMAL = 0,
    FUSION = 1
};
}

namespace Kokkos::Experimental::details::execution_space
{

template <Mode mode>
struct Property;

template <typename>
inline constexpr bool is_property_v = false;

template <Mode mode>
inline constexpr bool is_property_v<Property<mode>> = true;

template <typename T>
concept is_property = is_property_v<T>;

template <class Tag>
struct apply_sender_for;

template <class Tag, class Env>
struct transform_sender_for;

template <Kokkos::ExecutionSpace Exec, is_property props>
struct SchedulerEnv;

template <Kokkos::ExecutionSpace Exec, is_property props>
struct Scheduler;

//! Concept for a sender whose completion scheduler is @ref Kokkos::Experimental::details::execution_space::Scheduler.
template <class Sndr, class Env = ::stdexec::env<>>
concept execution_space_completing_sender = ::stdexec::sender<Sndr>
    && ::stdexec::__is_instance_of<std::invoke_result_t<
        ::stdexec::get_completion_scheduler_t<::stdexec::set_value_t>, ::stdexec::env_of_t<Sndr>, Env>,
        Scheduler
    >;

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
