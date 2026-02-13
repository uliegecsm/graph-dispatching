#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_RECEIVER_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_RECEIVER_HPP

#include "stdexec/execution.hpp"

#include "kokkos_ext/impl/execution_space/get_exec.hpp"

namespace Kokkos::Experimental::details::execution_space {
struct ReceiverBase {
    using receiver_concept = stdexec::receiver_t;
};

template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct Receiver : public ReceiverBase {
    template <typename Scheduler, typename Rec>
    Receiver(Scheduler&& scheduler, Rec&& receiver)
        : schd(std::forward<Scheduler>(scheduler))
        , rcvr(std::forward<Rec>(receiver)) {
    }

    Schd schd;
    Rcvr rcvr;

    //! Make others aware of which execution space instance it will submit its work onto.
    GRAPH_DISPATCHING_KOKKOS_EXT_UPSERT_EXEC(typename Schd::execution_space, schd.state->exec, Rcvr, rcvr)

    /**
     * When the downstream receiver environment cannot be queried for @ref Kokkos::Experimental::details::execution_space::get_exec_t,
     * it is unlikely that it will dispatch work onto an execution space instance.
     * In that case, synchronization must occur before invoking the downstream receiver.
     *
     * This situation arises, for example, when this scheduler is used in a @c stdexec::when_all branch. In such a situation,
     * the branch is not terminated by a @c stdexec::schedule_from, so we'd be missing a synchronization.
     *
     * According to https://eel.is/c++draft/exec.async.ops#10:
     *
     *     A scheduler is an abstraction of an execution resource with a uniform, generic interface for scheduling work onto that resource.
     *     It is a factory for senders whose asynchronous operations execute value completion operations on an execution agent
     *     belonging to the scheduler’s associated execution resource.
     *
     * Since we are not invoking completion functions from the device, but instead from the current host thread,
     * we must explicitly synchronize previously launched work before calling @c set_value when the downstream receiver execution
     * context is unknown.
     *
     * This approach is inspired by https://github.com/NVIDIA/stdexec/blob/485160802ee5ca42ca4915e3a2330579efae4ea3/include/nvexec/stream/common.cuh#L621-L629.
     */
    template <typename Tag, typename... Args>
    void propagate_completion_signal(Tag, Args&&... args) && noexcept {
        constexpr bool skip = stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>;
        if constexpr (!skip) {
            schd.state->exec
                .fence(std::format("{}: continuation", Kokkos::Impl::TypeInfo<typename Schd::execution_space>::name()));
        }
        Tag()(std::move(rcvr), std::forward<Args>(args)...);
    }
};

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTION_SPACE_RECEIVER_HPP
