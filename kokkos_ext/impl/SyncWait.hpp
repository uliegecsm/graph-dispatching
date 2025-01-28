#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNCWAIT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNCWAIT_HPP

#include "Kokkos_Core_fwd.hpp"

namespace Kokkos::Experimental::graph::details
{
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct SyncWaitReceiver
{
    Exec exec;

    void set_value() && { exec.fence(); }
};

//! @todo Short description.
struct SyncWait
{
    template <typename Sender> requires sender<std::remove_cvref_t<Sender>>
    void operator()(Sender&& sndr) const
    {
        std::forward<Sender>(sndr).connect(
            SyncWaitReceiver{sndr.get_env().exec}
        ).start();
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_SYNCWAIT_HPP
