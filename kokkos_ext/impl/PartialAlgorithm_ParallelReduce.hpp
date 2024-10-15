#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELREDUCE_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELREDUCE_HPP

#include "kokkos_ext/impl/PartialAlgorithm.hpp"

namespace Kokkos::Experimental::graph::details
{

//! Specialization for @c Kokkos parallel-reduce.
template <typename Policy, typename Functor, typename Reducer>
struct PartialAlgorithm<Kokkos::ParallelReduceTag, Policy, Functor, Reducer>
{
    Policy policy;
    Functor functor;
    Reducer reducer;

    template <typename Sender> requires is_graph_sender<std::remove_reference_t<Sender>>
    decltype(auto) operator()(Sender&& input)
    {
        return std::forward<Sender>(input).then_parallel_reduce(
            std::move(policy),
            std::move(functor),
            std::move(reducer)
        );
    }

    /// Case of input sender being a @c Kokkos execution space instance.
    /// We do eager execution, to get as close as possible from what a user currently
    /// expects with vanilla @c Kokkos code.
    template <typename Sender> requires (! is_graph_sender<std::remove_reference_t<Sender>>)
    decltype(auto) operator()(Sender&& exec)
    {
        /// @todo The policy should be modified to ensure we run the kernel on the given
        ///       execution space instance. This is currently not possible.
        Kokkos::parallel_reduce(
            std::move(policy),
            std::move(functor),
            std::move(reducer)
        );
        return std::forward<Sender>(exec);
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELREDUCE_HPP
