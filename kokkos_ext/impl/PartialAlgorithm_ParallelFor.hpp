#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP

#include "kokkos_ext/impl/Concepts.hpp"
#include "kokkos_ext/impl/PartialAlgorithm.hpp"

namespace Kokkos::Experimental::graph::details
{

//! Specialization for @c Kokkos parallel-for.
template <typename Policy, typename Functor>
struct PartialAlgorithm<Kokkos::ParallelForTag, std::string, Policy, Functor>
{
    std::string label;
    Policy policy;
    Functor functor;

    template <typename Sender> requires is_graph_sender<std::remove_reference_t<Sender>>
    decltype(auto) operator()(Sender&& input) &&
    {
        return std::forward<Sender>(input).then_parallel_for(
            std::move(label),
            std::move(policy),
            std::move(functor)
        );
    }

    /// Case of input sender being a @c Kokkos execution space instance.
    /// We do eager execution, to get as close as possible from what a user currently
    /// expects with vanilla @c Kokkos code.
    template <typename Sender> requires (! is_graph_sender<std::remove_reference_t<Sender>>)
    decltype(auto) operator()(Sender&& exec) &&
    {
        /// @todo The policy should be modified to ensure we run the kernel on the given
        ///       execution space instance. This is currently not possible.
        Kokkos::parallel_for(
            std::move(label),
            std::move(policy),
            std::move(functor)
        );
        return std::forward<Sender>(exec);
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_PARALLELFOR_HPP
