#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_THEN_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_THEN_HPP

#include "kokkos_ext/impl/PartialAlgorithm.hpp"

namespace Kokkos
{
//! Tag used to distinguish a @c then. It's not in @c Kokkos yet.
struct ThenTag {};
} // namespace Kokkos

namespace Kokkos::Experimental::graph::details
{

//! Specialization for @c Kokkos @c then when no execution space instance is passed.
template <typename Functor>
struct PartialAlgorithm<Kokkos::ThenTag, std::string, Functor>
{
    std::string label;
    Functor functor;

    template <typename Sender> requires is_graph_sender<std::remove_reference_t<Sender>>
    decltype(auto) operator()(Sender&& input) &&
    {
        return std::forward<Sender>(input).then(
            std::move(label),
            std::move(functor)
        );
    }

    /// Case of input sender being a @c Kokkos execution space instance.
    /// We do eager execution, to get as close as possible from what a user currently
    /// expects with vanilla @c Kokkos code.
    /// @todo For now, it's implemented as a parallel-for from 0 to 1.
    template <typename Sender> requires (! is_graph_sender<std::remove_reference_t<Sender>>)
    decltype(auto) operator()(Sender&& exec) &&
    {
        Kokkos::parallel_for(
            std::move(label),
            Kokkos::RangePolicy(exec, 0, 1),
            std::move(functor)
        );
        return std::forward<Sender>(exec);
    }
};

//! Specialization for @c Kokkos @c then when an execution space instance is passed.
template <typename Exec, typename Functor>
struct PartialAlgorithm<Kokkos::ThenTag, std::string, Exec, Functor>
{
    std::string label;
    Exec exec;
    Functor functor;

    template <typename Sender> requires is_graph_sender<std::remove_reference_t<Sender>>
    decltype(auto) operator()(Sender&& input) &&
    {
        return std::forward<Sender>(input).then(
            std::move(label),
            std::move(exec),
            std::move(functor)
        );
    }

    /// Case of input sender being a @c Kokkos execution space instance.
    /// We do eager execution, to get as close as possible from what a user currently
    /// expects with vanilla @c Kokkos code.
    /// @todo For now, it's implemented as a parallel-for from 0 to 1.
    // UNCLEAR WHAT TO DO -> needs continues on
    template <typename Sender> requires (! is_graph_sender<std::remove_reference_t<Sender>>)
    decltype(auto) operator()(Sender&& sender) &&
    {
        sender.fence();
        Kokkos::parallel_for(
            std::move(label),
            Kokkos::RangePolicy(exec, 0, 1),
            std::move(functor)
        );
        return std::forward<Sender>(sender);
    }
};

} // namespace Kokkos::Experimental::graph::details

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_PARTIALALGORITHM_THEN_HPP
