#ifndef GRAPH_DISPATCHING_EXAMPLES_KOKKOS_EXECUTION_DIAMOND_HPP
#define GRAPH_DISPATCHING_EXAMPLES_KOKKOS_EXECUTION_DIAMOND_HPP

#include "Kokkos_Core.hpp"

namespace examples::KokkosExecution::diamond {

//! Test values for each of the diamond nodes.
struct Values {
    static constexpr int value_A = 5, value_B = 42, value_C = 156, value_D = 453;

    template <Kokkos::ExecutionSpace Exec, typename ViewType>
    static bool check(const Exec& exec, const ViewType& data) {
        using policy_t = Kokkos::RangePolicy<Exec>;

        bool result = false;
        Kokkos::parallel_reduce(
            policy_t(exec, 0, data.size()),
            KOKKOS_LAMBDA(const typename policy_t::index_type index, bool& current) {
                const auto expt_value = value_A + (index >= data.size() / 2 ? value_C : value_B) + value_D;
                current = data(index) == expt_value && current;
            },
            Kokkos::LAnd<bool>(result));
        return result;
    }
};

} // namespace examples::KokkosExecution::diamond

#endif // GRAPH_DISPATCHING_EXAMPLES_KOKKOS_EXECUTION_DIAMOND_HPP
