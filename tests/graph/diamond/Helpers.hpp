#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_DIAMOND_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_DIAMOND_HELPERS_HPP

namespace tests::graph::diamond
{

//! Test values.
struct Values
{
    static constexpr int value_A = 5, value_B = 42, value_C = 156, value_D = 453;
};

//! Check that the data was correctly processed by the graph.
template <typename Exec, typename ViewType>
bool check_data(const Exec& exec, const ViewType& data)
{
    using policy_t = Kokkos::RangePolicy<Exec>;

    bool result = false;
    Kokkos::parallel_reduce(
        policy_t(exec, 0, data.size()),
        KOKKOS_LAMBDA(const typename policy_t::index_type index, bool& current) {
            const auto expt_value = Values::value_A
                + (index >= data.size() / 2 ? Values::value_C : Values::value_B)
                + Values::value_D;
            current = data(index) == expt_value;
        },
        Kokkos::LAnd<bool>(result)
    );
    return result;
}

} // namespace tests::graph::diamond

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_DIAMOND_HELPERS_HPP
