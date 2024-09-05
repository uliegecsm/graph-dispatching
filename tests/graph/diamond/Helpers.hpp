#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_DIAMOND_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_DIAMOND_HELPERS_HPP

namespace tests::graph::diamond
{

/**
 * @brief Add @ref value to @ref data.
 *
 * @note The @ref offset is needed for ranges that do not start at 0.
 *       Indeed, @c stdexec::bulk currently only takes an integer for the "shape".
 */
template <typename ViewType>
struct AddValueOffset
{
    ViewType data;
    typename ViewType::value_type value;
    typename ViewType::size_type offset = 0;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        data(offset + index) += value;
    }
};

//! Check that the data was correctly processed by the graph.
#define ASSERT_IT_WENT_FINE                               \
    bool result = false;                                  \
    Kokkos::parallel_reduce(                              \
        Kokkos::RangePolicy<execution_space>(0, size),    \
        KOKKOS_LAMBDA(const auto index, bool& current) {  \
            const auto expt_value = value_A               \
                + (index >= size / 2 ? value_C : value_B) \
                + value_D;                                \
            current = data(index) == expt_value;          \
        },                                                \
        Kokkos::LAnd<bool>(result)                        \
    );                                                    \
    ASSERT_TRUE(result);

} // tests::graph::diamond

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_DIAMOND_HELPERS_HPP
