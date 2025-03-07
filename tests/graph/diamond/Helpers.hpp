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

    /**
     * @note The @c noexcept is needed when using @c exec::any_receiver_ref (as of
     *       https://github.com/NVIDIA/stdexec/blob/8bc7c7f06fe39831dea6852407ebe7f6be8fa9fd/include/exec/any_sender_of.hpp).
     */
    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const noexcept{
        data(offset + index) += value;
    }
};

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
