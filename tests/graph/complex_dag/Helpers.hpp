#ifndef GRAPH_DISPATCHING_TESTS_GRAPH_COMPLEX_DAG_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_GRAPH_COMPLEX_DAG_HELPERS_HPP

namespace tests::graph::complex_dag
{

//! Fetch values at @ref indices in @ref data and contribute them at @c TargetIndex.
template <typename ViewType, size_t TargetIndex, size_t NumIndices = 0>
struct FetchValuesAndContribute
{
    static_assert(std::is_same_v<typename ViewType::value_type,
                                 typename ViewType::non_const_value_type>);

    ViewType data;
    typename ViewType::value_type value;
    std::array<size_t, NumIndices> indices{};

    FetchValuesAndContribute(ViewType data_,
                             std::integral_constant<size_t, TargetIndex>,
                             typename ViewType::value_type value_)
        : data(std::move(data_)), value(std::move(value_)) {}

    FetchValuesAndContribute(ViewType data_,
                             std::array<size_t, NumIndices> indices_,
                             std::integral_constant<size_t, TargetIndex>,
                             typename ViewType::value_type value_)
        : data(std::move(data_)), value(std::move(value_)), indices(std::move(indices_)) {}

    template <typename T>
    KOKKOS_FUNCTION void operator()(const T) const {
        for(const auto& index : indices) data(TargetIndex) += data(index);
        data(TargetIndex) += value;
    }
};

template <typename ViewType, size_t TargetIndex, size_t NumIndices>
FetchValuesAndContribute(ViewType, const size_t (&)[NumIndices],
                         std::integral_constant<size_t, TargetIndex>,
                         typename ViewType::non_const_value_type)
    -> FetchValuesAndContribute<ViewType, TargetIndex, NumIndices>;

#define DEFINE_VALUES                                                        \
    constexpr int value_A1 = 5, value_A2 = 9, value_A3 = 63;                 \
    constexpr int value_B1 = 6, value_B2 = 2, value_B3 = 93, value_B4 = 186; \
    constexpr int value_C1 = 7, value_C2 = 1;

#define DEFINE_INDICES                                    \
    constexpr std::integral_constant<size_t, 0> index_A1; \
    constexpr std::integral_constant<size_t, 1> index_A2; \
    constexpr std::integral_constant<size_t, 2> index_A3; \
    constexpr std::integral_constant<size_t, 3> index_B1; \
    constexpr std::integral_constant<size_t, 4> index_B2; \
    constexpr std::integral_constant<size_t, 5> index_B3; \
    constexpr std::integral_constant<size_t, 6> index_B4; \
    constexpr std::integral_constant<size_t, 7> index_C1; \
    constexpr std::integral_constant<size_t, 8> index_C2;

#define ASSERT_IT_WENT_FINE(data)                                                              \
    ASSERT_EQ(data(index_A1()), value_A1);                                                     \
    ASSERT_EQ(data(index_A2()), value_A2);                                                     \
    ASSERT_EQ(data(index_A3()), value_A3);                                                     \
    ASSERT_EQ(data(index_B1()), value_A1 + value_A2 + value_B1);                               \
    ASSERT_EQ(data(index_B2()), value_A1 + value_B2);                                          \
    ASSERT_EQ(data(index_B3()), value_A1 + value_A2 + value_B3);                               \
    ASSERT_EQ(data(index_B4()), value_A3 + value_B4);                                          \
    ASSERT_EQ(data(index_C1()), 2 * value_A1 + 2 * value_A2 + value_B1 + value_B3 + value_C1); \
    ASSERT_EQ(data(index_C2()), value_A1 + value_A3 + value_B2 + value_B4 + value_C2);

} // namespace tests::graph::complex_dag

#endif // GRAPH_DISPATCHING_TESTS_GRAPH_COMPLEX_DAG_HELPERS_HPP
