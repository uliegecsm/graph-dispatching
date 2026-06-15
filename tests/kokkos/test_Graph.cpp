#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos-utils/concepts/View.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "tests/sycl/ur_cuda_skip.hpp"
#include "tests/utils/Graph.hpp"

namespace tests::kokkos {

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename execution_space::memory_space;

struct GraphTest
    : public testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<execution_space> { };

template <Kokkos::utils::concepts::ViewOfRank<1> ViewType, std::integral IndexType, typename ValueType>
struct LoadAddStore {
    ViewType data;
    IndexType dst, src;
    ValueType value;

    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args&&...) const noexcept {
        data(dst) = data(src) + value;
    }
};

/**
 * @test A simple topology test.
 *
 * The topology looks like:
 * @verbatim
 *       root
 *        |
 *       [A]
 *      /   \
 *    [B]   [C]-----
 *   /   \    \     |
 * [D]   [E]   |   [F]
 *   \    |   /
 *    \   |  /
 *       [G]
 * @endverbatim
 */
void test_simple(const execution_space& exec) {
    constexpr unsigned short int num_nodes = 7;

    const Kokkos::View<int[num_nodes], memory_space> data(Kokkos::view_alloc("data", exec));
    const auto mirror = Kokkos::create_mirror_view(data);

    const auto device_handle = Kokkos::Experimental::get_device_handle(exec);

    const Kokkos::Experimental::Graph graph(device_handle);

    constexpr int index_A = 0, index_B = 1, index_C = 2, index_D = 3, index_E = 4, index_F = 5, index_G = 6;
    constexpr int value_A = 42, value_B = 41, value_C = 40, value_D = 39, value_E = 38, value_F = 37, value_G = 36;

    auto node_A = graph.root_node().then(KOKKOS_LAMBDA() { data(index_A) = value_A; });

    auto node_B = node_A.then(LoadAddStore{.data = data, .dst = index_B, .src = index_A, .value = value_B});
    auto node_C = node_A.then(LoadAddStore{.data = data, .dst = index_C, .src = index_A, .value = value_C});

    auto node_D = node_B.then(KOKKOS_LAMBDA() { data(index_D) = data(index_A) + data(index_B) + value_D; });
    auto node_E = node_B.then(KOKKOS_LAMBDA() { data(index_E) = data(index_A) + data(index_B) + value_E; });

    auto node_F = node_C.then(KOKKOS_LAMBDA() { data(index_F) = data(index_A) + data(index_C) + value_F; });

    auto node_G = Kokkos::Experimental::when_all(node_D, node_E, node_C).then(KOKKOS_LAMBDA() {
        data(index_G) = data(index_A) + data(index_B) + data(index_C) + data(index_D) + data(index_E) + value_G;
    });

    graph.submit(exec);
    Kokkos::deep_copy(exec, mirror, data);
    exec.fence();

    ASSERT_EQ(mirror(index_A), value_A);
    ASSERT_EQ(mirror(index_B), value_A + value_B);
    ASSERT_EQ(mirror(index_C), value_A + value_C);
    ASSERT_EQ(mirror(index_D), 2 * value_A + value_B + value_D);
    ASSERT_EQ(mirror(index_E), 2 * value_A + value_B + value_E);
    ASSERT_EQ(mirror(index_F), 2 * value_A + value_C + value_F);
    ASSERT_EQ(mirror(index_G), 7 * value_A + 3 * value_B + value_C + value_D + value_E + value_G);
}
TEST_F(GraphTest, simple) {
    test_simple(exec);
}

/**
 * @test Large topology with nested fork/joins.
 *
 * The topology looks like:
 * @verbatim
 *                         root
 *                          |
 *                         [0]
 *                    /     |     \
 *                  [1]    [2]    [3]
 *                   |      |      |
 *                  [4]    [6]    [9]
 *                   |      |      |
 *                pfor[5]  [7]     |
 *                   |      |      |
 *                   |   pfor[8]   |
 *                    \     |     /
 *                       pred[10]
 *               /      /       \      \
 *            [11]    [12]     [13]    [14]
 *             |        |       |       |    \
 *            [15]    [17]     [18]    [21]  [22]
 *             |        |       |       |     |
 *          pfor[16]    |      [19]     |     |
 *             |        |       |       |     |
 *              \       |    pfor[20]   |     |
 *               \      |       /       |     |
 *                   pred[23]          /     /
 *                      |             /    /
 *                     [24]          /   /
 *                      |          /   /
 *                   pfor[25]    /   /
 *                      \      /  /
 *                         [26]
 *                        /    \
 *                     [27]    [28]
 *                      |        |
 *                    [29]      [32]
 *                      |        |
 *                    [30]    pfor[33]
 *                      |       /
 *                    [31]     /
 *                      \     /
 *                      pred[34]
 *                    /    |    \     \
 *                [35]   [36]   [37]   \
 *                 |       |      |     |
 *                [38]   [40]   [43]   [53]
 *                 |       |      |     |
 *              pfor[39] [41]     |    [54]
 *                 |       |      |
 *                 |    pfor[42]  |
 *                  \      |     /
 *                      pred[44]
 *                         |
 *                       [45]
 *                         |
 *                       [46]
 *                      /    \
 *                   [47]    [48]
 *                    |        |
 *                 pfor[49]  [50]
 *                    |        |
 *                    |     pfor[51]
 *                     \      /
 *                     pred[52]
 * @endverbatim
 */
void test_large(const execution_space& exec) {
    using range_policy_t = Kokkos::RangePolicy<execution_space>;

    GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP

    constexpr unsigned short num_nodes = 55;

    const Kokkos::View<int[num_nodes], memory_space> data(Kokkos::view_alloc("data", exec));
    const auto mirror = Kokkos::create_mirror_view(data);

    const Kokkos::Experimental::Graph graph(Kokkos::Experimental::get_device_handle(exec));

    constexpr int index_0 = 0, index_1 = 1, index_2 = 2, index_3 = 3, index_4 = 4, index_5 = 5, index_6 = 6,
                  index_7 = 7, index_8 = 8, index_9 = 9, index_10 = 10, index_11 = 11, index_12 = 12, index_13 = 13,
                  index_14 = 14, index_15 = 15, index_16 = 16, index_17 = 17, index_18 = 18, index_19 = 19,
                  index_20 = 20, index_21 = 21, index_22 = 22, index_23 = 23, index_24 = 24, index_25 = 25,
                  index_26 = 26, index_27 = 27, index_28 = 28, index_29 = 29, index_30 = 30, index_31 = 31,
                  index_32 = 32, index_33 = 33, index_34 = 34, index_35 = 35, index_36 = 36, index_37 = 37,
                  index_38 = 38, index_39 = 39, index_40 = 40, index_41 = 41, index_42 = 42, index_43 = 43,
                  index_44 = 44, index_45 = 45, index_46 = 46, index_47 = 47, index_48 = 48, index_49 = 49,
                  index_50 = 50, index_51 = 51, index_52 = 52, index_53 = 53, index_54 = 54;

    constexpr int value_0 = 1, value_1 = 2, value_2 = 3, value_3 = 4, value_4 = 5, value_5 = 6, value_6 = 7,
                  value_7 = 8, value_8 = 9, value_9 = 10, value_10 = 11, value_11 = 12, value_12 = 13, value_13 = 14,
                  value_14 = 15, value_15 = 16, value_16 = 17, value_17 = 18, value_18 = 19, value_19 = 20,
                  value_20 = 21, value_21 = 22, value_22 = 23, value_23 = 24, value_24 = 25, value_25 = 26,
                  value_26 = 27, value_27 = 28, value_28 = 29, value_29 = 30, value_30 = 31, value_31 = 32,
                  value_32 = 33, value_33 = 34, value_34 = 35, value_35 = 36, value_36 = 37, value_37 = 38,
                  value_38 = 39, value_39 = 40, value_40 = 41, value_41 = 42, value_42 = 43, value_43 = 44,
                  value_44 = 45, value_45 = 46, value_46 = 47, value_47 = 48, value_48 = 49, value_49 = 50,
                  value_50 = 51, value_51 = 52, value_52 = 53, value_53 = 54, value_54 = 55;

    //! Reduce nodes write into subviews.
    const auto reduce_target_10 = Kokkos::subview(data, index_10);
    const auto reduce_target_23 = Kokkos::subview(data, index_23);
    const auto reduce_target_34 = Kokkos::subview(data, index_34);
    const auto reduce_target_44 = Kokkos::subview(data, index_44);
    const auto reduce_target_52 = Kokkos::subview(data, index_52);

    auto node_0 = graph.root_node().then(KOKKOS_LAMBDA() { data(index_0) = value_0; });

    auto node_1 = node_0.then(LoadAddStore{.data = data, .dst = index_1, .src = index_0, .value = value_1});
    auto node_2 = node_0.then(LoadAddStore{.data = data, .dst = index_2, .src = index_0, .value = value_2});
    auto node_3 = node_0.then(LoadAddStore{.data = data, .dst = index_3, .src = index_0, .value = value_3});

    auto node_4 = node_1.then(LoadAddStore{.data = data, .dst = index_4, .src = index_1, .value = value_4});
    auto node_5 = node_4.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_5, .src = index_4, .value = value_5});

    auto node_6 = node_2.then(LoadAddStore{.data = data, .dst = index_6, .src = index_2, .value = value_6});
    auto node_7 = node_6.then(LoadAddStore{.data = data, .dst = index_7, .src = index_6, .value = value_7});
    auto node_8 = node_7.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_8, .src = index_7, .value = value_8});

    auto node_9 = node_3.then(LoadAddStore{.data = data, .dst = index_9, .src = index_3, .value = value_9});

    auto node_10 = Kokkos::Experimental::when_all(node_5, node_8, node_9)
                       .then_parallel_reduce(
                           range_policy_t(0, 1),
                           KOKKOS_LAMBDA(const int, int& lsum) {
                               lsum += data(index_5) + data(index_8) + data(index_9) + value_10;
                           },
                           reduce_target_10);

    auto node_11 = node_10.then(LoadAddStore{.data = data, .dst = index_11, .src = index_10, .value = value_11});
    auto node_12 = node_10.then(LoadAddStore{.data = data, .dst = index_12, .src = index_10, .value = value_12});
    auto node_13 = node_10.then(LoadAddStore{.data = data, .dst = index_13, .src = index_10, .value = value_13});
    auto node_14 = node_10.then(LoadAddStore{.data = data, .dst = index_14, .src = index_10, .value = value_14});

    auto node_15 = node_11.then(LoadAddStore{.data = data, .dst = index_15, .src = index_11, .value = value_15});
    auto node_16 = node_15.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_16, .src = index_15, .value = value_16});

    auto node_17 = node_12.then(LoadAddStore{.data = data, .dst = index_17, .src = index_12, .value = value_17});

    auto node_18 = node_13.then(LoadAddStore{.data = data, .dst = index_18, .src = index_13, .value = value_18});
    auto node_19 = node_18.then(LoadAddStore{.data = data, .dst = index_19, .src = index_18, .value = value_19});
    auto node_20 = node_19.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_20, .src = index_19, .value = value_20});

    auto node_21 = node_14.then(LoadAddStore{.data = data, .dst = index_21, .src = index_14, .value = value_21});
    auto node_22 = node_14.then(LoadAddStore{.data = data, .dst = index_22, .src = index_14, .value = value_22});

    auto node_23 = Kokkos::Experimental::when_all(node_16, node_17, node_20)
                       .then_parallel_reduce(
                           range_policy_t(0, 1),
                           KOKKOS_LAMBDA(const int, int& lsum) {
                               lsum += data(index_16) + data(index_17) + data(index_20) + value_23;
                           },
                           reduce_target_23);

    auto node_24 = node_23.then(LoadAddStore{.data = data, .dst = index_24, .src = index_23, .value = value_24});
    auto node_25 = node_24.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_25, .src = index_24, .value = value_25});

    auto node_26 = Kokkos::Experimental::when_all(node_21, node_22, node_25).then(KOKKOS_LAMBDA() {
        data(index_26) = data(index_21) + data(index_22) + data(index_25) + value_26;
    });

    auto node_27 = node_26.then(LoadAddStore{.data = data, .dst = index_27, .src = index_26, .value = value_27});
    auto node_28 = node_26.then(LoadAddStore{.data = data, .dst = index_28, .src = index_26, .value = value_28});

    auto node_29 = node_27.then(LoadAddStore{.data = data, .dst = index_29, .src = index_27, .value = value_29});
    auto node_30 = node_29.then(LoadAddStore{.data = data, .dst = index_30, .src = index_29, .value = value_30});
    auto node_31 = node_30.then(LoadAddStore{.data = data, .dst = index_31, .src = index_30, .value = value_31});

    auto node_32 = node_28.then(LoadAddStore{.data = data, .dst = index_32, .src = index_28, .value = value_32});
    auto node_33 = node_32.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_33, .src = index_32, .value = value_33});

    auto node_34 = Kokkos::Experimental::when_all(node_31, node_33)
                       .then_parallel_reduce(
                           range_policy_t(0, 1),
                           KOKKOS_LAMBDA(const int, int& lsum) { lsum += data(index_31) + data(index_33) + value_34; },
                           reduce_target_34);

    auto node_35 = node_34.then(LoadAddStore{.data = data, .dst = index_35, .src = index_34, .value = value_35});
    auto node_36 = node_34.then(LoadAddStore{.data = data, .dst = index_36, .src = index_34, .value = value_36});
    auto node_37 = node_34.then(LoadAddStore{.data = data, .dst = index_37, .src = index_34, .value = value_37});

    auto node_38 = node_35.then(LoadAddStore{.data = data, .dst = index_38, .src = index_35, .value = value_38});
    auto node_39 = node_38.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_39, .src = index_38, .value = value_39});

    auto node_40 = node_36.then(LoadAddStore{.data = data, .dst = index_40, .src = index_36, .value = value_40});
    auto node_41 = node_40.then(LoadAddStore{.data = data, .dst = index_41, .src = index_40, .value = value_41});
    auto node_42 = node_41.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_42, .src = index_41, .value = value_42});

    auto node_43 = node_37.then(LoadAddStore{.data = data, .dst = index_43, .src = index_37, .value = value_43});

    auto node_44 = Kokkos::Experimental::when_all(node_39, node_42, node_43)
                       .then_parallel_reduce(
                           range_policy_t(0, 1),
                           KOKKOS_LAMBDA(const int, int& lsum) {
                               lsum += data(index_39) + data(index_42) + data(index_43) + value_44;
                           },
                           reduce_target_44);

    auto node_45 = node_44.then(LoadAddStore{.data = data, .dst = index_45, .src = index_44, .value = value_45});
    auto node_46 = node_45.then(LoadAddStore{.data = data, .dst = index_46, .src = index_45, .value = value_46});

    auto node_47 = node_46.then(LoadAddStore{.data = data, .dst = index_47, .src = index_46, .value = value_47});
    auto node_48 = node_46.then(LoadAddStore{.data = data, .dst = index_48, .src = index_46, .value = value_48});

    auto node_49 = node_47.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_49, .src = index_47, .value = value_49});

    auto node_50 = node_48.then(LoadAddStore{.data = data, .dst = index_50, .src = index_48, .value = value_50});
    auto node_51 = node_50.then_parallel_for(
        range_policy_t(0, 1), LoadAddStore{.data = data, .dst = index_51, .src = index_50, .value = value_51});

    auto node_52 = Kokkos::Experimental::when_all(node_49, node_51)
                       .then_parallel_reduce(
                           range_policy_t(0, 1),
                           KOKKOS_LAMBDA(const int, int& lsum) { lsum += data(index_49) + data(index_51) + value_52; },
                           reduce_target_52);

    auto node_53 = node_34.then(LoadAddStore{.data = data, .dst = index_53, .src = index_34, .value = value_53});
    auto node_54 = node_53.then(LoadAddStore{.data = data, .dst = index_54, .src = index_53, .value = value_54});

    graph.submit(exec);
    Kokkos::deep_copy(exec, mirror, data);
    exec.fence();

    constexpr int expected_34 = 1586;

    EXPECT_EQ(mirror(index_0), 1);
    EXPECT_EQ(mirror(index_1), 3);
    EXPECT_EQ(mirror(index_2), 4);
    EXPECT_EQ(mirror(index_3), 5);
    EXPECT_EQ(mirror(index_4), 8);
    EXPECT_EQ(mirror(index_5), 14);
    EXPECT_EQ(mirror(index_6), 11);
    EXPECT_EQ(mirror(index_7), 19);
    EXPECT_EQ(mirror(index_8), 28);
    EXPECT_EQ(mirror(index_9), 15);
    EXPECT_EQ(mirror(index_10), 68);
    EXPECT_EQ(mirror(index_11), 80);
    EXPECT_EQ(mirror(index_12), 81);
    EXPECT_EQ(mirror(index_13), 82);
    EXPECT_EQ(mirror(index_14), 83);
    EXPECT_EQ(mirror(index_15), 96);
    EXPECT_EQ(mirror(index_16), 113);
    EXPECT_EQ(mirror(index_17), 99);
    EXPECT_EQ(mirror(index_18), 101);
    EXPECT_EQ(mirror(index_19), 121);
    EXPECT_EQ(mirror(index_20), 142);
    EXPECT_EQ(mirror(index_21), 105);
    EXPECT_EQ(mirror(index_22), 106);
    EXPECT_EQ(mirror(index_23), 378);
    EXPECT_EQ(mirror(index_24), 403);
    EXPECT_EQ(mirror(index_25), 429);
    EXPECT_EQ(mirror(index_26), 667);
    EXPECT_EQ(mirror(index_27), 695);
    EXPECT_EQ(mirror(index_28), 696);
    EXPECT_EQ(mirror(index_29), 725);
    EXPECT_EQ(mirror(index_30), 756);
    EXPECT_EQ(mirror(index_31), 788);
    EXPECT_EQ(mirror(index_32), 729);
    EXPECT_EQ(mirror(index_33), 763);
    EXPECT_EQ(mirror(index_34), expected_34);
    EXPECT_EQ(mirror(index_35), 1622);
    EXPECT_EQ(mirror(index_36), 1623);
    EXPECT_EQ(mirror(index_37), 1624);
    EXPECT_EQ(mirror(index_38), 1661);
    EXPECT_EQ(mirror(index_39), 1701);
    EXPECT_EQ(mirror(index_40), 1664);
    EXPECT_EQ(mirror(index_41), 1706);
    EXPECT_EQ(mirror(index_42), 1749);
    EXPECT_EQ(mirror(index_43), 1668);
    EXPECT_EQ(mirror(index_44), 5163);
    EXPECT_EQ(mirror(index_45), 5209);
    EXPECT_EQ(mirror(index_46), 5256);
    EXPECT_EQ(mirror(index_47), 5304);
    EXPECT_EQ(mirror(index_48), 5305);
    EXPECT_EQ(mirror(index_49), 5354);
    EXPECT_EQ(mirror(index_50), 5356);
    EXPECT_EQ(mirror(index_51), 5408);
    EXPECT_EQ(mirror(index_52), 10815);
    EXPECT_EQ(mirror(index_53), expected_34 + value_53);
    EXPECT_EQ(mirror(index_54), expected_34 + value_53 + value_54);
}
TEST_F(GraphTest, large) {
    test_large(exec);
}

template <Kokkos::utils::concepts::ViewOfRank<1> ViewType>
struct ContributeFrom {
    ViewType::const_type data;

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T index, typename ViewType::non_const_value_type& local) const noexcept {
        local += data(index);
    }
};

/**
 * @test Fork-join topology, each node uses a parallel reduce.
 *
 * It should detect nodes using the same scratch view.
 */
void test_fork_join_reduce(const execution_space& exec) {
    using range_policy_t = Kokkos::RangePolicy<execution_space>;
    using view_t = Kokkos::View<int[10], memory_space>;

    const view_t data(Kokkos::view_alloc("data", exec));
    const auto mirror = Kokkos::create_mirror_view(data);

    const Kokkos::Experimental::Graph graph(Kokkos::Experimental::get_device_handle(exec));

    constexpr int value_0 = 42, value_1 = 666, value_2 = 128, value_3 = 7, value_4 = 5, value_5 = 3, value_6 = 9;

    //! Set values.
    auto memset = graph.root_node().then(KOKKOS_LAMBDA() {
        data(0) = value_0;
        data(1) = value_1;
        data(2) = value_2;
        data(3) = value_3;
        data(4) = value_4;
        data(5) = value_5;
        data(6) = value_6;
    });

    //! Sum elements 0, 1 and 2, write into element 3.
    auto node_A = memset.then_parallel_reduce(
        range_policy_t(0, 3), ContributeFrom<view_t>{.data = data}, Kokkos::subview(data, 3));

    //! Sum elements 1, 2 and 3, write into element 6.
    auto node_B = node_A.then_parallel_reduce(
        range_policy_t(1, 4), ContributeFrom<view_t>{.data = data}, Kokkos::subview(data, 6));

    //! Sum elements 2, 3 and 4, write into element 7.
    auto node_C = node_A.then_parallel_reduce(
        range_policy_t(2, 5), ContributeFrom<view_t>{.data = data}, Kokkos::subview(data, 7));

    //! Sum elements 3, 4 and 5, write into element 8.
    auto node_D = node_A.then_parallel_reduce(
        range_policy_t(3, 6), ContributeFrom<view_t>{.data = data}, Kokkos::subview(data, 8));

    //! Sum elements 1, 2, 3, 4, and 5, write into element 9.
    auto node_E = node_A.then_parallel_reduce(
        range_policy_t(1, 6), ContributeFrom<view_t>{.data = data}, Kokkos::subview(data, 9));

    //! Sum elements 6, 7, 8 and 9, write into element 0.
    auto node_F = Kokkos::Experimental::when_all(node_B, node_C, node_D, node_E)
                      .then_parallel_reduce(
                          range_policy_t(6, 10), ContributeFrom<view_t>{.data = data}, Kokkos::subview(data, 0));

    graph.submit(exec);
    Kokkos::deep_copy(exec, mirror, data);
    exec.fence();

    constexpr int expected_3 = value_0 + value_1 + value_2;
    constexpr int expected_6 = value_1 + value_2 + expected_3;
    constexpr int expected_7 = value_2 + expected_3 + value_4;
    constexpr int expected_8 = expected_3 + value_4 + value_5;
    constexpr int expected_9 = value_1 + value_2 + expected_3 + value_4 + value_5;
    constexpr int expected_0 = expected_6 + expected_7 + expected_8 + expected_9;

    constexpr std::array<int, 10> expected{
        expected_0, value_1, value_2, expected_3, value_4, value_5, expected_6, expected_7, expected_8, expected_9};

    if constexpr (tests::utils::is_graph_defaulted<execution_space>) {
        ASSERT_THAT((std::span{mirror.data(), mirror.size()}), testing::ElementsAreArray(expected));
    } else {
        for (size_t ielem = 0; ielem < mirror.size(); ++ielem) {
            if (mirror(ielem) != expected[ielem]) {
                std::cerr << "Value of: mirror(" << ielem << ")\n"
                          << "  Actual: " << mirror(ielem) << "\n"
                          << "Expected: " << expected[ielem] << std::endl;
            }
        }
        GTEST_SKIP() << "Backends for which the graph uses the vendor implementation are not expected to pass this "
                        "test due to race conditions in the usage of reduction scratch spaces.";
    }
}
TEST_F(GraphTest, fork_join_reduce) {
    test_fork_join_reduce(exec);
}

} // namespace tests::kokkos
