#include "gtest/gtest.h"

#include "kokkos_ext/Algorithms.hpp"
#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/kokkos_ext/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * @c Kokkos extensions for graph-compatible parallel-reduce construct
 * -------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos_Graph_Execution.hpp effectively
 * makes it possible to use a parallel-reduce construct in a templated code in either
 * graph or execution space instance mode transparently.
 *
 * The tests can be found in @ref kokkos_ext/test_parallel_reduce.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space    = typename Kokkos::DefaultExecutionSpace::memory_space;

namespace tests::kokkos_ext
{

//! Dummy function that can be transparently used with graph or execution space instance.
template <bool label, typename Sender, typename ViewType, typename ReducerType>
decltype(auto) my_function(Sender&& sender, const ViewType& data, ReducerType&& reducer)
{
    using policy_t = Kokkos::RangePolicy<typename std::remove_reference_t<Sender>::execution_space>;

    #define MY_FUNCTION_CORE(...)                                                           \
        return std::forward<Sender>(sender) | Kokkos::Experimental::graph::parallel_reduce( \
            __VA_ARGS__ __VA_OPT__(,)                                                       \
            policy_t(0, data.size()),                                                       \
            MyDummyFunctor{.data = data},                                                   \
            std::forward<ReducerType>(reducer)                                              \
        );
    if constexpr (label) MY_FUNCTION_CORE("this is a test parallel-reduce")
    else                 MY_FUNCTION_CORE()
}

template <typename T>
class ParallelReduceTest : public ::testing::Test
{
public:
    static constexpr size_t size = 2<<9;

    using view_t = Kokkos::View<const int[size], memory_space>;

    using result_t = typename view_t::non_const_value_type;

public:
    void SetUp() override
    {
        this->execs = Kokkos::Experimental::partition_space(execution_space {}, 1, 1);

        typename view_t::non_const_type tmp(Kokkos::view_alloc("data", Kokkos::WithoutInitializing, execs.at(0))); // NOLINT(misc-const-correctness)
        KokkosExt::fill_sequence(execs.at(0), tmp, 0);
        this->data = std::move(tmp); // NOLINT(performance-move-const-arg)
        execs.at(0).fence("Ensure that the setup is finished before running the test.");
    }
protected:
    std::vector<execution_space> execs;
    view_t                       data;
};

using ParallelReduceTestTypes = ::testing::Types<
    std::integral_constant<bool, true>,
    std::integral_constant<bool, false>
>;

TYPED_TEST_SUITE(ParallelReduceTest, ParallelReduceTestTypes);

//! Call @ref tests::kokkos_ext::my_function with a given reducer in @p __mem__ targeting @p __target__, scheduled on @p __on__.
#define CALL_FUNCTION(__on__, __mem__, __target__)                      \
    using sum_t = Kokkos::Sum<typename TestFixture::result_t, __mem__>; \
    decltype(auto) tail = my_function<TypeParam::value>(                \
        __on__, this->data,                                             \
        sum_t(__target__));

/**
 * @test Check the execution space instance mode for the parallel-reduce construct.
 *
 * @note It is fine to build the reduction target view from the host scalar. In fact, this is even required
 *       for this test to be semantically legal. Indeed, since @ref Kokkos::Experimental::graph::details::PartialAlgorithm
 *       cannot modify the policy to pass it the execution space instance, relying on the reduction target
 *       being a scalar ensures that @c Kokkos will internally fence before returning control flow.
 */
TYPED_TEST(ParallelReduceTest, exec)
{
    typename TestFixture::result_t reduced_sum = 0;
    CALL_FUNCTION(this->execs.at(0), Kokkos::HostSpace, reduced_sum)

    static_assert(std::same_as<decltype(tail), execution_space&>);

    ASSERT_EQ(std::addressof(this->execs.at(0)), std::addressof(tail)) << "You abused of the execution space instance.";

    Kokkos::Experimental::submit(this->execs.at(0), std::move(tail));

    this->execs.at(0).fence("Ensure reduction is finished before checking the result.");

    ASSERT_EQ(reduced_sum, TestFixture::size * (TestFixture::size - 1) / 2);
}

/**
 * @test Check the graph mode for the parallel-reduce construct.
 *
 * @note Compared to @ref ParallelReduceTest_exec_Test, we must store the result in a device view
 *       that is not built from a host pointer on our @c VOLTA70 machine.
 *       For our @c AMPERE86 machine, it would be fine using the same approach as in @ref ParallelReduceTest_exec_Test
 *       because it supports pageable memory access (see https://docs.nvidia.com/cuda/cuda-c-programming-guide/#system-requirements-for-unified-memory).
 *       Note that it's probably through HMM, see https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/.
 *       See also https://gist.github.com/romintomasetti/b8472f574e1407096466e55aede8bfd7.
 */
TYPED_TEST(ParallelReduceTest, graph)
{
    decltype(auto) root = Kokkos::Experimental::graph::create_graph(this->execs.at(0));

    typename TestFixture::result_t reduced_sum = 0;
    const Kokkos::View<typename TestFixture::result_t, memory_space> reduction_result(Kokkos::view_alloc("reduction result on device", this->execs.at(0)));
    CALL_FUNCTION(root, memory_space, reduction_result)

    static_assert(Kokkos::Impl::is_specialization_of<decltype(tail), Kokkos::Experimental::GraphNodeRef>::value);

    this->execs.at(0).fence("Ensure that the graph is ready to be submitted.");

    Kokkos::Experimental::graph::submit(this->execs.at(1), std::move(tail));

    Kokkos::deep_copy(this->execs.at(1), reduced_sum, reduction_result);
    this->execs.at(1).fence();
    ASSERT_EQ(reduced_sum, TestFixture::size * (TestFixture::size - 1) / 2);
}

#define CHECK_RESULT_AFTER_SUBMIT(__on__, __plus__)     \
    {                                                   \
        value_t tmp = 0;                                \
        Kokkos::deep_copy(__on__, tmp, result);         \
        exec.fence();                                   \
        std::cout << tmp << std::endl;                  \
        ASSERT_EQ(tmp, size * (size - 1) / 2 __plus__); \
    }

/**
 * @test Submit the graph twice, and ensure that the reducer behaves correctly.
 *
 * This test seeks to ensure that the reducer value from the first submission will be "reset" (or overriden)
 * during the second submission (they won't add up).
 */
TEST(ParallelReduce, submit_twice)
{
    constexpr size_t size = 2<<9;

    using value_t = int;
    using  view_t = Kokkos::View<value_t[size], memory_space>;

    const execution_space exec {};

    const view_t data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data", exec));
    KokkosExt::fill_sequence(exec, data, 0);

    const Kokkos::View<value_t, memory_space> result(Kokkos::view_alloc(Kokkos::WithoutInitializing, "result", exec));

    decltype(auto) root = Kokkos::Experimental::graph::create_graph(exec);

    decltype(auto) reduce = root | Kokkos::Experimental::graph::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, size),
        MyDummyFunctor{.data = data},
        Kokkos::Sum<value_t, memory_space>(result)
    );

    Kokkos::Experimental::graph::submit(exec, reduce);

    CHECK_RESULT_AFTER_SUBMIT(exec,)

    KokkosExt::fill_sequence(exec, data, 10);

    Kokkos::Experimental::graph::submit(exec, reduce);

    CHECK_RESULT_AFTER_SUBMIT(exec, + size * 10)
}

} // namespace tests::kokkos_ext
