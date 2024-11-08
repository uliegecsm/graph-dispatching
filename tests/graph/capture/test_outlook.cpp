#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "kokkos_ext/Algorithms.hpp"
#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

/**
 * @addtogroup unittests
 *
 * Capturing with @c Kokkos::Graph
 * -------------------------------
 *
 * Create a simple graph that will capture kernels from external libraries.
 *
 * For backends such as @c Cuda or @c HIP, it uses stream capture.
 *
 * The test can be found in @ref capture/test_outlook.cpp.
 */

namespace tests::graph::capture
{

namespace external
{
//! Helper for @ref SumWithState.
template <typename T>
__global__
void sum_with_state_impl(T const * data, T* sum, T const * buffer)
{
    Kokkos::atomic_add(
        sum,
        data[threadIdx.y] + buffer[threadIdx.y]
    );
}

} // namespace external

//! Helper for @ref SumWithState.
template <typename ViewType, typename BufferType>
struct SumWithStateImpl
{
    ViewType   data;
    BufferType state;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index, typename ViewType::non_const_value_type& current) const {
        current += state(index) + data(index);
    }
};

//! Sum of @ref data. We use @ref SumWithState so @ref buffer must be kept alive.
template <
    bool UseKokkos,
    typename ViewType,
    typename BufferType = Kokkos::View<typename ViewType::non_const_value_type*, typename ViewType::memory_space>
>
struct SumWithState
{
    ViewType   data;
    BufferType state {};

    template <typename Exec, typename Result>
    decltype(auto) apply(Exec&& exec, Result&& result)
    {
        //! Due to capture constraints, allocation must be done outside of capture scope.
        state = BufferType(Kokkos::view_alloc(Kokkos::Experimental::graph::get_exec(exec), "state"), data.size());

        std::cout << "> In class, state ptr is " << state.data() << std::endl;

        /// When we use @c Kokkos, the @c Kokkos parallel constructs will always create a graph node that stores the functor.
        /// Therefore, keeping the graph node alive is sufficient for @ref state to stay alive.
        if constexpr (UseKokkos) {
            return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_reduce(
                Kokkos::RangePolicy(0, data.size()),
                SumWithStateImpl{.data = data, .state = state},
                Kokkos::Sum(std::forward<Result>(result))
            );
        /// Otherwise, we mimic graph capture
        // using kkkos tools I see that the memory is deallocated but it still runs fine on Cuda.
        } else {
            const dim3 block(1, 128, 1);
            const dim3 grid (1,   1, 1);
            if(data.size() != 128) std::abort();
            external::sum_with_state_impl<<<grid, block, 0, Kokkos::Experimental::graph::get_exec(exec).cuda_stream()>>>(data.data(), result.data(), state.data());
            return std::forward<Exec>(exec);
        }
    }
};

template <typename T>
class GraphCaptureTest : public ::testing::Test
{
public:
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    using value_t = int;

    using data_t = Kokkos::View<value_t*, memory_space>;

    static constexpr size_t size = 128;

public:
    void SetUp() override
    {
        this->data = data_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data", exec), size);

        KokkosExt::fill_sequence(exec, data, 0);
    }
protected:
    execution_space exec {};
    data_t data;
};

using GraphCaptureTestTypes = ::testing::Types<
    std::integral_constant<bool, true>,
    std::integral_constant<bool, false>
>;

TYPED_TEST_SUITE(GraphCaptureTest, GraphCaptureTestTypes);

//! @test Graph capture with @c Kokkos.
TYPED_TEST(GraphCaptureTest, outlook)
{
    decltype(auto) root = Kokkos::Experimental::graph::create_graph(this->exec);

    Kokkos::View<typename TestFixture::value_t, Kokkos::SharedSpace> result(Kokkos::view_alloc(Kokkos::WithoutInitializing, "result", this->exec));

    void * ptr;

    {
        auto tmp = SumWithState<TypeParam::value, typename TestFixture::data_t>{.data = this->data};
        [[maybe_unused]]decltype(auto) node = tmp.apply(std::move(root), result);
        ptr = tmp.state.data();
    }

    std::cout << "> PTR is " << ptr << std::endl;
    cudaPointerAttributes attrs;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaPointerGetAttributes(&attrs, ptr));
    ASSERT_EQ(attrs.devicePointer, nullptr);

    for(size_t irep = 0 ; irep < 1; ++irep)
    {
        Kokkos::Experimental::graph::submit(this->exec, root);

        this->exec.fence("Waiting for the graph to complete.");

        ASSERT_EQ(result(), (TestFixture::size - 1) * TestFixture::size / 2);
    }
}

#if defined(NOT_DEFINED)
//! Add a task.
template <typename Exec, typename Task, typename Data>
decltype(auto) Kokkos::graph::task(const Exec& exec, Task&& task, Data&& data);

//! Specialization when @p exec is not a graph: "eager execution".
template <typename Exec, typename Task, typename Data> requires ( ! graph-like exec )
decltype(auto) Kokkos::graph::task(const Exec& exec, Task&& task, Data&& data)
{
    std::forward<Task>(task)(exec, std::forward<Data>(data));
    //! If data is given, we have guarenteed that it would be kept alive until completion.
    if constexpr (data not empty and not moved) exec.fence();
    return exec;
}

//! Specialization when @p exec is a @c Kokkos defaulted graph (@c Serial or @c OpenMP for instance).
template <typename Exec, typename Task, typename Data> requires ( defaulted graph exec )
decltype(auto) Kokkos::graph::task(const Exec& exec, Task&& task, Data&& data)
{
    /// 1. Create graph node with @p task and @p data. Add it to graph.
    ///    Since @p data is grafted it to graph, its lifetime is bounded to the one of graph (and thereby to the node's lifetime).
    return node;
}

//! Specialization when @p exec is a @c Kokkos specialized graph (@c Cuda or @c HIP for instance).
template <typename Exec, typename Task, typename Data> requires ( specialized graph exec )
decltype(auto) Kokkos::graph::task(const Exec& exec, Task&& task, Data&& data)
{
    /// 1. Create graph node(s) by executing @p task and using stream capture.
    std::forward<Task>(task)(exec, data);
    /// 2. Create data handle with @p data. Graft it to graph to bind its lifetime to the one of graph.
    return node;
}
#endif

} // namespace tests::graph::capture
