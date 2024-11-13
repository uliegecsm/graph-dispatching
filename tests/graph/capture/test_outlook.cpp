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

template <typename ViewType>
struct Increment
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const { ++data(index); }
};

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

//! We reset the @p result to 0, to make it work when we re-submit.
template <typename Exec, typename ViewType, typename ResultType, typename BufferType>
void sum_with_state_impl(const Exec& exec, const ViewType& data, const ResultType& result, const BufferType& buffer)
{
    if(data.size() < 128 || data.size() > 1024) Kokkos::abort("Unsupported size.");

    const dim3 block(1, data.size(), 1);
    const dim3 grid (1,           1, 1);

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemsetAsync(result.data(), 0, sizeof(typename ResultType::value_type), exec.cuda_stream()));
    external::sum_with_state_impl<<<grid, block, 0, exec.cuda_stream()>>>(data.data(), result.data(), buffer.data());
}
} // namespace external

namespace impl
{
//! Helper for @ref SumWithState, when we use a @c Kokkos parallel construct.
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
} // namespace impl

/**
 * @brief Sum of @ref data.
 *
 * If @c UseKokkos is @c true, we use the functor @ref impl::SumWithState. Therefore, @ref buffer will be kept alive
 * implicitly through @ref impl::SumWithState as long as the @c Kokkos node will live.
 *
 * If @c UseKokkos is @c false, we use the function @ref external::sum_with_state. Therefore, the @ref buffer ownership
 * has to be extended somehow.
 *
 * See definition of @ref apply for more details.
 */
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

        return apply_impl(std::forward<Exec>(exec), std::forward<Result>(result));
    }

    /// @brief For any execution space and using @c Kokkos.
    /// When we use @c Kokkos, the @c Kokkos parallel constructs will always create a graph node that stores the functor.
    /// Therefore, keeping the graph node alive is sufficient for @ref state to stay alive.
    template <typename Exec, typename Result> requires UseKokkos
    decltype(auto) apply_impl(Exec&& exec, Result&& result)
    {
        return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_reduce(
            Kokkos::RangePolicy<typename std::remove_cvref_t<Exec>::execution_space>(0, data.size()),
            impl::SumWithStateImpl{.data = data, .state = state},
            Kokkos::Sum(std::forward<Result>(result))
        );
    }

    /// @brief For @c Serial and not using @c Kokkos.
    /// The lambda will be the node's workload.
    template <typename Exec, typename Result> requires (std::same_as<typename std::remove_cvref_t<Exec>::execution_space, Kokkos::Serial> && !UseKokkos)
    decltype(auto) apply_impl(Exec&& exec, Result&& result)
    {
        return std::forward<Exec>(exec) | Kokkos::Experimental::graph::then<Kokkos::Serial>(
            [data_ = data, result_ = std::forward<Result>(result), state_ = state](const Kokkos::Serial&) {
                typename std::remove_cvref_t<Result>::value_type acc = 0;
                for(size_t ielem = 0; ielem < data_.size(); ++ielem)
                    acc += data_(ielem) + state_(ielem);
                result_() = acc;
        }); 
    }

    /// @brief For @c Cuda and not using @c Kokkos.
    /// We use graph capture to define the node's workload.
    template <typename Exec, typename Result> requires (std::same_as<typename std::remove_cvref_t<Exec>::execution_space, Kokkos::Cuda> && !UseKokkos)
    decltype(auto) apply_impl(Exec&& exec, Result&& result)
    {
        return std::forward<Exec>(exec) | Kokkos::Experimental::graph::then<Kokkos::Cuda>(
            [data_ = data, result_ = result, state_ = state](const Kokkos::Cuda& exec_){
            external::sum_with_state_impl(exec_, data_, result_, state_);
        });
    }
};

template <typename T>
class GraphCaptureTest : public ::testing::Test
{
public:
    using execution_space = std::tuple_element_t<1, T>;
    using memory_space    = typename execution_space::memory_space;

    using value_t = int;

    using data_t = Kokkos::View<value_t*, memory_space>;

    static constexpr bool use_kokkos_par = std::tuple_element_t<0, T>::value;

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
    std::tuple<std::integral_constant<bool, true> , Kokkos::Serial>,
    std::tuple<std::integral_constant<bool, false> , Kokkos::Serial>,
    std::tuple<std::integral_constant<bool, true> , Kokkos::DefaultExecutionSpace>,
    std::tuple<std::integral_constant<bool, false>, Kokkos::DefaultExecutionSpace>
>;

TYPED_TEST_SUITE(GraphCaptureTest, GraphCaptureTestTypes);

//! @test Graph capture with @c Kokkos.
TYPED_TEST(GraphCaptureTest, outlook)
{
    decltype(auto) root = Kokkos::Experimental::graph::create_graph(this->exec);

    decltype(auto) add = std::forward<decltype(root)>(root) | Kokkos::Experimental::graph::parallel_for(
        Kokkos::RangePolicy<typename TestFixture::execution_space>(0, this->data.size()),
        Increment{.data = this->data}
    );

    using result_mem_t = std::conditional_t<std::same_as<typename TestFixture::execution_space, Kokkos::Serial>, Kokkos::HostSpace, Kokkos::SharedSpace>;
    Kokkos::View<typename TestFixture::value_t, result_mem_t> result(Kokkos::view_alloc(Kokkos::WithoutInitializing, "result", this->exec));

    [[maybe_unused]]decltype(auto) node = SumWithState<
        TestFixture::use_kokkos_par,
        typename TestFixture::data_t
    >{.data = this->data}.apply(std::forward<decltype(add)>(add), result);

    for(size_t irep = 0 ; irep < 10; ++irep)
    {
        Kokkos::Experimental::graph::submit(this->exec, root);

        this->exec.fence("Waiting for the graph to complete.");

        auto sum_in_interval = [] <typename T, typename U> (const T bound_a, const U bound_b) {
            return bound_b * ( bound_b + 1 ) / 2 - bound_a * ( bound_a + 1 ) / 2 + bound_a;
        };

        ASSERT_EQ(result(), sum_in_interval(1 + irep, TestFixture::size + irep));
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


// Pseudo code - 0 - no graph
// --------------------------

// cuSparseSpVV(...)

// cuBLAS(...)

// Pseudo code - 1 - native graph capture
// --------------------------------------

// cudaGraph_t graph;

// stream_capture_start(...)
//     cuSparseSpVV(...)
//     cuBLAS(...)
// stream_capture_stop(...)

// gaph.add_node(captured_subgraph)

// Pseudo code - 2 - Kokkos captured
// ---------------------------------

// Kokkos::Graph graph;

// root = gaph.root;

// node_1 = root | then( cuSparseSpVV(...) )

// node_2 = node_1 | then( cuBLAS(...) )