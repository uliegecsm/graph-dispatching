#ifndef GRAPH_DISPATCHING_TESTS_CUDA_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_HELPERS_HPP

namespace tests::cuda
{

template <typename view_t>
struct MyFunctor
{
    view_t data;

    KOKKOS_FUNCTION
    void operator()(const unsigned int index) const {
        atomicAdd(&data(index), 1);
    }
};

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_HELPERS_HPP
