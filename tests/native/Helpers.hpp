#ifndef GRAPH_DISPATCHING_TESTS_CUDA_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_HELPERS_HPP

namespace tests::cuda
{

template <typename view_t, bool Atomic = true>
struct MyFunctor
{
    view_t data;

    KOKKOS_FUNCTION
    void operator()(const unsigned int index) const  requires Atomic {
        atomicAdd(&data(index), 1);
    }

    KOKKOS_FUNCTION
    void operator()(const unsigned int index) const requires ( !Atomic ) {
        ++data(index);
    }
};

template <typename ViewType>
struct SetToIndex
{
    ViewType data;

    template <typename T>
    KOKKOS_FUNCTION
    void operator()(const T index) const { data(index) = index; }
};

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_HELPERS_HPP
