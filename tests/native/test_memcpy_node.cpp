#include <filesystem>
#include <numeric>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda graph @c memcpy node
 * --------------------------------------
 *
 * This test shows how a @c Cuda graph @c memcpy node can be added.
 *
 * The test can be found in @ref native/test_memcpy_node.cpp.
 */

namespace tests::cuda
{
template <typename ViewType>
struct ElementWiseAdd
{
    ViewType src;
    ViewType dst;

    template <typename T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        dst(index) += src(index);
    }
};

//! @test Check how a @c Cuda graph @c memcpy node can be added.
TEST(cuda, memcpy_node)
{
    using scalar_t = int;
    using view_t   = View<scalar_t>;
    using sti_t    = SetToIndex<view_t>;
    using ewa_t    = ElementWiseAdd<view_t>;

    constexpr size_t size = 2<<11;

    const Stream stream;

    const view_t src(stream, size);
    const view_t dst(stream, size);

    const Graph graph;

    //! Initialize both views using @ref SetToIndex.
    const sti_t init_src_f{.data = src};
    GraphNodeKernel init_src_n(init_src_f, size);
    init_src_n.add(graph, {});

    const sti_t init_dst_f{.data = dst};
    GraphNodeKernel init_dst_n(init_dst_f, size);
    init_dst_n.add(graph, {});

    //! Index-wise add of elements in @c src into @c dst.
    const ewa_t ewa_f{.src = src, .dst = dst};
    GraphNodeKernel ewa_n(ewa_f, size);
    ewa_n.add(graph, {init_src_n, init_dst_n}); // NOLINT(cppcoreguidelines-slicing)

    //! Memory copy of @c dst into @c src.
    GraphNodeMemcpy<scalar_t> memcpy(src.buffer, dst.buffer, size, PREFIXED_API(MemcpyDeviceToDevice));
    memcpy.add(graph, {ewa_n}); // NOLINT(cppcoreguidelines-slicing)

    const GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);

    //! Check the values in @c dst and @c src are identical (and as expected).
    const auto src_h = src.get_host_copy(stream);
    const auto dst_h = dst.get_host_copy(stream);
    stream.fence();

    std::vector<scalar_t> expected(size);
    std::iota(expected.begin(), expected.end(), 0);

    ASSERT_EQ(dst_h, src_h);
    ASSERT_EQ(dst_h, expected);

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_memcpy_node.dot").c_str(), PREFIXED_API(GraphDebugDotFlagsVerbose));
}

} // namespace tests::cuda
