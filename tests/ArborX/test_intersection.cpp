#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/ArborX/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Simple example using @c ArborX
 * ------------------------------
 *
 * Simple integration test for @c ArborX, heavily inspired by
 * https://github.com/arborx/ArborX/blob/78e754883d8c1e6275034d62a419f62ad3da950b/examples/simple_intersection/example_intersection.cpp.
 *
 * The test can be found in @ref ArborX/test_intersection.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space    = typename execution_space::memory_space;

using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::ArborX
{

//! @test Find which boxes intersect points using @c ArborX queries.
TEST(ArborX, intersection)
{
    const execution_space      exec   {};
    const host_execution_space exec_h {};

    //! Create 2D boxes.
    const auto boxes = create_boxes<memory_space>(exec, exec_h);

    //! Create the queries for 2D points.
    using point_t          = ::ArborX::Point<2, double>;
    using intersect_view_t = Kokkos::View<::ArborX::Intersects<point_t>*, memory_space>;

    const intersect_view_t queries(Kokkos::view_alloc("queries", Kokkos::WithoutInitializing, exec), 3);

    const auto queries_h = Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec_h, Kokkos::HostSpace{}), queries);
    queries_h[0] = ::ArborX::intersects(point_t{1.8, 1.5});
    queries_h[1] = ::ArborX::intersects(point_t{1.3, 1.7});
    queries_h[2] = ::ArborX::intersects(point_t{1, 1});
    Kokkos::deep_copy(exec, queries, queries_h);

    /// Create a boundary volume hierarchy, used to accelerate the search for geometrical objects in space.
    const ::ArborX::BoundingVolumeHierarchy tree(exec, ::ArborX::Experimental::attach_indices(boxes));

    //! The query will resize indices and offsets accordingly.
    using int_view_t = Kokkos::View<int *, memory_space>;
    int_view_t indices("intersections - indices", 0);
    int_view_t offsets("intersections - offsets", 0);
    tree.query(exec, queries, indices, offsets);

    /// Expected output:
    ///   offsets: 0 1 2 6
    ///   indices: 3 3 0 2 1 3
    /// The order of the last 4 indices may vary.
    const auto offsets_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, offsets); Kokkos::deep_copy(exec, offsets_h, offsets);
    const auto indices_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, indices); Kokkos::deep_copy(exec, indices_h, indices);

    ASSERT_EQ(indices_h.size(), 6);
    ASSERT_EQ(offsets_h.size(), 4);

    ASSERT_EQ(indices_h(0), 3);
    ASSERT_EQ(indices_h(1), 3);
    ASSERT_THAT(
        (std::span{indices_h.data() + 2, 4}),
        ::testing::UnorderedElementsAre(0, 2, 1, 3)
    );
}

} // namespace tests::ArborX
