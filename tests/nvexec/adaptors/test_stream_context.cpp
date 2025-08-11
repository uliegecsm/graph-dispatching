#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsign-compare")
#include "exec/static_thread_pool.hpp"
#include "nvexec/stream_context.cuh"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c nvexec::stream_context
 * -----------------------------------
 *
 * This group of tests check the behavior of @c nvexec::stream_context.
 *
 * From our experiments, @c nvexec::stream_context is linked to a device (see
 * https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream_context.cuh#L306-L331
 * for instance). It has a pool of streams.
 * It seems that the execution resource of @c nvexec::stream_context is the whole device,
 * and its scheduler is allowed to create as many streams as it needs.
 *
 * The test can be found in @ref nvexec/adaptors/test_stream_context.cpp.
 */

namespace tests::nvexec::adaptors
{
class StreamContextTest : public ::testing::Test
{
protected:
    ::nvexec::stream_context stream_ctx{};
};

//! Load the value at @ref data and check it is equal to @ref prev. Then, add @ref value to it.
template <typename ValueType, bool OnDevice>
struct LoadCheckAddFunctor
{
    ValueType prev;
    ValueType value;
    ValueType* data;

    KOKKOS_FUNCTION
    void operator()() const
    {
        if constexpr (OnDevice) { KOKKOS_IF_ON_HOST  (Kokkos::abort("Bulk: you should not be running on host.");) }
        else                    { KOKKOS_IF_ON_DEVICE(Kokkos::abort("Bulk: you should not be running on device.");) }

        if(*data != prev) Kokkos::abort("Unexpected value.");
        *data += value;
    }
};

//! @test Check the forward progress guarantee of the @c nvexec::stream_context scheduler.
TEST_F(StreamContextTest, forward_progress_guarantee)
{
    const auto fpg = ::stdexec::get_forward_progress_guarantee(stream_ctx.get_scheduler());
    ASSERT_EQ(fpg, ::stdexec::forward_progress_guarantee::weakly_parallel);
}

/**
 * @test Simple test that checks that @c nvexec::stream_context respects workload dependencies.
 *
 * @note Because of https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/then.cuh#L28,
 *       we cannot pass a @c Kokkos::View to functors because it would make them non-trivially copyable.
 */
TEST_F(StreamContextTest, workload_dependencies)
{
    using value_t = unsigned int;
    using  view_t = Kokkos::View<value_t, Kokkos::SharedSpace>;

    const view_t witness(Kokkos::view_alloc("witness"));

    auto chain = stdexec::schedule(stream_ctx.get_scheduler())
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  0, .value = 4, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  4, .value = 2, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  6, .value = 6, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev = 12, .value = 9, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev = 21, .value = 8, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev = 29, .value = 1, .data = witness.data()});

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness(), 30);
}

//! Can be used for a @c bulk, only on device.
struct BulkFunctor
{
    unsigned short int id;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const
    {
        KOKKOS_IF_ON_HOST(Kokkos::abort("Bulk: This should not happen.");)
        Kokkos::printf("> %s with id %d: thread (%d,%d,%d): index %d\n", __PRETTY_FUNCTION__, id, threadIdx.x, threadIdx.y, threadIdx.z, index);
    }
};

//! Can be used for a @c then, only on device.
struct ThenFunctor
{
    unsigned short int id;

    KOKKOS_FUNCTION
    void operator()() const
    {
        KOKKOS_IF_ON_HOST(Kokkos::abort("Then: This should not happen.");)
        Kokkos::printf("> %s with id %d: thread (%d,%d,%d)\n", __PRETTY_FUNCTION__, id, threadIdx.x, threadIdx.y, threadIdx.z);
    }
};

/**
 * @test This test can be used to check how @c nvexec::stream_context deals with a @c stdexec::split.
 *
 * From inspection with @c nsys, it seems that in this case @c nvexec::stream_context will
 * create 4 streams.
 *
 * @note Heavily inspired by https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/examples/nvexec/split.cpp.
 */
TEST_F(StreamContextTest, split)
{
    ::stdexec::scheduler auto sch = stream_ctx.get_scheduler();

    constexpr size_t size = 4;

    auto fork = stdexec::schedule(sch) | stdexec::bulk(::stdexec::par, size, BulkFunctor{.id = 0}) | stdexec::split();

    /// Here, we mustn't use @c stdexec::when_all alone.
    /// Indeed, using @c stdexec::when_all alone would result in the last node being run on host.
    /// We could use @c nvexec::transfer_when_all, but it seems that a combination of @c stdexec::when_all
    /// and @c stdexec::continues_on works as well.
    ///
    /// This makes sense because, according to P2300, the sender returned by @c stdexec::when_all has no completion scheduler.
    /// We aren't sure why though, but we think it is because @c stdexec::when_all might receive several input senders that all
    /// use a different scheduler.
    ///
    /// References:
    ///     - https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/stdexec/__detail/__when_all.hpp#L453-L463
    ///     - https://github.com/pika-org/pika/blob/11dfb31d37e9395dc42416a53d39d46335f9fd70/libs/pika/execution/include/pika/execution/algorithms/transfer_when_all.hpp#L23
    auto snd = stdexec::when_all(
        fork | stdexec::bulk(::stdexec::par, size, BulkFunctor{.id = 1}),
        fork | stdexec::then(                      ThenFunctor{.id = 0}),
        fork | stdexec::bulk(::stdexec::par, size, BulkFunctor{.id = 2}))
             | stdexec::continues_on(sch)
             | stdexec::then(                      ThenFunctor{.id = 1});

    stdexec::sync_wait(std::move(snd));
}

//! @test Check that @c nvexec::stream_context correctly synchronizes the kernels before switching to @c exec::static_thread_pool.
TEST_F(StreamContextTest, move_to_static_thread_pool)
{
    using value_t = unsigned int;
    using  view_t = Kokkos::View<value_t, Kokkos::SharedSpace>;

    const view_t witness(Kokkos::view_alloc("witness"));

    exec::static_thread_pool pool {1};

    auto chain = stdexec::schedule(stream_ctx.get_scheduler())
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  0, .value = 4, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, true>{.prev =  4, .value = 2, .data = witness.data()})
        | stdexec::continues_on(pool.get_scheduler())
        | stdexec::then(LoadCheckAddFunctor<value_t, false>{.prev =  6, .value = 6, .data = witness.data()})
        | stdexec::then(LoadCheckAddFunctor<value_t, false>{.prev = 12, .value = 9, .data = witness.data()});

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness(), 21);
}

//! Transition from @p from to @p to, with a non-empty value channel.
template <typename From, typename To>
void test_transition_from_one_to_another_with_value(From&& from, To&& to)
{
    auto snd = ::stdexec::schedule(std::forward<From>(from))
        | ::stdexec::then([=] () -> int {
            if (::nvexec::is_on_gpu())
                return 1;
            else
                return 0;
        })
        | ::stdexec::continues_on(std::forward<To>(to))
        | ::stdexec::then([=](const int val) -> int {
            if (::nvexec::is_on_gpu() && val == 1)
                return val + 1;
            else
                return val;
        });

    const auto [result] = ::stdexec::sync_wait(std::move(snd)).value();

    ASSERT_EQ(result, 2);
}

//! Transition from @p from to @p to, with an empty value channel.
template <typename From, typename To>
void test_transition_from_one_to_another_no_value(From&& from, To&& to)
{
    Kokkos::View<int, Kokkos::SharedSpace> value("shared");

    auto* const ptr = value.data();

    auto snd = ::stdexec::schedule(std::forward<From>(from))
        | ::stdexec::then([=] {
            if (::nvexec::is_on_gpu())
                ++(*ptr);
        })
        | ::stdexec::continues_on(std::forward<To>(to))
        | ::stdexec::then([=] {
            if (::nvexec::is_on_gpu() && *ptr == 1)
                ++(*ptr);
        });

    ::stdexec::sync_wait(std::move(snd)).value();

    ASSERT_EQ(value(), 2);
}

//! @test Check that we can transition from one @c nvexec::stream_context to itself with a value channel.
TEST_F(StreamContextTest, transition_to_itself_with_value)
{
    test_transition_from_one_to_another_with_value(stream_ctx.get_scheduler(), stream_ctx.get_scheduler());
}

//! @test Check that we can transition from one @c nvexec::stream_context to itself without value channel.
TEST_F(StreamContextTest, transition_to_itself_no_value)
{
    test_transition_from_one_to_another_no_value(stream_ctx.get_scheduler(), stream_ctx.get_scheduler());
}

//! @test Check that we can transition from one @c nvexec::stream_context to another with value channel.
TEST_F(StreamContextTest, transition_to_another_with_value)
{
    /**
     * @todo Once https://github.com/NVIDIA/stdexec/issues/1563 is solved,
     *       we can really exercise this test.
     */
    GTEST_SKIP() << "It will seg. fault.";

    ::nvexec::stream_context stream_ctx_another{};

    test_transition_from_one_to_another_with_value(
        stream_ctx.get_scheduler(),
        stream_ctx_another.get_scheduler()
    );
}

//! @test Check that we can transition from one @c nvexec::stream_context to another without value channel.
TEST_F(StreamContextTest, transition_to_another_no_value)
{
    /**
     * @todo Once https://github.com/NVIDIA/stdexec/issues/1563 is solved,
     *       we can really exercise this test.
     */
    GTEST_SKIP() << "It will seg. fault.";

    ::nvexec::stream_context stream_ctx_another{};

    test_transition_from_one_to_another_no_value(
        stream_ctx.get_scheduler(),
        stream_ctx_another.get_scheduler()
    );
}

} // namespace tests::nvexec::adaptors
