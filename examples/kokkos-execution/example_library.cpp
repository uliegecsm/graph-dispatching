#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"
#include "kokkos_ext/impl/GraphContext.hpp"

/**
 * @addtogroup examples
 *
 * User code calling a library with @c Kokkos
 * ------------------------------------------
 *
 * Some user code calls a library. The same source code can map to either the execution space model
 * or the graph model,
 * using @c stdexec customizations for @c Kokkos execution space or @c Kokkos::Experimental::Graph.
 *
 * The examples can be found in @ref examples/kokkos-execution/example_library.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace examples::KokkosExecution::library {

template <char ID>
struct Greetings {
    int value;

    template <typename... Args>
    requires(std::integral<Args> && ...)
    KOKKOS_FUNCTION void operator()(const Args...) const {
        Kokkos::printf("Greetings from %c with value %d.\n", ID, value); // NOLINT(modernize-use-std-print)
    }
};

//! Some library function.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
auto library_func(const Schd& schd, Sndr&& sndr) -> stdexec::sender auto {
    return std::forward<Sndr>(sndr)
         | exec::fork_join(
               stdexec::continues_on(schd) | stdexec::bulk(stdexec::par, 1, Greetings<'B'>{.value = 42}),
               stdexec::continues_on(schd) | stdexec::then(Greetings<'C'>{.value = 123}));
}

//! Some user function.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
auto user_func(const Schd& schd, Sndr&& sndr) -> stdexec::sender auto {
    auto pre = std::forward<Sndr>(sndr) | stdexec::continues_on(schd)
             | stdexec::bulk(stdexec::par, 1, Greetings<'A'>{.value = 666});
    /// @note Though the @ref library_func returns a @c exec::fork_join, thus making the @c stdexec::continues_on optional,
    ///       it seems good practice to either unconditionally add @c stdexec::continues_on, or check if it's needed
    ///       by querying the completion scheduler of the sender returned by @ref library_func.
    return library_func(schd, std::move(pre)) | stdexec::continues_on(schd)
         | stdexec::bulk(stdexec::par, 1, Greetings<'D'>{.value = 31415});
}

#define EXPECTED_GREETINGS                                                                                             \
    {                                                                                                                  \
        const auto& output = ::testing::internal::GetCapturedStdout();                                                 \
        ASSERT_THAT(                                                                                                   \
            output,                                                                                                    \
            ::testing::MatchesRegex(                                                                                   \
                "Greetings from A with value 666.\nGreetings from (B|C) with value (42|123).\nGreetings from (B|C) "   \
                "with value (42|123).\nGreetings from D with value 31415.\n"));                                        \
        ASSERT_THAT(                                                                                                   \
            output,                                                                                                    \
            ::testing::AllOf(                                                                                          \
                ::testing::HasSubstr("Greetings from B with value 42."),                                               \
                ::testing::HasSubstr("Greetings from C with value 123.")));                                            \
    }

struct LibraryTest
    : public ::testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<execution_space> { };

//! @test Run @ref user_func with @ref Kokkos::Experimental::ExecutionSpaceContext.
TEST_F(LibraryTest, execution_space) {
    const Kokkos::Experimental::ExecutionSpaceContext<execution_space> ectx{exec};

    auto scheduler = ectx.get_scheduler();

    ::testing::internal::CaptureStdout();

    stdexec::sync_wait(user_func(scheduler, stdexec::just()));

    EXPECTED_GREETINGS
}

//! @test Run @ref user_func with @ref Kokkos::Experimental::GraphContext.
TEST_F(LibraryTest, graph) {
    const Kokkos::Experimental::GraphContext<execution_space> gctx{exec};

    auto scheduler = gctx.get_scheduler();

    ::testing::internal::CaptureStdout();

    stdexec::sync_wait(user_func(scheduler, stdexec::just()));

    EXPECTED_GREETINGS
}

} // namespace examples::KokkosExecution::library
