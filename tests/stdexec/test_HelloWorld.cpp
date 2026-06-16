#include <atomic>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Simple "Hello, world !" example with @c stdexec
 * -----------------------------------------------
 *
 * Reproduce a slightly adapted version of the "Hello, world !" example from https://github.com/NVIDIA/stdexec/blob/main/examples/hello_world.cpp.
 *
 * The test can be found in @ref stdexec/test_HelloWorld.cpp.
 */

namespace tests::stdexec {

constexpr size_t parallel = 16;

template <::stdexec::sender Sender>
::stdexec::sender auto tester(Sender&& start) {
    //! Enqueue a sender that prints to console, and returns an vector with a single value.
    ::stdexec::sender auto handshake = ::stdexec::then(std::forward<Sender>(start), []() -> std::vector<int> {
        std::printf("Hello world! Have a vector of int.\n"); // NOLINT(modernize-use-std-print)
        return {13};
    });

    /// Add 42 to the preceding sender's output's first element and return the vector.
    /// @note Please note that the lambda has to take by @c && because we're nicely moving things around.
    ::stdexec::sender auto add_42 = ::stdexec::then(std::move(handshake), [](std::vector<int> arg) -> std::vector<int> {
        arg.at(0) += 42;
        return arg;
    });

    //! Perform some bulky operation. Each work item atomically adds its index to the first element of the input.
    ::stdexec::sender auto bulky =
        ::stdexec::bulk(std::move(add_42), ::stdexec::par, parallel, [](size_t index, std::vector<int>& arg) {
            std::atomic_ref<int>(arg.at(0)) += index;
        });

    /// Retrieve the first element from the input.
    /// @note Again, things are nicely moved around.
    return ::stdexec::then(std::move(bulky), [](std::vector<int> arg) -> int { return arg.at(0); });
}

//! @test Simple @c stdexec test that just aims at showing what can be done.
TEST(stdexec, hello_world) {
    //! Retrieve the NUMA configuration, and get the number of CPUs on the first node for sizing the thread pool.
    const exec::numa_policy numa(exec::no_numa_policy{});
    exec::static_thread_pool context(numa.num_cpus(0)); // NOLINT(misc-const-correctness)

    //! Get a scheduler from the thread pool.
    ::stdexec::scheduler auto scheduler = context.get_scheduler();

    //! Begin chain of workloads.
    ::stdexec::sender auto begin = ::stdexec::schedule(scheduler);
    ::stdexec::sender auto contd = tester(begin);

    //! Run workloads. Retrieve and check the result.
    auto [result] = ::stdexec::sync_wait(std::move(contd)).value(); // NOLINT(performance-move-const-arg)

    static_assert(std::same_as<decltype(result), int>);

    auto sum_in_interval = []<typename T, typename U>(const T bound_a, const U bound_b) {
        return bound_b * (bound_b + 1) / 2 - bound_a * (bound_a + 1) / 2 + bound_a;
    };

    ASSERT_EQ(result, 13 + 42 + sum_in_interval(0, parallel - 1));
}

} // namespace tests::stdexec
