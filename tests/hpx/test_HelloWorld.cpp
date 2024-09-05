#include "hpx/execution.hpp"
#include "hpx/thread.hpp"

#include "tests/hpx/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Simple "Hello, world !" example with @c HPX
 * -------------------------------------------
 *
 * A very simple usage of @c HPX *à la* P2300.
 *
 * The test can be found in @ref hpx/test_HelloWorld.cpp.
 */

namespace tests::hpx
{

using ::utils::hpx::test::HPXTest;

TEST_F(HPXTest, hello_world)
{
    namespace execution = ::hpx::execution::experimental;

    //! Retrieve the default @c HPX thread pool.
    ::hpx::threads::thread_pool_base& pool = ::hpx::resource::get_thread_pool("default");

    std::cout << "> Pool name           : " << pool.get_pool_name()       << std::endl;
    std::cout << "> Pool OS thread count: " << pool.get_os_thread_count() << std::endl;
    pool.print_pool(std::cout);

    //! Get a scheduler for that thread pool with asynchronous execution.
    execution::thread_pool_scheduler scheduler(&pool, ::hpx::launch::async);

    //! Say hello to the world from within a chain of senders.
    std::atomic<std::size_t> count{0};

    auto work = execution::schedule(scheduler) | execution::then([&]() {
        std::cout << "Hello, world !" << std::endl;
        ++count;
    });

    ::hpx::this_thread::experimental::sync_wait(std::move(work));

    ASSERT_EQ(count, 1);
}

} // namespace tests::hpx
