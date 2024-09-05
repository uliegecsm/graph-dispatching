#include "gtest/gtest.h"

#include "hpx/execution.hpp"
#include "hpx/init.hpp"
#include "hpx/thread.hpp"

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
//! Initialize @c HPX. It could be a singleton.
struct HPX
{
    HPX()
    {
        ::hpx::start(nullptr, 0, nullptr, ::hpx::init_params {});
    }

    ~HPX()
    {
        ::hpx::post([]() { ::hpx::finalize(); });
        ::hpx::stop();
    }
};

class HPXTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        this->hpx = std::make_shared<HPX>();
    }
protected:
    //! Use a shared pointer to delay the initialization to @ref SetUp.
    std::shared_ptr<HPX> hpx;
};

TEST_F(HPXTest, hello_world)
{
    namespace execution = ::hpx::execution::experimental;

    //! Retrieve the default @c HPX thread pool.
    ::hpx::threads::thread_pool_base& pool = ::hpx::resource::get_thread_pool("default");

    std::cout << "> Pool name           : " << pool.get_pool_name()       << std::endl;
    std::cout << "> Pool OS thread count: " << pool.get_os_thread_count() << std::endl;
    pool.print_pool(std::cout);

    //! Get a scheduler for that thread pool with asynchronous execution.
    [[maybe_unused]]execution::thread_pool_scheduler scheduler(&pool, ::hpx::launch::async);

    //! Say hello to the world from within a chain of senders.
    std::atomic<std::size_t> count{0};

    auto work = execution::schedule(scheduler) | execution::then([&]() {
        std::cout << "Hello, world !" << std::endl;
        ++count;
    });

    ::hpx::this_thread::experimental::sync_wait(work);

    ASSERT_EQ(count, 1);
}

} // namespace tests::hpx
