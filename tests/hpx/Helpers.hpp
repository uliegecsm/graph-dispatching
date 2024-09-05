#ifndef GRAPH_DISPATCHING_TESTS_HPX_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_HPX_HELPERS_HPP

#include "gtest/gtest.h"

#include "hpx/init.hpp"

namespace utils::hpx
{

/**
 * @brief Initialize @c HPX.
 *
 * It will add a pool with a single thread. For that, it seems we need to allow the resource
 * partitioner to oversubscribe.
 *
 * @note It could be a singleton.
 */
struct HPX
{
    HPX()
    {
        ::hpx::init_params init_args;
        init_args.rp_mode     = ::hpx::resource::partitioner_mode::allow_oversubscription;
        init_args.rp_callback = std::bind(&HPX::init_resource_partitioner_handler, this, std::placeholders::_1);
        ::hpx::start(nullptr, 0, nullptr, init_args);
    }

    void init_resource_partitioner_handler(::hpx::resource::partitioner& rp)
    {
        rp.create_thread_pool(
            single_thread_pool_name,
            ::hpx::resource::scheduling_policy::unspecified,
            ::hpx::threads::policies::scheduler_mode::enable_stealing
        );
        rp.add_resource(::hpx::resource::pu(0), single_thread_pool_name);
    }

    ~HPX()
    {
        ::hpx::post([]() { ::hpx::finalize(); });
        ::hpx::stop();
    }

    static const std::string single_thread_pool_name;
};

const std::string HPX::single_thread_pool_name { "test-thread-pool-single" };

namespace test
{

/**
 * @brief Helper for tests that need to initialize/finalize @c HPX.
 *
 * It uses @ref utils::hpx::HPX under the hood.
 */
class HPXTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        this->hpx = std::make_shared<::utils::hpx::HPX>();
    }
protected:
    //! Use a shared pointer to delay the initialization to @ref SetUp.
    std::shared_ptr<::utils::hpx::HPX> hpx;
};

} // namespace test

} // namespace utils::hpx

#endif // GRAPH_DISPATCHING_TESTS_HPX_HELPERS_HPP
