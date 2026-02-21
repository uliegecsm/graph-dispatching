#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::let_value
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::let_value.
 *
 * The test can be found in @ref stdexec/adaptors/test_let_value.cpp.
 */

namespace tests::stdexec::adaptors {

class LetValueTest
    : public utils::StaticThreadPool<'A', 'B', 'C', 'D'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    static constexpr size_t index_of_C = index_of<'C'>();
    static constexpr size_t index_of_D = index_of<'D'>();
};

template <bool Throws>
struct Dummy {
    void operator()() const noexcept(!Throws) {
    }
};

/**
 * @test Completion signatures of @c stdexec::let_value.
 *
 * This test shows that the completion signature of @c stdexec::let_value depends
 * both on the @c noexcept -ness of its lambda, but also on the completion signatures
 * of the returned sender.
 */
constexpr bool test_completion_signatures() {
    //! The lambda is @c noexcept, the functor is @c noexcept callable.
    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(::stdexec::just() | ::stdexec::let_value([]() noexcept {
                               return ::stdexec::just() | ::stdexec::then(Dummy<false>{});
                           })),
                  ::stdexec::__mset<::stdexec::set_value_t()>,
                  ::stdexec::env<>
    >);

    //! The lambda is not @c noexcept, the functor is @c noexcept callable.
    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(::stdexec::just() | ::stdexec::let_value([]() {
                               return ::stdexec::just() | ::stdexec::then(Dummy<false>{});
                           })),
                  ::stdexec::__mset<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>,
                  ::stdexec::env<>
    >);

    //! The lambda is @c noexcept, the functor is not @c noexcept callable.
    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(::stdexec::just() | ::stdexec::let_value([]() noexcept {
                               return ::stdexec::just() | ::stdexec::then(Dummy<true>{});
                           })),
                  ::stdexec::__mset<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>,
                  ::stdexec::env<>
    >);

    return true;
}
static_assert(test_completion_signatures());

/**
 * @test Use @c stdexec::let_value to express branching instead of @c exec::split, as proposed in @cite P3682R0.
 *
 * The semantic seems different. With @c stdexec::let_value, the inner sender is connected "lately", only when the @c stdexec::let_value
 * receiver is consumed.
 */
TEST_F(LetValueTest, for_branching) {
    std::array<std::thread::id, 4> thrids;

    std::atomic<bool> consumed{false};

    auto shared = ::stdexec::schedule(pools.at(index_of_A).get_scheduler()) | THEN_STORE_ID(thrids[0])
                | ::stdexec::then([&consumed]() -> double {
                      std::printf("Preliminary 'then' returning.\n"); // NOLINT(modernize-use-std-print)
                      if (consumed)
                          throw std::runtime_error("Already consumed.");
                      consumed = true;
                      return 42.;
                  });

    ::stdexec::sync_wait(
        std::move(shared) // NOLINT(performance-move-const-arg)
        | ::stdexec::let_value([this, &consumed, &thrids](const double value) {
              std::cout << "Value received is " << value << ".\n";
              if (!consumed)
                  throw std::runtime_error("Not consumed yet.");
              auto br_b = ::stdexec::schedule(pools.at(index_of_B).get_scheduler()) | THEN_STORE_ID(thrids[1]);
              auto br_c = ::stdexec::schedule(pools.at(index_of_C).get_scheduler()) | THEN_STORE_ID(thrids[2]);
              auto br_d = ::stdexec::schedule(pools.at(index_of_D).get_scheduler()) | THEN_STORE_ID(thrids[3]);
              return ::stdexec::when_all(
                  std::move(br_b), std::move(br_c), std::move(br_d)); // NOLINT(performance-move-const-arg)
          }));

    ASSERT_EQ(thrids[0], threads.at(index_of_A));
    ASSERT_EQ(thrids[1], threads.at(index_of_B));
    ASSERT_EQ(thrids[2], threads.at(index_of_C));
    ASSERT_EQ(thrids[3], threads.at(index_of_D));

    ASSERT_TRUE(consumed);
}

} // namespace tests::stdexec::adaptors
