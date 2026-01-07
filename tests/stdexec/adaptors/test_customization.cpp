#include <future>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec customization points
 * -----------------------------------------
 *
 * This group of tests shows how we can customize @c stdexec algorithms.
 *
 * The tests can be found in @ref tests/stdexec/adaptors/test_customization.cpp.
 */

namespace tests::stdexec::adaptors
{
//! Default ID if a @c then is not launched through @ref tests::stdexec::adaptors::ThenReceiver::set_value.
inline constexpr char DEFAULT_ID = 'X';

//! Default customization (*i.e.* no customization).
inline constexpr char DEFAULT_CU = 'Y';

#define DEFAULT_PAIR {DEFAULT_ID, DEFAULT_CU}

//! @name Thread local variables used to check how @c then is launched.
///@{
thread_local char thread_id            = DEFAULT_ID;
thread_local char thread_customization = DEFAULT_CU;
///@}

/**
 * @brief Receiver for @c then.
 *
 * @todo Investigate if we can inherit from the @c Rcvr, as shown in the example
 *       at https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#example-then.
 */
template <char ID, char Customization, ::stdexec::receiver Rcvr, typename Functor>
struct ThenReceiver
{
    using receiver_concept = ::stdexec::receiver_t;

    Rcvr rcvr;
    Functor functor;

    template <class... Args> requires std::invocable<Functor, Args...>
    void set_value(Args&&... args) && noexcept
    {
        auto result = std::async(std::launch::async,
            [func = std::move(functor)] (Args&&... func_args) {
                thread_id = ID;
                thread_customization = Customization;
                return std::move(func)(std::forward<Args>(func_args)...);
            }, std::forward<Args>(args)...);

        result.wait();

        if constexpr (std::same_as<std::invoke_result_t<Functor, Args...>, void>)
            ::stdexec::set_value(std::move(rcvr));
        else
            ::stdexec::set_value(std::move(rcvr), std::move(result.get()));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        ::stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        ::stdexec::set_stopped(std::move(rcvr));
    }

    /// This is a game changer for @c starts_on. Without this @c get_env, the @c starts_on is not "propagated"
    /// all the way up to the "beginning" of the chain during the "connect" phase.
    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(rcvr); }
};

namespace details
{
//! @name We need this mess because we cannot have @c stdexec::set_value_t evaluated with @c void within a template alias.
///@{
template <bool, typename, typename...>
struct set_value;

template <typename Functor, typename... Args>
struct set_value<true, Functor, Args...> { using type = ::stdexec::completion_signatures<::stdexec::set_value_t()>; };

template <typename Functor, typename... Args>
struct set_value<false, Functor, Args...> { using type = ::stdexec::completion_signatures<::stdexec::set_value_t(std::invoke_result_t<Functor, Args...>)>; };

template <typename Functor, typename... Args> requires std::invocable<Functor, Args...>
using set_value_t = set_value<std::is_void_v<std::invoke_result_t<Functor, Args...>>, Functor, Args...>::type;
///@}
} // namespace details

//! Sender for @c then.
template <char ID, char Customization, ::stdexec::sender Sndr, typename Functor, class Schd>
struct ThenSender
{
    using sender_concept = ::stdexec::sender_t;

    /// @name Compute the completion signatures.
    ///
    /// We cannot hardcode the completion signatures, it has to depend on @c Sndr and @c Functor.
    /// We have implemented a simpler version than the one that can be found at
    /// https://github.com/NVIDIA/stdexec/blob/b888185d667f68b9a8bda5d0c81d03edf9ec3fe1/include/execpools/thread_pool_base.hpp#L320-L336.
    ///@{
    template <typename... Args>
    using set_value_t = details::set_value_t<Functor, Args...>;

    //! Check if a callable is invocable and does not throw, in which case there's no need to advertise completion with @c std::exception_ptr.
    template <typename Callable>
    using then_non_throwing = std::is_nothrow_invocable<Callable>;

    using with_error_invoke_t = std::conditional_t<
        then_non_throwing<Functor>::value,
        ::stdexec::completion_signatures<>,
        ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr)>
    >;

    template <class Self, typename... Env>
    using completion_signatures = ::stdexec::transform_completion_signatures<
        ::stdexec::completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>,
        with_error_invoke_t,
        set_value_t
    >;
    ///@}

    //! As required by https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/include/stdexec/__detail/__diagnostics.hpp#L266-L310.
    template <class... Env>
    auto get_completion_signatures(Env&&...) -> completion_signatures<ThenSender, Env...> { return {}; } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <::stdexec::receiver Rcvr>
    auto connect(Rcvr&& rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
    {
        using recv_t = ThenReceiver<ID, Customization, std::remove_cvref_t<Rcvr>, Functor>;

        //! @note We don't pass the @ref schd to the receiver because it serves no purpose in this test.
        return ::stdexec::connect(std::move(sndr), recv_t{.rcvr = std::forward<Rcvr>(rcvr), .functor = std::move(functor)});
    }

    Sndr sndr;
    Functor functor;
    Schd schd;

    decltype(auto) get_env() const noexcept { return ::stdexec::get_env(sndr); }
};

//! Part of our customization, used by @ref Domain.
template <char ID, char Customization, class Schd>
struct TransformThen
{
    Schd schd;

    template <typename Functor, ::stdexec::sender Sndr>
    auto operator()(::stdexec::then_t, Functor&& functor, Sndr&& sndr) && noexcept {
        return ThenSender<ID, Customization, Sndr, Functor, Schd>{
            .sndr    = std::forward<Sndr>(sndr),
            .functor = std::forward<Functor>(functor),
            .schd    = std::move(schd)
        };
    }
};

template <class Domain>
struct DomainSpecificScheduler;

template <char ID>
struct Domain
{
    template <::stdexec::sender_expr_for<::stdexec::then_t> Sndr, typename Env>
    auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env) const noexcept
    {
        std::cout << "> domain " << ID << ": transform_sender" << std::endl;

        if constexpr (::stdexec::__completes_on<Sndr, DomainSpecificScheduler<Domain>, Env>) {
            auto schd = ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(sndr), env);
            return sndr.apply(std::forward<Sndr>(sndr), TransformThen<ID, 'L', decltype(schd)>{.schd = std::move(schd)});
        } else {
            static_assert(::stdexec::__completes_on<Sndr, DomainSpecificScheduler<Domain>, Env>);
        }
    }
};

//! A "very basic" scheduler.
template <class Domain>
struct DomainSpecificScheduler
{
    struct Env
    {
        //! The accepted completion tag types must agree with @ref Sender::completion_signatures.
        template <::stdexec::__one_of<::stdexec::set_value_t> CompletionTag>
        DomainSpecificScheduler query(::stdexec::get_completion_scheduler_t<CompletionTag>) const noexcept { return {}; }
    };

    template <class R>
    struct Op
    {
        using operation_state_concept = ::stdexec::operation_state_t;

        R rcvr;

        //! @todo Check if the @ref rcvr must be moved or not.
        void start() & noexcept {
            ::stdexec::set_value(std::move(rcvr));
        }
    };

    struct Sender
    {
        using sender_concept = ::stdexec::sender_t;

        using completion_signatures = ::stdexec::completion_signatures<::stdexec::set_value_t()>;

        template <::stdexec::receiver_of<completion_signatures> R>
        Op<std::remove_cvref_t<R>> connect(R&& rcvr) const noexcept(std::is_nothrow_move_constructible_v<R>) {
            return {.rcvr = std::forward<R>(rcvr)};
        }

        Env get_env() const noexcept { return {}; }
    };

    ::stdexec::sender auto schedule() const noexcept { return Sender{}; }

    auto query(::stdexec::get_domain_t) const noexcept { return Domain{}; }

    auto query(::stdexec::get_completion_domain_t<::stdexec::set_value_t>) const noexcept { return Domain{}; }

    friend bool operator==(const DomainSpecificScheduler&, const DomainSpecificScheduler&) noexcept { return true; }
};

//! Helper to add a @c then.
#define ADD_THEN(__nth__)                                                       \
    ::stdexec::then([&trace]() noexcept {                                       \
        std::cout << "Then(" << thread_customization << ", " << __nth__ << "):" \
                  << " got thread ID " << thread_id                             \
                  << " in thread " << std::this_thread::get_id()                \
                  << std::endl;                                                 \
        trace[__nth__] = {thread_id, thread_customization};                     \
    })

//! Helper to setup the trace and the expected trace.
#define SETUP_TRACE(__size__, ...)                                \
    using trace_t = std::array<std::tuple<char, char>, __size__>; \
    constexpr trace_t expected{{__VA_ARGS__}};                    \
    trace_t trace;

//! Helper to check the trace content.
#define CHECK_TRACE ASSERT_THAT(trace, ::testing::ElementsAreArray(expected));

class CustomizationTest : public ::testing::Test {};

/// @test Check that the @c then is properly enqueued and executed
///       for a simple chain without any specified scheduler.
TEST_F(CustomizationTest, no_specified_scheduler)
{
    SETUP_TRACE(1, DEFAULT_PAIR)

    ::stdexec::sync_wait(::stdexec::just() | ADD_THEN(0));

    CHECK_TRACE
}

/// @test Check that the customized @c then is properly enqueued and executed
///       for a simple chain starting with a schedule sender. It uses early customization.
TEST_F(CustomizationTest, begins_with_schedule_sender_followed_by_custom_then)
{
    SETUP_TRACE(2, {'A', 'L'}, {'A', 'L'})

    ::stdexec::sender auto work = ::stdexec::schedule(DomainSpecificScheduler<Domain<'A'>>{})
        | ADD_THEN(0) | ADD_THEN(1);

    static_assert(std::same_as<::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(work)>, Domain<'A'>>);

    ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)

    CHECK_TRACE
}

/// @test The chain begins without any scheduler specified, so the first @c then
///       is using the default domain. Then, we transfer to a custom scheduler
///       with @c continues_on and the custom @c then is used. It uses early
///       customization.
TEST_F(CustomizationTest, continues_on_uses_custom_then)
{
    SETUP_TRACE(8, DEFAULT_PAIR, DEFAULT_PAIR, DEFAULT_PAIR, {'B', 'L'}, {'B', 'L'}, {'B', 'L'}, {'C', 'L'}, {'C', 'L'})

    ::stdexec::sender auto chain = ::stdexec::just() | ADD_THEN(0) | ADD_THEN(1) | ADD_THEN(2);

    ::stdexec::sender auto work = std::move(chain) // NOLINT(performance-move-const-arg)
        | ::stdexec::continues_on(DomainSpecificScheduler<Domain<'B'>>{})
        | ADD_THEN(3) | ADD_THEN(4) | ADD_THEN(5)
        | ::stdexec::continues_on(DomainSpecificScheduler<Domain<'C'>>{})
        | ADD_THEN(6) | ADD_THEN(7);

    ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)

    CHECK_TRACE
}

/// @test This shows that @c starts_on modifies all its predecessors, when the passed
///       sender is still on the default domain.
TEST_F(CustomizationTest, starts_on)
{
    SETUP_TRACE(5, {'C', 'L'}, {'C', 'L'}, {'C', 'L'}, {'D', 'L'}, {'D', 'L'})

    ::stdexec::sender auto begin = ::stdexec::just() | ADD_THEN(0) | ADD_THEN(1);

    ::stdexec::sender auto starts_on = ::stdexec::starts_on(DomainSpecificScheduler<Domain<'C'>>{}, std::move(begin)); // NOLINT(performance-move-const-arg)

    ::stdexec::sender auto work = std::move(starts_on) | ADD_THEN(2) // NOLINT(performance-move-const-arg)
        | ::stdexec::continues_on(DomainSpecificScheduler<Domain<'D'>>{})
        | ADD_THEN(3)
        | ADD_THEN(4);

    ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)

    CHECK_TRACE
}

/// @test Check that @c stdexec::on without closure works. It uses late customization.
TEST_F(CustomizationTest, on_no_closure)
{
    SETUP_TRACE(2, {'R', 'L'}, {'R', 'L'})

    ::stdexec::sync_wait(::stdexec::on(DomainSpecificScheduler<Domain<'R'>>{}, ::stdexec::just() | ADD_THEN(0) | ADD_THEN(1)));

    CHECK_TRACE
}

//! @test Check that the transition is back-and-forth, using the default otherwise.
TEST_F(CustomizationTest, on_in_the_middle_otherwise_default)
{
    SETUP_TRACE(3, DEFAULT_PAIR, {'R', 'L'}, DEFAULT_PAIR)

    ::stdexec::sync_wait(::stdexec::just() | ADD_THEN(0) | ::stdexec::on(DomainSpecificScheduler<Domain<'R'>>{}, ADD_THEN(1)) | ADD_THEN(2));

    CHECK_TRACE
}

/// @test Same purpose as @ref CustomizationTest_on_in_the_middle_otherwise_default_Test,
///       but the chain is started by a schedule sender on @ref DomainSpecificScheduler.
TEST_F(CustomizationTest, on_in_the_middle_otherwise_custom_scheduler)
{
    SETUP_TRACE(3, {'Q', 'L'}, {'R', 'L'}, {'Q', 'L'})

    ::stdexec::sync_wait(::stdexec::schedule(DomainSpecificScheduler<Domain<'Q'>>{})
        | ADD_THEN(0)
        | ::stdexec::on(DomainSpecificScheduler<Domain<'R'>>{}, ADD_THEN(1))
        | ADD_THEN(2));

    CHECK_TRACE
}

//! @test Test @c stdexec::on with a mix of different schedulers.
TEST_F(CustomizationTest, on)
{
    SETUP_TRACE(8, {'W', 'L'}, {'W', 'L'}, DEFAULT_PAIR, DEFAULT_PAIR, {'Y', 'L'}, DEFAULT_PAIR, {'Z', 'L'}, {'Z', 'L'})

    exec::static_thread_pool pool{1};

    ::stdexec::sender auto chain = ::stdexec::schedule(DomainSpecificScheduler<Domain<'W'>>{})
        | ADD_THEN(0) | ADD_THEN(1)
        | ::stdexec::continues_on(pool.get_scheduler())
        | ADD_THEN(2)
        | ADD_THEN(3)
        | ::stdexec::on(DomainSpecificScheduler<Domain<'Y'>>{}, ADD_THEN(4))
        | ADD_THEN(5)
        | ::stdexec::continues_on(DomainSpecificScheduler<Domain<'Z'>>{})
        | ADD_THEN(6)
        | ADD_THEN(7);

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    CHECK_TRACE
}

//! @test Check we're able to use our customization when we pass values in the value channel.
TEST_F(CustomizationTest, then_with_values)
{
    SETUP_TRACE(3, {'W', 'L'}, {'W', 'L'}, {'W', 'L'})

    std::vector<double> values(1);

    values.at(0) = 42;

    const double* ptr = values.data();

    #define ADD_THEN_WITH_DATA(__id__, __value__)              \
        ::stdexec::then([&trace](auto&& data) {                \
            if(data.at(0) != __value__)                        \
                throw std::runtime_error("The test failed.");  \
            ++data[0];                                         \
            trace[__id__] = {thread_id, thread_customization}; \
            return std::move(data);                            \
        })

    auto chain = ::stdexec::just(std::move(values))
        | ::stdexec::continues_on(DomainSpecificScheduler<Domain<'W'>>{})
        | ADD_THEN_WITH_DATA(0, 42)
        | ADD_THEN_WITH_DATA(1, 43)
        | ADD_THEN_WITH_DATA(2, 44);

    const auto res = std::get<0>(::stdexec::sync_wait(std::move(chain)).value());

    ASSERT_EQ(res.at(0), 45);
    ASSERT_EQ(res.data(), ptr) << "There shouldn't be any copy.";

    CHECK_TRACE
}

} // namespace tests::stdexec::adaptors
