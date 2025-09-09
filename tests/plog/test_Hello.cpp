#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"
#include "plog/Log.h"

/**
 * @addtogroup unittests
 *
 * Logging with @c plog
 * --------------------
 *
 * Inspired from https://github.com/SergiusTheBest/plog/blob/e5c033e317a01b2703d13aab42288d09b2efdafc/README.md.
 */

namespace tests
{

//! @test Check that the logging with @c plog can output something.
TEST(plog, stdout)
{
    ::testing::internal::CaptureStdout();

    ::plog::init<::plog::TxtFormatter>(::plog::debug, ::plog::streamStdOut);

    PLOG_DEBUG << "Hello log!";

    const std::string& output = ::testing::internal::GetCapturedStdout();

    ASSERT_THAT(output, ::testing::MatchesRegex(
        "[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+.[0-9]+ DEBUG \\[[0-9]+\\] \\[tests::plog_stdout_Test::TestBody@[0-9]+\\] Hello log!\n"
    ));
}

} // namespace tests
