#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

//! Entry point that will initialize both <tt>Google Test</tt> and @c Kokkos.
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    Kokkos::initialize(argc, argv);

    const auto code = RUN_ALL_TESTS();

    Kokkos::finalize();

    return code;
}
