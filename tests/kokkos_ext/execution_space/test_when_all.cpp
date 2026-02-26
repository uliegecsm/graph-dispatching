/**
 * @test A @c stdexec::when_all with three branches, all using a unique execution space instance.
 *
 * It is expected that *e.g.* for CUDa, it would allow for the kernels in different branches to overlap.
 */

TEST_F(WhenAllTest, concurrent_branches) {
    const auto [exec_A, exec_B, exec_C] = Kokkos::Experimental::partition_space(exec, 1, 1, 1);

    const context_t esc_A{exec_A}, esc_B{exec_B}, esc_C{exec_C};


    auto sndr = ::stdexec::when_all(
        ::stdexec::schedule(esc_A.get_scheduler()) | ADD_BULK(128),
        ::stdexec::schedule(esc_A.get_scheduler()) | ADD_BULK(128),
        ::stdexec::schedule(esc_A.get_scheduler()) | ADD_BULK(128)
    );

// no eager exec
   ::stdexec::sync_wait(std::move(sndr));

// Check with kk tools would that prevent overlap ? We should proably check in the log messages
// of then customization which thread ID is launching the kernels. Each branch should be using its own
// thread, otherwise we are serializing the branches.
}
