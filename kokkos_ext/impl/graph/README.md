# Integrate `Kokkos::Experimental::Graph` with `std::execution`

### A motivating example

```c++
// Some pre-processing is required before the graph. For instance, before a convergence loop, some data need to be (re-)initialized.
// It will not be resubmitted many or involves operations that cannot be put in the graph (MPI, I/O, and so on). On CPU.
stdexec::sender auto preprocess = stdexec::schedule(cpu.get_scheduler()) | stdexec::then(...) | ...;

// As a graph. On GPU.
auto graph_one = stdexec::then(...) | exec::fork_join(
    stdexec::then(...),
    stdexec::then(...))
| stdexec::then(...);

// Some intermediate processing before the next graph. On CPU.
auto interlude = stdexec::then(...);

// Another graph. On GPU.
auto graph_two = stdexec::then(...) | exec::fork_join(
    stdexec::then(...),
    stdexec::then(...),
    stdexec::bulk(...))
| stdexec::bulk(...);

// Some post-processing. On CPU.
auto postprocess = stdexec::then(...) | ...;

// Loop until convergence.
stdexec::sync_wait(exec::repeat_until(
    preprocess
    | stdexec::on(gpu.get_scheduler(), graph_one)
    | stdexec::on(cpu.get_scheduler(), interlude)
    | stdexec::on(gpu.get_scheduler(), graph_two)
    | stdexec::on(cpu.get_scheduler(), postprocess)
    | stdexec::then(... convergence criterion ...)
));
```

### Semantic and performance issues

* `exec::repeat_until` copies the input sender and connect it at each iteration.
  This would mean copying all sender data (functors) at each iteration, before the graph is submitted (so it's a cost that
  cannot be hidden).
* There is no way to create a shared state that would hold the underlying `Kokkos::Experimental::Graph` created during the
  first iteration to reuse it in subsequent iterations, because customization
  is called only at the `stdexec::connect` time (on the copy).
  The shared state should not be created under the hood because the subsequent `stdexec::connect` would not have the same
  "visible effects", but it's hard to tell from how the senders are chained.

### Device placement

First, we aknowledge that HPC libraries such as *Celerity* [ref] allow for automated device placement at runtime (load balancing and such).
This is a use case that must be supported.

The proposed API for such an automated device placement would be:
```c++
// Build a multi-device context, e.g. from device IDs.
MultiDeviceContext md_ctx {0, 2, 5};

// Build the sender that represents your work.
auto work = then_A | exec::fork_join(then_B, then_C) | then_D;

// The scheduler places nodes using some automated strategy, and it may use whichever strategy, such as a CUDA graph or CUDA streams.
stdexec::sync_wait(stdexec::on(md_ctx.get_scheduler(), work));
```

Yet, manual device placement and combination with automated device placement must also be allowed.
```c++
// Build a multi-device context, e.g. from device IDs.
MultiDeviceContext md_ctx {0, 2, 5};

// The work node A must be placed on device 0, other work nodes are unconstrained.
auto work = stdexec::on(md_ctx.get_scheduler(0), then_A) | exec::fork_join(then_B, then_C) | then_D;

// The scheduler places nodes using some automated strategy while observing node placement constraints,
// and it may use whichever strategy, such as a CUDA graph or CUDA streams.
stdexec::sync_wait(stdexec::on(md_ctx.get_scheduler(), work));
```

### Enforce the work to map to a `Kokkos::Experimental::Graph`

We'd like to enforce that a given portion of the work always directly maps to a `Kokkos::Experimental::Graph`.
```c++
// Build a multi-device context, e.g. from device IDs.
MultiDeviceContext<Cuda> md_ctx {0, 2, 5};

// Retrieve a graph-oriented context from the multi-device context.
auto graph_ctx = md_ctx.get_graph_context();

// The work node A must be placed on device 0, other work nodes are unconstrained.
auto work = on(graph_ctx.get_scheduler(0), then_A) | fork_join(then_B, then_C) | then_D;

// The scheduler places nodes using some automated strategy while observing node placement constraints,
// and it can only use the Kokkos::Experimental::Graph from the shared state.
stdexec::sync_wait(stdexec::on(graph_ctx.get_scheduler(), work));
```

### Amortized resubmission of the graph

The shared state containing the underlying `Kokkos::Experimental::Graph` ensures that a single graph is built when connecting the chain.
However, it is not yet clear what happens when the graph sender is re-connected.

Does it create a new graph instance ? Or does it reuse the graph from the previous reconnection ?

From the user expectations, the following code should create a fresh `Kokkos::Experimental::Graph` at each iteration:
```c++
// Build a multi-device context, e.g. from device IDs.
MultiDeviceContext<Cuda> md_ctx {0, 2, 5};

// Retrieve a graph-oriented context from the multi-device context.
auto graph_ctx = md_ctx.get_graph_context();

// The work node A must be placed on device 0, other work nodes are unconstrained.
auto work = on(graph_ctx.get_scheduler(0), then_A) | fork_join(then_B, then_C) | then_D;

// Create, instantiate and submit the underlying graph.
stdexec::sync_wait(stdexec::on(graph_ctx.get_scheduler(), work));

// Create, instantiate and submit a new underlying graph.
stdexec::sync_wait(stdexec::on(graph_ctx.get_scheduler(), work));
```

We therefore need a clear way to express the following intent:
> When connected the first time, create the underlying graph. But when reconnected later on,
  reuse the underlying graph from the previous connect.

Such a use case must be supported, as the primary benefit of using a `Kokkos::Experimental::Graph` is usually that subsequent
resubmissions of the same executable graph are must faster.

We propose an explicit wording to ensure that, from the API, it's clear that some memoization of the first `stdexec::connect`
result happens:
```c++
// Build a multi-device context, e.g. from device IDs.
MultiDeviceContext<Cuda> md_ctx {0, 2, 5};

// Retrieve a graph-oriented context from the multi-device context.
auto graph_ctx = md_ctx.get_graph_context();

// The work node A must be placed on device 0, other work nodes are unconstrained.
auto work = on(graph_ctx.get_scheduler(0), then_A) | fork_join(then_B, then_C) | then_D;

// Wrap the work explicitly
auto memoized = replayable_on(graph_ctx.get_scheduler(), work);

// Create, instantiate and submit the underlying graph.
stdexec::sync_wait(memoized);

// Submit the cached underlying graph.
stdexec::sync_wait(memoized);
```

### A concrete example

```c++
exec::static_thread_pool pool{4};

MultiDeviceContext md_ctx{0, 2, 5};

auto graph_ctx_one = md_ctx.get_graph_context();
auto graph_ctx_two = md_ctx.get_graph_context();

// Some pre-processing is required before the graph. For instance, before a convergence loop, some data need to be (re-)initialized.
// It will not be resubmitted many or involves operations that cannot be put in the graph (MPI, I/O, and so on). On CPU.
stdexec::sender auto preprocess = stdexec::schedule(pool.get_scheduler()) | stdexec::then(...) | ...;

// As a graph. On GPU. Automated device placement. Replayable (cached underlying graph).
auto graph_one = replayable_on(graph_ctx_one.get_scheduler(), stdexec::then(...) | exec::fork_join(
    stdexec::then(...),
    stdexec::then(...))
| stdexec::then(...));

// Some intermediate processing before the next graph. On CPU.
auto interlude = stdexec::on(pool.get_scheduler(), stdexec::then(...));

// Another graph. On GPU. Partially manual device placement. Replayable (cached underlying graph).
auto graph_two = replayable_on(graph_ctx_two.get_scheduler(), stdexec::then(...)) | exec::fork_join(
    std::exec::on(graph_ctx_two.get_scheduler(0), stdexec::then(...)),
    std::exec::on(graph_ctx_two.get_scheduler(2), stdexec::then(...)),
    std::exec::on(graph_ctx_two.get_scheduler(5), stdexec::bulk(...)))
| stdexec::bulk(...);

// Some post-processing. On CPU.
auto postprocess = stdexec::on(pool.get_scheduler(), stdexec::then(...) | ...);

// Loop until convergence.
stdexec::sync_wait(exec::repeat_until(
    preprocess
    | graph_one
    | interlude
    | graph_two
    | postprocess
    | stdexec::then(... convergence criterion ...)
));
```

### Key insights

1. Create a unique graph-oriented context per graph.


        A
       /  \
      B    C