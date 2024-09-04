# graph-dispatching

This repository is a playground for graph-based dispatching of asynchronous workloads using `Kokkos` and the P2300 formalism.

## Terminolgy

For a definition of:
- `execution context`
- `sender`

see https://en.cppreference.com/w/cpp/experimental/execution.

As opposed to a *single-shot sender* that can be consumed only *once*, a *multi-shot sender* can be consumed *many times*.

## What is a graph then ?

We can think of a graph (*e.g.* `Kokkos::Graph`) as a **multi-shot sender chain**.

## Repository structure

* [tests](./tests)
  * [stdexec](./tests/stdexec) is dedicated to pure `stdexec` testing and exploration.
  * [graph](./tests/graph) is dedicated to implementing `Kokkos::Graph` *à la* `stdexec` (P2300).
    Test files are always a triplet.
    The `.stdexec` is the `stdexec` version of the test. It is compiled and tested.
    The `.kokkos`is the `Kokkos` version of the test. It is compiled and tested.
    The `.outlook` is the *drafted à la P2300* `Kokkos` version. It cannot be compiled and should be considered pseudo-code (for now).
    For instance:
    - `test_ABC.stdexec.cpp`
    - `test_ABC.kokkos.cpp`
    - `test_ABC.outlook.cpp`
