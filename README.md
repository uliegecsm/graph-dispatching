# Dispatching of asynchronous workloads using `Kokkos::Graph` under `std::execution` formalism

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
    Test files are always a triplet, and each use case lives in its own subdirectory.
    The `test_stdexec.cpp` is the `stdexec` version of the test. It is compiled and tested.
    The `test_kokkos.cpp`is the `Kokkos` version of the test. It is compiled and tested.
    The `test_outlook.cpp` is the *drafted à la P2300* `Kokkos` version. It cannot be compiled and should be considered pseudo-code (for now).
    For instance:
    - `diamond/test_stdexec.cpp`
    - `diamond/test_kokkos.cpp`
    - `diamond/test_outlook.cpp`

## `Docker` images

This repository provides several `Docker` images. They have the same name, only the *tag* changes.

1. `gcc-OpenMP` uses the `g++` compiler, and the `Kokkos::OpenMP` backend is enabled.
2. `clang-OpenMP` uses the `clang++` compiler, and the `Kokkos::OpenMP` backend is enabled.
3. `clang-HPX` uses the `clang++` compiler, and the `Kokkos::Experimental::HPX` backend is enabled.

For the `clang-HPX` image, we also provide an `ARM64` version.
