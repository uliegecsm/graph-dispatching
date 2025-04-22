#---- Find Benchmark.
string(JSON Benchmark_REQUIRED_VERSION GET "${VERSION_JSON}" dependencies googlebenchmark value)

find_package(
    benchmark
    ${Benchmark_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)
