#ifndef GRAPH_DISPATCHING_TESTS_IGNOREWARNINGS_HPP
#define GRAPH_DISPATCHING_TESTS_IGNOREWARNINGS_HPP

#define DO_PRAGMA_(x) _Pragma (#x)

#if defined(__GNUC__) || defined(__GNUG__)
    #define PRAGMA_DIAGNOSTIC_POP           _Pragma("GCC diagnostic pop")
    #define PRAGMA_DIAGNOSTIC_PUSH          _Pragma("GCC diagnostic push")
    #define PRAGMA_DIAGNOSTIC_IGNORED(what) DO_PRAGMA_(GCC   diagnostic ignored what)
#else
    #error "Unsupported compiler."
#endif

#endif // GRAPH_DISPATCHING_TESTS_IGNOREWARNINGS_HPP
