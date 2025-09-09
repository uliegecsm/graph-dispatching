set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/test_Hello.cpp)
add_executable(tests_plog_hello test_Hello.cpp)
target_link_libraries(tests_plog_hello PRIVATE plog GTest::gtest_main)
add_test(NAME tests_plog_hello COMMAND $<TARGET_FILE:tests_plog_hello>)
