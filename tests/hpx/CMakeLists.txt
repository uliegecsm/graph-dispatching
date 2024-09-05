string(JSON HPX_REQUIRED_VERSION  GET "${VERSION_JSON}" dependencies hpx value)

find_package(
    HPX
    ${HPX_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

add_one_test(NAME HelloWorld ADDITIONAL_LINK_LIBRARIES HPX::hpx)
