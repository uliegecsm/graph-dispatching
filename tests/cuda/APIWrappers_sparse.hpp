#ifndef GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_SPARSE_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_SPARSE_HPP

#include "cusparse.h"

#include "tests/cuda/APIWrappers.hpp"

//! Check the return code of a sparse API call (*e.g.* @c cuSPARSE).
#define CHECK_SPARSE_CALL(call) CHECK_CALL_IMPL(call, CUSPARSE_STATUS_SUCCESS, cusparseGetErrorString)

namespace tests::cuda::sparse
{

/**
 * @brief Wrapper for @c cusparseHandle_t.
 *
 * @warning As of @c Cuda 12.6, trying to create a handle while stream capturing is activated results in an initialization error:
 *          @code
 *          On entry to cusparseCreate(): CUDA context cannot be initialized
 *          @endcode
 */
struct Handle
{
    cusparseHandle_t handle = nullptr;

    Handle()  { CHECK_SPARSE_CALL(cusparseCreate(&handle)); }
    ~Handle() { CHECK_SPARSE_CALL(cusparseDestroy(handle)); }

    void set_stream(const Stream& stream) const { CHECK_SPARSE_CALL(cusparseSetStream(handle, stream.stream)); }
};

//! Small helper to deal with a sparse view defined by the non-zero indices and their values.
template <typename T>
struct View
{
    ::tests::cuda::View<int> indices;
    ::tests::cuda::View<T>   values;
};

template <typename T>
struct DataType;

template <>
struct DataType<double> { static constexpr auto type = CUDA_R_64F; };

//! Wrapper for @c cusparseSpVecDescr_t.
template <typename T>
struct SparseVectorDescriptor
{
    cusparseSpVecDescr_t descr = nullptr;
    bool owning = false;

    SparseVectorDescriptor() = default;

    SparseVectorDescriptor(const View<T>& data) : owning(true)
    {
        CHECK_SPARSE_CALL(cusparseCreateSpVec(
            &this->descr,
            data.values.size, data.indices.size, data.indices.buffer, data.values.buffer,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, DataType<T>::type
        ));
    }

    SparseVectorDescriptor& operator=(SparseVectorDescriptor&& other)
    {
        this->descr = other.descr;
        this->owning = other.owning;
        if(other.owning) other.owning = false;
        return *this;
    }

    ~SparseVectorDescriptor() { if(owning) CHECK_SPARSE_CALL(cusparseDestroySpVec(descr)); }
};

//! Wrapper for @c cusparseDnVecDescr_t.
template <typename T>
struct DenseVectorDescriptor
{
    cusparseDnVecDescr_t descr = nullptr;
    bool owning = false;

    DenseVectorDescriptor() = default;

    DenseVectorDescriptor(const ::tests::cuda::View<T>& values) : owning(true)
    {
        CHECK_SPARSE_CALL(cusparseCreateDnVec(
            &this->descr,
            values.size, values.buffer,
            DataType<T>::type
        ));
    }

    DenseVectorDescriptor& operator=(DenseVectorDescriptor&& other)
    {
        this->descr = other.descr;
        this->owning = other.owning;
        if(other.owning) other.owning = false;
        return *this;
    }

    ~DenseVectorDescriptor() { if(owning) CHECK_SPARSE_CALL(cusparseDestroyDnVec(descr)); }
};

} // namespace tests::cuda::sparse

#endif // GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_SPARSE_HPP
