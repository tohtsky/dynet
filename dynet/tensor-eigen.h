#ifndef DYNET_TENSOR_EIGEN_H
#define DYNET_TENSOR_EIGEN_H

// This file includes all of the DyNet tensor functions that require
// Eigen to be importet.d. It should be included sparingly to prevent
// unnecessary compile time.

#include "dynet/tensor.h"

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include "dynet/tensor-eigen-helper.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace dynet {

/**
 * \brief Get the data as an Eigen matrix
 * \return Eigen matrix
 */
inline Eigen::Map<Eigen::MatrixXf> mat(Tensor& t) {
  DYNET_ARG_CHECK((t.d.batch_elems() == 1 && t.d.ndims() < 3),
                          "Attempted to access Tensor with more than one batch element or more than two dimensions in matrix form: " << t.d);
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols());
}
inline const Eigen::Map<Eigen::MatrixXf> mat(const Tensor& t) {
  DYNET_ARG_CHECK((t.d.batch_elems() == 1 && t.d.ndims() < 3),
                          "Attempted to access Tensor with more than one batch element or more than two dimensions in matrix form: " << t.d);
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols());
}
/**
 * \brief Get the data as an Eigen vector
 * \details This returns the full tensor contents even if it has many dimensions
 * \return Flattened tensor
 */
inline Eigen::Map<Eigen::VectorXf> vec(Tensor & t) {
  return Eigen::Map<Eigen::VectorXf>(t.v, t.d.size());
}
inline const Eigen::Map<Eigen::VectorXf> vec(const Tensor & t) {
  return Eigen::Map<Eigen::VectorXf>(t.v, t.d.size());
}

/**
 * \brief Get the data as an order 1 Eigen tensor
 * \details this returns the full tensor contents as a one dimensional Eigen tensor which can be used for on-device processing where dimensions aren't important
 * \return Eigen order 1 tensor
 */
inline Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec(Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, t.d.size());
}
inline const Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec(const Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, t.d.size());
}
/**
 * \brief Get the data as an order 2 tensor including batch size
 * \details this returns the full tensor contents as a two dimensional Eigen tensor where the first dimension is a flattened representation of each batch and the second dimension is the batches
 * \return batch size x elements per batch matrix
 */
inline Eigen::TensorMap<Eigen::Tensor<float, 2>> tbvec(Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, t.d.batch_size(), t.d.batch_elems());
}
inline const Eigen::TensorMap<Eigen::Tensor<float, 2>> tbvec(const Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, t.d.batch_size(), t.d.batch_elems());
}

// Get view as an Eigen Tensor (see specializations below-- this is to work Eigen's and DyNet's compile-type vs. run-time differences)
/**
 * \brief Get view as a Tensor
 * \tparam Order Tensor order. 
 * \return Eigen Tensor of the given order
 */


template<int Order>
struct _create_tensor_without_batch{
    using return_type = Eigen::TensorMap<Eigen::Tensor<float, Order>>;
    using fargs = std::tuple<float*>;
    using self_type = _create_tensor_without_batch<Order>; 

    template<int M, int N>
    static inline return_type supply_one_call(fargs&& args, const dynet::Dim &v){
        return _fill_one_call_f<self_type>:: template FillOne<M,N>(std::forward<fargs>(args), v);
    }
    template<typename ... Args>
    static inline return_type call(fargs&& fargs, Args ... args){ 
        return return_type{std::get<0>(fargs), args...}; 
    }

};

template<int Order>
struct _create_tensor_with_batch{
    using return_type = Eigen::TensorMap<Eigen::Tensor<float, Order+1>>;
    using fargs = std::tuple<float*, int>;
    using self_type = _create_tensor_with_batch<Order>; 

    template<int M, int N>
    static inline return_type supply_one_call(fargs&& args, const dynet::Dim &v){
        return _fill_one_call_f<self_type>:: template FillOne<M,N>(std::forward<fargs>(args), v);
    }
    template<typename ... Args>
    static inline return_type call(fargs&& fargs, Args ... args){ 
        return return_type{std::get<0>(fargs), args..., std::get<1>(fargs)}; 
    }

};


template <int Order>
inline void _check_t(const dynet::Dim &d){
    DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= Order,
            "Illegal access of tensor in function t<" << Order << ">(Tensor & t): dim=" << d); 
}

template <>
inline void _check_t<0>(const dynet::Dim &d){
    DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.size() == 1,
            "Illegal access of tensor in function t<0>(Tensor & t): dim=" << d); 
}

template <>
inline void _check_t<1>(const dynet::Dim &d){
    DYNET_ASSERT(t.d.batch_elems() == 1 && (t.d.ndims() == 1 || t.d.size() == t.d.rows()),
            "Illegal access of tensor in function t<1>(Tensor & t): dim=" << d);
} 

template <int Order> inline Eigen::TensorMap<Eigen::Tensor<float, Order>> t(Tensor & t){ 
    _check_t<Order>(t.d);
    std::tuple<float * > arg(t.v);
    return _branch_vector_size<Order,Order,_create_tensor_without_batch<Order>>::_call(std::make_tuple(t.v),t.d);
}

template <int Order> inline Eigen::TensorMap<Eigen::Tensor<float, Order>> t(const Tensor & t){ 
    _check_t<Order>(t.d); 
    std::tuple<float * > arg(t.v); 
    return _branch_vector_size<Order,Order,_create_tensor_without_batch<Order>>::_call(std::make_tuple(t.v),t.d);
}

template <int Order> inline Eigen::TensorMap<Eigen::Tensor<float, Order+1>> tb(Tensor & t){ 
    return _branch_vector_size<Order,Order,_create_tensor_with_batch<Order>>::_call(std::make_tuple(t.v,t.d.bd),t.d);
}

template <int Order> inline Eigen::TensorMap<Eigen::Tensor<float, Order+1>> tb(const Tensor & t){ 
    return _branch_vector_size<Order,Order,_create_tensor_with_batch<Order>>::_call(std::make_tuple(t.v,t.d.bd),t.d);

}


/**
* \brief Get the matrix for a particular batch
* \details Automatically broadcasting if the size is zero.
*
* \param bid Batch id requested
* \return Matrix at batch id `bid` (of shape `t.d.rows()` x `t.d.cols()`)
*/
inline Eigen::Map<Eigen::MatrixXf> batch_matrix(Tensor & t, unsigned bid) {
  return Eigen::Map<Eigen::MatrixXf>(t.v + (bid % t.d.bd) * t.d.batch_size(), t.d.rows(), t.d.cols());
}
inline const Eigen::Map<Eigen::MatrixXf> batch_matrix(const Tensor & t, unsigned bid) {
  return Eigen::Map<Eigen::MatrixXf>(t.v + (bid % t.d.bd) * t.d.batch_size(), t.d.rows(), t.d.cols());
}
/**
 * \brief Get the data as a matrix, where each "row" is the concatenation of rows and columns, and each "column" is batches
 * \return matrix of shape `t.d.rows() * t.d.cols()` x `t.d.batch_elems()`
 */
inline Eigen::Map<Eigen::MatrixXf> rowcol_matrix(Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows() * t.d.cols(), t.d.batch_elems());
}
inline const Eigen::Map<Eigen::MatrixXf> rowcol_matrix(const Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows() * t.d.cols(), t.d.batch_elems());
}

/**
 * \brief Get the data as a matrix, where each "row" is the concatenation of rows, and each "column" is the concatenation of columns and batches
 * \return matrix of shape `t.d.rows() * t.d.cols()` x `t.d.batch_elems()`
 */
inline Eigen::Map<Eigen::MatrixXf> colbatch_matrix(Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols() * t.d.batch_elems());
}
inline const Eigen::Map<Eigen::MatrixXf> colbatch_matrix(const Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols() * t.d.batch_elems());
}


}

#endif
