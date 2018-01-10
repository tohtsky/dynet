#ifndef TENSOR_EIGEN_HELPER
#define TENSOR_EIGEN_HELPER
#include "dynet/dim.h"

namespace dynet{
    namespace detail{
        template<int M, int N, class Function>
            struct branch_by_dim{
                inline typename Function::return_type static 
                    apply(typename Function::Fargs && other_args, const dynet::Dim & d){
                        if(d.nd==N)
                            return Function:: template supply_one_call<M, N>(std::forward<typename Function::Fargs>(other_args), d);
                        else 
                            return branch_by_dim<M,N-1,Function>::apply(std::forward<typename Function::Fargs>(other_args), d);
                    }
            };

        template<unsigned M, class Function>
            struct branch_by_dim<M, 0,Function>{
                inline typename Function::return_type static
                    apply(typename Function::Fargs && other_args, const dynet::Dim & d){
                        return Function::template supply_one_call<M, 0>(std::forward<typename Function::Fargs>(other_args), d);
                    }
            }; 

        template<class Function>
            struct fill_one_call_f{
                template<unsigned M, unsigned N, typename ...Args>
                    static inline typename std::enable_if< (N>sizeof...(Args)) ,typename Function::return_type>::type
                    apply(typename Function::Fargs&& func_args, const dynet::Dim& v, Args...args){ 
                        return fill_one_call_f<Function>::
                            apply<M,N>(std::forward<typename Function::Fargs>(func_args), v, args..., static_cast<int>(v[sizeof...(Args)]));
                    }

                template<unsigned M, unsigned N, typename ...Args>
                    static inline typename std::enable_if< (N<=sizeof...(Args)) && (sizeof...(Args)<M) ,typename Function::return_type>::type
                    apply(typename Function::Fargs && func_args, const dynet::Dim& v, Args...args){ 
                        return fill_one_call_f<Function>::
                            apply<M,N>(std::forward<typename Function::Fargs>(func_args), v, args..., (int) 1);
                    }

                template<unsigned M, unsigned N, typename ...Args>
                    static inline typename std::enable_if< (sizeof...(Args)==M) ,typename Function::return_type>::type
                    apply(typename Function::Fargs && func_args, const dynet::Dim& v, Args...args){ 
                        return Function::call(std::forward<typename Function::Fargs>(func_args), args...);
                    }
            };

        template<int Order, class Function>
            struct branch_then_supply_one{
                using return_type = typename Function::template return_type<Order>;
                using Fargs = typename Function::Fargs;
                using self_type = branch_then_supply_one<Order, Function>; 

                template<int M, int N>
                    static inline return_type supply_one_call(Fargs&& fargs_, const dynet::Dim &v){
                        return detail::fill_one_call_f<self_type>:: template apply<M,N>(std::forward<Fargs>(fargs_), v);
                    }

                template<typename ... Args>
                    static inline return_type call(Fargs&& fargs, Args ... args){ 
                        return Function::template call<Args...>(std::forward<Fargs>(fargs), args...); 
                    } 

                inline static return_type generate_loop(Fargs&& fargs, const dynet::Dim & v){
                    return branch_by_dim<Order, Order, self_type>::apply(std::forward<Fargs>(fargs),v);
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

        template <int Order>
            inline void _check_tb(const dynet::Dim &d){
                DYNET_ASSERT(t.d.ndims() <= Order,
                        "Illegal access of tensor in function t<" << Order << ">(Tensor & t): dim=" << d); 
            }

        template <>
            inline void _check_tb<0>(const dynet::Dim &d){
                DYNET_ASSERT(t.d.batch_elems() == 1,
                        "Illegal access of tensor in function t<0>(Tensor & t): dim=" << d); 
            }

        template <>
            inline void _check_tb<1>(const dynet::Dim &d){
                DYNET_ASSERT(t.d.ndims() == 1 || t.d.batch_size() == t.d.rows(),
                        "Illegal access of tensor in function t<1>(Tensor & t): dim=" << d);
            }
    }
}
#endif
