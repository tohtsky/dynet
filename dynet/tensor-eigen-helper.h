#ifndef TENSOR_EIGEN_HELPER
#define TENSOR_EIGEN_HELPER

template<int M, int N, class Function>
struct _branch_vector_size{
    inline typename Function::return_type static 
        _call(typename Function::fargs && other_args, const dynet::Dim & d){
        if(d.nd==N)
            return Function:: template supply_one_call<M, N>(std::forward<typename Function::fargs>(other_args), d);
        else 
            return _branch_vector_size<M,N-1,Function>::_call(std::forward<typename Function::fargs>(other_args), d);
    }
};

template<unsigned M, class Function>
struct _branch_vector_size<M, 0,Function>{
    inline typename Function::return_type static
        _call(typename Function::fargs && other_args, const dynet::Dim & d){
        return Function::template supply_one_call<M, 0>(std::forward<typename Function::fargs>(other_args), d);
    }
}; 

template<class Function>
struct _fill_one_call_f{
    template<unsigned M, unsigned N, typename ...Args>
        static inline typename std::enable_if< (N>sizeof...(Args)) ,typename Function::return_type>::type
        FillOne(typename Function::fargs&& func_args, const dynet::Dim& v, Args...args){ 
            return _fill_one_call_f<Function>::
                FillOne<M,N>(std::forward<typename Function::fargs>(func_args), v, args..., v[sizeof...(Args)]);
        }

    template<unsigned M, unsigned N, typename ...Args>
        static inline typename std::enable_if< (N<=sizeof...(Args)) && (sizeof...(Args)<M) ,typename Function::return_type>::type
        FillOne(typename Function::fargs && func_args, const dynet::Dim& v, Args...args){ 
            return _fill_one_call_f<Function>::
                FillOne<M,N>(std::forward<typename Function::fargs>(func_args), v, args..., (int) 1);
        }

    template<unsigned M, unsigned N, typename ...Args>
        static inline typename std::enable_if< (sizeof...(Args)==M) ,typename Function::return_type>::type
        FillOne(typename Function::fargs && func_args, const dynet::Dim& v, Args...args){ 
            return Function::call(std::forward<typename Function::fargs>(func_args), args...);
        }
};

#endif
