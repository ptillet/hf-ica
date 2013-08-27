#ifndef PARICA_UTILS_HPP_
#define PARICA_UTILS_HPP

namespace parica{

    template<int N>
    struct compile_time_pow{
        template<class ScalarType>
        ScalarType operator()(ScalarType v){
            return v*compile_time_pow<N-1>()(v);
        }
    };

    template<>
    struct compile_time_pow<0>{
        template<class ScalarType>
        ScalarType operator()(ScalarType v){
            return 1;
        }
    };

}

#endif
