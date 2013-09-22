#ifndef PARICA_UTILS_HPP_
#define PARICA_UTILS_HPP_

#include <cstddef>

namespace parica{

    namespace detail{

        std::size_t round_to_previous_multiple(std::size_t x, std::size_t multiple){
            if((x%multiple)==0)
                return x;
            return x/multiple * multiple;
        }

        template<class ScalarType>
        std::size_t shuffle(ScalarType* data, std::size_t NC, std::size_t NF){
            for(std::size_t c = 0 ; c < NC ; ++c){
                for(std::size_t f = 0 ; f < NF ; ++f){
                }
            }
        }
    }

}

#endif
