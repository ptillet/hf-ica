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
            std::size_t* perms = new std::size_t[NF];
            ScalarType * shuffled_va = new ScalarType[NF];

            for(std::size_t i = 0 ; i < NF ; ++i)
                perms[i] = i;
            for(std::size_t i = 0 ; i < NF ; ++i){
                std::size_t j = rand()%(NF-i)+i;
                std::swap(perms[i], perms[j]);
            }
            for(std::size_t c = 0 ; c < NC ; ++c){
                for(std::size_t f = 0 ; f < NF ; ++f){
                    shuffled_va[f] = data[c*NF+perms[f]];
                }
                for(std::size_t f = 0 ; f < NF ; ++f){
                    data[c*NF+f] = shuffled_va[f];
                }
            }

            delete[] shuffled_va;
            delete[] perms;
        }
    }

}

#endif
