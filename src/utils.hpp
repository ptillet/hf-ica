#ifndef curveica_UTILS_HPP_
#define curveica_UTILS_HPP_

#include <cstddef>

namespace curveica{

    namespace detail{

        template<class ScalarType>
        std::size_t shuffle(ScalarType* data, std::size_t NC, std::size_t NF){
            srand(0);

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
