#ifndef dshf_ica_UTILS_HPP_
#define dshf_ica_UTILS_HPP_

#include <cstddef>

namespace dshf_ica{

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
                for(std::size_t f = 0 ; f < NF ; ++f)
                    shuffled_va[f] = data[c*NF+perms[f]];
                for(std::size_t f = 0 ; f < NF ; ++f)
                    data[c*NF+f] = shuffled_va[f];
            }

            delete[] shuffled_va;
            delete[] perms;
        }

        template<class T>
        T round_to_previous_multiple(T const & val, unsigned int multiple){
          return val/multiple*multiple;
        }

        template<class T>
        T round_to_next_multiple(T const & val, unsigned int multiple){
          return (val+multiple-1)/multiple*multiple;
        }
    }

}

#endif
