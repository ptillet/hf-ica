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
    }

}

#endif
