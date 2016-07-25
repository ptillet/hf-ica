#ifndef UMINTL_TOOLS_EXCEPTION_HPP
#define UMINTL_TOOLS_EXCEPTION_HPP

#include <exception>
#include <string>

namespace umintl{

namespace exceptions{

/** @brief Exception class in the case of incompatible parameters*/
class incompatible_parameters : public std::exception
{
public:
  incompatible_parameters() : message_() {}
  incompatible_parameters(std::string message) : message_("UMinTL: Incompatible supplied parameters: " + message) {}
  virtual const char* what() const throw() { return message_.c_str(); }
  virtual ~incompatible_parameters() throw() {}
private:
  std::string message_;
};


}

}
#endif
