#ifndef UMINTL_TOOLS_IS_CALL_POSSIBLE_HPP
#define UMINTL_TOOLS_IS_CALL_POSSIBLE_HPP
#endif
 
namespace umintl{

namespace detail{

template <typename Type>
class has_member {
   class yes { char m;};
   class no { yes m[2];};
   struct BaseMixin { void operator()() {} };
   struct Base : public Type, public BaseMixin {};
   template <typename T, T t>  class Helper {};
   template <typename U>
   static no deduce(U*, Helper<void (BaseMixin::*)(), &U::operator()>* = 0);
   static yes deduce(...);
public:
   static const bool result = sizeof(yes) == sizeof(deduce((Base*)(0)));
};

namespace details {
   template <typename type> class void_exp_result {};
   template <typename type, typename U> U const& operator,(U const&, void_exp_result<type>);
   template <typename type, typename U> U& operator,(U&, void_exp_result<type>);
   template <typename src_type, typename dest_type>
   struct clone_constness {
     typedef dest_type type;
   };
   template <typename src_type, typename dest_type>
   struct clone_constness<const src_type, dest_type> {
     typedef const dest_type type;
   };
}

template <typename type, typename call_details>
struct is_call_possible {
private:
   class yes {};
   class no { yes m[2]; };
   struct derived : public type {
     using type::operator();
     no operator()(...) const;
   };
   typedef typename details::clone_constness<type, derived>::type derived_type;
   template <typename T, typename due_type>
   struct return_value_check {
     static yes deduce(due_type);
     static no deduce(...);
   };
   template <typename T>
   struct return_value_check<T, void> {
     static yes deduce(details::void_exp_result<type>);
     static no deduce(...);
   };

   template <class T> static T null_object() {}

   template <bool has, typename F>
   struct impl { static const bool value = false; };
   template <typename arg1, typename r>
   struct impl<true, r(arg1)> {
      static const bool value =
         sizeof(
            return_value_check<type, r>::deduce((
                     ((derived_type*)0)->operator()(null_object<arg1>()),
                     details::void_exp_result<type>()))
         ) == sizeof(yes);
   };
   template <typename arg1, typename arg2, typename r>
   struct impl<true, r(arg1, arg2)> {
      static const bool value =
         sizeof(
            return_value_check<type, r>::deduce((
                     ((derived_type*)0)->operator()(null_object<arg1>(), null_object<arg2>()),
                     details::void_exp_result<type>()))
         ) == sizeof(yes);
   };

   template <typename arg1, typename arg2, typename arg3, typename r>
   struct impl<true, r(arg1, arg2, arg3)> {
      static const bool value =
         sizeof(
            return_value_check<type, r>::deduce(
                  (((derived_type*)0)->operator()(null_object<arg1>(), null_object<arg2>(), null_object<arg3>()),
                     details::void_exp_result<type>()))
         ) == sizeof(yes);
   };

   template <typename arg1, typename arg2, typename arg3, typename arg4, typename r>
   struct impl<true, r(arg1, arg2, arg3, arg4)> {
      static const bool value =
         sizeof(
            return_value_check<type, r>::deduce(
                  (((derived_type*)0)->operator()(null_object<arg1>(), null_object<arg2>(), null_object<arg3>(), null_object<arg4>()),
                     details::void_exp_result<type>()))
         ) == sizeof(yes);
   };

   template <typename arg1, typename arg2, typename arg3, typename arg4, typename arg5, typename r>
   struct impl<true, r(arg1, arg2, arg3, arg4, arg5)> {
      static const bool value =
         sizeof(
            return_value_check<type, r>::deduce(
                  (((derived_type*)0)->operator()(null_object<arg1>(), null_object<arg2>(), null_object<arg3>(), null_object<arg4>(), null_object<arg5>()),
                     details::void_exp_result<type>()))
         ) == sizeof(yes);
   };
public:
   static const bool value = impl<has_member<type>::result, call_details>::value;
};

}

}
