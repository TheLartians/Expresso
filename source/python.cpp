
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/operators.hpp>

#include <Python.h>

#include <iostream>
#include <stdexcept>

#include <lars/to_string.h>
#include <lars/iterators.h>

#include <boost/python/implicit.hpp>

#include "expression/core.h"
#include "expression/evaluator.h"
#include "expression/algorithms.h"

namespace python = boost::python;


namespace symbols_wrapper {
  
  using namespace symbols;
  
  using expression_ptr = std::shared_ptr<const symbols::Expression>;
  using Object = symbols::Data<python::object>;

  expression create_symbol(const std::string &name){
    return make_expression<symbols::Symbol>(name) ;
  }
  
  expression create_wildcard_symbol(const std::string &name){
    return make_expression<symbols::WildcardSymbol>(name) ;
  }
  
  expression create_object(const python::object &o,const std::string &rep){
    return make_expression<Object>(o,rep);
  }

  argument_list get_argument_list(const python::object &args){
    return argument_list( (python::stl_input_iterator< expression >( args )), python::stl_input_iterator< expression >( ) );
  }
  
  expression create_call(const Function &F,const python::object &args){
    return F(get_argument_list(args))->shared_from_this();
  }
  
  template <class Function> expression call_function(const python::tuple &args,const python::dict & kwargs){
    if(len(kwargs) > 0) throw std::runtime_error("function does not accept named arguments");
    const Function & F = python::extract<Function &>(args[0]);
    return F(get_argument_list(args.slice(1, python::_)));
  }

  struct expression_to_ptr{
    static PyObject * convert(const expression & e){
      auto obj = python::object( e->shared_from_this() );
      return boost::python::incref(obj.ptr());
    }
  };
  
  python::object get_arguments(const expression &e){
    auto f = e->as<Function>();
    if(!f) return python::object();
    return python::object( boost::ref(f->arguments) );
  }
  
  boost::shared_ptr<Rule> create_rule(const expression &a,const expression &b,const python::object f){
    return boost::shared_ptr<Rule>(new Rule(a,b,[f](replacement_map &m,EvaluatorVisitor &v)->bool{
      auto res = f(boost::ref(m));
      if(res == python::object()) return true;
      return python::extract<bool>(res);
    }));
  }
  
  boost::shared_ptr<MatchCondition> create_match_condition(const std::string &name,const python::object f){
    return boost::shared_ptr<MatchCondition>(new MatchCondition(name,[f](const expression &expr){
      bool res = python::extract<bool>(f(boost::ref(expr)));
      return res;
    }));
  }
  
  class replacement_map_policies:public python::detail::final_map_derived_policies<replacement_map, false> {
    public:
    
    static bool compare_index(replacement_map& container, index_type a, index_type b){
      return a < b;
    }
    
  };
  
  replacement_map commutative_match(expression expr,expression search){
    replacement_map rep;
    if(symbols::commutative_match(expr, search, rep)){
      return std::move(rep);
    }
    throw std::runtime_error("expressions do not match");
  }
  
}


template <class C> void create_iterator(C & c){
  using namespace boost::python;
  
  std::string iterator_name = std::string(boost::python::extract<std::string>(c.attr("__name__"))) + "_iterator";
  using iterator = typename C::wrapped_type::iterator;
  using value_type = typename C::wrapped_type::iterator::value_type;
  
  class iterator_wrapper{
    iterator bit,eit;
  public:
    iterator_wrapper(const iterator & begin,const iterator & end):bit(begin),eit(end){}
    value_type next(){
      if(bit == eit){
        boost::python::objects::stop_iteration_error();
        throw std::runtime_error("boot python didn't abort iteration");
      }
      value_type v = *bit; ++bit; return v;
    }
  };
  
  class_<iterator_wrapper>(iterator_name.c_str(),no_init)
  .def("next",&iterator_wrapper::next)
  ;
  
  c.def("__iter__",+[](const typename C::wrapped_type &o){ return iterator_wrapper(o.begin(),o.end()); });
  
}

BOOST_PYTHON_MODULE(_symbols){
#pragma mark -

  using namespace boost::python;
  
  class_<symbols::argument_list,boost::noncopyable>("argument_list")
  .def(vector_indexing_suite<symbols::argument_list>())
  .def("__repr__",+[](const symbols::argument_list &l){ return str(list(l)); });
  
  class_<symbols::replacement_map>("replacement_map")
  .def(init<const symbols::replacement_map &>())
  .def(map_indexing_suite<symbols::replacement_map,false,symbols_wrapper::replacement_map_policies>());
  
  register_ptr_to_python<std::shared_ptr<const symbols::Expression>>();
  register_ptr_to_python<std::shared_ptr<const symbols::Function>>();
  register_ptr_to_python<std::shared_ptr<const symbols::UnaryOperator>>();
  register_ptr_to_python<std::shared_ptr<const symbols::BinaryOperator>>();
  
#pragma mark Expression

  class_<symbols::expression>("Expression",init<symbols::expression>())
  .def("__repr__",lars::to_string<symbols::expression>)
  .def("is_function",+[](const symbols::expression &e){ return e->is_function(); })
  .def("is_atomic",+[](const symbols::expression &e){ return e->is_atomic(); })
  .def("is_symbol",+[](const symbols::expression &e){ return e->is_atomic() && e->is<symbols::Symbol>(); })
  .def("is_wildcard_symbol",+[](const symbols::expression &e){ return e->is<symbols::WildcardSymbol>(); })
  .def("is_wildcard_function",+[](const symbols::expression &e){ return e->is<symbols::WildcardFunction>(); })
  .def("get_arguments",symbols_wrapper::get_arguments)
  .def("get_value",+[](const symbols::expression &e){
    auto o = e->as<symbols_wrapper::Object>();
    if(!o) return object();
    return o->get_value();
  })
  .def("__hash__",+[](const symbols::expression &e){ return e->get_hash().quick_hash; })
  .def(self == self)
  .def(self != self)
  .def(self < self)
  .def("function",+[](const symbols::expression &e){
    auto f = e->as<symbols::Function>();
    if(f) return object( f );
    return object();
  });
  
  def("create_symbol", symbols_wrapper::create_symbol);
  def("create_object", symbols_wrapper::create_object);
  def("create_wildcard_symbol", symbols_wrapper::create_wildcard_symbol);

  def("create_call",symbols_wrapper::create_call);
  
  def("match",symbols_wrapper::commutative_match);
  def("replace",+[](const symbols::expression &s,const symbols::replacement_map &r){ return symbols::replace(s,r); });
  
#pragma mark -
#pragma mark Function
  
  class_<symbols::Function>("Function",init<std::string>())
  .def("get_name", +[](const symbols::Function &f)->std::string{ return f.get_name(); } )
  .def("__call__", raw_function(symbols_wrapper::call_function<symbols::Function>,1))
  .def("__repr__",+[](const symbols::Function &f)->std::string{ return f.get_name(); })
  .def("get_symbol",+[](const symbols::Function &o)->std::string{ return ""; });
  
#pragma mark WildcardFunction
  
  class_<symbols::WildcardFunction,bases<symbols::Function>>("WildcardFunction",init<std::string>());
  
#pragma mark Operator
  
  class_<symbols::Operator,bases<symbols::Function>,boost::noncopyable>("Operator",no_init)
  .def("get_symbol",+[](const symbols::Operator &o)->std::string{ return o.get_symbol(); })
  .def("get_precedence",&symbols::Operator::get_precedence);
  
#pragma mark UnaryOperator
  
  enum_<symbols::UnaryOperator::fix_type>("fix_type")
  .value("prefix", symbols::UnaryOperator::fix_type::prefix)
  .value("postfix", symbols::UnaryOperator::fix_type::postfix);
  
  boost::python::scope().attr("prefix") = symbols::UnaryOperator::fix_type::prefix;
  boost::python::scope().attr("postfix") = symbols::UnaryOperator::fix_type::postfix;
  
  class_<symbols::UnaryOperator,bases<symbols::Operator>>("UnaryOperator",init<std::string,symbols::UnaryOperator::fix_type,int>())
  .def("is_prefix", &symbols::UnaryOperator::is_prefix)
  .def("is_postfix", &symbols::UnaryOperator::is_postfix);

#pragma mark BinaryOperator
  
  enum_<symbols::BinaryOperator::associativity_type>("associativity_type")
  .value("associative", symbols::BinaryOperator::associativity_type::associative)
  .value("non_associative", symbols::BinaryOperator::associativity_type::non_associative);
  boost::python::scope().attr("associative") = symbols::BinaryOperator::associativity_type::associative;
  boost::python::scope().attr("non_associative") = symbols::BinaryOperator::associativity_type::non_associative;
  
  enum_<symbols::BinaryOperator::commutativity_type>("commutativity_type")
  .value("commutative", symbols::BinaryOperator::commutativity_type::commutative)
  .value("non_commutative", symbols::BinaryOperator::commutativity_type::non_commutative);
  boost::python::scope().attr("commutative") = symbols::BinaryOperator::commutativity_type::commutative;
  boost::python::scope().attr("non_commutative") = symbols::BinaryOperator::commutativity_type::non_commutative;
  
  class_<symbols::BinaryOperator,bases<symbols::Operator>>("BinaryOperator",init<std::string,int>())
  .def(init<std::string,symbols::BinaryOperator::associativity_type,symbols::BinaryOperator::commutativity_type,int>())
  .def("is_associative", &symbols::BinaryOperator::is_associative)
  .def("is_commutative", &symbols::BinaryOperator::is_commutative);
  
#pragma mark MatchCondition

  class_<symbols::MatchCondition,bases<symbols::Function>>("MatchCondition",no_init)
  .def("__init__",make_constructor(symbols_wrapper::create_match_condition));
  
#pragma mark -
#pragma mark Evaluator
  
  class_<symbols::Evaluator,boost::noncopyable>("Evaluator",no_init)
  .def("__call__",+[](const symbols::Evaluator &r,const symbols::expression &e){ return r(e); });

#pragma mark MultiEvaluator
  
  class_<symbols::MultiEvaluator,bases<symbols::Evaluator>>("MultiEvaluator")
  .def("add_evaluator",+[](symbols::MultiEvaluator &m,symbols::Evaluator &e){ m.add_evaluator(&e); })
  ;
  
#pragma mark Rule
  
  class_<symbols::Rule>("Rule",init<symbols::expression,symbols::expression>())
  .def("__init__",make_constructor(symbols_wrapper::create_rule))
  .def(init<const symbols::Rule &>())
  .def("has_evaluator",+[](const symbols::Rule &r){ return bool(r.evaluator); })
  .def_readonly("search",&symbols::Rule::search)
  .def_readonly("replacement",&symbols::Rule::replacement)
  .def("__repr__",lars::to_string<symbols::Rule>);
  
#pragma mark RuleEvaluator
  
  class_<symbols::RuleEvaluator,bases<symbols::Evaluator>>("RuleEvaluator")
  .def_readwrite("recursive",&symbols::RuleEvaluator::recursive)
  .def("add_rule",&symbols::RuleEvaluator::add_rule<symbols::Rule>)
  .def("add_rule",+[](symbols::RuleEvaluator &e,const symbols::Rule &r,int p){
    e.add_rule(r,p);
  })
  .def("add_rule",&symbols::RuleEvaluator::add_rule<symbols::expression,symbols::expression>)
  .def("__len__",&symbols::RuleEvaluator::size)
  .def("get_rule",+[](const symbols::RuleEvaluator &r,size_t idx){
    if(idx < r.size()) return r.get_rule(idx);
    else throw std::range_error("invalid rule index");
  })
  .def("set_apply_callback",+[](symbols::RuleEvaluator &r,object f){
    if(f)r.apply_callback = [=](const symbols::Rule &r,const symbols::replacement_map &m){
      f(r,m);
    };
    else r.apply_callback = symbols::RuleEvaluator::CallbackFunction();
  })
  ;
  
#pragma mark Traversal
  
  class_<symbols::postorder_traversal> postorder_traversal("postorder_traversal",init<symbols::expression>());
  create_iterator(postorder_traversal);
  
  class_<symbols::preorder_traversal> preorder_traversal("preorder_traversal",init<symbols::expression>());
  create_iterator(preorder_traversal);
  
  class_<symbols::commutative_permutations> commutative_permutations("commutative_permutations",init<symbols::expression>());
  create_iterator(commutative_permutations);
  
#pragma mark groups and fields
  
  class_<symbols::group>("Group",init<const symbols::Function &,const symbols::Function &,const symbols::expression &>())
  .def("get_operation", +[](const symbols::group &g)->const symbols::Function &{
    return g.operation;
  },return_internal_reference<>())
  .def("get_inverse", +[](const symbols::group &g)->const symbols::Function &{
    return g.inverse;
  },return_internal_reference<>())
  .def_readonly("neutral", &symbols::group::neutral);

  class_<symbols::field>("Field",init<const symbols::group &,const symbols::group &>())
  .def_readonly("additive_group", &symbols::field::additive_group)
  .def_readonly("multiplicative_group", &symbols::field::multiplicative_group);
  
#pragma mark MulplicityList
  
  class_<symbols::mulplicity_list::value_type>("Mulplicity",init<symbols::expression,symbols::expression>())
  .def_readonly("value", &symbols::mulplicity_list::value_type::first)
  .def_readonly("mulplicity", &symbols::mulplicity_list::value_type::second);
  
  class_<symbols::mulplicity_list>("MulplicityList",init<const symbols::group &,const symbols::Function &,const symbols::field &>())
  .def(init<const symbols::expression &,const symbols::group &,const symbols::Function &,const symbols::field &>())
  .def("__iter__",iterator<symbols::mulplicity_list>())
  .def("__len__",&symbols::mulplicity_list::size)
  .def("__getitem__",+[](const symbols::mulplicity_list &m,size_t idx){ return m[idx]; })
  .def("intersection",+[](const symbols::mulplicity_list &m,const symbols::mulplicity_list &other){ return m.intersection(other); })
  .def("intersection",+[](const symbols::mulplicity_list &m,const symbols::mulplicity_list &other,object f){
    return m.intersection(other,[&](const symbols::expression &a,const symbols::expression &b)->symbols::expression{
      auto res =  f(a,b);
      if(res == object()) return symbols::expression();
      return extract<symbols::expression>(res);
    });
  })
  .def("difference",+[](const symbols::mulplicity_list &m,const symbols::mulplicity_list &other){ return m.difference(other); })
  .def("sum",+[](const symbols::mulplicity_list &m,const symbols::mulplicity_list &other){ return m.sum(other); })
  .def("power",&symbols::mulplicity_list::power)
  .def("as_expression",&symbols::mulplicity_list::as_expression)
  .def("get_real_field",+[](const symbols::mulplicity_list &m)->const symbols::field &{ return m.real_field; },return_internal_reference<>())
  .def("get_base",+[](const symbols::mulplicity_list &m)->const symbols::group &{ return m.base; },return_internal_reference<>())
  .def("get_mulplicity_function",+[](const symbols::mulplicity_list &m)->const symbols::Function &{ return m.mulplicity; },return_internal_reference<>())
  ;

  
}


