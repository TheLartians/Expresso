
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

#include "expresso/core.h"
#include "expresso/evaluator.h"
#include "expresso/algorithms.h"

namespace python = boost::python;

namespace boost { 
  
  template <class T> const T * get_pointer(const std::shared_ptr<const T> & ptr) {
    return ptr.get();
  }

  template <class T> const T * get_pointer(std::shared_ptr<const T> & ptr) {
    return ptr.get();
  }

}


namespace expresso_wrapper {
  
  using namespace expresso;
  
  using expression_ptr = std::shared_ptr<const expresso::Expression>;
  using Object = expresso::Data<python::object>;

  expression create_symbol(const std::string &name){
    return make_expression<expresso::Symbol>(name) ;
  }
  
  expression create_wildcard_symbol(const std::string &name){
    return make_expression<expresso::WildcardSymbol>(name) ;
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
  
  boost::shared_ptr<Rule> create_conditional_rule_f(const expression &a,const expression &b,const expression &c,const expression &t,const python::object f){
    return boost::shared_ptr<Rule>(new Rule(a,b,c,t,[f](replacement_map &m,EvaluatorVisitor &v)->bool{
      auto res = f(boost::ref(m));
      if(res == python::object()) return true;
      return python::extract<bool>(res);
    }));
  }
  
  boost::shared_ptr<Rule> create_conditional_rule(const expression &a,const expression &b,const expression &c,const expression &t){
    return boost::shared_ptr<Rule>(new Rule(a,b,c,t));
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
  
  python::object match(expression expr,expression search){
    replacement_map rep;
    if(expresso::match(expr, search, rep)){
      return python::object(rep);
    }
    return python::object();
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
      while(bit != eit){
        value_type v = *bit; ++bit; return v;
      }
      boost::python::objects::stop_iteration_error();
      throw std::runtime_error("boost didn't stop iteration");
    }
  };
  
  class_<iterator_wrapper>(iterator_name.c_str(),no_init)
  .def("next",&iterator_wrapper::next)
  ;
  
  c.def("__iter__",+[](const typename C::wrapped_type &o){ return iterator_wrapper(o.begin(),o.end()); });
}

BOOST_PYTHON_MODULE(_expresso){
#pragma mark -

  using namespace boost::python;
  
  class_<expresso::argument_list,boost::noncopyable>("argument_list")
  .def(vector_indexing_suite<expresso::argument_list>())
  .def("__repr__",+[](const expresso::argument_list &l){ return str(list(l)); });
  
  class_<expresso::replacement_map>("replacement_map")
  .def(init<const expresso::replacement_map &>())
  .def(map_indexing_suite<expresso::replacement_map,false,expresso_wrapper::replacement_map_policies>());
  
  register_ptr_to_python<std::shared_ptr<const expresso::Expression>>();
  register_ptr_to_python<std::shared_ptr<const expresso::Function>>();
  register_ptr_to_python<std::shared_ptr<const expresso::UnaryOperator>>();
  register_ptr_to_python<std::shared_ptr<const expresso::BinaryOperator>>();
  
#pragma mark Expression

  class_<expresso::expression>("Expression",init<expresso::expression>())
  .def("__repr__",lars::to_string<expresso::expression>)
  .def("is_function",+[](const expresso::expression &e){ return e->is_function(); })
  .def("is_atomic",+[](const expresso::expression &e){ return e->is_atomic(); })
  .def("is_symbol",+[](const expresso::expression &e){ return e->is_atomic() && e->is<expresso::Symbol>(); })
  .def("is_wildcard_symbol",+[](const expresso::expression &e){ return e->is<expresso::WildcardSymbol>(); })
  .def("is_wildcard_function",+[](const expresso::expression &e){ return e->is<expresso::WildcardFunction>(); })
  .def("get_arguments",expresso_wrapper::get_arguments)
  .def("get_value",+[](const expresso::expression &e){
    auto o = e->as<expresso_wrapper::Object>();
    if(!o) return object();
    return o->get_value();
  })
  .def("__hash__",+[](const expresso::expression &e){ return e->get_hash().quick_hash; })
  .def(self == self)
  .def(self != self)
  .def(self < self)
  .def("function",+[](const expresso::expression &e){
    auto f = e->as<expresso::Function>();
    if(f) return object( f );
    return object();
  });
  
  def("create_symbol", expresso_wrapper::create_symbol);
  def("create_object", expresso_wrapper::create_object);
  def("create_wildcard_symbol", expresso_wrapper::create_wildcard_symbol);

  def("create_call",expresso_wrapper::create_call);
  
  def("match",expresso_wrapper::match);
  def("replace",+[](const expresso::expression &s,const expresso::replacement_map &r){
    return expresso::replace(s,r);
  });
  
#pragma mark -
#pragma mark Function
  
  class_<expresso::Function>("Function",init<std::string>())
  .def("get_name", +[](const expresso::Function &f)->std::string{ return f.get_name(); } )
  .def("__call__", raw_function(expresso_wrapper::call_function<expresso::Function>,1))
  .def("__repr__",+[](const expresso::Function &f)->std::string{ return f.get_name(); })
  .def("get_symbol",+[](const expresso::Function &o)->std::string{ return ""; });
  
#pragma mark WildcardFunction
  
  class_<expresso::WildcardFunction,bases<expresso::Function>>("WildcardFunction",init<std::string>());
  
#pragma mark Operator
  
  class_<expresso::Operator,bases<expresso::Function>,boost::noncopyable>("Operator",no_init)
  .def("get_symbol",+[](const expresso::Operator &o)->std::string{ return o.get_symbol(); })
  .def("get_precedence",&expresso::Operator::get_precedence);
  
#pragma mark UnaryOperator
  
  enum_<expresso::UnaryOperator::fix_type>("fix_type")
  .value("prefix", expresso::UnaryOperator::fix_type::prefix)
  .value("postfix", expresso::UnaryOperator::fix_type::postfix);
  
  boost::python::scope().attr("prefix") = expresso::UnaryOperator::fix_type::prefix;
  boost::python::scope().attr("postfix") = expresso::UnaryOperator::fix_type::postfix;
  
  class_<expresso::UnaryOperator,bases<expresso::Operator>>("UnaryOperator",init<std::string,expresso::UnaryOperator::fix_type,int>())
  .def("is_prefix", &expresso::UnaryOperator::is_prefix)
  .def("is_postfix", &expresso::UnaryOperator::is_postfix);

#pragma mark BinaryOperator
  
  enum_<expresso::BinaryOperator::associativity_type>("associativity_type")
  .value("associative", expresso::BinaryOperator::associativity_type::associative)
  .value("non_associative", expresso::BinaryOperator::associativity_type::non_associative);
  boost::python::scope().attr("associative") = expresso::BinaryOperator::associativity_type::associative;
  boost::python::scope().attr("left_associative") = expresso::BinaryOperator::associativity_type::left_associative;
  boost::python::scope().attr("right_associative") = expresso::BinaryOperator::associativity_type::right_associative;
  boost::python::scope().attr("non_associative") = expresso::BinaryOperator::associativity_type::non_associative;
  
  enum_<expresso::BinaryOperator::commutativity_type>("commutativity_type")
  .value("commutative", expresso::BinaryOperator::commutativity_type::commutative)
  .value("non_commutative", expresso::BinaryOperator::commutativity_type::non_commutative);
  boost::python::scope().attr("commutative") = expresso::BinaryOperator::commutativity_type::commutative;
  boost::python::scope().attr("non_commutative") = expresso::BinaryOperator::commutativity_type::non_commutative;
  
  class_<expresso::BinaryOperator,bases<expresso::Operator>>("BinaryOperator",init<std::string,int>())
  .def(init<std::string,expresso::BinaryOperator::associativity_type,expresso::BinaryOperator::commutativity_type,int>())
  .def("is_associative", &expresso::BinaryOperator::is_associative)
  .def("is_commutative", &expresso::BinaryOperator::is_commutative);
  
#pragma mark MatchCondition

  class_<expresso::MatchCondition,bases<expresso::Function>>("MatchCondition",no_init)
  .def("__init__",make_constructor(expresso_wrapper::create_match_condition));
  
#pragma mark -
#pragma mark Evaluator
  
  class_<expresso::EvaluatorVisitor,boost::noncopyable>("EvaluatorVisitor",no_init)
  .def("evaluate",&expresso::EvaluatorVisitor::evaluate);
  
  class_<expresso::Evaluator::settings_t,boost::noncopyable>("Evaluator.settings",no_init)
  .def_readwrite("recursive",&expresso::Evaluator::settings_t::recursive)
  .def_readwrite("split_binary",&expresso::Evaluator::settings_t::split_binary)
  .def_readwrite("split_binary_size",&expresso::Evaluator::settings_t::split_binary_size)
  .def_readwrite("commutate_binary",&expresso::Evaluator::settings_t::commutate_binary)
  ;
  
  class_<expresso::Evaluator,boost::noncopyable>("Evaluator",no_init)
  .def_readwrite("settings",&expresso::Evaluator::settings)
  .def("__call__",+[](const expresso::Evaluator &r,const expresso::expression &e){ return r(e); })
  .def("__call__",+[](const expresso::Evaluator &r,const expresso::expression &e,expresso::replacement_map &m){ return r(e,m); })
  ;

#pragma mark MultiEvaluator
  
  class_<expresso::MultiEvaluator,bases<expresso::Evaluator>>("MultiEvaluator")
  .def("add_evaluator",+[](expresso::MultiEvaluator &m,expresso::Evaluator &e){ m.add_evaluator(&e); })
  ;
  
  class_<expresso::StepEvaluator,bases<expresso::Evaluator>>("StepEvaluator")
  .def("add_evaluator",+[](expresso::StepEvaluator &m,expresso::Evaluator &e){ m.add_evaluator(&e); })
  ;
  
#pragma mark ReplaceEvaluator
  
  class_<expresso::ReplaceEvaluator,bases<expresso::Evaluator>>("ReplaceEvaluator")
  .def(init<const expresso::replacement_map &>())
  .def("add_replacement",&expresso::ReplaceEvaluator::add_replacement)
  .def("clear",&expresso::ReplaceEvaluator::clear)
  ;
    
#pragma mark Rule
  
  class_<expresso::Rule>("Rule",init<expresso::expression,expresso::expression>())
  .def("__init__",make_constructor(expresso_wrapper::create_rule))
  .def("__init__",make_constructor(expresso_wrapper::create_conditional_rule))
  .def("__init__",make_constructor(expresso_wrapper::create_conditional_rule_f))
  .def(init<const expresso::Rule &>())
  .def("has_evaluator",+[](const expresso::Rule &r){ return bool(r.evaluator); })
  .def_readonly("search",&expresso::Rule::search)
  .def_readonly("replacement",&expresso::Rule::replacement)
  .def("get_condition",+[](const expresso::Rule &r){ if(r.condition) return object(r.condition); return object(); })
  .def("__repr__",lars::to_string<expresso::Rule>);
  
#pragma mark RuleEvaluator
  
  class_<expresso::RuleEvaluator,bases<expresso::Evaluator>>("RuleEvaluator")
  .def("add_rule",&expresso::RuleEvaluator::add_rule<expresso::Rule>)
  .def("add_rule",+[](expresso::RuleEvaluator &e,const expresso::Rule &r,int p){
    e.add_rule(r,p);
  })
  .def("add_rule",&expresso::RuleEvaluator::add_rule<expresso::expression,expresso::expression>)
  .def("__len__",&expresso::RuleEvaluator::size)
  .def("get_rule",+[](const expresso::RuleEvaluator &r,size_t idx){
    if(idx < r.size()) return r.get_rule(idx);
    else throw std::range_error("invalid rule index");
  })
  .def("set_apply_callback",+[](expresso::RuleEvaluator &r,object f){
    if(f)r.apply_callback = [=](const expresso::Rule &r,const expresso::replacement_map &m){
      f(r,m);
    };
    else r.apply_callback = expresso::RuleEvaluator::CallbackFunction();
  })
  ;
  
#pragma mark Traversal
  
  class_<expresso::postorder_traversal> postorder_traversal("postorder_traversal",init<expresso::expression>());
  create_iterator(postorder_traversal);
  
  class_<expresso::preorder_traversal> preorder_traversal("preorder_traversal",init<expresso::expression>());
  create_iterator(preorder_traversal);
  
  class_<expresso::commutative_permutations> commutative_permutations("commutative_permutations",init<expresso::expression>());
  create_iterator(commutative_permutations);
  
#pragma mark groups and fields
  
  class_<expresso::group>("Group",init<const expresso::Function &,const expresso::Function &,const expresso::expression &>())
  .def("get_operation", +[](const expresso::group &g)->const expresso::Function &{
    return g.operation;
  },return_internal_reference<>())
  .def("get_inverse", +[](const expresso::group &g)->const expresso::Function &{
    return g.inverse;
  },return_internal_reference<>())
  .def_readonly("neutral", &expresso::group::neutral);

  class_<expresso::field>("Field",init<const expresso::group &,const expresso::group &>())
  .def_readonly("additive_group", &expresso::field::additive_group)
  .def_readonly("multiplicative_group", &expresso::field::multiplicative_group);
  
#pragma mark MulplicityList
  
  class_<expresso::mulplicity_list::value_type>("Mulplicity",init<expresso::expression,expresso::expression>())
  .def_readonly("value", &expresso::mulplicity_list::value_type::first)
  .def_readonly("mulplicity", &expresso::mulplicity_list::value_type::second);
  
  class_<expresso::mulplicity_list>("MulplicityList",init<const expresso::group &,const expresso::Function &,const expresso::field &>())
  .def(init<const expresso::expression &,const expresso::group &,const expresso::Function &,const expresso::field &>())
  .def("__iter__",iterator<expresso::mulplicity_list>())
  .def("__len__",&expresso::mulplicity_list::size)
  .def("__getitem__",+[](const expresso::mulplicity_list &m,size_t idx){ return m[idx]; })
  .def("intersection",+[](const expresso::mulplicity_list &m,const expresso::mulplicity_list &other){ return m.intersection(other); })
  .def("intersection",+[](const expresso::mulplicity_list &m,const expresso::mulplicity_list &other,object f){
    return m.intersection(other,[&](const expresso::expression &a,const expresso::expression &b)->expresso::expression{
      auto res =  f(a,b);
      if(res == object()) return expresso::expression();
      return extract<expresso::expression>(res);
    });
  })
  .def("difference",+[](const expresso::mulplicity_list &m,const expresso::mulplicity_list &other){ return m.difference(other); })
  .def("sum",+[](const expresso::mulplicity_list &m,const expresso::mulplicity_list &other){ return m.sum(other); })
  .def("power",&expresso::mulplicity_list::power)
  .def("as_expression",&expresso::mulplicity_list::as_expression)
  .def("get_real_field",+[](const expresso::mulplicity_list &m)->const expresso::field &{ return m.real_field; },return_internal_reference<>())
  .def("get_base",+[](const expresso::mulplicity_list &m)->const expresso::group &{ return m.base; },return_internal_reference<>())
  .def("get_mulplicity_function",+[](const expresso::mulplicity_list &m)->const expresso::Function &{ return m.mulplicity; },return_internal_reference<>())
  ;

  
}


