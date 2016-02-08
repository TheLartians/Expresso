
// The MIT License (MIT)
//
// Copyright (c) 2016 Lars Melchior
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#pragma once

#include <memory>
#include <vector>
#include <string>
#include <ostream>
#include <initializer_list>
#include <sha256/sha256.h>
#include <unordered_map>
#include <type_traits>
#include <functional>
#include <typeinfo>

#include <lars/shared_object.h>
#include <lars/to_string.h>

namespace expresso {
  
  class Expression;
  class AtomicExpression;
  class Symbol;
  class WildcardSymbol;
  class Function;
  class WildcardFunction;
  class BinaryOperator;
  class UnaryOperator;
  class CompressedNode;
  class MatchCondition;
  class Tupel;
  class DataExpression;
  
  using string = lars::shared_object<std::string>;
  
  struct Visitor{
    virtual void visit(const Function * e) = 0;
    virtual void visit(const AtomicExpression * e) = 0;
    
    virtual void visit(const Tupel * e){ visit((const Function *)e); }
    virtual void visit(const BinaryOperator * e){ visit((const Function *)e); }
    virtual void visit(const UnaryOperator * e){ visit((const Function *)e); }
    virtual void visit(const Symbol * e){ visit((const AtomicExpression*)e); }
    virtual void visit(const WildcardSymbol * e){ visit((const AtomicExpression*)e); }
    virtual void visit(const WildcardFunction * e){ visit((const Function*)e); }
    virtual void visit(const MatchCondition * e){ visit((const Function*)e); }
    virtual void visit(const CompressedNode * e){ visit((const Function*)e); }
    virtual void visit(const DataExpression * e){ visit((const AtomicExpression*)e); }
  };
  
  template <class Expr> struct DataVisitor:public virtual Visitor{
    using Visitor::visit;
    virtual void visit(const Expr * e){ Visitor::visit((const DataExpression*)e); }
  };
  
  class Expression:public std::enable_shared_from_this<Expression>{
    
  public:
    using hash_t = sha256_hash;

  private:
    mutable hash_t hash;
    friend Function;
    
    public:
    
    //using shared = std::shared_ptr<const Expression>;
    
    class shared:private std::shared_ptr<const Expression>{
      public:
      shared(){}
      shared(const std::shared_ptr<const Expression> &e):std::shared_ptr<const Expression>(e){}
      template<class E>shared(const std::shared_ptr<const E> &e):std::shared_ptr<const Expression>(e){}
      
      using std::shared_ptr<const Expression>::operator->;
      using std::shared_ptr<const Expression>::operator*;
      using std::shared_ptr<const Expression>::operator bool;
      
      bool operator==(const Expression::shared &other)const{ if(!*this) return !other; return *other.get() == (*get()); }
      bool operator!=(const Expression::shared &other)const{ if(!*this) return bool(other); if(!other) return bool(*this); return *other.get() != (*get());  }
      bool operator!=(const Expression * other)const{ if(!*this) return true; return *other != (*get());  }
      bool operator<(const Expression::shared & other)const{ return *get() < *other; }
      
      shared & operator=(const shared &other){ std::shared_ptr<const Expression>::operator=(other); return *this; }

      using std::shared_ptr<const Expression>::get;
      using std::shared_ptr<const Expression>::reset;
      
    };//*/
    
    virtual void accept(Visitor * v) const = 0;
    virtual ~Expression(){}
    
    // type
    template <class R> const std::shared_ptr<const R> as() const { return std::dynamic_pointer_cast<const R>(shared_from_this()); }
    template <class R> const std::shared_ptr<const R> static_as() const { return std::static_pointer_cast<const R>(shared_from_this()); }
    template <class R> bool is() const { return dynamic_cast<const R*>(this); }
    
    shared get_shared()const{ return shared(shared_from_this()); }
    operator shared()const{ return get_shared(); }
    
    // properties
    virtual bool is_identical(const Expression * other) const = 0;
    bool is_identical(const shared &other)const{ return is_identical(other.get()); }
    
    // hash
    virtual void generate_hash(sha256_hash &hash) const = 0;
    void invalidate_hash()const{ hash.invalidate(); }
    
    const hash_t & get_hash()const{
      if(!hash) generate_hash(hash);
      return hash;
    }
    
    bool operator==(const Expression &other)const{ return other.get_hash() == get_hash(); }
    bool operator!=(const Expression &other)const{ return other.get_hash() != get_hash(); }
    bool operator<(const Expression & other)const;

    virtual bool is_function()const = 0;
    bool is_atomic()const{ return !is_function(); }
  };
  
  using expression = Expression::shared;
  
  template <class T,typename ... Args> expression make_expression(Args... args){
    return expression(std::make_shared<T>( std::forward<Args>(args)... ));
  }
  
  struct MinimalVisitor:public Visitor{
    using Visitor::visit;
    void visit(const Function * e)override;
    void visit(const AtomicExpression * e)override{ }
  };
  
  class AtomicExpression:public Expression{
    public:
    virtual string get_representation()const = 0;
    void accept(Visitor * v)const override{ v->visit(this); }
    void generate_hash(sha256_hash &hash)const override{ sha256(get_representation(), hash); }
    bool is_function()const override{ return false; }
  };
  
  class Symbol:public AtomicExpression{
    string name;
  public:
    Symbol(string _name):name(_name){}
    using AtomicExpression::accept;
    void accept(Visitor * v)const override{ v->visit(this); };
    using Expression::is_identical; bool is_identical(const Expression * other)const override{ if(auto o = other->as<Symbol>()) return o->name==name; return false; }
    string get_representation()const override{ return name; }
    const string & get_name()const{ return name; }
  };
  
  class WildcardSymbol:public AtomicExpression{
  private:
    string id;
  public:
    WildcardSymbol(const string &_id,bool _open = false):id(_id){}
    const string & get_id() const { return id; }
    void accept(Visitor * v)const override{ v->visit(this); }
    using Expression::is_identical; bool is_identical(const Expression * other)const override{ if(auto o = other->as<WildcardSymbol>()) return o->get_id()==get_id(); return false; }
    string get_representation()const override{ return string("$") + get_id(); }
  };
  
  class Function: public Expression{
  public:
    using argument_list = std::vector<shared>;
    
  private:
    
    const string name;
    void generate_hash(sha256_hash &hash)const override;
    
  protected:
    mutable argument_list argument_data;
    
  public:
    const argument_list &arguments;
    
    Function(const string &_name,argument_list && _args = argument_list());
    const string & get_name()const{ return name; }
    void accept(Visitor * v)const override{ v->visit(this); }
    using Expression::is_identical; bool is_identical(const Expression * other)const override{ if(auto o = other->as<Function>())return o->get_name() == get_name(); return false; }
    virtual shared clone(argument_list && args)const{ return make_expression<Function>(get_name(),std::forward<argument_list>(args)); }
    shared clone(expression arg)const{ return clone(argument_list{arg}); }
    
    template <typename ... Args> shared operator()(Args... args)const{ return clone(argument_list{args...}); }
    bool is_function()const override{ return true; }
  };
  
  using argument_list = Function::argument_list;
  
  class WildcardFunction:public Function{
    const string id;
  public:
    WildcardFunction(const string &_id,argument_list &&_args = argument_list()):Function(string("$") + _id,std::forward<argument_list>(_args)),id(_id){}
    const string & get_id() const { return id; }
    void accept(Visitor * v)const override{ v->visit(this); }
    shared clone(argument_list && args)const override{ return make_expression<WildcardFunction>(get_id(),std::forward<argument_list>(args)); }
    using Expression::is_identical; bool is_identical(const Expression * other)const override{ if(auto o = other->as<WildcardFunction>()) return o->get_id()==get_id(); return false; }
  };
  
  class Operator:public Function{
    const string symbol;
    const int precedence;
  public:
    Operator(const string &name,const string &_symbol,int _precedence,argument_list &&args = argument_list()):Function(name,std::forward<argument_list>(args)),symbol(_symbol),precedence(_precedence){ }
    int get_precedence()const{ return precedence; }
    const string & get_symbol()const{ return symbol; }
    virtual shared clone(argument_list && args)const override= 0;
  };
  
  class UnaryOperator:public Operator{
  public:
    enum fix_type{ prefix,postfix };
  private:
    const fix_type fix;
  public:
    UnaryOperator(const string &_name,const string &_symbol,fix_type _fix,int precedence,argument_list &&args):Operator(_name,_symbol,precedence,std::forward<argument_list>(args)),fix(_fix){ }
    UnaryOperator(const string &_symbol,fix_type _fix,int precedence,expression arg = expression()):Operator(string(_fix == prefix?"__prefix_":"__postfix_")+_symbol,_symbol,precedence,{arg}),fix(_fix){ }
    void accept(Visitor * v)const override{ v->visit(this); }
    using Function::clone;
    shared clone(argument_list && args)const override{ return make_expression<UnaryOperator>(get_name(),get_symbol(),fix,get_precedence(),std::forward<argument_list>(args)); }
    bool is_prefix()const{ return fix == prefix; }
    bool is_postfix()const{ return fix == postfix; }
  };
  
  class BinaryOperator:public Operator{
    
    void finalize_arguments(argument_list &args)const;
    
  public:
    enum associativity_type:char{ associative='a',left_associative = 'l',right_associative = 'r',non_associative='n' };
    enum commutativity_type:char{ non_commutative='n',commutative='c' };
    
    static string create_name(const string &_symbol,associativity_type a ,commutativity_type c){
      return "__binary_"+*string(a=='a'?"a":"")+*string(c=='c'?"c":"")+"_"+*_symbol;
    }
    
    const associativity_type associativity;
    const commutativity_type commutativity;
    
    bool is_associative()const{ return associativity == associative; }
    bool is_commutative()const{ return commutativity == commutative; }
    
    BinaryOperator(const string &name,const string &_symbol,int precedence,argument_list &&args,associativity_type a ,commutativity_type c,bool finalize = true):Operator(name,_symbol,precedence,std::forward<argument_list>(args)),associativity(a),commutativity(c){ if(finalize) finalize_arguments(argument_data); }
    BinaryOperator(const string &_symbol,associativity_type a,commutativity_type c,int precedence,argument_list &&args = argument_list()):Operator(create_name(_symbol,a,c),_symbol,precedence,std::forward<argument_list>(args)),associativity(a),commutativity(c){ finalize_arguments(argument_data); }
    BinaryOperator(const string &_symbol,int precedence,argument_list &&args = argument_list()):Operator(create_name(_symbol,non_associative,non_commutative),_symbol,precedence,std::forward<argument_list>(args)),associativity(non_associative),commutativity(non_commutative){ finalize_arguments(argument_data); }
    void accept(Visitor * v)const override{ v->visit(this); }
    shared clone(argument_list && args,bool finalize)const{
      if(args.size() == 1){ return args[0]; }
      return make_expression<BinaryOperator>(get_name(),get_symbol(),get_precedence(),std::forward<argument_list>(args),associativity,commutativity,finalize);
    }
    shared clone(argument_list && args)const override{ return clone(std::move(args),true); }

  };
  
  class Tupel:public Function{
    string open,close;
    public:
    Tupel(const string &_open,const string &_close,argument_list &&args = argument_list()):Function(string("__tupel_")+_open+string("_")+_close,std::forward<argument_list>(args)),open(_open),close(_close){}
    Tupel(const string &_open,const string &_close,const string &_name,argument_list &&args = argument_list()):Function(_name,std::forward<argument_list>(args)),open(_open),close(_close){}
    void accept(Visitor * v)const override{ v->visit(this); }
    const string & get_open()const{ return open; }
    const string & get_close()const{ return close; }
    shared clone(argument_list && args)const override{ return make_expression<Tupel>(get_open(),get_close(),get_name(),std::forward<argument_list>(args)); }
  };
  
  class DataExpression:public AtomicExpression{
    public:
    const std::type_info & type_id;
    DataExpression(const std::type_info & _type_id):type_id(_type_id){}
  };
  
  template <class D> class Data:public DataExpression{
  public:
    using value_type = D;
  private:
    const value_type value;
    string representation;
  public:
    Data(const value_type &_value):DataExpression(typeid(D)),value(_value),representation(lars::to_string(value)){ }
    Data(const value_type &_value,const string &rep):DataExpression(typeid(D)),value(_value),representation(rep){ }
    const value_type & get_value()const{ return value; }
    using AtomicExpression::accept;
    //using Expression::is_identical; bool is_identical(const Expression * other)const override{ if(auto o = other->as<Data<value_type>>()) return o->get_value() == get_value(); return false; }
    using Expression::is_identical; bool is_identical(const Expression * other)const override{
      if(auto o = other->as<Data<value_type>>()) return representation == o->representation;
      return false;
    }
    string get_representation()const override{ return representation; }
    void accept(Visitor * v)const override{ if(auto * dv = dynamic_cast<DataVisitor<Data>*>(v)) dv->visit(this); v->visit(this); };
  };
    
  class MatchCondition:public Function{
  public:
    using Condition = std::function<bool(const expression &)>;
    const Condition condition;
    
    MatchCondition(const string &name,const Condition &_condition,argument_list &&args = argument_list()):Function(name,std::forward<argument_list>(args)),condition(_condition){}
    
    void accept(Visitor * v)const override{ v->visit(this); };
    
    using Function::clone;
    shared clone(argument_list && args)const override{ return make_expression<MatchCondition>(get_name(),condition,args); }
  };
  
  class CompressedNode:public Function{
    using merge_function = std::function<bool(expression&,expression)>;
    using insert_function = std::function<void(expression&)>;
    
    static bool default_merge(expression& a,expression b){ return a->is_identical(b); }
    static void default_insert(expression& a){ }

    static string Name;

  public:
    
    using ID = unsigned;
    
    mutable std::vector<std::vector<ID>> indices;
    mutable std::unordered_map<ID, ID> reverse_indices;
    
    CompressedNode():Function(Name){  }
    CompressedNode(std::vector<std::vector<ID>> _indices,std::unordered_map<ID,ID> _reverse_indices,argument_list &&args):Function("__C__",std::forward<argument_list>(args)),indices(_indices),reverse_indices(_reverse_indices){  }
    
    unsigned size()const{ return reverse_indices.size(); }
    void accept(Visitor * v)const override{ v->visit(this); }
    void insert(expression,merge_function,insert_function,ID idx)const;
    void insert(expression e,merge_function merge = default_merge,insert_function ins = default_insert)const{ insert(e,merge,ins,size()); }
    
    shared clone(argument_list && args)const override{ return make_expression<CompressedNode>(indices,reverse_indices,arguments); }
    expression extract(ID idx)const;
  };
    
  std::ostream & operator<<( std::ostream &stream,Expression::shared expr );
  std::ostream & operator<<( std::ostream &stream,const Expression &expr );

}

#pragma mark -
  
namespace std{
  template<> struct hash<expresso::Expression::shared> {
    
    typedef expresso::Expression::shared argument_type;
    typedef std::size_t result_type;
    
    result_type operator()(argument_type const& s) const{
      return s->get_hash().quick_hash;
    }
    
  };
}

namespace expresso {
  struct replacement_map:public std::unordered_map<Expression::shared,Expression::shared>{
    using unordered_map::unordered_map;
    std::unordered_map<string,Expression::shared> functions;
  };
}




