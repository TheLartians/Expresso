
#include "core.h"

#include <lars/iterators.h>
#include <lars/hashers.h>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#include <iostream>

namespace expresso {
  using namespace lars;

  void MinimalVisitor::visit(const Function * e){
    for(auto arg:e->arguments) arg->accept(this);
  }
    
#pragma mark compare
  
  
  void check_less(const Expression &a,const Expression &b);
  
  bool less(const Expression &a,const Expression &b,bool check = true){
    
    if(check)check_less(a,b);
    
    return a.get_hash() < b.get_hash();
    
    
    if(auto A = a.as<AtomicExpression>()){
      if(auto B = b.as<AtomicExpression>()){
        return A->get_representation() < B->get_representation();
      }
      return true;
    }
    
    if(auto A = a.as<Function>()){
      if(auto B = b.as<Function>()){
        if(A->is_identical(B)){
          for(auto i:range(std::min(A->arguments.size(),B->arguments.size()))){
            if(A->arguments[i] != B->arguments[i]) return less(*A->arguments[i],*B->arguments[i]);
          }
        }
        else return A->get_name() < B->get_name();
      }
    }
    
    return false;
  }
  
  void check_less(const Expression &a,const Expression &b){
    
    if(!(less(a,b,false)^less(b,a,false))){
      if(a != b){
        std::cerr << "less brocken:\n" << a << std::endl;
        std::cerr << b << std::endl;
      }
    }
    else if(a == b){
      std::cerr << "less brocken:\n" << a << std::endl;
      std::cerr << b << std::endl;
    }
    
  }
  
  bool Expression::operator<(const Expression & other)const{
    return less(*this, other);
  }
  
#pragma mark Function
  
    void Function::generate_hash(sha256_hash &hash)const{
    auto ctx = SHA256();
    ctx.init();
    ctx.update( *get_name() );
    for(auto arg:arguments) ctx.update( arg->get_hash() );
    ctx.final(hash);
  }

  Function::Function(const string &_name,argument_list &&_args):name(_name),argument_data(std::forward<argument_list>(_args)),arguments(argument_data){
        
  }
  
  void BinaryOperator::finalize_arguments(argument_list &args)const{
    
    if(associativity == associative){
      auto it = args.begin(),end = args.end();
      while (it != end) {
        auto op = (*it)->as<BinaryOperator>();
        if(op && op->get_name() == get_name()){
          it = args.erase(it);
          
          auto i = it - args.begin() + op->arguments.size();
          args.insert(it,op->arguments.begin(), op->arguments.end());
          it = args.begin() + i;
          
          // it = args.insert(it,op->arguments.begin(), op->arguments.end()) + op->arguments.size(); // if gcc fails here, upgrade to at least version 4.9
          
          end = args.end();
        }
        else ++it;
      }
    }
    
    if(commutativity == commutative){
      std::sort(args.begin(), args.end());
    }
    
    //if(arguments.size() == 1) throw std::runtime_error("created binary operator with only one argument");
    
  }
  
#pragma mark compressed node
  
  string CompressedNode::Name = "__C__";
  
  void CompressedNode::insert(expression e,merge_function merge,insert_function ins,ID idx)const{
    unsigned i;
    auto f = e->as<Function>();
    
    if(argument_data.size() == 0) merge(e,this->get_shared());
    
    for(i=0;i<argument_data.size();++i){
      bool do_merge = merge(e,argument_data[i]);
      if(do_merge && (!f || (argument_data[i]->as<Function>()->arguments.size() == f->arguments.size()))) break;
    }
    
    if(i == argument_data.size()){
      indices.emplace_back();
      
      ins(e);
      if(f.get() != e.get()) f = e->as<Function>();

      if(f){
        argument_list args(f->arguments.size());
        for(auto i:range(args.size())) args[i] = make_expression<CompressedNode>();
        argument_data.push_back(f->clone(std::move(args)));
      }
      else argument_data.push_back(e);
    }
    
    indices[i].push_back(idx);
    reverse_indices[idx] = i;
    
    if(f){
      auto fi = arguments[i]->as<Function>();
      for(auto j:lars::indices(f->arguments)) fi->arguments[j]->as<CompressedNode>()->insert(f->arguments[j], merge, ins, idx );
    }
  }
  
  expression CompressedNode::extract(ID idx)const{
    auto i = reverse_indices[idx];
    auto f = arguments[i]->as<Function>();
    if(!f) return arguments[i];
    argument_list args(f->arguments.size());
    for(auto j:range(args.size())) args[j] = f->arguments[j]->as<CompressedNode>()->extract(idx);
    return f->clone(std::move(args));
  }
  
#pragma mark Printing
  
  bool needs_brackets_in(const Expression::shared &expr,const Operator * parent,bool sp){
    if(auto e = expr->as<Operator>() ) return sp?parent->get_precedence() <= e->get_precedence():parent->get_precedence() < e->get_precedence();
    return false;
  }
  
  struct BasicPrinterVisitor:public Visitor{
    std::ostream & stream;
    std::unordered_map<std::pair<string, string>, string,lars::tuple_hasher<string, string>> joint_operators;
    
    BasicPrinterVisitor(std::ostream & _stream):stream(_stream){
      //joint_operators[std::make_pair("*", "1/")] = "/";
      //joint_operators[std::make_pair("+", "-")] = "-";
    }
    
    void print_arguments(const Function * e){
      if(e->arguments.size() > 0){
        auto backit = e->arguments.end(); backit--;
        for (auto it = e->arguments.begin(), end = e->arguments.end(); it != end; ++it) {
          (*it)->accept(this);
          if(it != backit) stream << ',';
        }
      }
    }
    
    void visit(const Function * e){
      stream << e->get_name() << '(';
      print_arguments(e);
      stream << ')';
    }
    
    void visit(const Tupel * e){
      stream << e->get_open();
      print_arguments(e);
      stream << e->get_close();
    }
    
    void visit(const CompressedNode * e){
      if(e->arguments.size() == 1){
        e->arguments[0]->accept(this);
      }
      else{
        stream << '{';
        print_arguments(e);
        stream << '}';
      }
    }

    expression print_operator_symbol(const string &s,expression e, bool only_joint = false){
      
      if(auto o = e->as<UnaryOperator>()) if(o->is_prefix()) {
        auto it = joint_operators.find(std::make_pair(s, o->get_symbol()));
        if( it!=joint_operators.end() ){
          stream << it->second;
          return o->arguments[0];
        }
      }
      
      if(!only_joint) stream << s;
      return e;
    }
    
    //*
    void visit(const BinaryOperator * e){
      if(e->arguments.size() < 2){
        visit((const Function *)e);
        return;
      }
      print_with_brackets_in(print_operator_symbol(e->get_symbol(),e->arguments.front(),true), e);
      for (auto i : range<unsigned>(1,e->arguments.size())) {
        print_with_brackets_in(print_operator_symbol(e->get_symbol(),e->arguments[i]), e);
      }
    }

    void visit(const UnaryOperator * e){
      if(e->is_prefix()) stream << e->get_symbol();
      print_with_brackets_in(e->arguments.front(), e);
      if(e->is_postfix()) stream << e->get_symbol();
    }//*/
    
    void visit(const AtomicExpression * e){
      stream << e->get_representation();
    }
    
    /*
    void visit(const WildcardSymbol * e){
      stream << e->get_id();
    }
    
    void visit(const WildcardFunction * e){
      stream << e->get_id() << '(';
      print_arguments(e);
      stream << ')';
    }
    */
    
    void print_with_brackets_in(const Expression::shared & expr,const Operator * parent,bool sp = true){
      bool brackets =  needs_brackets_in(expr, parent, sp);
      if(brackets) stream << '(';
      expr->accept(this);
      if(brackets) stream << ')';
    }
  };
  
  std::ostream & operator<<( std::ostream &stream,const Expression &expr ){
    BasicPrinterVisitor v(stream);
    expr.accept(&v);
    return stream;
  }
  
  std::ostream & operator<<( std::ostream &stream,Expression::shared expr ){
    BasicPrinterVisitor v(stream);
    expr->accept(&v);
    return stream;
  }

}
