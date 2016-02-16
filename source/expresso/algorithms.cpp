
#include "algorithms.h"

#include <lars/iterators.h>
#include <lars/set_operations.h>

#include <stdexcept>
#include <deque>
#include <sstream>
#include <algorithm>

using namespace lars;

namespace expresso {
  
  Expression::shared copy(const Expression::shared &s){ replacement_map rep; return replace(s, rep); }

  void expression_exception::print_to_stream(std::ostream &stream)const{
    stream << "cause: " << get_cause();
  }
  
  const char * expression_exception::what()const noexcept{
    std::stringstream stream;
    print_to_stream(stream);
    what_string = stream.str();
    return what_string.c_str();
  }
    
  #pragma mark replace
  
  struct ReplaceVisitor:public Visitor{
    Expression::shared copy;
    const replacement_map &replacements;
    bool modified = false;
    
    ReplaceVisitor(const replacement_map &_replacements):replacements(_replacements){}
    
    bool replaced(const Expression * e){
      auto it = replacements.find(e->get_shared());
      if(it == replacements.end()) return false;
      copy = it->second;
      modified = true;
      return true;
    }
    
    void visit(const Function * e)override{
      
      if(replaced(e)) return;
      Function::argument_list args;
      args.reserve(e->arguments.size());
      
      auto m = modified;
      modified = false;
      for(auto a:e->arguments){ a->accept(this); args.push_back(copy); }
      if(modified) copy = e->clone(std::move(args));
      else copy = e->get_shared();

      modified |= m;
    }
    
    expression get_replacement(expression e){
      e->accept(this);
      return copy;
    }
    
    void visit(const WildcardFunction *e)override{
      auto it = replacements.functions.find(e->get_id());
      if(it == replacements.functions.end()){ copy = e->get_shared(); return; }
      auto f = it->second->as<WildcardFunction>();
      
      replacement_map arg_replace;
      for(auto i:indices(f->arguments)){
        arg_replace.insert(std::make_pair( get_replacement(f->arguments[i]) , get_replacement(e->arguments[i])));
      }
      
      copy = replace(replacements.find(it->second)->second, arg_replace);
      modified = true;
    }
    
    void visit(const AtomicExpression * e)override{
      if(replaced(e)) return;
      copy = e->get_shared();
    }
    
  };
  
  Expression::shared replace(const Expression::shared &s,const replacement_map &replacements ){
    ReplaceVisitor v(replacements);
    s->accept(&v);
    return v.copy;
  }
  
#pragma mark match
  
  struct MatchVisitor:public Visitor{
    replacement_map &wildcards;
    Expression::shared to_match;

    bool valid = true;
    
    MatchVisitor(Expression::shared _to_match,replacement_map &_wildcards):wildcards(_wildcards),to_match(_to_match){
    
    }
    
    void test(const Expression * e){
      if(!e->is_identical(to_match)) valid = false;
    }
    
    void visit(const Function * e)override{
      test(e);
      if(!valid) return;
      
      auto tm = to_match->as<Function>();
      
      if(tm->arguments.size() != e->arguments.size()){ valid = false; return; }
      
      for(auto i: indices(tm->arguments) ){
        to_match = tm->arguments[i];
        e->arguments[i]->accept(this);
        if(!valid) return;
      }
      
    }
    
    void visit(const MatchCondition * e)override{
      if(!e->condition(to_match)){ valid = false; return; }
      e->arguments[0]->accept(this);
    }
    
    void visit(const BinaryOperator * e)override{
      
      // TODO: add directional associativity to Binary operator?
      // if(e->associativity != BinaryOperator::associative){ visit((Function*)e); return; }
      
      test(e);
      if(!valid) return;

      auto tm = to_match->as<Function>();
      if(tm->arguments.size() < e->arguments.size()){ valid = false; return; }
      else if(tm->arguments.size() == e->arguments.size()) for(auto i: indices(tm->arguments) ){
        to_match = tm->arguments[i];
        e->arguments[i]->accept(this);
      }
      else {
        if(e->arguments.size() == 2){
        
          Expression::shared lhs,rhs;
          
          if(e->associativity == BinaryOperator::left_associative || e->associativity == BinaryOperator::associative){
            lhs = tm->arguments.front();
            to_match = lhs; e->arguments[0]->accept(this); if(!valid) return;
            rhs = tm->clone(Function::argument_list(tm->arguments.begin()+1,tm->arguments.end()));
            to_match = rhs; e->arguments[1]->accept(this); if(!valid) return;
          }
          else if(e->associativity == BinaryOperator::right_associative){
            lhs = tm->clone(Function::argument_list(tm->arguments.begin(),tm->arguments.end()-1));
            to_match = lhs; e->arguments[0]->accept(this); if(!valid) return;
            rhs = tm->arguments.back();
            to_match = rhs; e->arguments[1]->accept(this); if(!valid) return;
          }
          else {
            valid = false;
            return;
          };

        }
        else {
          valid = false;
          return;
        };
      }
    }
    
    bool inside_wildcard_function = false;
    
    void visit(const WildcardSymbol * e)override{
      if(inside_wildcard_function) return;
      
      auto it = wildcards.find(e->get_shared());
      if(it == wildcards.end()) wildcards.insert(std::make_pair(e->get_shared(),to_match));
      else {
        if( it->second !=  to_match->get_shared() ) valid = false;
      }
    }
    
    void visit(const WildcardFunction * e)override{
      auto it = wildcards.functions.find(e->get_id());
      
        if(it == wildcards.functions.end()){
        
        auto f = to_match->as<Function>();
        if(f && e->arguments.size() == f->arguments.size()){
          bool before = inside_wildcard_function;
          inside_wildcard_function = true;
          
          for(auto i:indices(e->arguments)){
            to_match = f->arguments[i];
            e->arguments[i]->accept(this);
            if(!valid) break;
          }
        
          to_match = f;
          inside_wildcard_function = before;
        }
        
        if(!valid) return;
        
        wildcards.functions.insert(std::make_pair(e->get_id(),e->get_shared()));
        wildcards.insert(std::make_pair(e->get_shared(),to_match));
      }
      else {
        auto it2 = wildcards.find(it->second);
        if(it2 == wildcards.end()){ valid = false; return; }
        auto wf2 = it2->first->as<WildcardFunction>();
        if(it2->first != e && wf2->arguments.size() == e->arguments.size()){
          replacement_map arg_replace;
          for(auto i:indices(e->arguments)) arg_replace.insert(std::make_pair(e->arguments[i], wf2->arguments[i]));
          if(replace(to_match, arg_replace) != it2->second) { valid = false; return; };
        }
        else{ valid = false; return; }
      }
    }
    
    void visit(const AtomicExpression * e)override{
      test(e);
    }
    
  };
    
  bool match( const Expression::shared &expr,const Expression::shared &search,replacement_map &wildcards ){
    MatchVisitor v(expr,wildcards);
    search->accept(&v);
    return v.valid;
  }
      
#pragma mark get matches
  
      class GetMatchesVisitor:public MatchVisitor{
        public:
        
        bool match = true;
        std::vector<CompressedNode::ID> &matches;
        
        GetMatchesVisitor(Expression::shared _to_match,replacement_map &_wildcards,std::vector<CompressedNode::ID> &_matches):MatchVisitor(_to_match,_wildcards),matches(_matches){}
        
        bool do_match(expression expr,expression search){
          bool prev = match;
          
          match = true;
          to_match = expr;
          
          search->accept(this);
          if(valid == false){
            match = false;
            valid = true;
          }
          
          std::swap(prev,match);
          
          return prev;
        }
        
        using MatchVisitor::visit;
        
        void visit(const CompressedNode * e)override{
          //
          // TODO: speed this up e.g. by sorting the list by function name, not calculating the whole intersection
          
          auto tm = to_match;
          
          for(auto i:indices(e->indices)){
            auto & index_list = e->indices[i];
            auto intersection = lars::set_intersection(index_list, matches);
            
            if(intersection.size() == 0) continue;
            
            if(!do_match(tm, e->arguments[i])){
              matches = lars::set_difference(matches, intersection);
            }
          }
        }
        
      };
      
      void get_matches( const Expression::shared &expr, std::shared_ptr<CompressedNode> searches, replacement_map &wildcards, std::vector<CompressedNode::ID> & matches ){
        GetMatchesVisitor v(expr,wildcards,matches);
        searches->accept(&v);
      }
      
      std::vector<CompressedNode::ID> get_matches( const Expression::shared &expr, std::shared_ptr<CompressedNode> searches, replacement_map &wildcards){
        std::vector<CompressedNode::ID> matches;
        for(auto &m: searches->reverse_indices ) matches.push_back(m.first);
        std::sort(matches.begin(),matches.end());
        get_matches(expr, searches, wildcards,matches);
        return matches;
      }
    
#pragma mark mulplicity
  
  mulplicity_list::mulplicity_list(const group &_base,const Function &_mulplicity,const field &_field):base(_base),mulplicity(_mulplicity),real_field(_field){}
  
  mulplicity_list::mulplicity_list(const expression &e,const group &_base,const Function &_mulplicity,const field &_field):base(_base),mulplicity(_mulplicity),real_field(_field){
    set_from_expression(e);
  };
  
  void mulplicity_list::set_from_argument_list(const argument_list &args){
    clear();
    
    if(args.size() == 0) return;
    
    auto as_mulplicity = [&](const expression &arg){
      
      if(mulplicity.is_identical(arg)){
        auto f = arg->as<Function>();
        auto a = f->arguments;
        
        if(auto b = f->as<BinaryOperator>()){
          if(b->commutativity == BinaryOperator::commutative){
            if(a.size() == 2) return std::make_pair(a[1], a[0]);
            argument_list new_args{a[0]};
            new_args.insert(new_args.begin(), a.begin()+2,a.end());
            return std::make_pair(a[1], f->clone(std::move(new_args)));
          }
        }
        
        if(a.size() == 2) return std::make_pair(a[0], a[1]);
        return std::make_pair(a[0], f->clone(argument_list(a.begin()+1,a.end())));
        
      }
      
      return std::make_pair(arg, real_field.multiplicative_group.neutral);
    };
    
    mulplicity_list::value_type * last_value = nullptr;
    
    auto emplace_back = [&](mulplicity_list::value_type && mul){
      this->emplace_back(std::move(mul)); last_value = &this->back();
    };
    
    std::function<void(mulplicity_list::value_type &&mul)> insert_into_list;
    
    insert_into_list = [&](mulplicity_list::value_type &&mul){
      if(base.operation.is_identical(mul.first)){
        for(auto &arg:mul.first->as<Function>()->arguments) emplace_back(mulplicity_list::value_type(arg,mul.second));
        return;
      }
      if(base.inverse.is_identical(mul.first)){
        auto inner_mul = as_mulplicity(mul.first->static_as<Function>()->arguments[0]);
        
        mul.first = inner_mul.first;
        
        expression mp;
        if(mul.second == real_field.multiplicative_group.neutral) mp = inner_mul.second;
        else if(inner_mul.second == real_field.multiplicative_group.neutral) mp = mul.second;
        else mp = real_field.multiplicative_group.operation(mul.second,inner_mul.second);

        mul.second = real_field.additive_group.inverse(mp);
        
        insert_into_list(std::move(mul));
        return;
      }
      if(last_value && last_value->first == mul.first){
        last_value->second = real_field.additive_group.operation(last_value->second,mul.second);
      }
      else{ emplace_back(std::move(mul)); }
    };
    
    for(const auto & arg: args){
      auto mul = as_mulplicity(arg);
      insert_into_list(std::move(mul));
    }
    
    std::sort(this->begin(), this->end(), [](const mulplicity_list::value_type &a,const mulplicity_list::value_type &b){ return a.first < b.first; });
  
    for(unsigned i=0; i+1<size();){
      auto &a = (*this)[i], &b = (*this)[i+1];
      
      if(a.first == b.first){
        a.second = real_field.additive_group.operation(a.second,b.second);
        erase(begin()+i+1);
      }
      else ++i;
    }
    
    
  }
  
  void mulplicity_list::set_from_expression(const expression &e){
    if(auto b = e->as<Function>()){
      if(base.operation.is_identical(e)) return set_from_argument_list(b->arguments);
    }
    return set_from_argument_list(argument_list{e});
  }
  
  mulplicity_list mulplicity_list::difference(const mulplicity_list &other)const{
    mulplicity_list res(base,mulplicity,real_field);
    
    auto it1 = begin(), end1 = end();
    auto it2 = other.begin(), end2 = other.end();
    
    while(it1 != end1 && it2 != end2){
      if(it1->first < *it2->first){
        res.emplace_back(std::make_pair(it1->first, it1->second));
        ++it1;
      }
      else{
        if(!(*it2->first < *it1->first)){
          if(it1->second != it2->second){
            res.emplace_back(std::make_pair(it1->first, real_field.additive_group.operation(it1->second,real_field.additive_group.inverse(it2->second))));
          }
          ++it1;
        }
        else{
          res.emplace_back(std::make_pair(it2->first, real_field.additive_group.inverse(it2->second)));
        }
        ++it2;
      }
    }
    
    while(it1 != end1){
      res.emplace_back(std::make_pair(it1->first, it1->second));
      ++it1;
    }
    
    while(it2 != end2){
      res.emplace_back(std::make_pair(it2->first, real_field.additive_group.inverse(it2->second)));
      ++it2;
    }
    
    return std::move(res);
  }
  
  mulplicity_list mulplicity_list::sum(const mulplicity_list &other)const{
    mulplicity_list res(base,mulplicity,real_field);
    
    auto it1 = begin(), end1 = end();
    auto it2 = other.begin(), end2 = other.end();
    
    while(it1 != end1 && it2 != end2){
      if(it1->first < *it2->first){
        res.emplace_back(std::make_pair(it1->first, it1->second));
        ++it1;
      }
      else{
        if(!(*it2->first < *it1->first)){
          res.emplace_back(std::make_pair(it1->first, real_field.additive_group.operation(it1->second,it2->second)));
          ++it1;
        }
        else{
          res.emplace_back(std::make_pair(it2->first, it2->second));
        }
        ++it2;
      }
    }
    
    while(it1 != end1){
      res.emplace_back(std::make_pair(it1->first, it1->second));
      ++it1;
    }
    
    while(it2 != end2){
      res.emplace_back(std::make_pair(it2->first, it2->second));
      ++it2;
    }
    
    return std::move(res);
  }
  
  mulplicity_list mulplicity_list::power(const expression &expr)const{
    mulplicity_list res(base,mulplicity,real_field);
    for(auto &mul:*this){
      res.emplace_back(mul.first,real_field.multiplicative_group.operation(expr,mul.second));
    }
    return res;
  }
  
  mulplicity_list mulplicity_list::intersection(const mulplicity_list &other, inner_intersection_function get_inner)const{
    mulplicity_list res(base,mulplicity,real_field);
    
    if(!get_inner){
      get_inner = [&](const expression &a,const expression &b){
        
        mulplicity_list aargs = mulplicity_list(a, real_field.additive_group, real_field.multiplicative_group.operation, real_field);
        mulplicity_list bargs = mulplicity_list(b, real_field.additive_group, real_field.multiplicative_group.operation, real_field);
        mulplicity_list intersection = aargs.intersection(bargs,[&](const expression &a,const expression &b){
          if(a < b) return a;
          return b;
        });
        
        if(intersection.size() == 0) return expresso::expression();
        
        return intersection.as_expression();
      };
    }
    
    auto it1 = begin(), end1 = end();
    auto it2 = other.begin(), end2 = other.end();
    
    while(it1 != end1 && it2 != end2){
      
      if(it1->first < *it2->first) ++it1;
      else{
        if(!(*it2->first < *it1->first)){
          auto inner = get_inner(it1->second,it2->second);
          if(inner) res.emplace_back(it1->first,inner);
        }
        ++it2;
      }
    }
    
    return std::move(res);
  }
  
  argument_list mulplicity_list::as_argument_list()const{
    
    std::unordered_map<expression, argument_list> inverted_mlist;
    argument_list res;
    
    for(const auto &arg:*this){
      if(arg.second == real_field.multiplicative_group.neutral){
        res.emplace_back(arg.first);
      }
      else{
        inverted_mlist[arg.second].emplace_back(arg.first);
      }
    }
    
    for(auto &im:inverted_mlist){
      res.emplace_back(mulplicity(base.operation(std::move(im.second)),im.first));
    }
    
    
    return std::move(res);
  }
  
  expression mulplicity_list::as_expression()const{
    auto arglist = as_argument_list();
    if(arglist.size() == 0) return base.neutral;
    if(arglist.size() == 1) return arglist.front();
    return base.operation.clone(std::move(arglist));
  }
  
  #pragma mark -
  
  std::ostream & operator<<(std::ostream & stream,const mulplicity_list &m){
    stream << '[';
    for(auto &v:m){
      stream << '(' << v.first << ')';
      stream << ',';
    }
    stream << ']';
    return stream;
  }
  
  void stack_based_expression_iterator::recursive_set_expression(const expression &e,unsigned idx){
    expression_stack.emplace_back(e);
    index_stack.emplace_back(idx);
    if(!e->is_function()) return;
    auto f = e->static_as<Function>();
    iterator_stack.emplace_back(f->arguments.begin(),f->arguments.end());
    recursive_set_expression(f->arguments.front(),0);
  }
  
  void stack_based_expression_iterator::set_expression(const expression &e,unsigned idx){
    expression_stack.emplace_back(e);
    index_stack.emplace_back(idx);
    if(!e->is_function()) return;
    auto f = e->static_as<Function>();
    iterator_stack.emplace_back(f->arguments.begin(),f->arguments.end());
  }
  
  expression expression_iterator::operator*()const{
    return expression_stack.back();
  }
  
  expression replace(const expression expr, std::function<expression(const expression_location &)> search){
    argument_list arguments;
    
    auto traversal = postorder_traversal(expr);
    unsigned modified_depth = 0;
    
    for(auto it = traversal.begin(),end = traversal.end(); it != end; ++it ){
      
      const expression replacement = search(it.location());
      
      if(auto f = (*it)->as<Function>()){
        auto arg_begin = arguments.end() - f->arguments.size();
        expression copy;
        
        if(replacement) {
          copy = *replacement;
          modified_depth = std::max( it.depth(), modified_depth );
        }
        else{
          if(it.depth() < modified_depth) copy = f->clone(argument_list(arg_begin,arguments.end()));
          else copy = *it;
        }
        arguments.erase(arg_begin, arguments.end());
        arguments.push_back(copy);
      }
      else{
        if(replacement) {
          modified_depth = std::max( it.depth(), modified_depth );
          arguments.push_back(*replacement);
        }
        else arguments.emplace_back(*it);
      }
      
    }
    
    assert(arguments.size() == 1);
    
    return arguments.front();
  }
  
  expression expression_iterator::replace_current(const expression &rep)const{
    return replace(outer(), [&](const expression_location &it){ if(it == location()) return rep; return expression(); }  );
  }
  
  preorder_traversal::iterator::iterator(const expression & e){
    set_expression(e,0);
  }

  preorder_traversal::iterator & preorder_traversal::iterator::operator++(){
    unsigned idx = 0;
    
    if (expression_stack.size() > iterator_stack.size()){
      idx = index_stack.back() + 1;
      expression_stack.pop_back();
      index_stack.pop_back();
    }
    
    if(iterator_stack.size()>0){
      while(iterator_stack.back().first == iterator_stack.back().second){
        iterator_stack.pop_back();
        idx = index_stack.back() + 1;
        expression_stack.pop_back();
        index_stack.pop_back();
        if(iterator_stack.size() == 0) return *this;
      }
      auto tmp = *iterator_stack.back().first;
      iterator_stack.back().first++;
      set_expression(tmp,idx);
    }
    
    return *this;
  }
  
  preorder_traversal::iterator preorder_traversal::iterator::operator++(int){
    preorder_traversal::iterator copy(*this);
    ++(*this);
    return copy;
  }
  
  postorder_traversal::iterator::iterator(const expression & e){
    recursive_set_expression(e,0);
  }
  
  postorder_traversal::iterator & postorder_traversal::iterator::operator++(){
    
    unsigned idx = 0;
    
    if(expression_stack.size() > iterator_stack.size()){
      expression_stack.pop_back();
      idx = index_stack.back() + 1;
      index_stack.pop_back();
      if(expression_stack.size() == 0) return *this;
    }
    
    iterator_stack.back().first++;
    
    if(iterator_stack.back().first == iterator_stack.back().second){
      iterator_stack.pop_back();
    }
    else recursive_set_expression(*iterator_stack.back().first,idx);
    
    return *this;
  }
  
  postorder_traversal::iterator postorder_traversal::iterator::operator++(int){
    postorder_traversal::iterator copy(*this);
    ++(*this);
    return copy;
  }
  
  commutative_permutations::iterator::commutable_expression::commutable_expression(std::shared_ptr<const BinaryOperator> e,const expression_location & p):expression(e),location(p){
    permutated_argument_indices.resize(e->static_as<Function>()->arguments.size());
    for(auto i:indices(permutated_argument_indices)) permutated_argument_indices[i] = i;
  }
  
  void commutative_permutations::iterator::add_expression(const expression &e,expression_location loc,unsigned parent_index){
    if(auto b = e->as<BinaryOperator>()){ if(b->is_commutative()){
      commutable_expressions.emplace_back(b,loc);
      if(parent_index != unsigned(-1)){
        expression_location inner_loc{0};
        inner_loc.insert(inner_loc.end(),loc.begin() + commutable_expressions[parent_index].location.size(),loc.end());
        commutable_expressions[parent_index].inner_commutable_expressions.emplace_back(commutable_expressions.size()-1,inner_loc);
      }
      else {
        outer_commutable_indices.emplace_back(commutable_expressions.size() - 1);
      }
      
      loc.emplace_back(0);
      parent_index = commutable_expressions.size() - 1;
      
      for(auto &arg:b->arguments){
        add_expression(arg, loc, parent_index);
        loc.back()++;
      }
      
      return;
    }
    }
    
    if(auto f = e->as<Function>()){
      loc.emplace_back(0);
      for(auto &arg:f->arguments){
        add_expression(arg, loc, parent_index);
        loc.back()++;
      }
    }
  }
  
  commutative_permutations::iterator::iterator(const expression & e):base(e){
    add_expression(e, {0});
  }
    
  expression commutative_permutations::iterator::operator*()const{
    
    return replace(base, [&](const expression_location &loc)->expression{
      for(auto &ce:outer_commutable_indices){
        if(commutable_expressions[ce].location == loc) return get_permutated_expression(commutable_expressions[ce]);
      }
      return expression();
    });
    
  }
  
  std::ostream & operator<<(std::ostream &stream,const expression_location &loc){
    stream << '[';
    for(auto i:indices(loc)){
      stream << loc[i] ;
      if(i != loc.size() - 1) stream << ',';
    }
    stream << ']';
    return stream;
  }
  
  expression commutative_permutations::iterator::get_permutated_expression(const commutable_expression & e)const{
    argument_list args(e.permutated_argument_indices.size());
    
    std::shared_ptr<const BinaryOperator> rep_expr;
    
    if(e.inner_commutable_expressions.size() == 0) rep_expr = e.expression;
    else{
      rep_expr = replace(e.expression, [&](const expression_location &loc)->expression{
      
        for(auto &ce:e.inner_commutable_expressions){
          if(ce.second == loc){
            return get_permutated_expression(commutable_expressions[ce.first]);
          }
        }
        return expression();
      })->as<BinaryOperator>();
    }
      
    for(auto i:indices(e.permutated_argument_indices)){
      if(!args[i]) args[i] = rep_expr->arguments[e.permutated_argument_indices[i]];
    }
    
    return rep_expr->clone(std::move(args), false);
  }
  
  commutative_permutations::iterator & commutative_permutations::iterator::operator++(){
    
    for(auto &com:commutable_expressions){
      if(std::next_permutation(com.permutated_argument_indices.begin(), com.permutated_argument_indices.end())) break;
      else if(&com == &commutable_expressions.back()){
        base.reset();
        commutable_expressions.clear();
        break;
      }
    }
    if(commutable_expressions.size() == 0){
      base.reset();
    }
    
    return *this;
  }
  
  commutative_permutations::iterator commutative_permutations::iterator::operator++(int){
    commutative_permutations::iterator copy(*this);
    ++(*this);
    return copy;
  }

  
  
}
























