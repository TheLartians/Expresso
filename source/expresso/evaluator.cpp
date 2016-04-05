#include "evaluator.h"
#include "algorithms.h"

#include <lars/iterators.h>

#include <algorithm>

//#define VERBOSE

#ifdef VERBOSE
#include <iostream>
#endif

namespace expresso {
  
  using namespace lars;
  
    EvaluatorVisitor::EvaluatorVisitor(const Evaluator &_evaluator,replacement_map & _cache):evaluator(_evaluator),cache(_cache){

    }
        
    bool EvaluatorVisitor::get_from_cache(expression e,expression &res){
      auto key = e->get_shared();
      auto cached = cache.find(key);
      if(cached != cache.end()) {
        res = cached->second;
        return true;
      }
      return false;
    }
    
    bool EvaluatorVisitor::is_cached(const Expression * e){
#ifdef VERBOSE
      std::cout << "entering: " << *e << std::endl;
#endif
      if(get_from_cache(e->get_shared(),copy)) {
        modified |= *e != *copy;
#ifdef VERBOSE
         std::cout << "cached: " << *e << " -> " << *copy << std::endl;
#endif
        return true;
      }
      //TODO: we should figure out how check for infinite recursion without screwing up the evaluation process
      //if(expression_stack.find(*e) != expression_stack.end()){
#ifdef VERBOSE
        //std::cout << "stopped recursion: " << *e << std::endl;
#endif
        //copy = *e;
        //return true;
      //}
      return false;
    }
    
    void EvaluatorVisitor::add_to_cache(expression e,expression res){
#ifdef VERBOSE
      std::cout << "caching: " << e << " -> " << res << std::endl;
#endif
      cache[e] = res;
    }
    
    void EvaluatorVisitor::finalize(const Expression * e){
      add_to_cache(e->get_shared(), copy);
      if(*copy != *e){
        modified = true;
        if(evaluator.settings.recursive){
#ifdef VERBOSE
          std::cout << "Revisit: " << copy << " != " << *e << std::endl;
#endif
          expression keep_reference = copy;
          copy->accept(this);
          add_to_cache(e->get_shared(), copy);
        }
      }
    }
  
  expression EvaluatorVisitor::evaluate(expression e){
#ifdef VERBOSE
    std::cout << "Evaluate: " << e << std::endl;
#endif
    copy = expression();
    e->accept(this);
    if(copy) return copy;
    return e;
  }
  
    bool EvaluatorVisitor::copy_function(const Function * e){
      
      if(is_cached(e)){
        return true;
      }
      
      expression_stack.emplace(*e);
      
      argument_list args(e->arguments.size());
      auto m = modified;
      modified = false;
      
      auto tmp = std::weak_ptr<const Function>( e->as<Function>() );
      
      for(auto a:enumerate(e->arguments)){
        a.value->accept(this);
        args[a.index] = copy;
      }
      if(modified){
        copy = e->clone(std::move(args));
        
#ifdef VERBOSE
        std::cout << "recursive copied expression: " << *e << " -> " << *copy << std::endl;
#endif
        
        if(is_cached(copy.get())){
          expression_stack.erase(expression_stack.find(*e));
          add_to_cache(e->get_shared(),copy);
          return true;
        }
      }
      
      else copy = e->get_shared();
      modified |= m;
      
      expression_stack.erase(expression_stack.find(*e));
      return false;
    }
    
    void EvaluatorVisitor::visit(const Function * e){
#ifdef VERBOSE
      std::cout << "visit function: " << *e << std::endl;
#endif
      if(copy_function(e)) return;
      auto tmp = copy;
      copy = evaluator.evaluate(copy,*this);
      if(*tmp!=*e) finalize(tmp.get());
      finalize(e);
    }
  
    void EvaluatorVisitor::visit_binary(const BinaryOperator * e){
      if(copy_function(e)) return;
      
      auto c = copy->static_as<BinaryOperator>();
      
      if(e->associativity == BinaryOperator::non_associative || c->arguments.size()<=evaluator.settings.split_binary_size){
#ifdef VERBOSE
        std::cout << "visit binary as function: " << *e << std::endl;
#endif
        auto tmp = copy;
        copy = evaluator.evaluate(copy,*this);
        if(*tmp!=*e) finalize(tmp.get());
        finalize(e);
        return;
      }
      
      std::unordered_set<unsigned> ignore_indices;
      std::vector<unsigned> new_indices;
      argument_list new_args;
      
      auto m = modified;
      
      std::unique_ptr<BinaryIterator> bit;
      
      if(e->is_commutative() && evaluator.settings.commutate_binary){
        bit.reset(new BinaryIterators::SingleOrdered(evaluator.settings.split_binary_size));
      }
      else{
          // TODO: add real associativity support
          bit.reset(new BinaryIterators::Window(evaluator.settings.split_binary_size));
      }
      bit->init(c.get());
      
      do {
        
        auto & indices = bit->get_indices();
        CAargs.resize(indices.size());
        
        bool invalid = false;
        for(auto i: enumerate( bit->get_indices() ) ){
          if(ignore_indices.find(i.value) != ignore_indices.end()){
            invalid = true; break;
          }
          CAargs[i.index] =  c->arguments[i.value];
        }
        if(invalid) continue;
        
        auto test = e->clone(std::move(CAargs));
        
#ifdef VERBOSE
        std::cout << "Binary window: " << *test << std::endl;
#endif
        
        expression res = evaluate(test);
        
        modified = res != test;
        
        if(modified){
          ignore_indices.insert(bit->get_indices().begin(),bit->get_indices().end());
          if(e->is_identical(res)){
            auto resf = res->as<Function>();
            for(auto & arg:resf->arguments){
              new_args.push_back(arg);
              new_indices.emplace_back(bit->get_indices().front());
            }
          }
          else{
            new_args.push_back(res);
            new_indices.emplace_back(bit->get_indices().front());
          }
        }

      } while (bit->step());
      
      if(ignore_indices.size() != 0){
          for(auto arg:enumerate(c->arguments)) if(ignore_indices.find(arg.index) == ignore_indices.end()) {
            if(c->is_commutative()) new_args.emplace_back(arg.value);
            else{
              unsigned idx = std::lower_bound(new_indices.begin(), new_indices.end(), arg.index) - new_indices.begin();
              new_args.insert(new_args.begin()+idx, arg.value);
              new_indices.insert(new_indices.begin()+idx, arg.index);
            }
          }
          copy = c->clone(std::move(new_args));
          modified = true;
      }
      else{
        copy = c;
        modified = m;
      }
      
      finalize(e);
    }
    
    void EvaluatorVisitor::visit(const BinaryOperator * e){
#ifdef VERBOSE
      std::cout << "visit binary: " << *e << std::endl;
#endif
      if(evaluator.settings.split_binary) visit_binary(e);
      else visit((const Function *)e);
    }
    
    void EvaluatorVisitor::visit(const AtomicExpression * e){
#ifdef VERBOSE
      std::cout << "visit atomic: " << *e << std::endl;
#endif
      if(is_cached(e)) return;
      expression_stack.emplace(*e);
      copy = evaluator.evaluate(e->get_shared(),*this);
      expression_stack.erase(expression_stack.find(*e));
      modified |= copy != e;
      finalize(e);
    }
  
  expression Evaluator::run(expression e)const{
    replacement_map cache;
    return run(e,cache);
  }
  
  expression Evaluator::run(expression e,replacement_map &cache)const{
    EvaluatorVisitor v(*this,cache);
    return v.evaluate(e);
  }
  
#pragma mark ReplaceEvaluator
  
  expression ReplaceEvaluator::evaluate(expression e,EvaluatorVisitor &)const{
    auto it = replacements.find(e);
    if(it != replacements.end()) return it->second;
    return e;
  }
  
  void ReplaceEvaluator::extend(){
    auto old_replacements = replacements;
    for(auto r:old_replacements){
      auto res = run(r.first), rep = run(r.second);
      replacements[r.first] = rep;
      if(res != rep) replacements[res] = rep;
    }
  }
  
  void ReplaceEvaluator::add_replacement(expression search,expression replace){
    replacements[search] = replace;
  }
  
#pragma mark RuleEvaluator
    
  std::ostream & operator<<(std::ostream &stream,const Rule &rule){
    stream << rule.search << " -> " << rule.replacement;
    if(rule.condition) stream << " if " << rule.condition << " -> " << rule.valid;
    if(rule.evaluator) stream << " ...";
    return stream;
  }
  
  void RuleEvaluator::insert_rules(const RuleEvaluator & e,int priority){
    for(const auto &rule:e.rules) insert_rule(rule.rule,priority);
  }
  
  void RuleEvaluator::add_rule(const Rule &r,int priority){

    Rule copy = r;
    replacement_map wc;
    std::vector<expression> inserted;
    
    for(auto p:commutative_permutations(r.search)){
      bool valid = true;
      
      for(auto &s:inserted){
        wc.clear();
        if(match(s, p, wc)){
          valid = false;
          break;
        }
      }
      
      if(!valid) continue;
      
      copy.search = p;
      insert_rule(copy,priority);
      inserted.emplace_back(p);
    }
    
  }
  
  RuleEvaluator::rule_id RuleEvaluator::insert_rule(const Rule &r,int p){
    
    RuleEvaluator::rule_id id = rules.size();
    rules.emplace_back(r,p);
    auto & current = rules.back();
    
    auto merge_function = [&](expression &a,expression b){
      if(a->is_identical(b)) {
        return true;
      }
      
      if(auto wa = a->as<WildcardSymbol>()){
        auto ba = b->is<WildcardSymbol>();
        if(ba) current.wildcard_mapping[b] = a;
        return ba;
      }
      
      if(auto wa = a->as<WildcardFunction>()){
        auto ba = b->as<WildcardFunction>();
        auto match = ba && ba->arguments.size() == wa->arguments.size();
        if(match) current.wildcard_mapping[b] = a;
        return match;
      }
      
      return false;
    };
    
    auto insert_function = [&](expression &a){
      if(a->is<WildcardSymbol>()){
        auto b = make_expression<WildcardSymbol>(std::to_string(wc_count++));
        current.wildcard_mapping.insert(std::make_pair(b, a));
        a = b;
      }
      if(auto af = a->as<WildcardFunction>()){
        auto b = make_expression<WildcardFunction>(std::to_string(wc_count++),af->arguments);
        current.wildcard_mapping.functions.insert(std::make_pair(b->as<WildcardFunction>()->get_id(), b));
        current.wildcard_mapping.insert(std::make_pair(b, a));
        a = b;
      }
    };
    
    search_tree->insert(current.rule.search,merge_function,insert_function);

    for(auto &child:child_evaluators){
      child.first->insert_rule(r,child.second);
    }
    
    return id;
  }
  
  void RuleEvaluator::add_evaluator(RuleEvaluator & other,int priority){
    insert_rules(other);
    other.child_evaluators.emplace_back(this,priority);
  }

  expression RuleEvaluator::evaluate(expression e,EvaluatorVisitor &v)const{
    auto r = range<CompressedNode::ID>(0, rules.size());
    return evaluate(e,v,std::vector<rule_id>(r.begin(),r.end()));
  }
  
  expression RuleEvaluator::evaluate(expression e,EvaluatorVisitor &v,std::vector<rule_id> m)const{
    
#ifdef VERBOSE
    std::cout << "entering rule evaluator: " << e << std::endl;
#endif

    
    replacement_map raw_wildcards;
    
    get_matches(e, search_tree, raw_wildcards, m);
    
    /*
    std::sort(m.begin(), m.end(), [this](rule_id a, rule_id b){
      auto pa = rules[a].priority,pb = rules[b].priority; if(pa == pb) return a<b;
      return pa < pb;
    });
    */
     
    std::vector<expression> wc_functions;
    replacement_map wildcards;

    for(auto i:m){
      
      wc_functions.clear();
      wildcards.clear();
      
      auto & current = rules[i];

      //std::cout << "Candidate: " << current.rule << std::endl;
      bool valid = true;
      
      for(auto w:current.wildcard_mapping){
        auto it = wildcards.find(w.second);
        
        if(it != wildcards.end()){
          auto it2 = raw_wildcards.find(w.first);

          if(it2 != raw_wildcards.end() && it->second != it2->second){
            valid = false;
            break;
          }
        }
        else{
          if(auto wf = w.first->as<WildcardFunction>()){
            auto it = raw_wildcards.functions.find(wf->get_id());
            if(it == raw_wildcards.functions.end()){ valid = false; break; }
            wildcards.insert(std::make_pair(w.second,raw_wildcards[it->second]));
            wildcards.functions.insert(std::make_pair(w.second->as<WildcardFunction>()->get_id(), w.second));
            if(wildcards.find(w.second) == wildcards.end()){ valid = false; break; }
            wc_functions.emplace_back(w.second);
          }
          else{
            auto it = raw_wildcards.find(w.first);
            if(it == raw_wildcards.end()){ continue; }
            wildcards.insert(std::make_pair(w.second,it->second));
          }
        }
      }
      
      for(auto &f: wc_functions){
        auto wf = f->as<WildcardFunction>();
        if( wf->arguments.size() == 1 ){
          auto it = wildcards.find(wf->arguments[0]);
          if(it != wildcards.end() && it->second == wildcards[f]){ valid = false; break; }
        }
      }
      
      if(!valid) continue;
      
#ifdef VERBOSE
      std::cout << "matched " << current.rule.search << ", wildcards: ";
      for(auto r:wildcards){
        std::cout << "(" << r.first << "," << r.second << "),";
      }
#endif
      
      if(current.rule.condition){
#ifdef VERBOSE
        std::cout << "checking condition on rule " << current.rule << std::endl;
        std::cout << replace(current.rule.condition,wildcards) << std::endl;
#endif
        if(v.evaluate(replace(current.rule.condition,wildcards)) == current.rule.valid){
          if(current.rule.evaluator) for(auto &rep:wildcards) rep.second = v.evaluate(rep.second);
        }
        else continue;
      }
      
      if(current.rule.evaluator) if(!current.rule.evaluator(wildcards,v)) continue;
      
      auto res = replace(current.rule.replacement, wildcards);
      
      if(res != e){
#ifdef VERBOSE
        std::cout << "Apply: " << current.rule << ":" << std::endl;
        std::cout << replace(current.rule.search,wildcards) << " => " << replace(current.rule.replacement,wildcards) << std::endl;
#endif
        
        if(apply_callback) apply_callback(current.rule,wildcards);
        return res;
      }
    }
    
    return e;
  }
  
  #pragma mark evaluator evaluator
  
  expression MultiEvaluator::evaluate(expression expr,EvaluatorVisitor &v)const{
    
    expression tmp = expr;
    
    for(auto & evaluator:evaluators){
      expr = evaluator->evaluate(expr,v);
      if(tmp != expr){
        return expr;
      }
    }
    
    return expr;
  }
  
  void MultiEvaluator::add_evaluator(Evaluator*e){
    if(auto me = dynamic_cast<MultiEvaluator*>(e)){
      for(auto eval:me->evaluators) add_evaluator(eval);
      return;
    }
    for(auto eval:evaluators){
      if(eval == e) return;
    }
    evaluators.emplace_back(e);
  }


  expression StepEvaluator::evaluate(expression expr,EvaluatorVisitor &v)const{
    for(auto & evaluator:evaluators) expr = evaluator->evaluate(expr,v);
    return expr;
  }
  


  
}