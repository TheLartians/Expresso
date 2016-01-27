#include "evaluator.h"
#include "algorithms.h"

#include <lars/iterators.h>

#include <iostream>
#include <algorithm>

#define VERBOSE

namespace symbols {
  
  using namespace lars;
  

    EvaluatorVisitor::EvaluatorVisitor(const Evaluator &_evaluator,replacement_map & _cache):evaluator(_evaluator),cache(_cache){

    }
        
    bool EvaluatorVisitor::get_from_cache(expression e,expression &res){
      auto key = e->get_shared();
      auto cached = cache.find(key);
      if(cached != cache.end()) {
        res = cached->second;
        //if(e != res) std::cout << "cached: " << e << " -> " << res << std::endl;
        return true;
      }
      return false;
    }
    
    bool EvaluatorVisitor::is_cached(const Expression * e){
      //std::cout << "entering: " << *e << std::endl;
      if(get_from_cache(e->get_shared(),copy)) {
        modified |= *e != *copy;
        return true;
      }
      return false;
    }
    
    void EvaluatorVisitor::add_to_cache(expression e,expression res){
      //std::cout << "caching: " << e << " -> " << res << std::endl;
      cache[e] = res;
    }
    
    void EvaluatorVisitor::finalize(const Expression * e){
      add_to_cache(e->get_shared(), copy);
      if(*copy != *e){
        modified = true;
        if(evaluator.recursive){
          //std::cout << "Revisit: " << copy << " != " << *e << std::endl;
          expression keep_reference = copy;
          copy->accept(this);
          add_to_cache(e->get_shared(), copy);
        }
      }
    }
    
    void EvaluatorVisitor::copy_function(const Function * e){
      
      argument_list args(e->arguments.size());
      auto m = modified;
      modified = false;
      
      auto tmp = std::weak_ptr<const Function>( e->as<Function>() );
      
      for(auto a:enumerate(e->arguments)){
        a.value->accept(this);
        args[a.index] = copy;
      }
      if(modified) copy = e->clone(std::move(args));
      else copy = e->get_shared();
      modified |= m;
    }
    
    void EvaluatorVisitor::visit(const Function * e){
      if(is_cached(e)) return;
      copy_function(e);
      copy = evaluator.evaluate(copy,*this);
      finalize(e);
    }
    
    void EvaluatorVisitor::visit_binary(const BinaryOperator * e){
      copy_function(e);

      auto c = copy->as<BinaryOperator>();
      
      std::unordered_set<unsigned> ignore_indices;
      std::vector<unsigned> new_indices;
      argument_list new_args;
      
      auto m = modified;
      
      std::unique_ptr<BinaryIterator> bit;
      
      if(e->is_commutative()) bit.reset(new BinaryIterators::SingleOrdered(2));
      else bit.reset(new BinaryIterators::Window(2));
      bit->init(c.get());
      
      std::cout << "Visiting: " << *c << std::endl;
      std::cout << "commutative: " << c->is_commutative() << std::endl;
      
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
        expression res;
        
        if(!get_from_cache(test,res)){
          res = evaluator.evaluate(test,*this);
          add_to_cache(test,res);
        }
                //auto res = evaluator.evaluate(test);
        
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
        if(ignore_indices.size() == c->arguments.size() && new_args.size() == 1) copy = new_args.front();
        else {
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
      }
      else{
        copy = c;
        modified = m;
      }
    }
    
    void EvaluatorVisitor::visit(const BinaryOperator * e){
      if(is_cached(e)) return;
      visit_binary(e);
      finalize(e);
    }
    
    void EvaluatorVisitor::visit(const AtomicExpression * e){
      if(is_cached(e)) return;
      copy = evaluator.evaluate(e->get_shared(),*this);
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
  
  void RuleEvaluator::verbose_apply_callback(const Rule &rule,const replacement_map &wildcards){
    std::cout << "Apply: " << replace(rule.search,wildcards) << " => " << replace(rule.replacement,wildcards) << std::endl;
  };
    
  std::ostream & operator<<(std::ostream &stream,const Rule &rule){
    stream << rule.search << " -> " << rule.replacement;
    if(rule.evaluator) stream << " ...";
    return stream;
  }
  
  void RuleEvaluator::insert_rules(const RuleEvaluator & e,int priority){
    for(const auto &rule:e.rules) insert_rule(rule.rule,priority);
  }
  
  void RuleEvaluator::add_rule(const Rule &r,int priority){

    Rule copy = r;
    replacement_map wc;
    bool first = true;
    
    for(auto p:commutative_permutations(r.search)){
      wc.clear();
      if(!first && match(r.search, p, wc)){
        continue;
      }
      first = false;
      copy.search = p;
      insert_rule(copy,priority);
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
    replacement_map raw_wildcards;
    
    get_matches(e, search_tree, raw_wildcards, m);
    
    std::sort(m.begin(), m.end(), [this](rule_id a, rule_id b){
      auto pa = rules[a].priority,pb = rules[b].priority; if(pa == pb) return a<b;
      return pa < pb;
    });
    
    std::vector<expression> wc_functions;
    replacement_map wildcards;

    //std::cout << "Matching: " << e << std::endl;

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
      
      if(valid && current.rule.evaluator) valid = current.rule.evaluator(wildcards,v);
      if(!valid) continue;
      
      auto res = replace(current.rule.replacement, wildcards);
      
      if(res != e){
        if(apply_callback) apply_callback(current.rule,wildcards);
        return res;
      }
    }
    
    return e;
  }
  
  #pragma mark evaluator evaluator
  
  expression MultiEvaluator::evaluate(expression expr,EvaluatorVisitor &v)const{
    
    bool repeat;
    //std::cout << "enter: " << expr << std::endl;
    
    do{
      repeat = false;
      expression tmp = expr;
      for(auto & evaluator:evaluators){
        expr = evaluator->evaluate(expr,v);
        if(tmp != expr){
          repeat = true;
          break;
        }
      }
    }
    while (repeat);
    
    return expr;
  }

  expression StepEvaluator::evaluate(expression expr,EvaluatorVisitor &v)const{
    for(auto & evaluator:evaluators) expr = evaluator->evaluate(expr,v);
    return expr;
  }

  
}