
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

#include "core.h"
#include <unordered_map>
#include <unordered_set>
#include <array>

#include <lars/iterators.h>

namespace expresso {
  
  struct BinaryIterator{
    virtual void init(const BinaryOperator *) = 0;
    virtual const std::vector<unsigned> & get_indices() = 0;
    virtual bool step() = 0;
    virtual std::shared_ptr<BinaryIterator> clone()const = 0;
  };
  
  namespace BinaryIterators {
    
    using namespace lars;
    
    struct All:public BinaryIterator{
      std::vector<unsigned> indices;
      void init(const BinaryOperator * op)override{ indices.resize(op->arguments.size()); for(auto i:range(op->arguments.size())) indices[i] = i; }
      const std::vector<unsigned> & get_indices()override{ return indices; }
      bool step()override{ return false; }
      std::shared_ptr<BinaryIterator> clone()const override{ return std::make_shared<All>(); }
    };
    
    struct Window:public BinaryIterator{
      std::vector<unsigned> indices;
      unsigned N;
      unsigned size;
      Window(unsigned _N = 2):N(_N){}
      void init(const BinaryOperator * op)override{ size = op->arguments.size(); indices.resize(std::min(size,N)); for(auto i:range(std::min(size,N))) indices[i] = i; }
      const std::vector<unsigned> & get_indices()override{ return indices; }
      bool step()override{ for(auto &i:indices)i++; return indices.back() != size;  }
      std::shared_ptr<BinaryIterator> clone()const override{ return std::make_shared<Window>(N); }
    };
    
    struct SingleOrdered:public BinaryIterator{
      subarray_indices::iterator it;
      unsigned N;
      SingleOrdered(unsigned _N = 2):N(_N){}
      void init(const BinaryOperator * op)override{ unsigned s = op->arguments.size(); it.init(std::min(s,N),s); }
      const std::vector<unsigned> & get_indices()override{ return *it; }
      bool step()override{ return it.step(); }
      std::shared_ptr<BinaryIterator> clone()const override{ return std::make_shared<SingleOrdered>(N); }
    };
    
    struct SingleUnordered:public BinaryIterator{
      permutated_subarray_indices::iterator it;
      unsigned N;
      SingleUnordered(unsigned _N = 2):N(_N){}
      void init(const BinaryOperator * op)override{ unsigned s = op->arguments.size(); it.init(std::min(s,N),s); }
      const std::vector<unsigned> & get_indices()override{ return *it; }
      bool step()override{ return it.step(); }
      std::shared_ptr<BinaryIterator> clone()const override{ return std::make_shared<SingleUnordered>(N); }
    };
    
  }
  
  class Evaluator;
  
  class EvaluatorVisitor:public Visitor{
    Expression::shared copy;
    const Evaluator & evaluator;
    bool modified = false;
    argument_list CAargs;
    
    bool get_from_cache(expression e,expression &res);
    bool is_cached(const Expression * e);
    void add_to_cache(expression e,expression res);
    void finalize(const Expression * e);
    bool copy_function(const Function * e);
    void visit(const Function * e)override;
    void visit_binary(const BinaryOperator * e);
    void visit(const BinaryOperator * e)override;
    void visit(const AtomicExpression * e)override;
    
    public:
    replacement_map & cache;
    std::unordered_set<expression> expression_stack;

    EvaluatorVisitor(const Evaluator &_evaluator,replacement_map & _cache);
    expression evaluate(expression e);
  };
  
  class Evaluator{
    public:
    
    struct settings_t{
      bool recursive = false;
      bool split_binary = true;
      unsigned split_binary_size = 2;
      bool commutate_binary = true;
    } settings;
    
    using ignore_set = std::unordered_set<expression>;
    virtual expression evaluate(expression,EvaluatorVisitor &) const = 0;
    
    virtual expression run(expression e)const;
    virtual expression run(expression e,replacement_map &cache)const;

    expression operator()(expression e)const{ return run(e); }
    expression operator()(expression e,replacement_map &cache)const{ return run(e,cache); }
  };
  
  template <typename F> class FunctionEvaluator:public Evaluator{
    F function;
  public:
    FunctionEvaluator(F f):function(f){  }
    virtual expression evaluate(expression expr)const{ return function(expr); }
  };
  
  template <typename F> FunctionEvaluator<F> create_function_evaluator(F f){ return FunctionEvaluator<F>(f); }
  
  class ReplaceEvaluator:public Evaluator{
    replacement_map replacements;
  public:
    ReplaceEvaluator(){}
    ReplaceEvaluator(const replacement_map &rep):replacements(rep){}
    void clear(){ replacements.clear(); }
    void add_replacement(expression search,expression replace);
    expression evaluate(expression,EvaluatorVisitor &)const override;
    void extend();
  };
  
  struct Rule{
    using expression_evaluator = std::function<bool(replacement_map &,EvaluatorVisitor &)>;
    using minimal_expression_evaluator = std::function<bool(replacement_map &)>;
    expression search,replacement,condition,valid;
    expression_evaluator evaluator;
    
    Rule(expression _search,expression _replacement,expression_evaluator ev = expression_evaluator()):search(_search),replacement(_replacement),evaluator(ev){}
    Rule(expression _search,expression _replacement,minimal_expression_evaluator ev):search(_search),replacement(_replacement),evaluator([ev](replacement_map &m,EvaluatorVisitor &){ return ev(m); }){}
    Rule(expression _search,expression _replacement,expression _condition,expression _valid,expression_evaluator ev = expression_evaluator()):search(_search),replacement(_replacement),condition(_condition),valid(_valid),evaluator(ev){}
    Rule(expression _search,expression _replacement,expression _condition,expression _valid,minimal_expression_evaluator ev):search(_search),replacement(_replacement),condition(_condition),valid(_valid),evaluator([ev](replacement_map &m,EvaluatorVisitor &){ return ev(m); }){}
  };
  

  std::ostream & operator<<(std::ostream &stream,const Rule &rule);

  
  /*
   TODO: add lazy evaluation
   */
  
  
  class RuleEvaluator:public Evaluator{
    using expression_evaluator = std::function<void(replacement_map &)>;

    struct CompressedRule{
      Rule rule;
      replacement_map wildcard_mapping;
      int priority;
      
      CompressedRule(const Rule &r,int p):rule(r),priority(p){ }
    };
    
    std::vector<CompressedRule> rules;
    std::shared_ptr<CompressedNode> search_tree = std::make_shared<CompressedNode>();
    
    unsigned wc_count = 0;
    int parent_evaluator_priority = 0;

    std::vector<std::pair<RuleEvaluator *,int>> child_evaluators;
    
  public:
    
    using rule_id = CompressedNode::ID;
        
    using CallbackFunction = std::function<void(const Rule &,const replacement_map &)>;
    CallbackFunction apply_callback;
        
    CompressedNode & get_search_tree(){ return *search_tree; }
        
    rule_id insert_rule(const Rule &r,int priority);
    const Rule & get_rule(rule_id idx)const{ return rules[idx].rule; }
    size_t size()const{ return rules.size(); }
    
    void add_rule(const Rule &r,int priority = 0);
    template <typename ... Args> void add_rule(Args... args){ add_rule(Rule(args...)); }

    expression evaluate(expression e,EvaluatorVisitor &)const override;
    expression evaluate(expression,EvaluatorVisitor &,std::vector<rule_id>)const;
    
    void insert_rules(const RuleEvaluator & e,int priority = 0);
    
    void add_evaluator(RuleEvaluator & e,int priority);
    void add_evaluator(RuleEvaluator & e){ add_evaluator(e,++parent_evaluator_priority); }
    
  };
  
  class MultiEvaluator:public Evaluator{
  protected:
    std::vector<Evaluator*> evaluators;
  public:
    void add_evaluator(Evaluator*e);
    expression evaluate(expression expr,EvaluatorVisitor &)const override;
  };

  
  class StepEvaluator:public Evaluator{
  protected:
    std::vector<Evaluator*> evaluators;
  public:
    void add_evaluator(Evaluator*e){ evaluators.emplace_back(e); }
    expression evaluate(expression expr,EvaluatorVisitor &)const override;
  };

  
  
}