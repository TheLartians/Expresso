
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

#include <lars/no_content.h>

#include <unordered_map>
#include <stdexcept>
#include <functional>
#include <iterator>

namespace expresso{
  
  class expression_exception:public std::exception{
    const Expression * cause;
    mutable std::string what_string;
  public:
    expression_exception(const Expression * _cause):cause(_cause){}
    Expression::shared get_cause()const{ return cause->get_shared(); }
    virtual void print_to_stream(std::ostream &stream)const;
    const char * what()const noexcept override;
  };
    
#pragma mark match and replace
  
  expression rhs_associative(expression x);
  
  Expression::shared replace(const Expression::shared &s,const replacement_map &replacements );
  bool match( const Expression::shared &expr,const Expression::shared &search,replacement_map &wildcards);
  
  void get_matches( const Expression::shared &expr, std::shared_ptr<CompressedNode> searches, replacement_map &wildcards, std::vector<CompressedNode::ID> & matches );
  std::vector<CompressedNode::ID> get_matches( const Expression::shared &expr, std::shared_ptr<CompressedNode> searches, replacement_map &wildcards );
  
#pragma mark mulplicity list
  
  
  struct group{
    const Function &operation,&inverse;
    expression neutral;
    group(const Function &O,const Function &I,expression N):operation(O),inverse(I),neutral(N){ }
  };
  
  struct field{
    group additive_group,multiplicative_group;
    field(const group &_addition,const group &_multiplication):additive_group(_addition),multiplicative_group(_multiplication){ }
  };
  
  class mulplicity_list:public std::vector<std::pair<expression,expression>>{
  private:
    
  public:
    
    using inner_intersection_function = std::function<expression(const expression &,const expression &)>;
    
    const group & base;
    const Function & mulplicity;
    const field & real_field;
    
    mulplicity_list(const group &base,const Function &mulplicity,const field &real_field);
    mulplicity_list(const expression &e,const group &base,const Function &mulplicity,const field &real_field);
    
    void set_from_argument_list(const argument_list &e);
    void set_from_expression(const expression &e);

    mulplicity_list sum(const mulplicity_list &other)const;
    mulplicity_list difference(const mulplicity_list &other)const;
    mulplicity_list power(const expression &expr)const;
    
    mulplicity_list intersection(const mulplicity_list &other, inner_intersection_function get_inner = inner_intersection_function())const;
    
    argument_list as_argument_list()const;
    expression as_expression()const;
    
  };
  
  std::ostream & operator<<(std::ostream & stream,const mulplicity_list &l);
    
#pragma mark Expression iterators
  
  using expression_location = std::vector<unsigned>;
  expression replace(const expression expr, std::function<expression(const expression_location &)> search);

  std::ostream & operator<<(std::ostream &,const expression_location &);
  
  class expression_iterator:public std::iterator<std::forward_iterator_tag,expression>{
  protected:
    public:
    
    expression_location index_stack;
    std::vector<expression> expression_stack;
    
  public:
    expression replace_current(const expression &)const;
    
    expression operator*()const;
    virtual expression_iterator &operator++() = 0;
    
    bool operator==(const expression_iterator &other)const{ return other.location() == location() && other.expression_stack == expression_stack; }
    bool operator!=(const expression_iterator &other)const{ return !(other == *this); }

    const expression_location & location()const{ return index_stack; }
    
    const expression &outer()const{ return expression_stack.front(); }
    unsigned depth(){ return expression_stack.size(); }
  };
  
  class stack_based_expression_iterator:public expression_iterator{
    protected:
    std::vector<std::pair<argument_list::const_iterator, argument_list::const_iterator>> iterator_stack;
    void set_expression(const expression &,unsigned idx);
    void recursive_set_expression(const expression &,unsigned idx );
  };
  
  
  class preorder_traversal{
    expression top;
    
  public:
    class iterator:public stack_based_expression_iterator{
    public:
      iterator(const expression & e);
      iterator(){}
      iterator &operator++()override;
      iterator operator++(int);
    };
    
    preorder_traversal(expression _top):top(_top){}
    iterator begin()const{ return iterator(top); }
    iterator end()const{ return iterator(); }
  };
  
  class postorder_traversal{
    expression top;

  public:
    class iterator:public stack_based_expression_iterator{
    public:
      iterator(const expression & e);
      iterator(){}
      iterator &operator++()override;
      iterator operator++(int);
    };

    postorder_traversal(expression _top):top(_top){}
    iterator begin()const{ return iterator(top); }
    iterator end()const{ return iterator(); }
  };
  
#pragma mark permutations
  
  class commutative_permutations{
    expression top;
    
  public:
    
    class iterator:public std::iterator<std::forward_iterator_tag,expression>{

      struct commutable_expression{
        std::shared_ptr<const BinaryOperator> expression;
        std::vector<std::pair<unsigned,expression_location>> inner_commutable_expressions;

        expression_location location;
        std::vector<unsigned> permutated_argument_indices;
        
        commutable_expression(std::shared_ptr<const BinaryOperator> e,const expression_location & p);
        
        bool operator==(const commutable_expression &other)const{
          return expression == other.expression && location == other.location && permutated_argument_indices == other.permutated_argument_indices;
        }
        bool operator!=(const commutable_expression &other)const{ return !(*this == other); }
        
      };
      
      expression get_permutated_expression(const commutable_expression & e)const;
      
      std::vector< commutable_expression > commutable_expressions;
      std::vector<unsigned> outer_commutable_indices;
      
      expression base;
      
      void add_expression(const expression &,expression_location loc,unsigned parent_index = -1);
      
    public:
      iterator(const expression & e);
      iterator(){}
      
      expression operator*()const;
      
      iterator &operator++();
      iterator operator++(int);
      
      bool operator==(const iterator &other)const{ return other.commutable_expressions == commutable_expressions && base == other.base; }
      bool operator!=(const iterator &other)const{ return !(other == *this); }
      
    };
    
    commutative_permutations(expression _top):top(_top){}
    iterator begin()const{ return iterator(top); }
    iterator end()const{ return iterator(); }

    
  };
  
}






