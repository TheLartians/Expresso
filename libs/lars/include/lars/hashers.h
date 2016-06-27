
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

#include <functional>

namespace lars {
  
  template <class array,class scalar> struct array_hasher {
    std::hash<scalar> scalar_hasher;
    std::size_t operator()(const array& a) const {
      size_t seed = 0;
      for(int i=0;i<a.size();++i) seed ^= scalar_hasher(a[i]) + 0x9e3779b9 + (seed<<6) + (seed>>2);
      return seed;
    }
  };
  
  template <class T> inline void hash_combine(std::size_t& seed, T const& v) {
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }

  namespace{
  // Recursive template code derived from Matthieu M.
  template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1> struct HashValueImpl{
    static void apply(size_t& seed, Tuple const& tuple){
      HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
      hash_combine(seed, std::get<Index>(tuple));
      }
    };
  
    template <class Tuple> struct HashValueImpl<Tuple,0>{
      static void apply(size_t& seed, Tuple const& tuple){
        hash_combine(seed, std::get<0>(tuple));
      }
    };
  }
  
  template <typename ... TT> struct tuple_hasher {
    size_t operator()(std::tuple<TT...> const& tt) const{
      size_t seed = 0;
      HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
      return seed;
    }
  };

  
}