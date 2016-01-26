
import pysymbols.visitor as visitor
from pysymbols.printer import Printer

from .functions import *
from .expression import *

class CodePrinter(Printer):

    def __init__(self):
        super(CodePrinter,self).__init__(expression_converter)

    @visitor.on('expr',parent = Printer)
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr)
    
    def print_wildcard_symbol(self,expr):
        raise ValueError('cannot compile wildcard %s' % expr)

    def print_wildcard_function(self,expr):
        raise ValueError('cannot compile wildcard %s' % expr)

    @visitor.function(CustomFunction)
    def visit(self,expr):
        raise ValueError('cannot compile python function %s' % expr)

    @visitor.function(ArrayAccess)
    def visit(self,expr):
        raise ValueError('cannot compile python function %s' % expr)


import ctypes

class c_complex(ctypes.Structure):
    _fields_ = [('real',ctypes.c_double),('imag',ctypes.c_double)]
    def __str__(self):
        return str(self.real) + '+' + str(self.imag) + 'j'
    def __repr__(self):
        return '(' + str(self.real) + ',' + str(self.imag) + ')'
    def __complex__(self):
        return complex(self.real,self.imag)
    def is_complex(self):
        return True
    @staticmethod
    def np_type():
        import numpy
        return numpy.complex128

class CCodePrinter(CodePrinter):

    def __init__(self):
        super(CCodePrinter,self).__init__()
        self.includes = {'cmath','complex','thread','future','vector'}
        self.namespaces = {'std'}

        self.typenames = {
          Types.Boolean:'bool',
          Types.Natural:'unsigned',
          Types.Integer:'int',
          Types.Rational:'double',
          Types.Real:'double',
          Types.Complex:'complex<double>',
          None:'complex<double>'
        }

        self.ctype_map = {
            'bool':ctypes.c_bool,
            'unsigned':ctypes.c_uint,
            'int':ctypes.c_int,
            'double':ctypes.c_double,
            'complex<double>':c_complex
        }

        self.type_converters = {}
        self.need_conversion = {}
        self.preamble = set()
        self.globals = set()

        complex_operators = '''
inline complex<double> operator{0}(complex<double> lhs, const double & rhs){{
    return lhs {0} rhs;
}}
inline complex<double> operator{0}(const double & lhs,complex<double> rhs){{
    return lhs {0} rhs;
}}
        '''

        self.preamble.update(set([complex_operators.format(op) for op in ['+', '-', '*', '/']]))

        parallel_for = '''
  inline unsigned hardware_thread_count(){ return std::thread::hardware_concurrency(); }

template<typename C1,typename C2,typename F> void parallel_for(C1 start,C2 end,F f,uintptr_t thread_count = hardware_thread_count()){
    if(end-start < thread_count) thread_count = end-start;
    std::vector<std::future<void>> handles(thread_count);
    C2 block_size = (end - start)/thread_count;
    for(uintptr_t i=0;i<thread_count-1;++i){
      handles[i] = std::async(std::launch::async,[=](){ for(C2 j=start+block_size*i;j!=start+block_size*(i+1);++j){ f(j); } });
    }
    handles[thread_count-1] = std::async([&](){ for(C2 j=start+block_size*(thread_count-1);j!=end;++j)f(j); });
    for(auto & handle:handles) handle.wait();
}
        '''

        self.preamble.add(parallel_for)

        ndarray = '''
template<size_t _size, size_t... sizes> struct ndarray_index_calculator {
  using rest = ndarray_index_calculator<sizes...>;
  static size_t size(){ return _size; }
  template <typename ... Args> static bool is_valid(size_t idx,Args ... args){ if(!rest::is_valid(args...)) return false; return idx < size(); }
  template <typename ... Args> static size_t get_index(size_t idx,Args ... args){ return idx + rest::size() * rest::get_index(args...); }
};
template<size_t _size> struct ndarray_index_calculator <_size> {
  static size_t size(){ return _size; }
  static bool is_valid(size_t idx){ return idx < size(); }
  static size_t get_index(size_t idx){ return idx; }
};
template <class T,size_t ... size> struct mapped_ndarray{
  T * data;
  T default_value;
  using index_calculator = ndarray_index_calculator<size...>;
  mapped_ndarray(T * d,const T &_default_value = 0):data(d),default_value(_default_value){ }
  template <typename ... Args> T & operator()(Args ... indices){
    if(!index_calculator::is_valid(indices...)){ return default_value; }
    return data[index_calculator::get_index(indices...)];
  }
};
        '''

        self.preamble.add(ndarray)


    def needs_brackets_in(self,expr,parent):
        if expr.is_atomic:
            return False
        return expr.function.is_operator

    @visitor.on('expr',parent = CodePrinter)
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr)

    @visitor.function(CustomFunction)
    def visit(self,expr):
        f = expr.args[0].value
        if hasattr(f,'ccode'):
            self.preamble.add(f.ccode)
        else:
            raise ValueError('cannot compile custom function %s' % expr)
        return "%s(%s)" % (f.name,','.join([self(arg) for arg in expr.args[1:]]))

    @visitor.function(Exponentiation)
    def visit(self,expr):
        return 'pow(%s,%s)' % (self(expr.args[0]),self(expr.args[1]))

    @visitor.atomic(I)
    def visit(self,expr):
        self.preamble.add("std::complex<double> __I(0,1);")
        return "__I"

    @visitor.function(Xor)
    def visit(self,expr):
        return self.print_binary_operator(expr,symbol='^')

    @visitor.function(Not)
    def visit(self,expr):
        return "!(%s)" % self(expr.args[0])

    @visitor.function(Equal)
    def visit(self,expr):
        return self.print_binary_operator(expr,'==')

    @visitor.function(Fraction)
    def visit(self,expr):
        return "1./(%s)" % self(expr.args[0])

    @visitor.function(Piecewise)
    def visit(self,expr):
        parts = ['(%s)?(%s):' % (self(arg.args[1]),self(arg.args[0])) for arg in expr.args]
        return '(%s%s)' % (''.join(parts),self(S(0)))

    @visitor.symbol
    def visit(self,expr):
        if expr in self.need_conversion:
            return self.need_conversion[expr](expr)
        return expr.name

    @visitor.atomic(S(True))
    def visit(self,expr):
        return 'true'

    @visitor.atomic(S(False))
    def visit(self,expr):
        return 'false'

    def print_includes(self):
        return '\n'.join(['#include <%s>' % name for name in self.includes])

    def print_namespaces(self):
        return '\n'.join(['using namespace %s;' % namespace for namespace in self.namespaces])

    def print_auxiliary_code(self):
        return '%s\n%s' % ('\n'.join(self.preamble),'\n'.join(self.globals))

    def print_file(self,*function_definitions):

        function_code = [self.generate_function(f) for f in function_definitions]
        function_code += [self.generate_vector_function(f,use_previous_definition=True) for f in function_definitions]

        return "\n\n".join([self.print_includes(),
                            self.print_namespaces(),
                            self.print_auxiliary_code()] + function_code )

    def print_typename(self,expr):
        return self.typenames.get(expr,self.typenames[None])

    def print_vector_typename(self,expr):
        return "%s*" % self.typenames.get(expr,self.typenames[None])

    def get_ctype(self,typename):
        if typename[-1] == '*':
            return ctypes.POINTER(self.get_ctype(typename[:-1]))
        return self.ctype_map[typename]

    @visitor.function(ArrayAccess)
    def visit(self,expr):
        arr = expr.args[0].value
        pointer = arr.ctypes.data
        type = numpy_converters.numpy_c_typenames[arr.dtype.name]
        size = ','.join([str(s) for s in arr.shape])
        name = expr.args[0].name
        self.globals.add('mapped_ndarray<%s,%s> %s((%s*)%s);' % (type,size,name,type,pointer))
        return "%s(%s)" % (name,','.join([self(arg) for arg in expr.args[1:]]))

    def generate_function(self,definition):

        if definition.return_type == None:
            return_type = self.print_typename(Type(definition.expr).evaluate())
        else:
            return_type = self.print_typename(definition.return_type)

        args = definition.args

        if definition.arg_types == None:
            argument_types = [self.print_typename(Type(arg).evaluate()) for arg in args]
        else:
            argument_types = [self.print_typename(Type(arg).evaluate()) for arg in definition.arg_types]

        self.need_conversion = {arg:self.type_converters[t]
                                for arg,t in zip(args,argument_types)
                                if t in self.type_converters}

        f_code = self(definition.expr)
        if return_type in self.type_converters:
            f_code = "%s(%s)" % (self.type_converters[return_type],f_code)

        formatted = (return_type, definition.name,
                    ','.join(['%s %s' % (type,arg.name) for arg,type in zip(args,argument_types)]),
                    f_code)

        definition.c_return_type = self.get_ctype(return_type)
        definition.c_arg_types = [self.get_ctype(arg_type) for arg_type in argument_types]

        return 'extern "C"{\n%s %s(%s){\n\treturn %s;\n}\n}' % formatted

    def vectorized_name(self,name):
        return "__%s_vector" % name

    def generate_vector_function(self,definition,use_previous_definition = False):

        if definition.return_type == None:
            return_type = self.print_vector_typename(Type(definition.expr).evaluate())
        else:
            return_type = self.print_vector_typename(definition.return_type)

        args = definition.args

        if definition.arg_types == None:
            argument_types = [self.print_vector_typename(Type(arg).evaluate()) for arg in args]
        else:
            argument_types = [self.print_vector_typename(Type(arg).evaluate()) for arg in definition.arg_types]

        self.need_conversion.update({arg:lambda a:'%s[__i]' % a.name
                                     for arg in args})

        argument_types = ['unsigned',return_type] + argument_types

        if not use_previous_definition :
            f_code = self(definition.expr)
            if return_type in self.type_converters:
                f_code = "%s(%s)" % (self.type_converters[return_type],f_code)
        else:
            f_code = '%s(%s)' % (definition.name,','.join(self(arg) for arg in definition.args))

        if definition.use_parallel:
            f_code = 'parallel_for(0,__size,[&](unsigned __i){ __res[__i] = %s; }); ' % f_code
        else:
            f_code = 'for(unsigned __i = 0; __i<__size;++__i) __res[__i] = %s;' % f_code

        formatted_args = ','.join(['%s %s' % vardef for vardef in
                                   zip(argument_types,['__size','__res'] + list(args))])


        formatted = (self.vectorized_name(definition.name), formatted_args, f_code)

        definition.c_vectorized_arg_types = [self.get_ctype(arg_type) for arg_type in argument_types]

        return 'extern "C"{ void %s(%s){\n\t%s\n} }' % formatted


