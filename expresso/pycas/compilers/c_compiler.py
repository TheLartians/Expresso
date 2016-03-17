
import ctypes
from ..codeprinter import CodePrinter
from .. import expression as e
from .. import functions as f
import expresso.visitor as visitor
from mpmath import mp


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
          f.Types.Boolean:'bool',
          f.Types.Natural:'unsigned',
          f.Types.Integer:'int',
          f.Types.Rational:'double',
          f.Types.Real:'double',
          f.Types.Complex:'c_complex',
          None:'c_complex'
        }

        self.ctype_map = {
            'bool':ctypes.c_bool,
            'unsigned':ctypes.c_uint,
            'int':ctypes.c_int,
            'double':ctypes.c_double,
            'c_complex':c_complex
        }

        self.type_converters = {}
        self.need_conversion = {}
        self.preamble = set()
        self.globals = set()

        c_complex_type = '''
struct c_complex{ double real; double imag; };
inline std::complex<double> to_cpp_complex(c_complex z){ return std::complex<double>(z.real,z.imag); }
inline c_complex to_c_complex(std::complex<double> z){ return c_complex{z.real(),z.imag()}; }
        '''

        self.preamble.add(c_complex_type)
        self.type_converters['c_complex'] = (lambda x:'to_cpp_complex(%s)' % x,lambda x:'to_c_complex(%s)' % x)

        complex_operators = '''
inline complex<double> operator{0}(complex<double> lhs, const double & rhs){{
    return lhs {0} complex<double>(rhs);
}}
inline complex<double> operator{0}(const double & lhs,complex<double> rhs){{
    return complex<double>(lhs) {0} rhs;
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
      handles[i] = std::async(std::launch::async,[=](){
        C2 begin = start+block_size*i, end = start+block_size*(i+1);
        for(C2 j=begin;j<end;++j){ f(j); }
      });
    }
    handles[thread_count-1] = std::async([&](){
      C2 begin = start+block_size*(thread_count-1);
      for(C2 j=begin;j<end;++j)f(j);
     });
    for(auto & handle:handles) handle.wait();
}
        '''

        self.preamble.add(parallel_for)

        ndarray = '''
template<size_t _size,size_t _stride, size_t... sizes> struct ndarray_index_calculator {
  using rest = ndarray_index_calculator<sizes...>;
  static size_t size(){ return _size; }
  template <typename ... Args> static bool is_valid(size_t idx,Args ... args){ if(!rest::is_valid(args...)) return false; return idx < size(); }
  static size_t stride(){ return _stride; }
  template <typename ... Args> static size_t get_index(size_t idx,Args ... args){ return stride() * idx + rest::get_index(args...); }
};

template<size_t _size,size_t _stride> struct ndarray_index_calculator <_size,_stride> {
  static size_t size(){ return _size; }
  static bool is_valid(size_t idx){ return idx < size(); }
  static size_t stride(){ return _stride; }
  static size_t get_index(size_t idx){ return idx; }
};

template <class T,size_t ... size_stride> struct mapped_ndarray{
  T * data;
  T default_value;
  using index_calculator = ndarray_index_calculator<size_stride...>;
  mapped_ndarray(T * d,const T &_default_value = 0):data(d),default_value(_default_value){ }
  template <typename ... Args> T & operator()(Args ... indices){
    if(!index_calculator::is_valid(indices...)){ return default_value; }
    return data[index_calculator::get_index(indices...)];
  }
};
        '''

        self.preamble.add(ndarray)

        self.preamble.add('''template <typename T> int sign(T val) { return (T(0) <= val) - (val < T(0)); }''')


    def needs_brackets_in(self,expr,parent):
        if expr.is_atomic:
            return False
        return expr.function.is_operator

    @visitor.on('expr',parent = CodePrinter)
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr)

    @visitor.function(f.CustomFunction)
    def visit(self,expr):
        f = expr.args[0].value
        if hasattr(f,'ccode'):
            self.preamble.add(f.ccode)
        else:
            raise ValueError('cannot compile custom function %s' % expr)
        return "%s(%s)" % (f.name,','.join([self(arg) for arg in expr.args[1:]]))

    @visitor.function(f.exponentiation)
    def visit(self,expr):
        return 'pow(%s,%s)' % (self(expr.args[0]),self(expr.args[1]))

    @visitor.atomic(f.I)
    def visit(self,expr):
        return "std::complex<double>(0,1)"

    @visitor.atomic(f.pi)
    def visit(self,expr):
        return "M_PI"

    @visitor.atomic(f.e)
    def visit(self,expr):
        return "M_E"

    @visitor.function(f.Xor)
    def visit(self,expr):
        return self.print_binary_operator(expr,symbol='^')

    @visitor.function(f.Not)
    def visit(self,expr):
        return "!(%s)" % self(expr.args[0])

    @visitor.function(f.equal)
    def visit(self,expr):
        return self.print_binary_operator(expr,'==')

    @visitor.function(f.fraction)
    def visit(self,expr):
        return "1./(%s)" % self(expr.args[0])

    @visitor.function(f.mod)
    def visit(self,expr):
        return "fmod(%s,%s)" % (self(expr.args[0]),self(expr.args[1]))

    @visitor.function(f.InnerPiecewise)
    def visit(self,expr):
        parts = ['(%s)?(%s):' % (self(arg.args[1]),self(arg.args[0])) for arg in expr.args]
        return '(%s%s)' % (''.join(parts),self(e.S(0)))

    @visitor.symbol
    def visit(self,expr):
        converter = self.need_conversion.get(expr)
        if converter:
            if isinstance(converter,tuple):
                return converter[0](expr)
            else:
                return converter(expr)
        return expr.name

    @visitor.atomic(e.S(True))
    def visit(self,expr):
        return 'true'

    @visitor.atomic(e.S(False))
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

    @visitor.function(f.unfoldable)
    def visit(self,expr):
        return self.visit(expr.args[0])

    @visitor.function(f.ArrayAccess)
    def visit(self,expr):
        arr = expr.args[0].value
        pointer = arr.ctypes.data
        type = f.type_converters.numpy_c_typenames[arr.dtype.name]
        size = ','.join(["%s,%s" % (size,stride/arr.itemsize) for size,stride in zip(arr.shape,arr.strides)])
        name = expr.args[0].name
        self.globals.add('mapped_ndarray<%s,%s> %s((%s*)%s);' % (type,size,name,type,pointer))
        return "%s(%s)" % (name,','.join([self(arg) for arg in reversed(expr.args[1:])]))

    @visitor.obj(mp.mpf)
    def visit(self,expr):
        return repr(float(expr.value))

    @visitor.obj(mp.mpc)
    def visit(self,expr):
        v = expr.value
        return "complex<double>(%s,%s)" % (repr(float(v.real)),repr(float(v.imag)))

    def optimize_function(self,expr):
        from expresso.pycas.evaluators.optimizers import optimize_for_compilation
        return optimize_for_compilation(expr)

    def get_body_code(self,definition):
        if definition.return_type == None:
            return_type = self.print_typename(f.Type(definition.expr).evaluate())
        else:
            return_type = self.print_typename(definition.return_type)
        f_code = self(self.optimize_function(definition.expr))
        if return_type in self.type_converters and isinstance(self.type_converters[return_type],tuple):
            f_code = self.type_converters[return_type][1](f_code)
        return f_code

    def generate_function(self,definition):

        if definition.return_type == None:
            return_type = self.print_typename(f.Type(definition.expr).evaluate())
        else:
            return_type = self.print_typename(definition.return_type)

        args = definition.args

        if definition.arg_types == None:
            argument_types = [self.print_typename(f.Type(arg).evaluate()) for arg in args]
        else:
            argument_types = [self.print_typename(f.Type(arg).evaluate()) for arg in definition.arg_types]

        self.need_conversion = {arg:self.type_converters[t]
                                for arg,t in zip(args,argument_types)
                                if t in self.type_converters}

        f_code = self.get_body_code(definition)

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
            return_type = self.print_vector_typename(f.Type(definition.expr).evaluate())
        else:
            return_type = self.print_vector_typename(definition.return_type)

        args = definition.args

        if definition.arg_types == None:
            argument_types = [self.print_vector_typename(f.Type(arg).evaluate()) for arg in args]
        else:
            argument_types = [self.print_vector_typename(f.Type(arg).evaluate()) for arg in definition.arg_types]

        self.need_conversion.update({arg:lambda a:'%s[__i]' % a for arg in args})

        argument_types = ['unsigned',return_type] + argument_types

        if not use_previous_definition :
            f_code = self.get_body_code(definition)
        else:
            f_code = '%s(%s)' % (definition.name,','.join(self(arg) for arg in definition.args))

        if definition.parallel:
            f_code = 'parallel_for(0,__size,[&](unsigned __i){ __res[__i] = %s; }); ' % f_code
        else:
            f_code = 'for(unsigned __i = 0; __i<__size;++__i) __res[__i] = %s;' % f_code

        rargument_types = [argument_types[0]] + ['%s __restrict__ ' % t for t in argument_types[1:]]
        formatted_args = ','.join(['%s %s' % vardef for vardef in
                                   zip(rargument_types,['__size','__res'] + list(args))])

        formatted = (self.vectorized_name(definition.name), formatted_args, f_code)

        definition.c_vectorized_arg_types = [self.get_ctype(arg_type) for arg_type in argument_types]

        return 'extern "C"{\nvoid %s(%s){\n\t%s\n}\n}' % formatted


def ccompile(*function_definitions,**kwargs):
    import tempfile
    import shutil
    import ctypes
    import numpy as np
    from subprocess import Popen, PIPE
    from os import environ

    ccode_printer = CCodePrinter()
    code  = ccode_printer.print_file(*function_definitions)

    output_directory = tempfile.mkdtemp()

    object_file = output_directory+'/'+'pycas_compiled_expression.o'

    flags = kwargs.pop('flags',[])
    p = Popen([environ.get('CXX','g++'),'-o',object_file] + flags + ['-c','-xc++','-std=c++11','-ffast-math','-O3','-fPIC','-'],stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.stdin.write(code)
    p.stdin.close()

    return_code = p.wait()
    if(return_code!=0):
        raise RuntimeError("Cannot compile expression: " + p.stderr.read() )

    print_output = kwargs.pop('print_output',False)
    print_warnings = print_output or kwargs.pop('print_warnings',False)

    if print_warnings:
        print p.stderr.read()

    if print_output:
        print p.stdout.read()

    shared_library = output_directory+'/'+'pycas_compiled_expression.so'
    p = Popen(['g++','-shared','-o',shared_library,object_file],stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.stdin.close()

    return_code = p.wait()
    if(return_code!=0):
        raise RuntimeError("Cannot convert to shared library: " + p.stderr.read())

    if print_warnings:
        print p.stderr.read()

    if print_output:
        print p.stdout.read()

    lib = ctypes.cdll.LoadLibrary(shared_library)
    shutil.rmtree(output_directory)

    compiled_functions = {}

    class CompiledFunction(object):
        def __init__(self,cf,cf_vector):
            self.cf = cf
            self.cf_vector = cf_vector

        def __call__(self,*args,**kwargs):
            if(len(args) == 0):
                return self.cf()
            if any([isinstance(arg,(list,tuple)) for arg in args]):
                argtypes = self.cf_vector.argtypes
                args = [np.array(arg,dtype=t) for t,arg in zip(argtypes[2:],args)]
            if any([isinstance(arg,np.ndarray) for arg in args]):
                argtypes = self.cf_vector.argtypes

                shape = None
                for arg in args:
                    if isinstance(arg,np.ndarray):
                        if shape == None:
                            shape = arg.shape
                        else:
                            if arg.shape != shape:
                                raise AttributeError('c function got arguments with different shapes')

                args = [arg if isinstance(arg,np.ndarray) else arg * np.ones(shape) for arg in args]
                args = [np.ascontiguousarray(arg,dtype=t._type_) for t,arg in zip(argtypes[2:],args)]

                if argtypes[1]._type_ == c_complex:
                    restype = c_complex.np_type()
                else:
                    restype = argtypes[1]._type_

                res = kwargs.get('res')
                if res is None:
                    res = np.zeros(args[0].shape,dtype=restype)
                else:
                    assert res.dtype == restype
                    assert res.shape == shape
                    assert res.flags['C_CONTIGUOUS']

                call_args  = [res.size,res.ctypes.data_as(argtypes[1])]
                call_args += [arg.ctypes.data_as(t) for t,arg in zip(argtypes[2:],args)]
                self.cf_vector(*call_args)
                return res
            return self.cf(*args)

        def address(self):
            return ctypes.cast(self.cf, ctypes.c_void_p).value

    class CompiledLibrary(object):
        def __init__(self,lib,code):
            self.lib = lib
            self.code = code

    res = CompiledLibrary(lib,code)

    for definition in function_definitions:
        f = getattr(lib,definition.name)
        f.argtypes = definition.c_arg_types
        f.restype  = definition.c_return_type

        f_vector = getattr(lib, ccode_printer.vectorized_name(definition.name))
        f_vector.argtypes = definition.c_vectorized_arg_types
        f_vector.restype  = None

        setattr(res,definition.name,CompiledFunction(f,f_vector))

    return res


