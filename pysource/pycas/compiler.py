
import pysymbols.visitor as visitor
from .expression import *
from .functions import *

class LambdaCompiler(object):

    def __init__(self,value_converter = lambda x:x,function_module = None,value_module = None):
        self.dispatcher = self.visit.dispatcher
        self.value_converter = value_converter
        if function_module == None:
            import math
            function_module = math
        if value_module == None:
            value_module = function_module
        self.function_module = function_module
        self.value_module = value_module

    @visitor.on('expr')
    def visit(self,expr):
        raise ValueError('cannot compile expression: %s' % expr)

    def get_function(self,name):
        if name in self.function_module.__dict__:
            return self.function_module.__dict__[name]

    @visitor.function
    def visit(self,expr):
        func = self.get_function(expr.function.name)
        if func == None:
            raise ValueError('unknown function: %s' % expr.function.name)
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:func(*[arg(args) for arg in cargs])

    @visitor.function(Addition)
    def visit(self,expr):
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:sum([arg(args) for arg in cargs])

    @visitor.function(Multiplication)
    def visit(self,expr):
        from operator import mul
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(mul,[arg(args) for arg in cargs])

    @visitor.function(Negative)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:-arg(args)

    @visitor.function(Fraction)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        one = self.value_converter(1.)
        return lambda args:one/arg(args)

    @visitor.function(Exponentiation)
    def visit(self,expr):
        from operator import pow
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(pow,[arg(args) for arg in cargs])

    @visitor.function(Equal)
    def visit(self,expr):
        from operator import eq
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(eq,[arg(args) for arg in cargs])

    @visitor.function(Less)
    def visit(self,expr):
        from operator import lt
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(lt,[arg(args) for arg in cargs])

    @visitor.function(Greater)
    def visit(self,expr):
        from operator import gt
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(gt,[arg(args) for arg in cargs])

    @visitor.function(LessEqual)
    def visit(self,expr):
        from operator import le
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(le,[arg(args) for arg in cargs])

    @visitor.function(GreaterEqual)
    def visit(self,expr):
        from operator import ge
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(ge,[arg(args) for arg in cargs])

    @visitor.function(And)
    def visit(self,expr):
        from operator import __and__
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(__and__,[arg(args) for arg in cargs])

    @visitor.function(Or)
    def visit(self,expr):
        from operator import __or__
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(__or__,[arg(args) for arg in cargs])

    @visitor.function(Xor)
    def visit(self,expr):
        from operator import __xor__
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(__xor__,[arg(args) for arg in cargs])

    @visitor.function(Real)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:arg(args).real

    @visitor.function(Imag)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:arg(args).imag

    @visitor.function(Conjugate)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:arg(args).conjugate()

    @visitor.function(Abs)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:abs(arg(args))

    @visitor.function(Not)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:not arg(args)

    @visitor.function(ArrayAccess)
    def visit(self,expr):
        array = expr.args[0].value
        indices = [self.visit(arg) for arg in expr.args[1:]]

        def access(args):
            idx = [int(arg(args)) for arg in indices[::-1]]
            for i,s in zip(idx,array.shape):
                if (i<0 or i>=s):
                    return self.value_converter(0)

            return array[tuple(idx)]

        return access

    @visitor.function(Piecewise)
    def visit(self,expr):
        cond_args = [self.visit(arg.args[1]) for arg in expr.args]
        eval_args = [self.visit(arg.args[0]) for arg in expr.args]

        def evaluate(args):
            for i in xrange(len(cond_args)):
                if cond_args[i](args):
                    return eval_args[i](args)
            return 0

        return evaluate

    @visitor.symbol
    def visit(self,expr):
        name = expr.name

        def get_symbol(args):
            try:
                return self.value_converter(args[name])
            except KeyError:
                pass
            try:
                return self.value_converter(self.value_module.__dict__[name])
            except KeyError:
                pass
            raise ValueError('undefined symbol %s' % name)

        return get_symbol

    @visitor.atomic(S(True))
    def visit(self,expr):
        return lambda args:True

    @visitor.atomic(S(False))
    def visit(self,expr):
        return lambda args:False

    @visitor.atomic(I)
    def visit(self,expr):
        im = self.value_converter(1j)
        return lambda args:im

    @visitor.obj(integer_type)
    def visit(self,expr):
        v = self.value_converter(expr.value)
        return lambda args:v

    @visitor.function(CustomFunction)
    def visit(self,expr):
        cargs = [self.visit(arg) for arg in expr.args[1:]]
        f = expr.args[0].value
        if not hasattr(f,'python_function'):
                raise ValueError("custom function %s has no attribute 'python_function'" % f.name)
        callback = f.python_function
        return lambda args:callback(*[arg(args) for arg in cargs])

def lambdify(expr,**kwargs):
    compiler = LambdaCompiler(**kwargs)
    compiled = compiler.visit(S(expr))
    return lambda **args:compiled(args)

def mpmathify(expr):
    from mpmath import mp
    vc = lambda x:mp.mpc(x) if isinstance(x,(complex,mp.mpc)) else mp.mpf(x)
    compiler = LambdaCompiler(value_converter=vc,function_module=mp)
    f = compiler.visit(S(expr))
    return lambda **args:f(args)

def N(expr,mp_dps = None,**kwargs):
    from mpmath import mp
    if mp_dps != None:
        mp.dps = mp_dps;
    f = mpmathify(expr)
    res = f(**kwargs)
    return res

import numpy as np

class NumpyCompiler(LambdaCompiler):

    def __init__(self,dtype):
        self.dtype = dtype
        super(NumpyCompiler,self).__init__()

    @visitor.on('expr',parent = LambdaCompiler)
    def visit(self,expr):
        raise ValueError('cannot compile expression: %s' % expr)

    def get_function(self,name):
        func = None
        if name in np.__dict__:
            return np.__dict__[name]
        if name[0] == 'a':
            arcname = 'arc' + name[1:]
            if arcname in np.__dict__:
                return np.__dict__[arcname]
        return None

    @visitor.function(Piecewise)
    def visit(self,expr):

        cond_args = [self.visit(arg.args[1]) for arg in expr.args ]
        eval_args = [self.visit(arg.args[0]) for arg in expr.args ]

        def evaluate(args):

            dtype = args['_dtype']
            shape = args['_shape']
            res = np.zeros(shape,dtype = dtype)
            unset = np.ones(shape,dtype = bool)

            for cond,val in zip(cond_args,eval_args):

                valid = cond(args)

                if not isinstance(valid,np.ndarray) or valid.shape == [1]:
                    if valid == False:
                        continue
                    if valid == True:
                        valid = unset
                else:
                    valid &= unset

                new_args = { name:arg[valid] if isinstance(arg,np.ndarray) and name[0]!='_' else arg
                            for name,arg in args.iteritems() }

                res[valid] = val(new_args)
                unset &= np.logical_not(valid)

            return res

        return evaluate

    @visitor.atomic(pi)
    def visit(self,expr):
        import numpy as np
        return lambda s,e:lambda args:np.pi

    @visitor.atomic(e)
    def visit(self,expr):
        import numpy as np
        return lambda s,e:lambda args:np.e

    @visitor.obj(integer_type)
    def visit(self,expr):
        v = self.dtype(expr.value)
        return lambda args:v

    @visitor.function(Not)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:np.logical_not( arg(args) )

    @visitor.function(Max)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        return lambda args:np.maximum( *[arg(args) for arg in arguments] )

    @visitor.function(Min)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        return lambda args:np.minimum( *[arg(args) for arg in arguments] )

    @visitor.function(ArrayAccess)
    def visit(self,expr):
        array = expr.args[0].value

        indices = [self.visit(arg) for arg in expr.args[1:]]

        def access_function(args):

            idx = [arg(args) for arg in indices[::-1]]

            shape = None
            for i in idx:
                if isinstance(i,np.ndarray):
                    shape = i.shape
                    break

            if shape != None:
                idx = [arg.astype(int) if isinstance(arg,np.ndarray) else int(arg)*np.ones(shape,dtype=int)
                       for arg in idx]

            valid =  reduce(np.logical_and,[ (i >= 0) & (i<s) for i,s in zip(idx,array.shape) ])

            if np.all(valid):
                return array[tuple(idx)]
            if np.all(valid == False):
                return self.value_converter(0)

            res = np.zeros(valid.shape,dtype = array.dtype)
            idx = [i[valid] for i in idx]
            res[valid] = array[tuple(idx)]

            return res

        return access_function


class FunctionDefinition(object):

    def __init__(self,name,args,expr,return_type = None,arg_types = None,use_parallel=True):
        self.name = name
        self.expr = S(expr)
        self.args = args
        self.return_type = return_type
        self.arg_types = arg_types
        self.use_parallel = use_parallel


def make_parallel(f):

    import threading
    from multiprocessing import cpu_count

    def run_parallel_thread(_processes = cpu_count(),**args):

        example_arg = args.iteritems().next()[1]

        size = len(example_arg) if hasattr(example_arg,'__len__') else 1
        _processes = min(size,_processes)

        if _processes == 1:
            return f(**args)

        example_arg = np.array(example_arg)

        step = int(size/_processes)
        slices = [[i*step,(i+1)*step] for i in range(_processes)]
        slices[-1][1] = size
        result = np.zeros(example_arg.shape,dtype = f.dtype)

        def worker(s,args):
            args = {name:value[s[0]:s[1]] for name,value in args.iteritems()}
            args['_slice'] = s
            result[s[0]:s[1]] = f(**args)

        threads = []

        for s in slices:
            t = threading.Thread(target=worker,args=[s,args])
            threads.append(t)
            t.start()

        for r in threads:
            t.join()

        return result

    return run_parallel_thread

def numpyfy(expr,dtype = float,parallel = False):

    compiler = NumpyCompiler(dtype)
    res = compiler.visit(S(expr))

    def call(**args):

        example_arg = args.itervalues().next()
        to_number = not hasattr(example_arg,"__len__")

        if to_number:
            args = { name:np.array([arg]) for name,arg in args.iteritems() }
            example_arg = args.itervalues().next()
        else:
            args = {name:np.array(arg) for name,arg in args.iteritems() }
            example_arg = args.itervalues().next()

        if '_shape' not in args:
            args['_shape'] = example_arg.shape
        if '_dtype' not in args:
            args['_dtype'] = dtype

        cres = res(args)

        if not isinstance(cres,np.ndarray):
            cres = np.ones(example_arg.shape,dtype=dtype) * cres

        if to_number:
            cres = cres[0]

        return cres

    call.dtype = dtype

    if parallel:
        return make_parallel(call)
    else:
        return call


def ccompile(*function_definitions,**kwargs):
    from .codeprinter import CCodePrinter,c_complex
    import tempfile
    import shutil
    import ctypes
    from subprocess import Popen, PIPE

    ccode_printer = CCodePrinter()

    output_directory = tempfile.mkdtemp()

    object_file = output_directory+'/'+'pycas_compiled_expression.o'
    p = Popen(['g++','-o',object_file,'-c','-xc++','-std=c++11','-funsafe-math-optimizations','-O3','-fPIC', '-'],stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.stdin.write(ccode_printer.print_file(*function_definitions))
    p.stdin.close()

    return_code = p.wait()
    if(return_code!=0):
        raise RuntimeError("Cannot compile expression: " + p.stderr.read() )

    print_warnings = kwargs.get('print_warnings')

    if print_warnings:
        print p.stderr.read()

    shared_library = output_directory+'/'+'pycas_compiled_expression.so'
    p = Popen(['g++','-shared','-o',shared_library,object_file],stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.stdin.close()

    return_code = p.wait()
    if(return_code!=0):
        raise RuntimeError("Cannot convert to shared library: " + p.stderr.read())

    if print_warnings:
        print p.stderr.read()

    lib = ctypes.cdll.LoadLibrary(shared_library)
    shutil.rmtree(output_directory)

    compiled_functions = {}

    class CompiledFunction(object):
        def __init__(self,cf,cf_vector):
            self.cf = cf
            self.cf_vector = cf_vector

        def __call__(self,*args):
            if(len(args) == 0):
                return self.cf()
            if isinstance(args[0],(list,tuple)):
                argtypes = self.cf_vector.argtypes
                args = [np.array(arg,dtype=t) for t,arg in zip(argtypes[2:],args)]
            if isinstance(args[0],np.ndarray):
                argtypes = self.cf_vector.argtypes
                args = [np.ascontiguousarray(arg,dtype=t._type_) for t,arg in zip(argtypes[2:],args)]

                if argtypes[1]._type_ == c_complex:
                    restype = c_complex.np_type()
                else:
                    restype = argtypes[1]._type_

                res = np.zeros(args[0].shape,dtype=restype)

                call_args = [res.size,res.ctypes.data_as(argtypes[1])]
                call_args += [arg.ctypes.data_as(t) for t,arg in zip(argtypes[2:],args)]
                self.cf_vector(*call_args)
                return res
            return self.cf(*args)

        def address(self):
            return ctypes.cast(self.cf, ctypes.c_void_p).value



    class CompiledLibrary(object):
        def __init__(self,lib):
            self.lib = lib

    res = CompiledLibrary(lib)

    for definition in function_definitions:
        f = getattr(lib,definition.name)
        f.argtypes = definition.c_arg_types
        f.restype  = definition.c_return_type

        f_vector = getattr(lib, ccode_printer.vectorized_name(definition.name))
        f_vector.argtypes = definition.c_vectorized_arg_types
        f_vector.restype  = None

        setattr(res,definition.name,CompiledFunction(f,f_vector))

    return res


