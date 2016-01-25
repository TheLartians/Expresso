
import pysymbols.visitor as visitor
from expression import *
from functions import *

class LambdaCompiler(object):

    def __init__(self):
        self.dispatcher = self.visit.dispatcher

    @visitor.on('expr')
    def visit(self,expr):
        raise ValueError('cannot compile expression: %s' % expr)

    def get_function(self,name):
        import math
        if name in math.__dict__:
            return math.__dict__[name]

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
        return lambda args:1/arg(args)

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

    @visitor.function(Not)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:not arg(args)

    @visitor.function(Piecewise)
    def visit(self,expr):
        cond_args = [self.visit(arg.args[1]) for arg in expr.args if arg.args[1] != otherwise]
        eval_args = [self.visit(arg.args[0]) for arg in expr.args if arg.args[1] != otherwise]
        cond_args += [lambda a:True for arg in expr.args if arg.args[1] == otherwise]
        eval_args += [self.visit(arg.args[0]) for arg in expr.args if arg.args[1] == otherwise]

        def evaluate(args):
            for i in xrange(len(cond_args)):
                if cond_args[i](args):
                    return eval_args[i](args)
        return evaluate

    @visitor.symbol
    def visit(self,expr):
        return lambda args:args[expr.name]

    @visitor.atomic(I)
    def visit(self,expr):
        return lambda args:1j

    @visitor.obj(integer_type)
    def visit(self,expr):
        v = float(expr.value)
        return lambda args:v

    @visitor.function(CustomFunction)
    def visit(self,expr):
        cargs = [self.visit(arg) for arg in expr.args[1:]]
        f = expr.args[0].value
        if not hasattr(f,'python_function'):
                raise ValueError("custom function %s has no attribute 'python_function'" % f.name)
        callback = f.python_function
        return lambda args:callback(*[arg(args) for arg in cargs])

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

                if not isinstance( c,np.ndarray):
                    if valid == False:
                        continue
                    if valid == True:
                        valid = unset
                else:
                    valid &= unset

                unset -= valid

                new_args = { name:arg[valid] if isinstance(arg,np.ndarray) and name[0]!='_' else arg
                            for name,arg in args.iteritems() }
                res[valid] = val(new_args)

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

def lambdify(expr):
    compiler = LambdaCompiler()
    compiled = compiler.visit(expr)
    return lambda **args:compiled(args)

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
    res = compiler.visit(expr)

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


def ccompile(function_definitions,print_warnings = False):
    from .codeprinter import CCodePrinter
    import tempfile
    import shutil
    import ctypes
    from subprocess import Popen, PIPE

    if not isinstance(function_definitions,(list,tuple)):
        function_definitions = [function_definitions]

    ccode_printer = CCodePrinter()

    output_directory = tempfile.mkdtemp()

    object_file = output_directory+'/'+'pycas_compiled_expression.o'
    p = Popen(['g++','-o',object_file,'-c','-xc++','-std=c++11','-O3','-fPIC', '-'],stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.stdin.write(ccode_printer.print_file(*function_definitions))
    p.stdin.close()

    return_code = p.wait()
    if(return_code!=0):
        raise RuntimeError("Cannot compile expression: " + p.stderr.read() )

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

    for definition in function_definitions:
        f = getattr(lib,definition.name)
        f.argtypes = definition.c_arg_types
        f.restype  = definition.c_return_type

    return lib


