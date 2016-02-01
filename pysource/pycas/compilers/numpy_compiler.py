
from .lambda_compiler import LambdaCompiler,visitor
import numpy as np
import pycas.expression as e
import pycas.functions as f
from mpmath import mp

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

    @visitor.function(f.Piecewise)
    def visit(self,expr):

        res_type = f.Type(expr).evaluate(cache=self.cache).value
        if isinstance(res_type,f.TypeInfo):
            ptype = res_type.__dict__.get('python_type')
        else:
            ptype = None

        cond_args = [self.visit(arg.args[1]) for arg in expr.args ]
        eval_args = [self.visit(arg.args[0]) for arg in expr.args ]

        def evaluate(args):

            dtype = ptype if ptype is not None else args['_dtype']
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

    @visitor.obj(e.Number)
    def visit(self,expr):
        v = self.dtype(expr.value)
        return lambda args:v

    @visitor.function(f.Not)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:np.logical_not( arg(args) )

    @visitor.function(f.Max)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        return lambda args:np.maximum( *[arg(args) for arg in arguments] )

    @visitor.function(f.Min)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        return lambda args:np.minimum( *[arg(args) for arg in arguments] )

    @visitor.function(f.ArrayAccess)
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

    @visitor.obj(mp.mpc)
    def visit(self,expr):
        return lambda args:complex(expr.value)

    @visitor.obj(mp.mpf)
    def visit(self,expr):
        return lambda args:float(expr.value)


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

        result = np.zeros(example_arg.shape,dtype = f.restype)

        def worker(s,args):
            args = {name:value[s[0]:s[1]] for name,value in args.iteritems()}
            args['_slice'] = s
            result[s[0]:s[1]] = f(**args)

        threads = []

        for s in slices:
            t = threading.Thread(target=worker,args=[s,args])
            threads.append(t)
            t.start()

        for t in threads:
            t.join()


        return result

    return run_parallel_thread


def numpyfy(expr,dtype = float,parallel = False):

    from pycas.evaluators.optimizers import optimize_for_compilation

    compiler = NumpyCompiler(dtype)
    res = compiler.visit(optimize_for_compilation(e.S(expr)))

    res_type = f.Type(expr).evaluate(cache=compiler.cache).value
    if isinstance(res_type,f.TypeInfo):
        restype = res_type.__dict__.get('python_type')
        if restype == None:
            res_type = dtype
    else:
        restype = dtype

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
            args['_dtype'] = restype

        cres = res(args)

        if not isinstance(cres,np.ndarray):
            cres = np.ones(example_arg.shape,dtype=dtype) * cres

        if to_number:
            cres = cres[0]

        return cres

    call.restype = restype

    if parallel:
        return make_parallel(call)
    else:
        return call
