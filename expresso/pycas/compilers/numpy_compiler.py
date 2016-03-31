
from .lambda_compiler import LambdaCompiler,visitor
import numpy as np
from .. import expression as e
from .. import functions as f
from mpmath import mp

class NumpyCompiler(LambdaCompiler):

    def __init__(self):
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

    @visitor.function(f.InnerPiecewise)
    def visit(self,expr):

        restype = f.Type(expr).evaluate(cache=self.cache).value
        if isinstance(restype ,f.TypeInfo):
            ptype = restype .__dict__.get('python_type')
        else:
            ptype = None

        cond_args = [self.visit(arg.args[1]) for arg in expr.args ]
        eval_args = [self.visit(arg.args[0]) for arg in expr.args ]

        def evaluate(args):

            #dtype = ptype if ptype is not None else args['_dtype']
            #shape = args['_shape']
            #res = np.zeros(shape,dtype = dtype)
            #unset = np.ones(shape,dtype = bool)

            is_arr = False

            for cond,val in zip(cond_args,eval_args):

                valid = cond(args)

                if not isinstance(valid,np.ndarray) or valid.shape == [1]:
                    if valid == False:
                        continue
                    if valid == True:
                        if not is_arr:
                            return val(args)
                        valid = unset
                else:
                    if not is_arr:
                        if np.all(valid):
                            return val(args)
                        if np.all(valid==False):
                            continue
                        shape = valid.shape
                        is_arr = True
                        unset = np.ones(shape,dtype = bool)
                        if ptype:
                            res = np.zeros(shape,dtype = ptype)
                        else:
                            res = np.zeros(shape)

                    valid &= unset

                new_args = { name:arg[valid] if isinstance(arg,np.ndarray) and arg.shape==shape else arg
                            for name,arg in args.iteritems() }

                values = np.array(val(new_args))

                if not np.can_cast(values.dtype,res.dtype):
                    res = res.astype(values.dtype)

                res[valid] = values

                unset &= np.logical_not(valid)

            return res

        return evaluate

    @visitor.obj(e.Number)
    def visit(self,expr):
        v = expr.value
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


def get_example_arg(args):
    for arg in args.values():
        if isinstance(arg,np.ndarray):
            return arg
    for arg in args.values():
        if hasattr(arg,"__len__"):
            return arg
    return args.values()[0]


def prepare_arguments(args):
    example_arg = get_example_arg(args)

    if not hasattr(example_arg,"__len__"):
        shape = None
        args = { name:np.array([arg]) for name,arg in args.iteritems() }
    else:
        shape = np.array(example_arg).shape
        args = { name:(np.array(arg) if hasattr(arg,"__len__") else arg) for name,arg in args.iteritems() }

    return args,shape


def make_parallel(f):

    import threading
    from multiprocessing import cpu_count

    def run_parallel_thread(_processes = cpu_count(),**args):

        args,shape = prepare_arguments(args)
        size = shape[0] if shape else 1
        _processes = min(size,_processes)

        if _processes == 1:
            return f(**args)

        step = int(size/_processes)
        slices = [[i*step,(i+1)*step] for i in range(_processes)]
        slices[-1][1] = size

        result = np.zeros(shape,dtype = f.restype)

        def worker(s,args):
            args = {name:(value[s[0]:s[1]]) if isinstance(value,np.ndarray) and value.shape==shape else value for name,value in args.iteritems()}
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

def numpyfy(expr,parallel = False,restype = None):

    from expresso.pycas.evaluators.optimizers import optimize_for_compilation

    compiler = NumpyCompiler()
    res = compiler.visit(optimize_for_compilation(e.S(expr)))

    if restype is None:
        restype = f.Type(expr).evaluate(cache=compiler.cache).value
        if isinstance(restype,f.TypeInfo):
            restype = restype.__dict__.get('python_type')
            if restype == None:
                restype = complex
        else:
            restype = complex

    def call(**args):

        args,shape = prepare_arguments(args)
        cres = np.array(res(args)).astype(restype)

        if not shape:
            if cres.shape:
                cres = cres[0]
        else:
            if shape and cres.shape != shape:
                cres = np.ones(shape)*cres

        return cres

    call.restype = restype

    if parallel:
        return make_parallel(call)
    else:
        return call


def ncompile(*function_definitions):

    functions = {}

    for definition in function_definitions:

        if definition.return_type is not None:
            restype = definition.return_type.value.__dict__.get('python_type')
        else:
            restype = None

        f = numpyfy(definition.expr,parallel=definition.parallel,restype=restype)

        if definition.arg_types:
            # TODO: implement argument type conversions
            arg_types = [arg.value.__dict__.get('python_type') for arg in definition.arg_types]

        arg_names = [arg.name for arg in definition.args]

        class Delegate(object):
            def __init__(self,f,arg_names):
                self.f = f
                self.arg_names = arg_names
            def __call__(self, *args, **kwargs):
                args = { n:a for n,a in zip(self.arg_names,args) }
                res = kwargs.pop('res',None)
                if res is not None:
                    res[:] = self.f(**args)
                    return res
                else:
                    return self.f(**args)

        functions[definition.name] = Delegate(f,arg_names)

    class lib(object):
        def __init__(self,functions):
            self.__dict__.update(functions)

    return lib(functions)











