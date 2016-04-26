
import expresso.visitor as visitor
from .. import expression as e
from .. import functions as f
from mpmath import mp

class LambdaCompiler(object):

    def __init__(self,value_converter = lambda x:x,function_module = None,value_module = None,cache=None,folding=False):
        self.dispatcher = self.visit.dispatcher
        self.value_converter = value_converter
        self.folding = folding
        if function_module == None:
            import math
            function_module = math
        if value_module == None:
            value_module = function_module
        self.function_module = function_module
        self.value_module = value_module

        if cache:
            self.cache = cache
        else:
            self.cache = {}

    def logic_reduce(self,op,args):
        return reduce(op,args)

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

    @visitor.function(f.OuterPiecewise)
    def visit(self,expr):
        return self.visit(expr.args[0])

    @visitor.function(f.addition)
    def visit(self,expr):
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:sum([arg(args) for arg in cargs])

    @visitor.function(f.multiplication)
    def visit(self,expr):
        from operator import mul
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(mul,[arg(args) for arg in cargs])

    @visitor.function(f.negative)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:-arg(args)

    @visitor.function(f.fraction)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        one = self.value_converter(1.)
        return lambda args:one/arg(args)

    @visitor.function(f.exponentiation)
    def visit(self,expr):
        from operator import pow
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:reduce(pow,[arg(args) for arg in cargs])

    @visitor.function(f.equal)
    def visit(self,expr):
        from operator import eq
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(eq,[arg(args) for arg in cargs])

    @visitor.function(f.unequal)
    def visit(self,expr):
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args: self.logic_reduce(lambda x,y:x!=y ,[arg(args) for arg in cargs])

    @visitor.function(f.Less)
    def visit(self,expr):
        from operator import lt
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(lt,[arg(args) for arg in cargs])

    @visitor.function(f.Greater)
    def visit(self,expr):
        from operator import gt
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(gt,[arg(args) for arg in cargs])

    @visitor.function(f.LessEqual)
    def visit(self,expr):
        from operator import le
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(le,[arg(args) for arg in cargs])

    @visitor.function(f.GreaterEqual)
    def visit(self,expr):
        from operator import ge
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(ge,[arg(args) for arg in cargs])

    @visitor.function(f.And)
    def visit(self,expr):
        from operator import __and__
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(__and__,[arg(args) for arg in cargs])

    @visitor.function(f.Or)
    def visit(self,expr):
        from operator import __or__
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(__or__,[arg(args) for arg in cargs])

    @visitor.function(f.Xor)
    def visit(self,expr):
        from operator import __xor__
        cargs = [self.visit(arg) for arg in expr.args]
        return lambda args:self.logic_reduce(__xor__,[arg(args) for arg in cargs])

    @visitor.function(f.real)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:arg(args).real

    @visitor.function(f.imag)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:arg(args).imag

    @visitor.function(f.conjugate)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:arg(args).conjugate()

    @visitor.function(f.Abs)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        def evaluate(args):
            return abs(arg(args))
        return evaluate

    @visitor.function(f.Min)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        return lambda args:min(*[arg(args) for arg in arguments])

    @visitor.function(f.Max)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        return lambda args:max(*[arg(args) for arg in arguments])

    @visitor.function(f.Not)
    def visit(self,expr):
        arg = self.visit(expr.args[0])
        return lambda args:not arg(args)

    @visitor.function(f.unfoldable)
    def visit(self,expr):
        if self.folding:
            raise RuntimeError('folding unfoldable value')
        return self.visit(expr.args[0])

    @visitor.function(f.Tuple)
    def visit(self,expr):
        arguments = [self.visit(arg) for arg in expr.args]
        def make_tuple(args):
            return tuple((arg(args) for arg in arguments))
        return make_tuple

    @visitor.function(f.ArrayAccess)
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

    @visitor.function(f.InnerPiecewise)
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
                raise ValueError('undefined symbol %r' % name)

        return get_symbol

    @visitor.obj(f.SymbolicConstant)
    def visit(self,expr):
        try:
            v = self.value_converter(self.value_module.__dict__[expr.value.name])
        except KeyError:
            raise ValueError('cannot compile %s' % expr)
        return lambda args: v

    @visitor.atomic(e.S(True))
    def visit(self,expr):
        return lambda args:True

    @visitor.atomic(e.S(False))
    def visit(self,expr):
        return lambda args:False

    @visitor.atomic(e.I)
    def visit(self,expr):
        im = self.value_converter(1j)
        return lambda args:im

    @visitor.obj(e.Number)
    def visit(self,expr):
        v = self.value_converter(expr.value)
        return lambda args:v

    @visitor.function(f.CustomFunction)
    def visit(self,expr):
        cargs = [self.visit(arg) for arg in expr.args[1:]]
        f = expr.args[0].value
        if not hasattr(f,'python_function'):
                raise ValueError("custom function %s has no attribute 'python_function'" % f.name)
        callback = f.python_function
        return lambda args:callback(*[arg(args) for arg in cargs])

    @visitor.obj
    def visit(self,expr):
        value = self.value_converter(expr.value)
        return lambda args:value

def lambdify(expr,**kwargs):
    compiler = LambdaCompiler(**kwargs)
    compiled = compiler.visit(e.S(expr))
    return lambda **args:compiled(args)

def mpmathify(expr,**kwargs):
    from mpmath import mp
    vc = lambda x:mp.mpc(x) if isinstance(x,(complex,mp.mpc)) else mp.mpf(x)
    compiler = LambdaCompiler(value_converter=vc,function_module=mp,**kwargs)
    f = compiler.visit(e.S(expr))
    return lambda **args:f(args)

def N(expr,mp_dps = None,**kwargs):
    from mpmath import mp
    if mp_dps != None:
        mp.dps = mp_dps;
    f = mpmathify(expr,**kwargs)
    res = f()
    return res
