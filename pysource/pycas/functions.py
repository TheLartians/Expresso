from expression import Function,BinaryOperator,UnaryOperator,Symbol,Tuple,Wildcard
import pysymbols

# Symbols
# -------

e = Symbol('e')
pi = Symbol('pi')
oo = Symbol('oo')

# Various Functions
# -----------------

factorial = UnaryOperator('!',pysymbols.postfix,0)

sign = Function('sign',argc = 1)
Floor = Function('floor',argc = 1)
Ceil = Function('ceil',argc = 1)
Round = Function('round',argc = 1)

Mod = Function('mod',argc = 2)

Real = Function('real',argc = 1)
Imag = Function('imag',argc = 1)
Conjugate = Function('conjugate',argc = 1)

Indicator = Function('indicator',argc = 1)
Piecewise = BinaryOperator('}{',-20)

# Calculus
# --------

derivative = Function('derivative',argc = 2)
evaluated_at = Function('evaluated_at',argc = 3)
tmp = UnaryOperator('tmp_',pysymbols.prefix,0)

# Trigonometric Functions
# -----------------------

sqrt = Function('sqrt',argc = 1)
exp = Function('exp',argc = 1)
log = Function('log',argc = 1)
sin = Function('sin',argc = 1)
cos = Function('cos',argc = 1)
asin = Function('asin',argc = 1)
acos = Function('acos',argc = 1)
tan = Function('tan',argc = 1)
atan = Function('atan',argc = 1)
atan2 = Function('atan2',argc = 2)
cot = Function('cot',argc = 1)
acot = Function('acot',argc = 1)

sinh = Function('sinh',argc = 1)
cosh = Function('cosh',argc = 1)
asinh = Function('asinh',argc = 1)
acosh = Function('acosh',argc = 1)
tanh = Function('tanh',argc = 1)
atanh = Function('atanh',argc = 1)
coth = Function('coth',argc = 1)
acoth = Function('acoth',argc = 1)

# Type
# ----

Type = Function('Type',argc = 1)
DominantType = BinaryOperator('<>',pysymbols.associative,pysymbols.commutative,0)
OperationType = Function('OperationType',argc = 1)

class Types:
  Boolean = Symbol('Boolean')
  Natural = Symbol('Natural')
  Integer = Symbol('Integer')
  Rational = Symbol('Rational')
  Real = Symbol('Real')
  Complex = Symbol('Complex')
  Imaginary = Symbol('Imaginary')
  Type = Symbol('Type')


# Custom Function
# ---------------

CustomFunction = Function("CustomFunction")

def custom_function(name,argc = None,return_type = None,**kwargs):
    
    class CustomFunctionData(object):
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs)

    func_obj = pysymbols.create_object(CustomFunctionData(name=name,**kwargs),name)
    
    if argc is not None and not isinstance(argc,(tuple,list)):
        argc = [argc]
        
    class CustomFunctionDelegate(object):
        
        def __init__(self,func_obj,name,argc):
            self.func_obj = func_obj
            self.name = name
            self.argc = argc
        
        @property
        def __repr__(self):
            return name
            
        def __call__(self,*args):
            if self.argc != None and len(args) not in self.argc:
                raise ValueError('%s takes %s args' % (self.name,' or '.join([str(s) for s in self.argc])))
            args = [self.func_obj] + list(args)
            return CustomFunction(*args)

    res = CustomFunctionDelegate(func_obj,name,argc)

    if return_type != None:
        if not argc:
            raise ValueError('argc needs to be defined to register result type')
        from evaluate import type_evaluator
        for c in argc:
            type_evaluator.add_rule(Type(res(*[Wildcard(str(s)) for s in range(c)])),return_type)

    return res




