from .expression import Function,BinaryOperator,UnaryOperator,Symbol,Wildcard,Number,Expression,S
from .expression import One,Zero,NaN,I,addition,negative,multiplication,fraction,exponentiation,addition_group,multiplication_group,real_field,complex_field,Or,And,Xor,Not,mod,equal,unequal,In,NotIn,Less,LessEqual,Greater,GreaterEqual,Abs,Tuple

import expresso
import math

# Symbols
# -------

class SymbolicConstant(object):
    def __init__(self,name):
        self.name = name

def create_symbolic_constant(name):
    return S(expresso.create_object(SymbolicConstant(name),'symbolic constant %s' % name))

e = create_symbolic_constant('e')
pi = create_symbolic_constant('pi')
oo = create_symbolic_constant('infinity')

# Various Functions
# -----------------

factorial = UnaryOperator('!',expresso.postfix,0)

sign = Function('sign',argc = 1)
floor = Function('floor',argc = 1)
ceil = Function('ceil',argc = 1)
round = Function('round',argc = 1)

Max = Function('max',argc = 2)
Min = Function('min',argc = 2)

real = Function('real',argc = 1)
imag = Function('imag',argc = 1)
conjugate = Function('conjugate',argc = 1)

Indicator = Function('indicator',argc = 1)

InnerPiecewise = BinaryOperator('}{', expresso.associative, expresso.non_commutative, -14)
OuterPiecewise = Function('outer piecewise')

def piecewise(*args):
    return OuterPiecewise(InnerPiecewise(*args))

# Calculus
# --------

derivative = Function('derivative',argc = 2)
evaluated_at = Function('evaluated_at',argc = 3)
tmp = UnaryOperator('tmp_',expresso.prefix,0)

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
DominantType = BinaryOperator('<>',expresso.left_associative,expresso.commutative,0)
OperationType = Function('OperationType',argc = 1)


class TypeInfo(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return repr('pyCAS.TypeInfo<%s>' % self.__dict__)


inverse_python_types = {}

def create_type(name,**kwargs):
    res = S(expresso.create_object(TypeInfo(name=name,**kwargs),'pyCAS type ' + name))
    if 'python_type' in kwargs:
        inverse_python_types[kwargs['python_type']] = res
    return res

class Types:
  Boolean = create_type('Boolean',python_type=bool,c_type='bool')
  Natural = create_type('Natural',python_type=long,c_type='unsigned')
  Integer = create_type('Integer',python_type=long,c_type='long')
  Rational = create_type('Rational',python_type=float,c_type='double')
  Real = create_type('Real',python_type=float,c_type='double')
  Complex = create_type('Complex',python_type=complex,c_type='c_complex')
  Imaginary = create_type('Imaginary',python_type=complex,c_type='c_complex')
  Unit = create_type('Unit')
  Type = create_type('Type')


def is_in_type(expression,type):
    return equal(DominantType(Type(expression),type),type)

# Custom Function
# ---------------

CustomFunction = Function("CustomFunction")

def custom_function(name,argc = None,return_type = None,**kwargs):
    
    class CustomFunctionData(object):
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs)

    func_obj = expresso.create_object(CustomFunctionData(name=name,**kwargs),name)
    
    if argc is not None and not isinstance(argc,(tuple,list)):
        argc = [argc]
        
    class CustomFunctionDelegate(object):
        
        def __init__(self,func_obj,name,argc):
            self.func_obj = func_obj
            self.name = name
            self.argc = argc
        
        def __repr__(self):
            return name
            
        def __call__(self,*args):
            if self.argc != None and len(args) not in self.argc:
                raise ValueError('%s takes %s arguments' % (self.name,' or '.join([str(s) for s in self.argc])))
            args = [self.func_obj] + list(args)
            return CustomFunction(*args)

    res = CustomFunctionDelegate(func_obj,name,argc)

    if return_type != None:
        if not argc:
            raise ValueError('argc needs to be defined to register result type')

        # TODO: register in local context, not global evaluators
        from evaluators.type_evaluator import evaluator
        for c in argc:
            evaluator.add_rule(Type(res(*[Wildcard(str(s)) for s in range(c)])),return_type)

    return res


ArrayAccess = Function("ArrayAccess")


class type_converters:

    import numpy as np

    numpy_c_typenames = {
        np.bool.__name__:'bool',
        np.int.__name__:'int',
        np.int8.__name__:'int8_t',
        np.int16.__name__:'int16_t',
        np.int32.__name__:'int32_t',
        np.int64.__name__:'int64_t',
        np.float32.__name__:'float',
        np.float.__name__:'double',
        np.float64.__name__:'double',
        np.complex:'complex<double>',
        np.complex128.__name__:'complex<double>'
    }


def array(name,inarray):
    import numpy as np

    if not inarray.dtype.name in type_converters.numpy_c_typenames:
        raise ValueError('Array dtype must be one of the following: %s' % ','.join(type_converters.numpy_c_typenames.keys()))

    array = inarray #np.ascontiguousarray(inarray,dtype=cast_type)
    pointer = array.ctypes.data

    class ArrayAccessDelegate(object):

        def __init__(self,name,array):
            self.name = name
            self.array_obj = expresso.create_object(array,'%s__id_%r' % (name,id(array)))
            self.argc = len(array.shape)

        def __repr__(self):
            return self.name

        def __call__(self,*args):
            if len(args) != self.argc:
                raise ValueError('%s takes %s arguments' % (self.name,self.argc))
            args = [self.array_obj] + list(args)
            return ArrayAccess(*args)

    for t,v in inverse_python_types.iteritems():
        if isinstance(inarray.dtype,t):
            # TODO: register in local context, not global evaluators
            from evaluators.type_evaluator import evaluator
            evaluator.add_rule(ArrayAccessDelegate([Wildcard(str(i)) for i in range(array.shape)]),v)
            break

    return ArrayAccessDelegate(name,array)

unfoldable = Function('unfoldable',argc=1)

def parameter(name,value):
    import numpy as np
    value = np.array([value])

    def set_value(new_value):
        value[0] = new_value
    def get_value():
        return value[0]

    svalue = array(name,value)(unfoldable(0))

    from .expression import latex,printer
    @latex.register_target(svalue)
    @printer.register_target(svalue)
    def print_value(printer,expr):
        return printer.format_name(name)

    svalue.set_value = set_value
    svalue.get_value = get_value

    return svalue



















