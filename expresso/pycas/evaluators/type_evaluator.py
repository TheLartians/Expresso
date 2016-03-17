
import expresso.pycas as pc
import rule_symbols as s

ordered_types = (pc.Types.Boolean,pc.Types.Natural,pc.Types.Integer,pc.Types.Rational,pc.Types.Real,pc.Types.Complex)

evaluator = pc.RewriteEvaluator(recursive=True, split_binary=True)

from .logic_evaluator import is_explicit_natural,is_function_type

evaluator.add_rule(pc.DominantType(s.x),s.x)

for i in range(len(ordered_types)):
    evaluator.add_rule(pc.DominantType(pc.Types.Imaginary,ordered_types[i]),pc.Types.Complex)
    evaluator.add_rule(pc.Type(ordered_types[i]),pc.Types.Type)
    for j in range(i):
      evaluator.add_rule(pc.DominantType(ordered_types[j],ordered_types[i]),ordered_types[i])

def eval_type_equality(m):
    tx = m[s.x]
    ty = m[s.y]
    if tx in ordered_types and ty in ordered_types:
        m[s.z] = tx == ty
        return True
    return False

evaluator.add_rule(pc.equal(s.x,s.y),False,eval_type_equality)

evaluator.add_rule(pc.Type(pc.Types.Imaginary*pc.Types.Complex),pc.Types.Complex)
evaluator.add_rule(pc.DominantType(s.x,s.x),s.x)

evaluator.add_rule(pc.Type(pc.Types.Imaginary),pc.Types.Type)
evaluator.add_rule(pc.Type(pc.Type(s.x)),pc.Types.Type)

evaluator.add_rule(pc.Type(True),pc.Types.Boolean)
evaluator.add_rule(pc.Type(False),pc.Types.Boolean)
evaluator.add_rule(pc.Type(s.x),pc.Types.Natural,condition=is_explicit_natural(s.x))
evaluator.add_rule(pc.Type(1/s.x),pc.DominantType(pc.Types.Rational,pc.Type(s.x)))

evaluator.add_rule(pc.Type(s.x**s.y),pc.OperationType(pc.Type(s.x)**pc.Type(s.y)))
evaluator.add_rule(pc.Type(s.x*s.y),pc.OperationType(pc.Type(s.x)*pc.Type(s.y)))

for t in (pc.Types.Natural,pc.Types.Integer,pc.Types.Rational,pc.Types.Real):
    evaluator.add_rule(pc.OperationType(pc.Types.Imaginary*t),pc.Types.Imaginary)
evaluator.add_rule(pc.OperationType(pc.Types.Imaginary*pc.Types.Imaginary),pc.Types.Real)

evaluator.add_rule(pc.OperationType(s.x**pc.Types.Natural),s.x)
evaluator.add_rule(pc.OperationType(s.x**pc.Types.Integer),pc.DominantType(s.x,pc.Types.Rational))
evaluator.add_rule(pc.OperationType(pc.Types.Natural**pc.Types.Rational),pc.Types.Real)

evaluator.add_rule(pc.OperationType(s.x**s.y),pc.Types.Complex)


evaluator.add_rule(pc.OperationType(s.x*s.y),pc.DominantType(s.x,s.y),condition=pc.Not(pc.Or(is_function_type(s.x,pc.Type),is_function_type(s.y,pc.Type))))
evaluator.add_rule(pc.OperationType(s.x)**s.y,pc.DominantType(s.x,s.y),condition=pc.Not(pc.Or(is_function_type(s.x,pc.Type),is_function_type(s.y,pc.Type))))


evaluator.add_rule(pc.Type(s.x+s.y),pc.DominantType(pc.Type(s.x),pc.Type(s.y)))
evaluator.add_rule(pc.Type(-s.x),pc.DominantType(pc.Types.Integer,pc.Type(s.x)))
evaluator.add_rule(pc.Type(pc.pi),pc.Types.Real)
evaluator.add_rule(pc.Type(pc.e),pc.Types.Real)
evaluator.add_rule(pc.Type(pc.I),pc.Types.Imaginary)

evaluator.add_rule(pc.Type(pc.factorial(s.x)),pc.Types.Natural)
evaluator.add_rule(pc.Type(pc.sign(s.x)),pc.Types.Integer)
evaluator.add_rule(pc.Type(pc.floor(s.x)),pc.Types.Integer)
evaluator.add_rule(pc.Type(pc.ceil(s.x)),pc.Types.Integer)
evaluator.add_rule(pc.Type(pc.round(s.x)),pc.Types.Integer)
evaluator.add_rule(pc.Type(pc.mod(s.x,s.y)),pc.Types.Integer)

evaluator.add_rule(pc.Type(pc.Abs(s.x)),pc.OperationType(pc.Abs(pc.Type(s.x))))
evaluator.add_rule(pc.OperationType(pc.Abs(pc.Types.Complex)),pc.Types.Real)
evaluator.add_rule(pc.OperationType(pc.Abs(pc.Types.Imaginary)),pc.Types.Real)
evaluator.add_rule(pc.OperationType(pc.Abs(pc.Types.Real)),pc.Types.Real)
evaluator.add_rule(pc.OperationType(pc.Abs(pc.Types.Rational)),pc.Types.Rational)
evaluator.add_rule(pc.OperationType(pc.Abs(pc.Types.Integer)),pc.Types.Natural)
evaluator.add_rule(pc.OperationType(pc.Abs(pc.Types.Natural)),pc.Types.Natural)


evaluator.add_rule(pc.Type(pc.real(s.x)),pc.Types.Real)
evaluator.add_rule(pc.Type(pc.imag(s.x)),pc.Types.Real)
evaluator.add_rule(pc.Type(pc.conjugate(s.x)),pc.Types.Complex)

evaluator.add_rule(pc.Type(pc.Indicator(s.x)),pc.Types.Natural)
evaluator.add_rule(pc.Type(pc.OuterPiecewise(s.x)),pc.Type(s.x))
evaluator.add_rule(pc.Type(pc.InnerPiecewise((s.a,s.b),s.x)),pc.DominantType(pc.Type(s.a),pc.Type(pc.InnerPiecewise(s.x))))
evaluator.add_rule(pc.Type(pc.InnerPiecewise((s.a,s.b))),pc.Type(s.a))


evaluator.add_rule(pc.Type(pc.derivative(s.x,s.y)),pc.Type(s.x))
evaluator.add_rule(pc.Type(pc.evaluated_at(s.x,s.y,s.z)),pc.DominantType(pc.Type(s.x),pc.Type(s.z)))

evaluator.add_rule(pc.Type(pc.tmp(s.x)),pc.Type(s.x))
evaluator.add_rule(pc.Type(pc.sqrt(s.x)),pc.Type(s.x**(1/pc.S(2))))

evaluator.add_rule(pc.Type(pc.atan2(s.x,s.y)),pc.DominantType(pc.Type(s.x),pc.Type(s.x),pc.Types.Rational))

for f in [pc.exp,pc.log,pc.sin,pc.cos,pc.asin ,pc.acos,pc.tan,pc.atan,pc.cot,pc.acot,pc.sinh,pc.cosh,pc.asinh,pc.acosh,pc.tanh,pc.atanh,pc.coth,pc.acoth]:
    evaluator.add_rule(pc.Type(f(s.x)),pc.DominantType(pc.Type(s.x),pc.Types.Rational))

def issubtype(x,t):
    return pc.equal(pc.DominantType(pc.Type(x),t),t)

from .logic_evaluator import is_mpmath
from mpmath import mp

def mpmath_type_evaluator(m):
    v = m[s.x].value
    if isinstance(v,mp.mpf):
        m[s.y] = pc.Types.Real
    elif isinstance(v,mp.mpc):
        m[s.y] = pc.Types.Complex
    else:
        raise AttributeError('unknown mpmath type')

evaluator.add_rule(pc.Type(s.x),s.y,condition=is_mpmath(s.x))


from .canonical_form import canonical_form
from .logic_evaluator import logic_evaluator

type_evaluator = pc.MultiEvaluator(recursive=True, split_binary=True)
type_evaluator.add_evaluator(canonical_form)
type_evaluator.add_evaluator(logic_evaluator)
type_evaluator.add_evaluator(evaluator)




















