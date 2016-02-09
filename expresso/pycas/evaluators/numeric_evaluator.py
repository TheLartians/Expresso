
import expresso.pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True, split_binary=True)

from .logic_evaluator import is_explicit_natural

def unary_rule(f):

    def evaluator(m):
        try:
            res = f(m[s.x].value)
        except:
            return False

        if res is None:
            return False
        else:
            m[s.z] = pc.S(res)

    return evaluator

def binary_rule(f):

    def evaluator(m):
        try:
            res = f(m[s.x].value,m[s.y].value)
        except:
            return False
        if res is None:
            return False
        else:
            m[s.z] = pc.S(res)

    return evaluator

def are_explicit_numbers(*args):
    return pc.And(*[is_explicit_natural(arg) for arg in args])

evaluator.add_rule(s.x + s.y, s.z, binary_rule(lambda x, y: x + y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(-s.x-s.y, -s.z, binary_rule(lambda x, y: x + y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x - s.y, s.z, binary_rule(lambda x, y: x - y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x * s.y, s.z, binary_rule(lambda x, y: x * y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x ** s.a * s.y ** s.a, s.z ** s.a, binary_rule(lambda x, y: x * y), condition=are_explicit_numbers(s.x, s.y))

def exp_length(a,b):
    import math as m
    return m.log10(a)*b

evaluator.add_rule(s.x ** s.y, s.z, binary_rule(lambda x, y: x ** y if exp_length(x,y)<100 else None), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x ** -s.y, s.z ** -1, binary_rule(lambda x, y: x ** y if exp_length(x,y)<100 else None), condition=are_explicit_numbers(s.x, s.y))

evaluator.add_rule(pc.mod(s.x,s.y), s.z, binary_rule(lambda x, y: x % y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(pc.mod(s.x,-s.y), s.z, binary_rule(lambda x, y: x % (-y)), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(pc.mod(-s.x,s.y), s.z, binary_rule(lambda x, y: (-x) % (-y)), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(pc.equal(s.x,s.y), s.z, binary_rule(lambda x, y: x == y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(pc.equal(s.x,-s.y), False, condition=are_explicit_numbers(s.x, s.y))

evaluator.add_rule(s.x < s.y, s.z, binary_rule(lambda x, y: x < y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x < -s.y, s.z, binary_rule(lambda x, y: x < -y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(-s.x < s.y, s.z, binary_rule(lambda x, y: -x < y), condition=are_explicit_numbers(s.x, s.y))


#evaluator.add_rule(s.a**s.x*s.b**-s.x, (s.a*s.b**-1)**(s.x),condition=are_explicit_numbers(s.a, s.b))

def is_even(x):
    return pc.equal(pc.mod(x,2),0)

from type_evaluator import issubtype

def is_uneven(x):
    return pc.And(issubtype(x,pc.Types.Integer),pc.Not(is_even(x)))


evaluator.add_rule((-s.x)**s.y, s.x**s.y,condition=is_even(s.y))
evaluator.add_rule((-s.x)**s.y, -(s.x**s.y),condition=is_uneven(s.y))

evaluator.add_rule(s.x<s.y*s.z**-1,s.z*s.x<s.y ,condition=are_explicit_numbers(s.y,s.z))
evaluator.add_rule(s.x*s.z**-1<s.y,s.x<s.y*s.z ,condition=are_explicit_numbers(s.x,s.z))
evaluator.add_rule(s.x<-(s.y*s.z**-1),s.z*s.x<-s.y ,condition=are_explicit_numbers(s.y,s.z))
evaluator.add_rule(-(s.x*s.z**-1)<s.y,-s.x<s.y*s.z ,condition=are_explicit_numbers(s.x,s.z))

evaluator.add_rule(s.x<s.z**-1,s.z*s.x<1 ,condition=is_explicit_natural(s.z))
evaluator.add_rule(s.z**-1<s.y,1<s.y*s.z ,condition=is_explicit_natural(s.z))
evaluator.add_rule(s.x<-(s.z**-1),s.z*s.x<-1 ,condition=is_explicit_natural(s.z))
evaluator.add_rule(-(s.z**-1)<s.y,-1<s.y*s.z ,condition=is_explicit_natural(s.z))


evaluator.add_rule(pc.equal(s.x,s.y*s.z**-1),pc.equal(s.z*s.x,s.y),condition=are_explicit_numbers(s.y,s.z))
evaluator.add_rule(pc.equal(s.x,-(s.y*s.z**-1)),pc.equal(s.z*s.x,-s.y),condition=are_explicit_numbers(s.y,s.z))

evaluator.add_rule(pc.equal(s.x,-(s.y*s.z**-1)),pc.equal(s.z*s.x,-s.y),condition=are_explicit_numbers(s.y,s.z))


from .logic_evaluator import is_pure_numeric,is_numeric

evaluator.add_rule(s.a+s.b*s.c**-1,(s.a*s.c+s.b)*s.c**-1,condition=pc.And(is_pure_numeric(s.a),are_explicit_numbers(s.b,s.c)))
evaluator.add_rule(s.a-s.b*s.c**-1,(s.a*s.c-s.b)*s.c**-1,condition=pc.And(is_pure_numeric(s.a),are_explicit_numbers(s.b,s.c)))

evaluator.add_rule(s.a+s.c**-1,(s.a*s.c+1)*s.c**-1,condition=pc.And(is_pure_numeric(s.a),is_explicit_natural(s.c)))
evaluator.add_rule(s.a-s.c**-1,(s.a*s.c-1)*s.c**-1,condition=pc.And(is_pure_numeric(s.a),is_explicit_natural(s.c)))


evaluator.add_rule((s.a+s.b)**2,(s.a**2+2*s.a*s.b+s.b**2),condition=pc.And(is_numeric(s.a),is_numeric(s.b)))
evaluator.add_rule(s.a*(s.b+s.c),s.a*s.b+s.a*s.c,condition=pc.And(is_pure_numeric(s.a),is_numeric(s.b),is_numeric(s.c)))

evaluator.add_rule(pc.I**s.n,(-1)**(s.n/2),condition=is_even(s.n))
evaluator.add_rule(pc.I**s.n,pc.I*(-1)**(s.n/2-0.5),condition=is_uneven(s.n))

evaluator.add_rule(pc.sign(-s.x),-1,condition=is_explicit_natural(s.x))
evaluator.add_rule(pc.sign(s.x),1,condition=is_explicit_natural(s.x))

import fractions

def evaluate_fraction(m):
    vx = m[s.x].value
    vy = m[s.y].value
    res = fractions.Fraction(vx,vy)
    if (res.numerator,res.denominator) == (vx,vy):
        return False
    m[s.a] = res.numerator
    m[s.b] = res.denominator

evaluator.add_rule(s.x*s.y**-1,s.a*s.b**-1,evaluate_fraction,condition=are_explicit_numbers(s.x, s.y))

evaluator.add_rule(s.x**s.c*s.y**-s.c,s.a**s.c*s.b**-s.c,evaluate_fraction,condition=are_explicit_numbers(s.x, s.y))



from .canonical_form import canonical_form
from .logic_evaluator import logic_evaluator

numeric_evaluator = pc.MultiEvaluator(recursive=True, split_binary=True)
numeric_evaluator.add_evaluator(canonical_form)
numeric_evaluator.add_evaluator(logic_evaluator)
numeric_evaluator.add_evaluator(evaluator)





