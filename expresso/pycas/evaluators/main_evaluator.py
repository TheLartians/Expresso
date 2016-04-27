
import expresso.pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True, split_binary=True)
factor_evaluator = pc.RewriteEvaluator(recursive=True, split_binary=True)

evaluator.add_rule(s.x+0, s.x)
evaluator.add_rule(s.x-s.x, 0)
evaluator.add_rule(s.x+s.x, 2*s.x)

evaluator.add_rule(s.x*1, s.x)
evaluator.add_rule(s.x*0, 0)

evaluator.add_rule(s.x**1, s.x)
evaluator.add_rule(s.x**0, 1)
evaluator.add_rule(1**s.x, 1)
evaluator.add_rule(0**s.x, 0,condition=s.x>0)

factor_evaluator.add_rule(s.x*s.x, s.x**2)
evaluator.add_rule(s.x*s.x**-1, 1)

evaluator.add_rule((s.x**s.a)**s.b, s.x**(s.a*s.b),condition=pc.equal(pc.DominantType(pc.Type(s.b),pc.Types.Integer),pc.Types.Integer))

from .numeric_evaluator import is_even,is_uneven
from .type_evaluator import issubtype

evaluator.add_rule((s.x**s.a)**(s.a**-1), abs(s.x),condition=pc.And( issubtype(s.x,pc.Types.Real),is_even(s.a) ) )
evaluator.add_rule((s.x**s.a)**(s.a**-1), s.x,condition=pc.And( issubtype(s.x,pc.Types.Real),is_uneven(s.a)))


evaluator.add_rule((-s.x)**(s.a), s.x**s.a ,condition=is_even(s.a))
evaluator.add_rule((-s.x)**(s.a), -(s.x)**s.a ,condition=is_uneven(s.a))



factor_evaluator.add_rule(s.x**s.n*s.y**-s.n,(s.x*s.y**-1)**s.n,condition=pc.Or(s.y>0,issubtype(s.n,pc.Types.Integer)))


factor_evaluator.add_rule(s.x**s.a/s.y**s.a,(s.x/s.y)**s.a,condition=pc.Or(s.x>0,s.y>0,issubtype(s.a,pc.Types.Integer)))
factor_evaluator.add_rule(s.x**s.a*s.y**s.a,(s.x*s.y)**s.a,condition=pc.Or(s.x>0,s.y>0,issubtype(s.a,pc.Types.Integer)))
factor_evaluator.add_rule(s.x**s.a*s.y**-s.a,(s.x/s.y)**s.a,condition=pc.Or(s.x>0,s.y>0,issubtype(s.a,pc.Types.Integer)))


factor_evaluator.add_rule(s.x**s.n*s.x, s.x**(s.n+1))
factor_evaluator.add_rule(s.x**s.n*s.x**s.m, s.x**(s.n+s.m))


from logic_evaluator import is_explicit_natural

evaluator.add_rule(s.a**s.x*s.b**s.x, (s.a*s.b)**(s.x), condition=pc.And(pc.Not(pc.Or(is_explicit_natural(s.a),is_explicit_natural(s.b))),pc.Or(issubtype(s.x,pc.Types.Natural),s.a>0,s.b>0)))

evaluator.add_rule(-(s.x+s.y), -s.x-s.y)
evaluator.add_rule(s.x*-1, -s.x)
evaluator.add_rule(-(-s.x), s.x)
evaluator.add_rule((-s.x)*s.y, -(s.x*s.y))
evaluator.add_rule(1 / -s.x, -(1 / s.x))

evaluator.add_rule(-pc.S(0), 0)

def extract_intersection(m):

    ma = pc.MulplicityList(m[s.x],pc.multiplication_group,pc.exponentiation,pc.real_field)
    mb = pc.MulplicityList(m[s.y],pc.multiplication_group,pc.exponentiation,pc.real_field)

    common = ma.intersection(mb)
    if len(common) == 0:
        return False

    m[s.a] = (ma-common).as_expression()
    m[s.b] = (mb-common).as_expression()
    m[s.c] = common.as_expression()



factor_evaluator.add_rule(s.x+s.y, s.c*(s.a+s.b), extract_intersection)
factor_evaluator.add_rule(s.x-s.y, s.c*(s.a-s.b), extract_intersection)
factor_evaluator.add_rule(-s.x-s.y, -s.c*(s.a+s.b), extract_intersection)

def extract_comp_mul_intersection(m):

    ma = pc.MulplicityList(m[s.x],pc.multiplication_group,pc.exponentiation,pc.real_field)
    mb = pc.MulplicityList(m[s.y],pc.multiplication_group,pc.exponentiation,pc.real_field)

    common = ma.intersection(mb)
    if len(common) == 0:
        return False

    for v,mul in common:
        if v.function == pc.sign:
            return False

    m[s.a] = (ma-common).as_expression()
    m[s.b] = (mb-common).as_expression()
    m[s.c] = common.as_expression()

evaluator.add_rule(-s.x<-s.y, s.y<s.x)


evaluator.add_rule(s.x<s.y, pc.sign(s.c)*s.a<pc.sign(s.c)*s.b, extract_comp_mul_intersection)
evaluator.add_rule(s.x<-s.y, pc.sign(s.c)*s.a<-pc.sign(s.c)*s.b, extract_comp_mul_intersection)
evaluator.add_rule(-s.x<s.y, -pc.sign(s.c)*s.a<pc.sign(s.c)*s.b, extract_comp_mul_intersection)


evaluator.add_rule(pc.sign(s.a),1,condition=s.a>=0)
evaluator.add_rule(pc.sign(s.a),-1,condition=s.a<0)
evaluator.add_rule(pc.sign(s.a*s.b),pc.sign(s.a)*pc.sign(s.b))

evaluator.add_rule(pc.sign(s.a**s.n),1,condition=is_even(s.n))
evaluator.add_rule(pc.sign(s.a**s.n),pc.sign(s.a),condition=is_uneven(s.n))




from .logic_evaluator import is_function_type

def evaluate_fraction(m):
    ex,ey = m[s.x],m[s.y]**m[s.n]
    ma = pc.MulplicityList(ex,pc.multiplication_group,pc.exponentiation,pc.real_field)
    mb = pc.MulplicityList(ey,pc.multiplication_group,pc.exponentiation,pc.real_field)

    mbs = {k for k,v in mb}

    valid = False

    if not valid:
        for k,v in ma:
            if k in mbs:
                valid = True
                break
            if valid:
                break

    if valid == False:
        return False

    m[s.c] = (ma+mb).as_expression()

evaluator.add_rule(s.x*s.y**s.n, s.c, evaluate_fraction,condition=issubtype(s.n,pc.Types.Integer))


evaluator.add_rule(pc.log(pc.e), 1)
evaluator.add_rule(pc.log(1), 0)
evaluator.add_rule(pc.sin(0), 0)
evaluator.add_rule(pc.cos(0), 1)
evaluator.add_rule(pc.tan(0), 0)
evaluator.add_rule(pc.atan(0), 0)

evaluator.add_rule(pc.Indicator(True), 1)
evaluator.add_rule(pc.Indicator(False), 0)


evaluator.add_rule(pc.InnerPiecewise((s.a,True),s.x),(s.a,True))
evaluator.add_rule(pc.InnerPiecewise((s.a,False),s.x),s.x)
evaluator.add_rule(pc.InnerPiecewise(s.x,(s.a,False)),s.x)
evaluator.add_rule(pc.InnerPiecewise((s.a,s.x),(s.b,s.x)),(s.a,s.x))
evaluator.add_rule(pc.InnerPiecewise((s.x,s.a),(s.x,s.b)),(s.x,pc.Or(s.a,s.b)))
evaluator.add_rule(pc.InnerPiecewise(pc.InnerPiecewise(s.x)),pc.InnerPiecewise(s.x))

evaluator.add_rule(pc.OuterPiecewise((s.a,s.b)),s.a*pc.Indicator(s.b))


from .logic_evaluator import contains_atomic


excluded_derivatives = {pc.InnerPiecewise,pc.Tuple}

def check_if_excluded_derivative(m):
    return not m[s.y].function in excluded_derivatives

evaluator.add_rule(pc.derivative(s.x,s.x),1)
evaluator.add_rule(pc.derivative(s.y,s.x),0,check_if_excluded_derivative,condition=pc.Not(contains_atomic(s.y,s.x)));

evaluator.add_rule(pc.derivative(s.a+s.b,s.x),pc.derivative(s.a,s.x)+pc.derivative(s.b,s.x))
evaluator.add_rule(pc.derivative(s.a*s.b,s.x),pc.derivative(s.a,s.x)*s.b+pc.derivative(s.b,s.x)*s.a)
evaluator.add_rule(pc.derivative(-s.x,s.x),-1)
evaluator.add_rule(pc.derivative(1/s.x,s.x),-s.x**-2)
evaluator.add_rule(pc.derivative(pc.log(s.x),s.x),1/s.x)


evaluator.add_rule(pc.derivative(pc.sin(s.x),s.x),pc.cos(s.x))
evaluator.add_rule(pc.derivative(pc.cos(s.x),s.x),-pc.sin(s.x))

# TODO: Add assumptions to expressions: the following is only valid if x != -1,1 or x != -i,i
evaluator.add_rule(pc.derivative(pc.asin(s.x),s.x),1/pc.sqrt(1-s.x**2))
evaluator.add_rule(pc.derivative(pc.acos(s.x),s.x),-1/pc.sqrt(1-s.x**2))
evaluator.add_rule(pc.derivative(pc.atan(s.x),s.x),1/(1+s.x**2))
evaluator.add_rule(pc.derivative(pc.acot(s.x),s.x),-1/(1+s.x**2))



evaluator.add_rule(pc.derivative(s.x**s.n,s.x),s.n*s.x**(s.n-1),condition=(pc.equal(pc.Type(s.n))));
evaluator.add_rule(pc.derivative(s.a**s.b,s.x),pc.derivative(s.b*pc.log(s.a),s.x)*s.a**s.b);

evaluator.add_rule(pc.derivative(pc.OuterPiecewise(s.a),s.x),pc.OuterPiecewise(pc.derivative(s.a,s.x)))
evaluator.add_rule(pc.derivative(pc.InnerPiecewise((s.a,s.b),s.c),s.x),pc.InnerPiecewise((pc.derivative(s.a,s.x),s.b),pc.derivative(pc.InnerPiecewise(s.c),s.x)),condition=pc.Not(contains_atomic(s.b,s.x)))
evaluator.add_rule(pc.derivative(pc.InnerPiecewise((s.a,s.b)),s.x),(pc.derivative(s.a,s.x),s.b),condition=pc.Not(contains_atomic(s.b,s.x)))


evaluator.add_rule(pc.OuterPiecewise(s.a)**s.x,pc.OuterPiecewise(s.a**s.x))
evaluator.add_rule(pc.InnerPiecewise((s.a,s.b),s.c)**s.x,pc.InnerPiecewise((s.a**s.x,s.b),pc.InnerPiecewise(s.c)**s.x))
evaluator.add_rule(pc.InnerPiecewise((s.a,s.b))**s.x,(s.a**s.x,s.b))

evaluator.add_rule(pc.real(s.x),(s.x + pc.conjugate(s.x))/2)
evaluator.add_rule(pc.imag(s.x),(s.x - pc.conjugate(s.x))/2j)

evaluator.add_rule(pc.conjugate(s.x),s.x,condition=issubtype(s.x,pc.Types.Real))
evaluator.add_rule(pc.conjugate(s.x),-s.x,condition=issubtype(s.x,pc.Types.Imaginary))
evaluator.add_rule(pc.conjugate(s.x+s.y),pc.conjugate(s.x)+pc.conjugate(s.y))
evaluator.add_rule(pc.conjugate(s.x*s.y),pc.conjugate(s.x)*pc.conjugate(s.y))
evaluator.add_rule(pc.conjugate(-s.x),-pc.conjugate(s.x))
evaluator.add_rule(pc.conjugate(1/s.x),1/pc.conjugate(s.x))
evaluator.add_rule(pc.conjugate(s.x**-1),pc.conjugate(s.x)**-1)


def create_tmp_x(m):
    m[c] = pc.tmp(s.x)

evaluator.add_rule( pc.derivative(s.f(s.g(s.x)),s.x) ,
                    pc.evaluated_at( pc.derivative(s.f(pc.tmp(s.x)),pc.tmp(s.x)), pc.tmp(s.x), s.g(s.x) ) * pc.derivative(s.g(s.x),s.x));

evaluator.add_rule(pc.evaluated_at( s.f(s.x), s.x, s.c ), s.f(s.c), condition = pc.Not(is_function_type(s.f(s.x),pc.derivative)) )

evaluator.add_rule(pc.Min(s.a,s.b),s.a,condition=s.a<=s.b)
evaluator.add_rule(pc.Min(s.a,s.b),s.b,condition=s.b<=s.a)


from .canonical_form import canonical_form
from .logic_evaluator import logic_evaluator
from .numeric_evaluator import numeric_evaluator
from .type_evaluator import type_evaluator


main_evaluator = pc.MultiEvaluator(recursive = True, split_binary=True)
main_evaluator.add_evaluator(canonical_form)
main_evaluator.add_evaluator(numeric_evaluator)
main_evaluator.add_evaluator(evaluator)
main_evaluator.add_evaluator(factor_evaluator)
main_evaluator.add_evaluator(type_evaluator)
main_evaluator.add_evaluator(logic_evaluator)











