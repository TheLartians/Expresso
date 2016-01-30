
import pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True, split_binary=True)

evaluator.add_rule(s.x+0, s.x)
evaluator.add_rule(s.x-s.x, 0)
evaluator.add_rule(s.x+s.x, 2*s.x)

evaluator.add_rule(s.x*1, s.x)
evaluator.add_rule(s.x*0, 0)

evaluator.add_rule(s.x**1, s.x)
evaluator.add_rule(s.x**0, 1)
evaluator.add_rule(1**s.x, 1)
evaluator.add_rule(0**s.x, 0)

evaluator.add_rule(s.x*s.x, s.x**2)
evaluator.add_rule(s.x*s.x**-1, 1)

evaluator.add_rule((s.x**s.a)**s.b, s.x**(s.a*s.b))
evaluator.add_rule(s.x**s.n*s.x, s.x**(s.n+1))
evaluator.add_rule(s.x**s.n*s.x**s.m, s.x**(s.n+s.m))


from logic_evaluator import is_explicit_natural
evaluator.add_rule(s.a**s.x*s.b**s.x, (s.a*s.b)**(s.x), condition=pc.Not(pc.Or(is_explicit_natural(s.a),is_explicit_natural(s.b))))

evaluator.add_rule(-(s.x+s.y), -s.x-s.y)
evaluator.add_rule(s.x*-1, -s.x)
evaluator.add_rule(-(-s.x), s.x)
evaluator.add_rule((-s.x)*s.y, -(s.x*s.y))
evaluator.add_rule(1 / -s.x, -(1 / s.x))
evaluator.add_rule((-s.x)**-1, -((s.x)**-1))
evaluator.add_rule(-pc.S(0), 0)

evaluator.add_rule(abs(s.x), s.x,condition=s.x>=0)
evaluator.add_rule(abs(s.x), -s.x,condition=s.x<0)

def extract_intersection(m):

    ma = pc.MulplicityList(m[s.x],pc.MultiplicationGroup,pc.Exponentiation,pc.RealField)
    mb = pc.MulplicityList(m[s.y],pc.MultiplicationGroup,pc.Exponentiation,pc.RealField)

    common = ma.intersection(mb)
    if len(common) == 0:
        return False
    m[s.a] = (ma-common).as_expression()
    m[s.b] = (mb-common).as_expression()
    m[s.c] = common.as_expression()

evaluator.add_rule(s.x+s.y, s.c*(s.a+s.b), extract_intersection)
evaluator.add_rule(s.x-s.y, s.c*(s.a-s.b), extract_intersection)
evaluator.add_rule(-s.x-s.y, -s.c*(s.a+s.b), extract_intersection)

from .logic_evaluator import is_function_type

def evaluate_fraction(m):
    ex,ey = m[s.x],m[s.y]**m[s.z]
    ma = pc.MulplicityList(ex,pc.MultiplicationGroup,pc.Exponentiation,pc.RealField)
    mb = pc.MulplicityList(ey,pc.MultiplicationGroup,pc.Exponentiation,pc.RealField)

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

evaluator.add_rule(s.x*s.y**s.z, s.c, evaluate_fraction)


evaluator.add_rule(pc.log(pc.e), 1)
evaluator.add_rule(pc.log(1), 0)
evaluator.add_rule(pc.sin(0), 0)
evaluator.add_rule(pc.cos(0), 1)

evaluator.add_rule(pc.Indicator(True), 1)
evaluator.add_rule(pc.Indicator(False), 0)


pp = pc.PiecewisePart
evaluator.add_rule(pc.Piecewise(s.x,(s.a,s.b)),pc.Piecewise(s.x,pp(s.a,s.b)))

evaluator.add_rule(pc.Piecewise(pp(s.a,True),s.x),pp(s.a,True))
evaluator.add_rule(pc.Piecewise(pp(s.a,False),s.x),s.x)
evaluator.add_rule(pc.Piecewise(s.x,pp(s.a,False)),s.x)



from .logic_evaluator import contains_atomic


evaluator.add_rule(pc.derivative(s.x,s.x),1)
evaluator.add_rule(pc.derivative(s.y,s.x),0,condition=pc.Not(contains_atomic(s.y,s.x)));

evaluator.add_rule(pc.derivative(s.a+s.b,s.x),pc.derivative(s.a,s.x)+pc.derivative(s.b,s.x))
evaluator.add_rule(pc.derivative(s.a*s.b,s.x),pc.derivative(s.a,s.x)*s.b+pc.derivative(s.b,s.x)*s.a)
evaluator.add_rule(pc.derivative(-s.x,s.x),-1)
evaluator.add_rule(pc.derivative(1/s.x,s.x),-s.x**-2)
evaluator.add_rule(pc.derivative(pc.log(s.x),s.x),1/s.x)


evaluator.add_rule(pc.derivative(pc.sin(s.x),s.x),pc.cos(s.x))
evaluator.add_rule(pc.derivative(pc.cos(s.x),s.x),-pc.sin(s.x))

evaluator.add_rule(pc.derivative(s.x**s.n,s.x),s.n*s.x**(s.n-1),condition=(pc.Equal(pc.Type(s.n))));
evaluator.add_rule(pc.derivative(s.a**s.b,s.x),pc.derivative(s.b*pc.log(s.a),s.x)*s.a**s.b);

def create_tmp_x(m):
    m[c] = pc.tmp(s.x)

evaluator.add_rule( pc.derivative(s.f(s.g(s.x)),s.x) ,
                    pc.evaluated_at( pc.derivative(s.f(pc.tmp(s.x)),pc.tmp(s.x)), pc.tmp(s.x), s.g(s.x) ) * pc.derivative(s.g(s.x),s.x));

evaluator.add_rule(pc.evaluated_at( s.f(s.x), s.x, s.c ), s.f(s.c), condition = pc.Not(is_function_type(s.f(s.x),pc.derivative)) )





from .canonical_form import canonical_form
from .logic_evaluator import logic_evaluator
from .numeric_evaluator import numeric_evaluator
from .type_evaluator import type_evaluator


main_evaluator = pc.MultiEvaluator(recursive = True, split_binary=True)
main_evaluator.add_evaluator(canonical_form)
main_evaluator.add_evaluator(evaluator)
main_evaluator.add_evaluator(type_evaluator)
main_evaluator.add_evaluator(logic_evaluator)
main_evaluator.add_evaluator(numeric_evaluator)













