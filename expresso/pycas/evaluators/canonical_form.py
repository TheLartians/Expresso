


import expresso.pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)
canonical_form = evaluator

def normalize_exponentiation(m):
    f = m[s.x]
    if f.function != pc.multiplication:
        return False

    candidates = [arg for arg in f.args if arg.function == pc.exponentiation]
    candidates += [arg for arg in f.args if isinstance(arg.value, pc.Number) or arg == pc.I]

    if len(candidates) == 0:
        return False

    e = m[s.y]
    exponents = [(arg.args[0]**arg.args[1])**e for arg in f.args if arg.function == pc.exponentiation]
    exponents += [arg ** e for arg in f.args if isinstance(arg.value, pc.Number) or arg == pc.I]

    if len(candidates) != len(f.args):
        exponents += [pc.multiplication(*[arg for arg in f.args if arg not in candidates])**e]

    m[s.z] = pc.multiplication(*exponents)

canonical_form.add_rule(pc.multiplication(s.x),s.x)
canonical_form.add_rule(pc.addition(s.x),s.x)
canonical_form.add_rule(pc.exponentiation(s.x),s.x)

canonical_form.add_rule(1/s.x, s.x**-1)
canonical_form.add_rule(pc.exp(s.x), pc.e**s.x)
canonical_form.add_rule(s.x**s.y, s.z, normalize_exponentiation,condition=pc.equal(pc.DominantType(pc.Type(s.y),pc.Types.Real),pc.Types.Real))

canonical_form.add_rule(pc.exp(s.x),pc.e**s.x)
canonical_form.add_rule(pc.sqrt(s.x),s.x**(1/pc.S(2)))


canonical_form.add_rule(s.x>s.y,s.y<s.x)
canonical_form.add_rule(s.x>=s.y,s.y<=s.x)
canonical_form.add_rule(s.x<=s.y,pc.Or(s.x<s.y,pc.equal(s.x,s.y)))
canonical_form.add_rule(pc.unequal(s.x,s.y),pc.Not(pc.equal(s.x,s.y)));


canonical_form.add_rule(abs(s.x),pc.Max(s.x,-s.x),condition=pc.equal(pc.DominantType(pc.Type(s.x),pc.Types.Real),pc.Types.Real))

canonical_form.add_rule(pc.Max(s.a,s.b),-pc.Min(-s.a,-s.b))

canonical_form.add_rule(pc.tan(s.x),pc.sin(s.x)/pc.cos(s.x))
canonical_form.add_rule(pc.cot(s.x),pc.cos(s.x)/pc.sin(s.x))


#canonical_form.add_rule(abs(s.x)<s.y,pc.And(s.x<s.y,-s.x<s.y),condition=pc.And(s.y>0,pc.equal(pc.DominantType(pc.Type(s.x),pc.Types.Real),pc.Types.Real)))

from .type_evaluator import issubtype

format_evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)
format_evaluator.add_rule(s.x**-1,1/s.x)
format_evaluator.add_rule(s.x ** -s.y, 1 / s.x ** s.y, lambda m:isinstance(m[s.y].value, pc.Number))
format_evaluator.add_rule(pc.e**s.x,pc.exp(s.x))
format_evaluator.add_rule(s.x**(1/pc.S(2)),pc.sqrt(s.x))


format_evaluator.add_rule(pc.sqrt(s.x)*pc.sqrt(s.y),pc.sqrt(s.x*s.y),condition=pc.Or(s.x>0,s.y>0))
format_evaluator.add_rule(pc.sqrt(s.x)/pc.sqrt(s.y),pc.sqrt(s.x/s.y),condition=pc.Or(s.x>0,s.y>0))


format_evaluator.add_rule(s.x**s.a/s.y**s.a,(s.x/s.y)**s.a,condition=pc.Or(s.x>0,s.y>0,issubtype(s.a,pc.Types.Integer)))
format_evaluator.add_rule(s.x**s.a*s.y**s.a,(s.x*s.y)**s.a,condition=pc.Or(s.x>0,s.y>0,issubtype(s.a,pc.Types.Integer)))
format_evaluator.add_rule(s.x**s.a*s.y**-s.a,(s.x/s.y)**s.a,condition=pc.Or(s.x>0,s.y>0,issubtype(s.a,pc.Types.Integer)))

format_evaluator.add_rule(pc.Min(s.a,-s.a),-abs(s.a))
format_evaluator.add_rule(pc.Min(-s.a,s.a),-abs(s.a))
format_evaluator.add_rule(pc.Min(s.a,s.b),-abs(s.a),condition=pc.equal(-s.b,s.a))
format_evaluator.add_rule(-(-s.a),s.a)

format_evaluator.add_rule(pc.equal(s.a,s.a),True)
format_evaluator.add_rule(-(s.a+s.b),-s.a-s.b)

format_evaluator.add_rule(pc.sin(s.x)/pc.cos(s.x),pc.tan(s.x))
format_evaluator.add_rule(pc.cos(s.x)/pc.sin(s.x),pc.cot(s.x))

format_evaluator.add_rule((-s.a)*s.b,-(s.a*s.b))
format_evaluator.add_rule(pc.Not(pc.equal(s.x,s.y)),pc.unequal(s.x,s.y));

format_evaluator.add_rule(pc.Or(s.x<s.y,pc.equal(s.x,s.y)),s.x<=s.y)

format_evaluator.add_rule((s.x + pc.conjugate(s.x)),2*pc.real(s.x))
format_evaluator.add_rule((s.x - pc.conjugate(s.x)),2j*pc.imag(s.x))
format_evaluator.add_rule(s.x/s.x,1)
format_evaluator.add_rule(1*s.x,s.x)
format_evaluator.add_rule(s.x/1,s.x)
format_evaluator.add_rule(-1*s.x,-s.x)
format_evaluator.add_rule(-1*-1,1)
format_evaluator.add_rule(pc.I*pc.I,-1)




