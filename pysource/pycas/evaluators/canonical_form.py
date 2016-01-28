
import pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True,split_binary=False)
canonical_form = evaluator


def normalize_exponentiation(m):
    f = m[s.x]
    if f.function != pc.Multiplication:
        return False

    e = m[s.y]

    inner_exponents = [arg.args[0]**(arg.args[1]*e) for arg in f.args if arg.function == pc.Exponentiation]

    if len(inner_exponents) == 0:
        return False

    if len(inner_exponents) == len(f.args):
        m[s.z] = pc.Multiplication(*inner_exponents)
    else:
        other_args = [arg for arg in f.args if arg.function != pc.Exponentiation]
        other_expr = pc.Multiplication(*other_args)**e
        m[s.z] = pc.Multiplication(*(inner_exponents + [other_expr]))

canonical_form.add_rule(1 / s.x, s.x ** -1)
canonical_form.add_rule(pc.exp(s.x), pc.e ** s.x)
canonical_form.add_rule(s.x ** s.y, s.z, normalize_exponentiation)

canonical_form.add_rule(pc.exp(s.x),pc.e**s.x)
canonical_form.add_rule(pc.sqrt(s.x),s.x**(pc.S(1)/2))

pp = pc.PiecewisePart
canonical_form.add_rule(pc.Piecewise((s.a,s.b),s.x),pc.Piecewise(pp(s.a,s.b),s.x))
