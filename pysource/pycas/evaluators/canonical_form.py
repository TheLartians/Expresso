
pc = __import__(__name__.split('.')[0])
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)
canonical_form = evaluator

def normalize_exponentiation(m):
    f = m[s.x]
    if f.function != pc.Multiplication:
        return False

    candidates = [arg for arg in f.args if arg.function == pc.Exponentiation]
    candidates += [arg for arg in f.args if isinstance(arg.value, pc.Number) or arg == pc.I]

    if len(candidates) == 0:
        return False

    e = m[s.y]
    exponents = [(arg.args[0]**arg.args[1])**e for arg in f.args if arg.function == pc.Exponentiation]
    exponents += [arg ** e for arg in f.args if isinstance(arg.value, pc.Number) or arg == pc.I]

    if len(candidates) != len(f.args):
        exponents += [pc.Multiplication(*[arg for arg in f.args if arg not in candidates])**e]

    m[s.z] = pc.Multiplication(*exponents)

canonical_form.add_rule(pc.Multiplication(s.x),s.x)
canonical_form.add_rule(pc.Addition(s.x),s.x)
canonical_form.add_rule(pc.Exponentiation(s.x),s.x)

canonical_form.add_rule(1/s.x, s.x**-1)
canonical_form.add_rule(pc.exp(s.x), pc.e**s.x)
canonical_form.add_rule(s.x**s.y, s.z, normalize_exponentiation,condition=pc.Equal(pc.DominantType(pc.Type(s.y),pc.Types.Real),pc.Types.Real))

canonical_form.add_rule(pc.exp(s.x),pc.e**s.x)
canonical_form.add_rule(pc.sqrt(s.x),s.x**(pc.S(1)/2))


canonical_form.add_rule(s.x>s.y,s.y<s.x)
canonical_form.add_rule(s.x>=s.y,s.y<=s.x)
canonical_form.add_rule(s.x<=s.y,pc.Or(s.x<s.y,pc.Equal(s.x,s.y)))


canonical_form.add_rule(abs(s.x),pc.Max(s.x,-s.x),condition=pc.Equal(pc.DominantType(pc.Type(s.x),pc.Types.Real),pc.Types.Real))


canonical_form.add_rule(pc.Max(s.a,s.b),-pc.Min(-s.a,-s.b))


#canonical_form.add_rule(abs(s.x)<s.y,pc.And(s.x<s.y,-s.x<s.y),condition=pc.And(s.y>0,pc.Equal(pc.DominantType(pc.Type(s.x),pc.Types.Real),pc.Types.Real)))

pp = pc.PiecewisePart
canonical_form.add_rule(pc.InnerPiecewise((s.a,s.b),(s.x,s.y)),pc.InnerPiecewise(pp(s.a,s.b),pp(s.x,s.y)))
canonical_form.add_rule(pc.InnerPiecewise((s.a,s.b),s.x),pc.InnerPiecewise(pp(s.a,s.b),s.x))
canonical_form.add_rule(pc.InnerPiecewise(s.x,(s.a,s.b)),pc.InnerPiecewise(s.x,pp(s.a,s.b)))




format_evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)
format_evaluator.add_rule(s.x**-1,1/s.x)
format_evaluator.add_rule(s.x ** -s.y, 1 / s.x ** s.y, lambda m:isinstance(m[s.y].value, pc.Number))
format_evaluator.add_rule(pc.e**s.x,pc.exp(s.x))
format_evaluator.add_rule(s.x**(1/pc.S(2)),pc.sqrt(s.x))

format_evaluator.add_rule(s.x**s.a/s.y**s.a,(s.x/s.y)**s.a)
format_evaluator.add_rule(s.x**s.a*s.y**s.a,(s.x*s.y)**s.a)
format_evaluator.add_rule(s.x**s.a*s.y**-s.a,(s.x/s.y)**s.a)

format_evaluator.add_rule(pc.Min(s.a,-s.a),-abs(s.a))
format_evaluator.add_rule(pc.Min(-s.a,s.a),-abs(s.a))
format_evaluator.add_rule(pc.Min(s.a,s.b),-abs(s.a),condition=pc.Equal(-s.b,s.a))
format_evaluator.add_rule(-(-s.a),s.a)

format_evaluator.add_rule(pc.Equal(s.a,s.a),True)
format_evaluator.add_rule(-(s.a+s.b),-s.a-s.b)

format_evaluator.add_rule((-s.a)*s.b,-(s.a*s.b))



format_evaluator.add_rule(pc.Or(s.x<s.y,pc.Equal(s.x,s.y)),s.x<=s.y)




