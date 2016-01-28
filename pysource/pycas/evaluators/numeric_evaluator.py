
import pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True, split_binary=True)

from .logic_evaluator import is_explicit_natural

def numeric_rule(f):

    def evaluator(m):
        res = f(m[s.x].value,m[s.y].value)
        if res is None:
            return False
        else:
            m[s.z] = pc.S(res)

    return evaluator

def are_explicit_numbers(*args):
    return pc.And(*[is_explicit_natural(arg) for arg in args])

evaluator.add_rule(s.x + s.y, s.z, numeric_rule(lambda x, y: x + y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x - s.y, s.z, numeric_rule(lambda x, y: x - y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x * s.y, s.z, numeric_rule(lambda x, y: x * y), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule(s.x ** s.y, s.z, numeric_rule(lambda x, y: x ** y if y <= 100 else None), condition=are_explicit_numbers(s.x, s.y))
evaluator.add_rule((-s.x) ** s.y, s.z, numeric_rule(lambda x, y: x ** y if y <= 100 else None), condition=are_explicit_numbers(s.x, s.y))

from .canonical_form import canonical_form
from .logic_evaluator import logic_evaluator

numeric_evaluator = pc.MultiEvaluator(recursive=True, split_binary=True)
numeric_evaluator.add_evaluator(canonical_form)
numeric_evaluator.add_evaluator(logic_evaluator)
numeric_evaluator.add_evaluator(evaluator)

