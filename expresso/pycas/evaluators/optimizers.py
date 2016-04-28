


import expresso.pycas as pc
import rule_symbols as s
from .logic_evaluator import is_numeric,is_atomic

from .canonical_form import format_evaluator
from .logic_evaluator import evaluator as logic_evaluator

from .logic_evaluator import is_mpmath

compile_evaluator = pc.RewriteEvaluator(recursive=False,split_binary=True)

fold_accuracy = 20

def fold(m):
    try:
        res = m[s.x].N(fold_accuracy,folding=True)
        if type(res) == tuple:
            return False
        m[s.y] = pc.expresso.create_object(res)
    except Exception as e:
        return False

compile_evaluator.add_rule(s.x,s.y,fold)
compile_evaluator.add_rule(s.x**2,s.x*s.x,condition=is_atomic(s.x))

compile_evaluator.add_rule(s.x*abs(s.y),abs(s.x*s.y),condition=is_mpmath(s.x))
compile_evaluator.add_rule(s.x*(s.y+s.z),(s.x*s.y+s.x*s.z),condition=is_mpmath(s.x))

compiler_opt_evaluator = pc.MultiEvaluator(recursive = True, split_binary=True)
compiler_opt_evaluator.add_evaluator(compile_evaluator)
compiler_opt_evaluator.add_evaluator(logic_evaluator)

def optimize_for_compilation(expr,cache = None,prec=20):
    global fold_accuracy
    fold_accuracy = prec
    opt = None

    # TODO: why do we need to evaluate multiple times?
    N = 10
    while opt != expr and N>0:
        N -= 1
        opt = expr
        expr = compiler_opt_evaluator(expr, cache = cache)

    return format_evaluator(expr, cache = cache)

