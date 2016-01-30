


import pycas as pc
import rule_symbols as s
from .logic_evaluator import is_numeric,is_atomic

compile_evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)

def fold_binary(f):

    def fold(m):
        try:
            m[s.z] = pc.pysymbols.create_object(f(m[s.x],m[s.y]).N(20))
        except:
            return False

    return fold

def fold_unary(f):

    def fold(m):
        try:
            m[s.z] = pc.pysymbols.create_object(f(m[s.x]).N(20))
        except:
            return False

    return fold

compile_evaluator.add_rule(s.x*s.y,s.z,fold_binary(lambda x,y:x*y))
compile_evaluator.add_rule(s.x+s.y,s.z,fold_binary(lambda x,y:x+y))
compile_evaluator.add_rule(s.x**s.y,s.z,fold_binary(lambda x,y:x**y))

compile_evaluator.add_rule(s.x**2,s.x*s.x,condition=is_atomic(s.x))


from .canonical_form import canonical_form,format_evaluator
from .logic_evaluator import logic_evaluator
from .numeric_evaluator import numeric_evaluator
from .type_evaluator import type_evaluator

compiler_opt_evaluator = pc.MultiEvaluator(recursive = True, split_binary=True)

compiler_opt_evaluator.add_evaluator(canonical_form)
compiler_opt_evaluator .add_evaluator(compile_evaluator)
#compiler_opt_evaluator .add_evaluator(type_evaluator)
compiler_opt_evaluator .add_evaluator(logic_evaluator)
#optimize_for_c_evaluator.add_evaluator(numeric_evaluator)

def optimize_for_compilation(expr):
    return format_evaluator(compiler_opt_evaluator (expr))