


pc = __import__(__name__.split('.')[0])
import rule_symbols as s
from .logic_evaluator import is_numeric,is_atomic

compile_evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)

fold_accuracy = 20


def fold_unary(f):

    def fold(m):
        try:
            m[s.z] = pc.pysymbols.create_object(f(m[s.x]).N(fold_accuracy))
        except:
            return False

    return fold

import math

compile_evaluator.add_rule(pc.floor(s.x),s.z,fold_unary(lambda x:math.floor(x)))
compile_evaluator.add_rule(pc.ceil(s.x),s.z,fold_unary(lambda x:math.ceil(x)))
compile_evaluator.add_rule(pc.round(s.x),s.z,fold_unary(lambda x:math.round(x)))


def fold_binary(f):

    def fold(m):
        try:
            m[s.z] = pc.pysymbols.create_object(f(m[s.x],m[s.y]).N(fold_accuracy))
        except:
            return False

    return fold


compile_evaluator.add_rule(s.x*s.y,s.z,fold_binary(lambda x,y:x*y))
compile_evaluator.add_rule(s.x+s.y,s.z,fold_binary(lambda x,y:x+y))
compile_evaluator.add_rule(s.x**s.y,s.z,fold_binary(lambda x,y:x**y))
compile_evaluator.add_rule(s.x<s.y,s.z,fold_binary(lambda x,y:x<y))
compile_evaluator.add_rule(pc.equal(s.x,s.y),s.z,fold_binary(lambda x,y:pc.equal(x,y)))
compile_evaluator.add_rule(s.x&s.y,s.z,fold_binary(lambda x,y:x&y))
compile_evaluator.add_rule(s.x|s.y,s.z,fold_binary(lambda x,y:x|y))

compile_evaluator.add_rule(s.x**2,s.x*s.x,condition=is_atomic(s.x))

folded_half = pc.pysymbols.create_object((1/pc.S(2)).N(fold_accuracy))
compile_evaluator.add_rule(s.x**folded_half,pc.sqrt(s.x))

from .canonical_form import canonical_form,format_evaluator
from .logic_evaluator import logic_evaluator
from .numeric_evaluator import numeric_evaluator
from .type_evaluator import type_evaluator

compiler_opt_evaluator = pc.MultiEvaluator(recursive = True, split_binary=True)

compiler_opt_evaluator.add_evaluator(canonical_form)
compiler_opt_evaluator.add_evaluator(compile_evaluator)
compiler_opt_evaluator.add_evaluator(type_evaluator)
compiler_opt_evaluator.add_evaluator(logic_evaluator)
compiler_opt_evaluator.add_evaluator(numeric_evaluator)

def optimize_for_compilation(expr,cache = None):
    return format_evaluator(compiler_opt_evaluator(expr, cache = cache))