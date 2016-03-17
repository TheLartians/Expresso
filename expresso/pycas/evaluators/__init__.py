
import expresso.pycas as pc
import canonical_form
import logic_evaluator
import numeric_evaluator
import type_evaluator
import main_evaluator
import expand_evaluator

__cached_evaluators = {}
__use_global_cache = False

def evaluate(expr,context = pc.global_context,cache = None,format = True):

    if context in __cached_evaluators:
        main = __cached_evaluators[context]
    else:
        main = pc.MultiEvaluator(recursive=True,split_binary=True)
        main.add_evaluator(context)
        main.add_evaluator(main_evaluator.main_evaluator)
        __cached_evaluators[context] = main

    if cache == None and __use_global_cache:
        cache = __cached_evaluators

    expr = main(expr,cache = cache)
    if format:
        expr = canonical_form.format_evaluator(expr,cache = cache)
    return expr

def use_global_cache(v):
    __use_global_cache = v

def set_debug(v):

    def callback(r,m):
        from IPython.display import display_latex

        print "appy rule: %s" % r

        lt = pc.latex(r.search.subs(m,evaluate=False)),\
             pc.latex(r.replacement.subs(m,evaluate=False)),\
             r"\;\text{ if }\;%s" % pc.latex(r.condition.subs(m,evaluate=False)) if r.condition is not None else ''

        display_latex(r"$$%s \rightarrow %s%s$$" % lt,raw=True)

    if v:
        main_evaluator.main_evaluator.set_rule_callback(callback)
        canonical_form.format_evaluator.set_rule_callback(callback)
        expand_evaluator.expand_evaluator.set_rule_callback(callback)
    else:
        main_evaluator.main_evaluator.set_rule_callback(None)
        canonical_form.format_evaluator.set_rule_callback(None)
        expand_evaluator.expand_evaluator.set_rule_callback(None)

def expand(expr,**kwargs):
    return canonical_form.format_evaluator(expand_evaluator.expand_evaluator(expr,**kwargs),**kwargs)

