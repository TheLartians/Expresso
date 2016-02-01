
import pycas as pc

import canonical_form
import logic_evaluator
import numeric_evaluator
import type_evaluator
import main_evaluator

def evaluate(expr,context = pc.global_context,cache = None,format = True):
    main = pc.MultiEvaluator(recursive=True,split_binary=True)
    main.add_evaluator(context)
    main.add_evaluator(main_evaluator.main_evaluator)
    expr = main(expr,cache)
    if format:
        expr = canonical_form.final_evaluator(canonical_form.format_evaluator(expr))
    return expr

def set_debug(v):

    def callback(r,m):
        from IPython.display import display_latex
        lt = pc.latex(r.search.subs(m,evaluate=False)),\
             pc.latex(r.replacement.subs(m,evaluate=False)),\
             r"\;\text{ if }\;%s" % pc.latex(r.condition.subs(m,evaluate=False)) if r.condition is not None else ''

        display_latex(r"$$%s \rightarrow %s%s$$" % lt,raw=True)

    if v:
        main_evaluator.main_evaluator.set_rule_callback(callback)
        canonical_form.format_evaluator.set_rule_callback(callback)
    else:
        main_evaluator.main_evaluator.set_rule_callback(None)
        canonical_form.format_evaluator.set_rule_callback(None)