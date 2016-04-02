

def get_symbols_in(expr):
    import expresso.pycas as pc

    symbols = set()
    for e in pc.postorder_traversal(expr):
        if e.is_symbol:
            symbols.add(e)
    return symbols

def apply_all_arguments(expr,function):
    return expr.function(*[function(arg) for arg in expr.args])

def create_conditional_factor_evaluator(condition):

    from .expression import MulplicityList,MultiEvaluator,RewriteEvaluator,wildcard_symbols
    from .functions import multiplication_group,exponentiation,real_field

    from .evaluators.main_evaluator import evaluator as main_evaluator

    a,b,c,d = wildcard_symbols('a,b,c,d')

    def group_evaluator(m):
        e1 = m[a]
        e2 = m[b]
        m1 = MulplicityList(e1,multiplication_group,exponentiation,real_field)
        m2 = MulplicityList(e2,multiplication_group,exponentiation,real_field)
        common = m1.intersection(m2)

        commondict = dict(common)

        if len(common) == 0:
            return False
        else:
            if condition(commondict):
                m[a] = (m1 - common).as_expression()
                m[b] = (m2 - common).as_expression()
                m[c] = common.as_expression()
                return True
        return False

    evaluator = RewriteEvaluator(recursive=True,split_binary=True)
    evaluator.add_rule(a+b,c*(a+b),group_evaluator)
    evaluator.add_rule(a-b,c*(a-b),group_evaluator)
    evaluator.add_rule(-a+b,c*(-a+b),group_evaluator)
    evaluator.add_rule(-a-b,-c*(a+b),group_evaluator)

    from evaluators.canonical_form import canonical_form,format_evaluator
    from evaluators.logic_evaluator import logic_evaluator
    from evaluators.numeric_evaluator import numeric_evaluator
    from evaluators.type_evaluator import type_evaluator
    from evaluators.main_evaluator import evaluator as main_evaluator

    conditional_factor = MultiEvaluator(recursive = True, split_binary=True)
    conditional_factor.add_evaluator(canonical_form)
    conditional_factor.add_evaluator(numeric_evaluator)
    conditional_factor.add_evaluator(main_evaluator)
    conditional_factor.add_evaluator(evaluator)
    conditional_factor.add_evaluator(type_evaluator)
    conditional_factor.add_evaluator(logic_evaluator)

    def conditional_evaluate(expr,**kwargs):
        return format_evaluator(conditional_factor(expr,**kwargs),**kwargs)

    return conditional_evaluate

def expand(*args,**kwargs):
    from .evaluators import expand
    return expand(*args,**kwargs)

def split(args,condition):
    r1 = []
    r2 = []
    for arg in args:
        if condition(arg):
            r1.append(arg)
        else:
            r2.append(arg)
    return r1,r2

def term_decomposition(expr):
    import expresso.pycas as pc

    if expr.function == pc.multiplication:
        return [item for term in expr.args for item in term_decomposition(term)]
    elif expr.function == pc.exponentiation:
        e = expr.args[1]
        return [term**e for term in term_decomposition(expr.args[0])]
    elif expr.function == pc.negative:
        return [-1] + term_decomposition(expr.args[0])
    elif expr.function == pc.fraction:
        return [1/term for term in term_decomposition(expr.args[0])]
    else:
        return [expr]

def get_coefficient(expr,term):
    t = split(expr.args,lambda e:term in term_decomposition(e))[0]
    if len(t) == 0:
        return 0
    return (expr.function(*t)/term).evaluate()




