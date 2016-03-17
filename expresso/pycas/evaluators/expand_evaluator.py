

import expresso.pycas as pc
import rule_symbols as s

evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)

evaluator.add_rule(s.a*(s.b+s.c),s.a*s.b+s.a*s.c)
evaluator.add_rule(s.a**-1*(s.b+s.c)**-1,(s.a*s.b+s.a*s.c)**-1)

from .logic_evaluator import is_pure_numeric,is_atomic

is_positive_numeric_integer = pc.And(is_pure_numeric(s.n), pc.is_in_type(s.n,pc.Types.Integer), s.n > 1)

evaluator.add_rule((s.a+s.b)**s.n, (s.a+s.b) * (s.a+s.b)**(s.n-1), condition=is_positive_numeric_integer)
evaluator.add_rule((s.a+s.b)**-s.n, (s.a+s.b)**(-s.n+1)*(s.a+s.b)**-1,condition=is_positive_numeric_integer)

evaluator.add_rule(s.x*s.x,s.x**2,condition=is_atomic(s.x))
evaluator.add_rule(s.x*s.x**s.n,s.x**(s.n+1),condition=is_atomic(s.x))
evaluator.add_rule(s.x**s.m*s.x**s.n,s.x**(s.n+s.m),condition=is_atomic(s.x))

evaluator.add_rule((s.a*s.b)**s.c,s.a**s.c*s.b**s.c)

from .canonical_form import canonical_form
from .logic_evaluator import logic_evaluator
from .numeric_evaluator import numeric_evaluator
from .type_evaluator import type_evaluator
from .main_evaluator import evaluator as main_evaluator

expand_evaluator = pc.MultiEvaluator(recursive = True, split_binary=True)
expand_evaluator.add_evaluator(canonical_form)
expand_evaluator.add_evaluator(numeric_evaluator)
expand_evaluator.add_evaluator(evaluator)
expand_evaluator.add_evaluator(main_evaluator)
expand_evaluator.add_evaluator(type_evaluator)
expand_evaluator.add_evaluator(logic_evaluator)

