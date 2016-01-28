
import pycas as pc

from canonical_form import canonical_form
from logic_evaluator import logic_evaluator
from numeric_evaluator import numeric_evaluator
from type_evaluator import type_evaluator
from main_evaluator import main_evaluator

def evaluate(expr,context = pc.global_context):
    main = pc.MultiEvaluator(recursive=True,split_binary=True)
    main.add_evaluator(context)
    main.add_evaluator(main_evaluator)
    return main(expr)