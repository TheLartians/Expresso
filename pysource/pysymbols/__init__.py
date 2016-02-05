from expression import core,WrappedExpression
import expression
import evaluator
import printer
import visitor

from expression import associative,non_associative,left_associative,right_associative,commutative,non_commutative,postfix,prefix

def create_object(data,unique_name=None):
    '''
Create an atomic expression that holds data as the value.
Note that two objects will be treated the the same if they
have the same 'unique_name' (by default given by the type and
rerp(data)).
    '''
    if unique_name is None:
        unique_name = '%s:(%s)' % (type(data).__name__, repr(data))
    return core.create_object(data,unique_name)

def create_symbol(name):
    return core.create_symbol(name)

class WrappedExpressionTypes(object):

    def __init__(self,Expression):
        
        S = Expression._get_wrapped_parameter('S')

        self.Function = expression.WrappedFunction(core.Function,S)
        self.BinaryOperator = expression.WrappedFunction(core.BinaryOperator,S)
        self.UnaryOperator = expression.WrappedFunction(core.UnaryOperator,S,argc = 1)
        
        self.Group = expression.WrappedGroup(S)
        self.Field = expression.WrappedField(S)
        self.ReplacementMap = expression.WrappedReplacementMap(S)
        self.MulplicityList = expression.WrappedMulplicityList(S)

        self.MatchCondition = evaluator.WrappedMatchCondition(S)
        self.Rule = evaluator.WrappedRule(S)
        self.RewriteEvaluator = evaluator.WrappedRewriteEvaluator(S)
        self.MultiEvaluator = evaluator.WrappedMultiEvaluator(S)
        self.StepEvaluator = evaluator.WrappedStepEvaluator(S)
        self.ReplaceEvaluator = evaluator.WrappedReplaceEvaluator(S)

        self.postorder_traversal = expression.wrapped_postorder_traversal(S)
        self.preorder_traversal = expression.wrapped_preorder_traversal(S)    
        self.commutative_permutations = expression.wrapped_commutative_permutations(S) 
        
def default_converter(expr):    
    if isinstance(expr,core.Expression):
        return Expression(expr)
    elif isinstance(expr,Expression):
        return expr
    elif isinstance(expr,str):
        return Expression(create_symbol(expr))
    else:
        return Expression(create_object(expr))
    
def comp(a,b):
    return core.Expression.__lt__(a,b)
    
#Expression = WrappedExpression(default_converter)
#locals().update(WrappedExpressionTypes(Expression).__dict__)

