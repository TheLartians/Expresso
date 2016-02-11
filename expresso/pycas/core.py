
import expresso as ex
from .printer import latex

def S(expr):
    if isinstance(expr,ex.core.Expression):
        return Expression(expr)
    if isinstance(expr,Expression):
        return expr
    if isinstance(expr,str):
        return Expression(ex.core.create_symbol(expr))
    if isinstance(expr,int):
        if expr >= 0:
            return Expression(ex.core.create_object(expr,str(expr)))
        else:
            expr = abs(expr)
            return negative(ex.core.create_object(expr,str(expr)))

    raise ValueError('Unsupported expression type: %s' % type(expr))

def W(name):
    return S(ex.core.create_wildcard_symbol(name))

def WF(name):
    return ex.Function(ex.core.WildcardFunction(name),S)

class Expression(ex.WrappedExpression(S)):
    
    def __add__(self, other):
        return addition(self,self.S(other))

    def __radd__(self, other):
        return addition(self.S(other),self)

    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self
    
    def __sub__(self, other):
        return addition(self, negative(self.S(other)))

    def __rsub__(self, other):
        return addition(self.S(other), negative(self))

    def __mul__(self, other):
        return multiplication(self,self.S(other))

    def __rmul__(self, other):
        return multiplication(self.S(other),self)

    def __div__(self, other):
        return multiplication(self, fraction(self.S(other)))

    def __rdiv__(self, other):
        other = self.S(other)
        if other == One:
            return fraction(self)
        return multiplication(self.S(other), fraction(self))
    
    def __pow__(self,other):
        return exponentiation(self,self.S(other))

    def __rpow__(self,other):
        return exponentiation(self.S(other),self)

    def _repr_latex_(self):
         return "$$%s$$" % latex(self)

One = S(1)
Zero = S(0)

Function = ex.WrappedFunction(ex.core.Function,S)
BinaryOperator = ex.WrappedFunction(ex.core.BinaryOperator,S)
UnaryOperator = ex.WrappedFunction(ex.core.UnaryOperator,S)

from evaluator import WrappedRule,WrappedRuleEvaluator

Rule = WrappedRule(S)
RuleEvaluator = WrappedRuleEvaluator(S)

MatchCondition = ex.WrappedMatchCondition(S)
MulplicityList = ex.WrappedMulplicityList(S)



equal = BinaryOperator("=",ex.core.associative,ex.core.commutative,-6)


addition = BinaryOperator("+",ex.core.associative,ex.core.commutative,-11)
negative = UnaryOperator("-",ex.core.prefix,-12)
multiplication = BinaryOperator("*",ex.core.associative,ex.core.commutative,-13)
fraction = UnaryOperator("1/",ex.core.prefix,-14)
exponentiation = BinaryOperator("**",-15)

Group = ex.WrappedGroup(S)
Field = ex.WrappedField(S)


addition_group = Group(addition,negative,Zero)
multiplication_group = Group(multiplication,fraction,One)

real_field = Field(addition_group,multiplication_group)










def __latex_print_addition(printer,expr):
    neg_args = [arg for arg in expr.args if arg.function == negative]
    covered = set(neg_args)
    rest = [arg for  arg in expr.args if arg not in covered]
    rest_str = '+'.join(printer._printed_operator_arguments(expr,rest))
    if len(neg_args) == 0:
        return rest_str
    neg_str = '-'.join(printer._printed_operator_arguments(expr,[arg.args[0] for arg in neg_args]))
    return rest_str + '-' + neg_str

latex.register_printer(addition,__latex_print_addition)

def __latex_print_multiplication(printer,expr):
    denominators = [arg for arg in expr.args if arg.function == fraction]
    if len(denominators)>0:
        numerators = [arg for arg in expr.args if arg.is_atomic]
        if len(numerators) == 0:
            numerators = [One]
            
        covered = set(numerators + denominators)
        rest = [arg for  arg in expr.args if arg not in covered]

        denom_str = printer(multiplication(*[arg.args[0] for arg in denominators]))
        num_str =   printer(multiplication(*numerators))
        
        if len(rest) == 0:
            rest_str = ""
        elif len(rest) == 1:
            rest_str = printer(rest[0])
            if printer._needs_brackets_in(rest[0],expr):
                rest_str = printer._bracket_format() % rest_str
        else:
            rest = multiplication(*rest)
            rest_str =  printer(rest)
        
        return r'\frac{%s}{%s} \, %s ' % (num_str,denom_str,rest_str)

    is_numeric = lambda x: x.value != None or (x.function == exponentiation and x.args[0].value != None)
    
    numeric = [x for x in expr.args if is_numeric(x)]
    non_numeric = [x for x in expr.args if not is_numeric(x)]

    res  = '\cdot '.join(printer._printed_operator_arguments(expr,numeric)) 
    res += '\, '.join(printer._printed_operator_arguments(expr,non_numeric))

    return res

latex.register_printer(multiplication,__latex_print_multiplication)

def __latex_print_fraction(printer,expr):
    return r'\frac{1}{%s}' % printer(expr.args[0])

latex.register_printer(fraction,__latex_print_fraction)

# Here brackets are not set correctly
def __latex_print_exp(printer,expr):
    parg = printer._printed_operator_arguments(expr,end=-1)
    parg += [printer(expr.args[-1])]
    return '^'.join(['{%s}' % arg for arg in parg])

latex.register_printer(exponentiation,__latex_print_exp)



postorder_traversal = ex.wrapped_postorder_traversal(S)
preorder_traversal = ex.wrapped_postorder_traversal(S)

