
import pysymbols as ex
from printer import latex

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
            return Negative(ex.core.create_object(expr,str(expr)))

    raise ValueError('Unsupported expression type: %s' % type(expr))

def W(name):
    return S(ex.core.create_wildcard_symbol(name))

def WF(name):
    return ex.Function(ex.core.WildcardFunction(name),S)

class Expression(ex.WrappedExpression(S)):
    
    def __add__(self, other):
        return Addition(self,self.S(other))

    def __radd__(self, other):
        return Addition(self.S(other),self)

    def __neg__(self):
        return Negative(self)

    def __pos__(self):
        return self
    
    def __sub__(self, other):
        return Addition(self, Negative(self.S(other)))

    def __rsub__(self, other):
        return Addition(self.S(other), Negative(self))

    def __mul__(self, other):
        return Multiplication(self,self.S(other))

    def __rmul__(self, other):
        return Multiplication(self.S(other),self)

    def __div__(self, other):
        return Multiplication(self, Fraction(self.S(other)))

    def __rdiv__(self, other):
        other = self.S(other)
        if other == One:
            return Fraction(self)
        return Multiplication(self.S(other), Fraction(self))
    
    def __pow__(self,other):
        return Exponentiation(self,self.S(other))

    def __rpow__(self,other):
        return Exponentiation(self.S(other),self)

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



Equal = BinaryOperator("=",ex.core.associative,ex.core.commutative,-6)


Addition = BinaryOperator("+",ex.core.associative,ex.core.commutative,-11)
Negative = UnaryOperator("-",ex.core.prefix,-12)
Multiplication = BinaryOperator("*",ex.core.associative,ex.core.commutative,-13)
Fraction = UnaryOperator("1/",ex.core.prefix,-14)
Exponentiation = BinaryOperator("**",-15)

Group = ex.WrappedGroup(S)
Field = ex.WrappedField(S)


AdditionGroup = Group(Addition,Negative,Zero)
MultiplicationGroup = Group(Multiplication,Fraction,One)

RealField = Field(AdditionGroup,MultiplicationGroup)










def __latex_print_addition(printer,expr):
    neg_args = [arg for arg in expr.args if arg.function == Negative]
    covered = set(neg_args)
    rest = [arg for  arg in expr.args if arg not in covered]
    rest_str = '+'.join(printer._printed_operator_arguments(expr,rest))
    if len(neg_args) == 0:
        return rest_str
    neg_str = '-'.join(printer._printed_operator_arguments(expr,[arg.args[0] for arg in neg_args]))
    return rest_str + '-' + neg_str

latex.register_printer(Addition,__latex_print_addition)

def __latex_print_multiplication(printer,expr):
    denominators = [arg for arg in expr.args if arg.function == Fraction]
    if len(denominators)>0:
        numerators = [arg for arg in expr.args if arg.is_atomic]
        if len(numerators) == 0:
            numerators = [One]
            
        covered = set(numerators + denominators)
        rest = [arg for  arg in expr.args if arg not in covered]

        denom_str = printer(Multiplication(*[arg.args[0] for arg in denominators]))
        num_str =   printer(Multiplication(*numerators))
        
        if len(rest) == 0:
            rest_str = ""
        elif len(rest) == 1:
            rest_str = printer(rest[0])
            if printer._needs_brackets_in(rest[0],expr):
                rest_str = printer._bracket_format() % rest_str
        else:
            rest = Multiplication(*rest)
            rest_str =  printer(rest)
        
        return r'\frac{%s}{%s} \, %s ' % (num_str,denom_str,rest_str)

    is_numeric = lambda x: x.value != None or (x.function == Exponentiation and x.args[0].value != None)
    
    numeric = [x for x in expr.args if is_numeric(x)]
    non_numeric = [x for x in expr.args if not is_numeric(x)]

    res  = '\cdot '.join(printer._printed_operator_arguments(expr,numeric)) 
    res += '\, '.join(printer._printed_operator_arguments(expr,non_numeric))

    return res

latex.register_printer(Multiplication,__latex_print_multiplication)

def __latex_print_fraction(printer,expr):
    return r'\frac{1}{%s}' % printer(expr.args[0])

latex.register_printer(Fraction,__latex_print_fraction)

# Here brackets are not set correctly
def __latex_print_exp(printer,expr):
    parg = printer._printed_operator_arguments(expr,end=-1)
    parg += [printer(expr.args[-1])]
    return '^'.join(['{%s}' % arg for arg in parg])

latex.register_printer(Exponentiation,__latex_print_exp)



postorder_traversal = ex.wrapped_postorder_traversal(S)
preorder_traversal = ex.wrapped_postorder_traversal(S)

