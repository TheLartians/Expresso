
import pysymbols

def Symbol(name,type=None,positive = False):
    s = Expression(pysymbols.create_symbol(name))
    if type != None:
        from functions import Type
        global_context.add_definition(Type(s),type)
    if positive == True:
        global_context.add_definition(s>0,True)
    return s

integer_type = long

def expression_converter(expr):
    if isinstance(expr,pysymbols.core.Expression):
        return Expression(expr)
    if isinstance(expr,Expression):
        return expr
    if isinstance(expr,bool):
        if expr == True:
            return Symbol('True')
        if expr == False:
            return Symbol('False')
    if isinstance(expr,int):
        expr = integer_type(expr)
    if isinstance(expr,integer_type):
        if expr >= 0:
            return Expression(pysymbols.create_object(expr))
        else:
            expr = abs(expr)
            return Negative(pysymbols.create_object(expr))
    if isinstance(expr,float):
        s = repr(expr)
        split = s.split('e')
        if len(split) == 2:
            mantissa,exponent = (split[0],integer_type(split[1]))
        else:
            exponent = 0
            mantissa = split[0]
        decimal = mantissa.find('.')
        mantissa = mantissa.replace('.','')
        if decimal != -1:
            exponent -= len(mantissa) - decimal
        mantissa = mantissa.lstrip('0')
        while len(mantissa) > 1 and mantissa[-1] == '0':
            exponent += 1
            mantissa = mantissa[:-1]
        if len(mantissa) == 0:
            return Zero
        mantissa = integer_type(mantissa)
        test_str = "%se%s" % (mantissa,exponent)
        assert eval(test_str) == expr, "float conversion error: %s != %s" % (eval(test_str),expr)
        if exponent == 0:
            return S(mantissa)
        return S(mantissa) * 10**S(exponent)
    if isinstance(expr,complex):
        if expr.real == 0:
            if expr.imag == 0:
                return Zero
            return I * S(float(expr.imag))
        if expr.imag == 0:
            return S(float(expr.real))
        return S(float(expr.real)) + I * S(float(expr.imag))
    if isinstance(expr,tuple):
        return Tuple(*expr)
    raise ValueError('Unsupported expression type: %s' % type(expr))

def S(value):
    if isinstance(value,str):
        return Symbol(value)
    return expression_converter(value)
    
def Wildcard(name):
    return S(pysymbols.core.create_wildcard_symbol(name))

def WildcardFunction(name):
    return pysymbols.expression.Function(pysymbols.core.WildcardFunction(name),S=expression_converter)

def symbols(string,**kwargs):
    string = string.replace(" ", "")
    return [Symbol(s,**kwargs) for s in string.split(',')]

def wildcard_symbols(string):
    string = string.replace(" ", "")
    return [Wildcard(s) for s in string.split(',')]

printer = pysymbols.printer.Printer(expression_converter)
latex = pysymbols.printer.LatexPrinter(expression_converter)

class Expression(pysymbols.WrappedExpression(expression_converter)):
    
    def __add__(self, other):
        return Addition(self,other)

    def __radd__(self, other):
        return Addition(other,self)

    def __neg__(self):
        return Negative(self)

    def __pos__(self):
        return self
    
    def __sub__(self, other):
        return Addition(self, Negative(other))

    def __rsub__(self, other):
        return Addition(other, Negative(self))

    def __mul__(self, other):
        return Multiplication(self,other)
    
    def __rmul__(self, other):
        return Multiplication(other,self)

    def __div__(self, other):
        return Multiplication(self, Fraction(other))
    
    def __rdiv__(self, other):
        other = self.S(other)
        if other == One:
            return Fraction(self)
        return Multiplication(other, Fraction(self))
    
    def __pow__(self,other):
        return Exponentiation(self,other)

    def __rpow__(self,other):
        return Exponentiation(other,self)
    
    def __lt__(self, other):
        return Less(self,other)

    def __le__(self, other):
        return LessEqual(self,other)
    
    def __gt__(self, other):
        return Greater(self,other)

    def __ge__(self, other):
        return GreaterEqual(self,other)

    def __or__(self, other):
        return Or(self,other)

    def __xor__(self, other):
        return Xor(self,other)

    def __and__(self, other):
        return And(self,other)

    def __max__(self, other):
        return Max(self,other)

    def __abs__(self):
        return Abs(self)
        
    def __nonzero__(self):
        raise ValueError('Cannot determine truth value of Expression. Perhaps you are using a python operator incorrectly?')
    
    def _repr_latex_(self):
         return "$$%s$$" % latex(self)

    def __repr__(self):
         return printer(self)

    def evaluate(self,context = None):
        if context == None:
            context = global_context
        from evaluate import evaluate
        return evaluate(self,global_context)
    
    def subs(self,*args,**kwargs):
        do_evaluate = kwargs.pop('evaluate',True)
        res = self.substitute(*args)
        if do_evaluate:
            return res.evaluate()
        return res

locals().update(pysymbols.WrappedExpressionTypes(Expression).__dict__)

class Context(ReplaceEvaluator):

    def add_definition(self,search,replacement):
        self.add_replacement(search.evaluate(),replacement)

global_context = Context()

One = S(1)
Zero = S(0)
I = Symbol('_I')

Addition = BinaryOperator("+",pysymbols.associative,pysymbols.commutative,-11)
Negative = UnaryOperator("-",pysymbols.prefix,-12)
Multiplication = BinaryOperator("*",pysymbols.associative,pysymbols.commutative,-13)
Fraction = UnaryOperator("1/",pysymbols.prefix,-14)
Exponentiation = BinaryOperator("**",-15)

AdditionGroup = Group(Addition,Negative,Zero)
MultiplicationGroup = Group(Multiplication,Fraction,One)
RealField = Field(AdditionGroup,MultiplicationGroup)
ComplexField = Field(AdditionGroup,MultiplicationGroup)

Or = BinaryOperator("|",pysymbols.associative,pysymbols.commutative,-3)
And = BinaryOperator("&",pysymbols.associative,pysymbols.commutative,-3)
Xor = BinaryOperator(" XOR ",pysymbols.associative,pysymbols.commutative,-3)

Not = UnaryOperator("~",pysymbols.prefix,-7)

Equal = BinaryOperator("=",pysymbols.associative,pysymbols.commutative,-6)
NotEqual = BinaryOperator("!=",pysymbols.associative,pysymbols.commutative,-6);

In = BinaryOperator(" in ",-6)
NotIn = BinaryOperator(" not in ",-6)

Less = BinaryOperator("<",pysymbols.associative,pysymbols.non_commutative,-6)
LessEqual = BinaryOperator("<=",pysymbols.associative,pysymbols.non_commutative,-6)
Greater = BinaryOperator(">",pysymbols.associative,pysymbols.non_commutative,-6)
GreaterEqual = BinaryOperator(">=",pysymbols.associative,pysymbols.non_commutative,-6)

Abs = Function('abs',argc = 1)
Tuple = Function('tuple')


