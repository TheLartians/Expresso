
import expresso

Number = long

def Symbol(name,type=None,positive = False,latex = None,repr = None):
    s = Expression(expresso.create_symbol(name))
    if type != None:
        from functions import Type
        global_context.add_definition(Type(s),type)
    if positive == True:
        from .functions import sign
        global_context.add_definition(s>=0,True)
        global_context.add_definition(sign(s),1)
    if latex is not None:
        latex_rep = latex
        from printer import latex,add_target
        @add_target(latex,s)
        def print_latex(printer,expr):
            return latex_rep
    if repr is not None:
        from printer import printer,add_target
        @add_target(printer,s)
        def evaluate(printer,expr):
            return repr
    return s

def expression_converter(expr):
    if isinstance(expr,expresso.core.Expression):
        return Expression(expr)
    if isinstance(expr,Expression):
        return expr
    if isinstance(expr,bool):
        if expr == True:
            return Expression(expresso.create_object(expr))
        if expr == False:
            return Expression(expresso.create_object(expr))
    if isinstance(expr,(int)):
        expr = Number(expr)
    if isinstance(expr, Number):
        if expr >= 0:
            return Expression(expresso.create_object(expr))
        else:
            expr = abs(expr)
            return negative(expresso.create_object(expr))
    if isinstance(expr,float):
        import fractions
        f = fractions.Fraction(repr(expr))
        if f.denominator == 1:
            return expression_converter(f.numerator)
        if f.numerator == 1:
            return fraction(f.denominator)
        else:
            return f.numerator * fraction(f.denominator)
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
    raise ValueError('Unsupported expression type: %s (%s)' % (type(expr),expr))

def S(value):
    if isinstance(value,str):
        return Symbol(value)
    return expression_converter(value)
    
def Wildcard(name):
    return S(expresso.core.create_wildcard_symbol(name))

def WildcardFunction(name):
    return expresso.expression.Function(expresso.core.WildcardFunction(name),S=expression_converter)

def symbols(string,**kwargs):
    return [Symbol(s.strip(),**kwargs) for s in string.split(',')]

def wildcard_symbols(string):
    string = string.replace(" ", "")
    return [Wildcard(s.strip()) for s in string.split(',')]

printer = expresso.printer.Printer(expression_converter)
latex = expresso.printer.LatexPrinter(expression_converter)

class Expression(expresso.WrappedExpression(expression_converter)):
    
    def __add__(self, other):
        return addition(self,other)

    def __radd__(self, other):
        return addition(other,self)

    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self
    
    def __sub__(self, other):
        return addition(self, negative(other))

    def __rsub__(self, other):
        return addition(other, negative(self))

    def __mul__(self, other):
        return multiplication(self,other)
    
    def __rmul__(self, other):
        return multiplication(other,self)

    def __div__(self, other):
        return multiplication(self, fraction(other))
    
    def __rdiv__(self, other):
        other = self.S(other)
        if other == One:
            return fraction(self)
        return multiplication(other, fraction(self))
    
    def __pow__(self,other):
        return exponentiation(self,other)

    def __rpow__(self,other):
        return exponentiation(other,self)

    def __mod__(self,other):
        return mod(self,other)

    def __rmod__(self,other):
        return mod(other,self)

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

    def __abs__(self):
        return Abs(self)
        
    def __nonzero__(self):
        raise ValueError('Cannot determine truth value of Expression. Perhaps you are using a python operator incorrectly?')
    
    def _repr_latex_(self):
         return "$$%s$$" % latex(self)

    def __repr__(self):
         return printer(self)

    def __iter__(self):
        from .functions import Tuple
        if self.function == Tuple:
            return self.args.__iter__()
        raise AttributeError('%s object has no attribute __iter__' % type(self))

    def evaluate(self,context = None,**kwargs):
        if context == None:
            context = global_context
        from evaluators import evaluate
        return evaluate(self,context = context,**kwargs)
    
    def subs(self,*args,**kwargs):
        #do_evaluate = kwargs.pop('evaluate',True)
        res = self.replace(*args)
        return res

    def N(self,prec = 16,**kwargs):
        from compilers import N
        return N(self,mp_dps=prec,**kwargs)

    def approximate(self,prec = 16,**kwargs):
        from expresso.pycas.evaluators.optimizers import optimize_for_compilation
        return optimize_for_compilation(self,prec=prec,**kwargs)

    def __float__(self):
        v = self.evaluate().N()
        try:
            return float(v)
        except:
            raise RuntimeError('expression %s is not convertable to float' % self)

    def __complex__(self):
        return complex(self.evaluate().N())

    def __int__(self):
        from compilers import lambdify
        v = lambdify(self.evaluate())()
        try:
            return int(v)
        except:
            raise RuntimeError('expression %s is not convertable to int' % self)

    def __long__(self):
        from compilers import lambdify
        v = lambdify(self.evaluate())()
        try:
            return long(v)
        except:
            raise RuntimeError('expression %s is not convertable to long' % self)

locals().update(expresso.WrappedExpressionTypes(Expression).__dict__)

class Context(ReplaceEvaluator):

    def add_definition(self,search,replacement):
        self.add_replacement(search,replacement)

global_context = Context()

One = S(1)
Zero = S(0)
NaN = S( expresso.create_object(float('nan'),'undefined value') )
I = S( expresso.create_object(1j,'imaginary unit') )

addition = BinaryOperator("+",expresso.associative,expresso.commutative,-11)
negative = UnaryOperator("-",expresso.prefix,-12)
multiplication = BinaryOperator("*",expresso.associative,expresso.commutative,-13)
fraction = UnaryOperator("1/",expresso.prefix,-14)
exponentiation = BinaryOperator("**",-15)

addition_group = Group(addition,negative,Zero)
multiplication_group = Group(multiplication,fraction,One)
real_field = Field(addition_group,multiplication_group)
complex_field = Field(addition_group,multiplication_group)

Or = BinaryOperator("|",expresso.associative,expresso.commutative,-3)
And = BinaryOperator("&",expresso.associative,expresso.commutative,-3)
Xor = BinaryOperator(" XOR ",expresso.associative,expresso.commutative,-3)

Not = UnaryOperator("~",expresso.prefix,-7)
mod = Function('mod',argc = 2)

equal = BinaryOperator("=",expresso.associative,expresso.commutative,-6)
unequal = BinaryOperator("!=",expresso.associative,expresso.commutative,-6);

In = BinaryOperator(" in ",-6)
NotIn = BinaryOperator(" not in ",-6)

Less = BinaryOperator("<",expresso.associative,expresso.non_commutative,-6)
LessEqual = BinaryOperator("<=",expresso.associative,expresso.non_commutative,-6)
Greater = BinaryOperator(">",expresso.associative,expresso.non_commutative,-6)
GreaterEqual = BinaryOperator(">=",expresso.associative,expresso.non_commutative,-6)

Abs = Function('abs',argc = 1)
Tuple = Function('tuple')


