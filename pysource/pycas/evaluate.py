
__all__ =['prepare_evaluator','main_evaluator','intermediate_evaluator','final_evaluator','primefactor_evaluator','expand_evaluator','expand','evaluate']

from .expression import *
from .functions import *
import pysymbols

prepare_evaluator = RewriteEvaluator()
main_evaluator = RewriteEvaluator(recursive=True)
type_evaluator = RewriteEvaluator(recursive=True)
intermediate_evaluator = RewriteEvaluator(recursive=True)
final_evaluator = RewriteEvaluator(recursive=True)
primefactor_evaluator = RewriteEvaluator(recursive=True)
expand_evaluator = RewriteEvaluator(recursive=True)

PiecewisePart = Function('piecewise_part',argc = 2)

def add_prepare_evaluator_rules(evaluator):
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    evaluator.add_rule(exp(x),e**x)
    evaluator.add_rule(sqrt(x),x**(S(1)/2))

match_int = MatchCondition('match_int',lambda e:isinstance(e.value,integer_type))

def is_even_number(expr):
    v = expr.value
    return isinstance(v,integer_type) and v % 2 == 0
is_even_number = MatchCondition('is_even_number',is_even_number)

def is_odd_number(expr):
    v = expr.value
    return isinstance(v,integer_type) and v % 2 != 0
is_odd_number = MatchCondition('is_odd_number',is_odd_number)

def add_basic_simplification_rules(evaluator):
    
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    
    evaluator.add_rule(x+0,x)
    evaluator.add_rule(x-x,0)
    evaluator.add_rule(x*1,x)
    evaluator.add_rule(x*0,0)
    evaluator.add_rule(x/x,1)
    evaluator.add_rule(1/S(1),1)
    evaluator.add_rule(1/S(-1),-1)

    evaluator.add_rule(x**1,x)
    evaluator.add_rule(x**0,1)
    evaluator.add_rule(1**x,1)
    evaluator.add_rule(0**x,0)

    evaluator.add_rule(x*x**-1,1)
    
    evaluator.add_rule(x+x,2*x)
    evaluator.add_rule(-(x+y),-x-y)
    evaluator.add_rule(x*-1,-x)
    evaluator.add_rule(-(-x),x)
    evaluator.add_rule((-x)*y,-(x*y))
    evaluator.add_rule(1/-x,-(1/x))
    evaluator.add_rule((-x)**-1,-((x)**-1))
    evaluator.add_rule(-S(0),0)

    evaluator.add_rule(1/(y/z),z/y)
    evaluator.add_rule(1/(1/z),z)
    evaluator.add_rule(1/-x,-(1/x))
        
    evaluator.add_rule((-x)**is_even_number(n),x**n)
    
    evaluator.add_rule((I)**is_even_number(n),(-1)**(n/2))
    evaluator.add_rule((I)**is_odd_number(n),I*(-1)**((n-1)/2))
    evaluator.add_rule((I)**is_odd_number(n),I * (-1)**((n+1)/2))
    evaluator.add_rule((I)**is_even_number(n), (-1)**(n/2+1))
    evaluator.add_rule(1/I,-I)
    
    evaluator.add_rule(log(e),1)
    evaluator.add_rule(log(1),0)

    evaluator.add_rule(sin(0),0)
    evaluator.add_rule(sin(pi/2),1)
    evaluator.add_rule(sin(2*pi),0)
    evaluator.add_rule(cos(0),1)
    evaluator.add_rule(sin(pi/2),0)
    evaluator.add_rule(cos(2*pi),1)
    
    evaluator.add_rule(Indicator(True),1)
    evaluator.add_rule(Indicator(False),0)
    
def add_logic_rules(evaluator):
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    f = WildcardFunction("f")

    evaluator.add_rule(Xor(False,False),False);
    evaluator.add_rule(Xor(True,False),True);
    evaluator.add_rule(Xor(False,True),True);
    evaluator.add_rule(Xor(True,True),False);

    evaluator.add_rule(And(x,True),x);
    evaluator.add_rule(And(x,False),False);
    evaluator.add_rule(Or(x,True),True);
    evaluator.add_rule(Or(x,False),x);
    evaluator.add_rule(Not(True),False);
    evaluator.add_rule(Not(False),True);
    evaluator.add_rule(Not(Not(x)),x);
    evaluator.add_rule(And(x,Not(x)),False);
    evaluator.add_rule(Or(x,Not(x)),True);
    evaluator.add_rule(Not(Or(x,y)),And(Not(x),Not(y)));
    evaluator.add_rule(Not(And(x,y)),Or(Not(x),Not(y)));
    evaluator.add_rule(And(x,x),x);
    evaluator.add_rule(Or(x,x),x);
    evaluator.add_rule(NotEqual(x,y),Not(Equal(x,y)));
    evaluator.add_rule(Equal(x,x),True);
    evaluator.add_rule(x<x,False);
    evaluator.add_rule(-x<-y,y<x);
    evaluator.add_rule(x<-x,False);
    evaluator.add_rule(-x<x,False);
    evaluator.add_rule(x<=x,True);
    
    evaluator.add_rule(x>y,y<x);
    evaluator.add_rule(x>=y,y<=x);
    evaluator.add_rule(x<=y,Or(x<y,Equal(x,y)));

    evaluator.add_rule(And(x<y,y<x),False);

    evaluator.add_rule(x<oo,True);
    evaluator.add_rule(-oo<x,True);
    evaluator.add_rule(oo<x,False);
    evaluator.add_rule(x<-oo,False);
    
    evaluator.add_rule(And(Equal(x,y),f(x)),And(Equal(x,y),f(y)));
    evaluator.add_rule(And(Or(x,y),z),Or(And(x,z),And(y,z)));
    
    evaluator.add_rule(abs(x)<y,And(x<y,-x<y));
    evaluator.add_rule(x<abs(y),Or(x<y,x<-y));
    evaluator.add_rule(Equal(abs(x),y),Or(Equal(x,y),Equal(x,-y)));

def primesfrom2to(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    import numpy as np
    sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in xrange(integer_type(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)/3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))/3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

primes = [integer_type(x) for x in primesfrom2to(1000000)]

def primefactor(x):
    if x <= 0:
        return [x]
    res = []
    for p in primes:
        while x % p == 0:
            x = x/p
            res.append(p)
        if p*p > x:
            break
    if x != 1:
        res.append(x)
    return res        

def add_numeric_evaluation_rules(evaluator,include):
    
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    
    def calculate_sum(m):
        res = m[x].value + m[y].value
        if not isinstance(res,integer_type):
            return False
        m[z] = res

    def calculate_difference(m):
        res = m[x].value - m[y].value
        if not isinstance(res,integer_type):
            return False
        m[z] = res
    
    def calculate_product(m):
        res = m[x].value * m[y].value
        if not isinstance(res,integer_type):
            return False
        m[z] = res
    
    def calculate_fraction(m):
        vx = m[x].value
        vy = m[y].value
        if vx % vy != 0:
            return False
        res = vx / vy
        if not isinstance(res,integer_type):
            return False
        m[z] = res
        
    def calculate_power(m):
        vx = m[x].value
        vy = m[y].value
        if vy > 100:
            return False
        res = vx**vy
        if not isinstance(res,integer_type):
            return False
        m[z] = res
    
    def calculate_negative_power(m):
        vx = -m[x].value
        vy = m[y].value
        if vy > 100:
            return False
        res = vx**vy
        if not isinstance(res,integer_type):
            return False
        m[z] = res


    def factor_primes(m):
        primes = primefactor(m[a].value)
        if len(primes) <= 1:
            return False
        m[b] = Multiplication(*primes)
    
    def factor_primes(m):
        primes = primefactor(m[a].value)
        if len(primes) <= 1:
            return False
        m[b] = Multiplication(*primes)

    if 'sum' in include:
        evaluator.add_rule(match_int(x)+match_int(y),z,calculate_sum)
        evaluator.add_rule(-match_int(x)-match_int(y),-z,calculate_sum)
        evaluator.add_rule(match_int(x)-match_int(y),z,calculate_difference)
        include.remove('sum')
    if 'product' in include:
        evaluator.add_rule(match_int(x)*match_int(y),z,calculate_product)
        evaluator.add_rule(match_int(x)/match_int(y),z,calculate_fraction)
        
        evaluator.add_rule(match_int(y)*match_int(x)**-1+match_int(z),(y+z*x)*(x)**-1)
        evaluator.add_rule(match_int(y)*match_int(x)**-1-match_int(z),(y-z*x)*(x)**-1)
        evaluator.add_rule(match_int(y)*match_int(x)**-1+match_int(a)*match_int(b)**-1,(y*b+a*x)*(b*x)**-1)
        evaluator.add_rule(match_int(y)*match_int(x)**-1-match_int(a)*match_int(b)**-1,(y*b-a*x)*(b*x)**-1)

        evaluator.add_rule(match_int(y)/match_int(x)+match_int(z),(y+z*x)/(x))
        evaluator.add_rule(match_int(y)/match_int(x)-match_int(z),(y-z*x)/(x))
        evaluator.add_rule(match_int(y)/match_int(x)+match_int(a)/match_int(b),(y*b+a*x)/(b*x))
        evaluator.add_rule(match_int(y)/match_int(x)-match_int(a)/match_int(b),(y*b-a*x)/(b*x))
        
        include.remove('product')
    if 'power' in include:
        evaluator.add_rule(match_int(x)**match_int(y),z,calculate_power)
        evaluator.add_rule((-match_int(x))**match_int(y),z,calculate_negative_power)
        evaluator.add_rule((match_int(x)*a)**match_int(y),a**y*x**y)
        evaluator.add_rule(match_int(x)**-match_int(y),z**-1,calculate_power)
        evaluator.add_rule((-match_int(x))**-match_int(y),z**-1,calculate_negative_power)
        include.remove('power')
    if 'primes' in include:
        evaluator.add_rule(match_int(a),b,factor_primes)   
        include.remove('primes')
    if 'logic' in include:
        
        def calculate_less(m):
            m[c] = m[a].value < m[b].value

        def calculate_rhs_negative_less(m):
            m[c] = m[a].value < -m[b].value

        def calculate_lhs_negative_less(m):
            m[c] = -m[a].value < m[b].value
                      
        #TODO: smarter way to create the rules for solving

        evaluator.add_rule(Equal(match_int(a),match_int(a)),True)
        evaluator.add_rule(Equal(match_int(a),match_int(b)),False)   
        evaluator.add_rule(Equal(match_int(a),-match_int(b)),False)   
        
        evaluator.add_rule(a<b/match_int(c),a*c<b)
        evaluator.add_rule(Equal(a,b/match_int(c)),Equal(a*c,b))
        evaluator.add_rule(a<1/match_int(c),a*c<1)
        evaluator.add_rule(Equal(a,1/match_int(c)),Equal(a*c,1))
        evaluator.add_rule(a<b*match_int(c)**-1,a*c<b)
        evaluator.add_rule(Equal(a,match_int(c)**-1),Equal(a*c,1))
        evaluator.add_rule(a<match_int(c)**-1,a*c<1)
        evaluator.add_rule(a<-match_int(c)**-1,a*c<-1)
        evaluator.add_rule(a<b+match_int(c)**-1,a-b<c**-1)

        evaluator.add_rule(match_int(a)+b*match_int(c)**-1,c**-1*(a*c+b))
        evaluator.add_rule(match_int(a)+match_int(c)**-1,c**-1*(a*c+1))
        evaluator.add_rule(match_int(a)-b*match_int(c)**-1,c**-1*(a*c-b))
        evaluator.add_rule(match_int(a)-match_int(c)**-1,c**-1*(a*c-1))

        evaluator.add_rule(match_int(a)<match_int(b),c,calculate_less)   
        evaluator.add_rule(match_int(a)<-match_int(b),c,calculate_rhs_negative_less)   
        evaluator.add_rule(-match_int(a)<match_int(b),c,calculate_lhs_negative_less)   

        include.remove('logic')
        
    if len(include)!=0:
        raise AttributeError('add_numeric_evaluation_rules got unknown includes: ' % include )
    
def add_fraction_reduction_rules(evaluator):
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    
    def extract_multiplication(m):
        ex,ey = m[x],m[y]
        if not any(e.function == Exponentiation for e in [ex,ey]):
            return False
        ma = MulplicityList(ex,MultiplicationGroup,Exponentiation,RealField)
        mb = MulplicityList(ey,MultiplicationGroup,Exponentiation,RealField)
        
        m[c] = (ma+mb).as_expression()

    evaluator.add_rule(1/x,x**-1)
    evaluator.add_rule(x*y,c,extract_multiplication)

def add_main_evaluator_rules(evaluator):
    
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')

    def extract_intersection(m):
                
        ma = MulplicityList(m[x],MultiplicationGroup,Exponentiation,RealField)
        mb = MulplicityList(m[y],MultiplicationGroup,Exponentiation,RealField)
        
        common = ma.intersection(mb)
        if len(common) == 0:
            return False
        m[a] = (ma-common).as_expression()
        m[b] = (mb-common).as_expression()
        m[c] = common.as_expression()

    evaluator.add_rule(1/x,x**-1)
    
    evaluator.add_rule(x*x,x**2)
    evaluator.add_rule(x**n/x,x**(n-1))
    evaluator.add_rule(x*x**n,x**(n+1))
    evaluator.add_rule(1/x**n,x**(-n))
    evaluator.add_rule((1/x)**n,x**(-n))
    evaluator.add_rule(x**m*x**n,x**(m+n))
    evaluator.add_rule(x**m/x**n,x**(m-n))
    evaluator.add_rule((x**m)**n,x**(m*n))    
    
    evaluator.add_rule(x+y,c*(a+b),extract_intersection)
    evaluator.add_rule(x-y,c*(a-b),extract_intersection)
    evaluator.add_rule(-x-y,-c*(a+b),extract_intersection)
    
    evaluator.add_rule(x**m*y**m,(x*y)**(m))
    
    pp = PiecewisePart
    evaluator.add_rule(Piecewise((a,b),x),Piecewise(pp(a,b),x))
    evaluator.add_rule(Piecewise(x,(a,b)),Piecewise(x,pp(a,b)))

    evaluator.add_rule(Piecewise(pp(a,True),x),pp(a,True))
    evaluator.add_rule(Piecewise(pp(a,False),x),x)
    evaluator.add_rule(Piecewise(x,pp(a,False)),x)

def add_basic_derivative_rules(evaluator):

    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')

    def contains_no_x(m):
        vx = m[x]
        for e in postorder_traversal(m[y]):
            if e == vx:
                return False
        return True
    
    evaluator.add_rule(derivative(x,x),1) 
    evaluator.add_rule(derivative(y,x),0,contains_no_x);

    evaluator.add_rule(derivative(a+b,x),derivative(a,x)+derivative(b,x))
    evaluator.add_rule(derivative(a*b,x),derivative(a,x)*b+derivative(b,x)*a)
    evaluator.add_rule(derivative(-x,x),-1)
    evaluator.add_rule(derivative(1/x,x),-x**-2)
    evaluator.add_rule(derivative(log(x),x),1/x)


    evaluator.add_rule(derivative(sin(x),x),cos(x))
    evaluator.add_rule(derivative(cos(x),x),-sin(x))

    evaluator.add_rule(derivative(x**match_int(n),x),n*x**(n-1));
    evaluator.add_rule(derivative(a**b,x),derivative(b*log(a),x)*a**b);
    
    
    f = WildcardFunction("f")
    g = WildcardFunction("g")
    
    def create_tmp_x(m):
        m[c] = tmp(m[x]) 
    
    evaluator.add_rule( derivative(f(g(x)),x) , 
                        evaluated_at( derivative(f(c),c), c, g(x) ) * derivative(g(x),x),
                        create_tmp_x);
    
    def do_evaluate(m):
        p = m[f(x)]
        if p.is_function and p.function == derivative:
            return False
    
    evaluator.add_rule(evaluated_at( f(x), x, c ), f(c), do_evaluate )
    
def add_expand_rules(evaluator):
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    
    evaluator.add_rule(a*(x+y),a*x+a*y)
    evaluator.add_rule((x*y)**(m),x**m*y**m)
    evaluator.add_rule((x**y)**(z),x**(y*z))

    evaluator.add_rule((x+y)**match_int(n),(x+y) * (x+y)**(n-1))
    
    no_sum = MatchCondition('no_sum',lambda e:e.is_atomic or e.function != Addition)
    
    evaluator.add_rule(no_sum(x)*x,x**2)
    evaluator.add_rule(no_sum(x)**n*x,x**(n+1))
    evaluator.add_rule(no_sum(x)**n*x**m,x**(n+m))

    def extract_intersection(m):
                
        ma = MulplicityList(m[x],MultiplicationGroup,Exponentiation,RealField)
        mb = MulplicityList(m[y],MultiplicationGroup,Exponentiation,RealField)
        
        common = ma.intersection(mb)
        if len(common) == 0:
            return False
        
        sa = (ma-common)
        sb = (mb-common)
        
        for arg in sa:
            if not isinstance(arg[0].value,integer_type):
                return False
            
        for arg in sb:
            if not isinstance(arg[0].value,integer_type):
                return False
        
        m[a] = sa.as_expression()
        m[b] = sb.as_expression()
        m[c] = common.as_expression()
    
    evaluator.add_rule(x+y,c*(a+b),extract_intersection)
    evaluator.add_rule(x-y,c*(a-b),extract_intersection)
    evaluator.add_rule(-x-y,-c*(a+b),extract_intersection)

def add_intermediate_evaluator_rules(evaluator):
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')
    evaluator.add_rule(Piecewise(PiecewisePart(a,b),x),Piecewise((a,b),x))
    evaluator.add_rule(Piecewise(x,PiecewisePart(a,b)),Piecewise(x,(a,b)))


def add_type_evaluator_rules(evaluator):
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')

    ordered_types = (Types.Boolean,Types.Natural,Types.Integer,Types.Rational,Types.Real,Types.Complex)

    for i in range(len(ordered_types)):
        evaluator.add_rule(DominantType(Types.Imaginary,ordered_types[i]),Types.Complex)
        evaluator.add_rule(Type(ordered_types[i]),Types.Type)
        for j in range(i):
          evaluator.add_rule(DominantType(ordered_types[j],ordered_types[i]),ordered_types[i])

    evaluator.add_rule(Type(Types.Imaginary*Types.Complex),Types.Complex)
    evaluator.add_rule(DominantType(x,x),x)

    evaluator.add_rule(Type(Types.Imaginary),Types.Type)
    evaluator.add_rule(Type(Type(x)),Types.Type)

    evaluator.add_rule(Type(True),Types.Boolean)
    evaluator.add_rule(Type(False),Types.Boolean)
    evaluator.add_rule(Type(match_int(x)),Types.Natural)
    evaluator.add_rule(Type(1/x),DominantType(Types.Rational,Type(x)))

    evaluator.add_rule(Type(x**y),OperationType(Type(x)**Type(y)))
    evaluator.add_rule(Type(x*y),OperationType(Type(x)*Type(y)))

    for t in (Types.Natural,Types.Integer,Types.Rational,Types.Real):
        evaluator.add_rule(OperationType(Types.Imaginary*t),Types.Imaginary)
    evaluator.add_rule(OperationType(Types.Imaginary*Types.Imaginary),Types.Real)

    evaluator.add_rule(OperationType(x**Types.Natural),x)
    evaluator.add_rule(OperationType(x**Types.Integer),DominantType(x,Types.Rational))
    evaluator.add_rule(OperationType(Types.Natural**Types.Rational),Types.Real)

    for t in (Types.Rational,Types.Real,Types.Complex):
        evaluator.add_rule(OperationType(x**t),Types.Complex)

    no_type_function = MatchCondition('no_type_function',lambda e:e.function != Type)

    evaluator.add_rule(OperationType(no_type_function(x)*no_type_function(y)),DominantType(x,y))
    evaluator.add_rule(OperationType(no_type_function(x)**no_type_function(y)),DominantType(x,y))

    evaluator.add_rule(Type(x+y),DominantType(Type(x),Type(y)))
    evaluator.add_rule(Type(-x),DominantType(Types.Integer,Type(x)))
    evaluator.add_rule(Type(pi),Types.Real)
    evaluator.add_rule(Type(e),Types.Real)
    evaluator.add_rule(Type(I),Types.Imaginary)

    evaluator.add_rule(Type(factorial(x)),Types.Natural)
    evaluator.add_rule(Type(sign(x)),Types.Integer)
    evaluator.add_rule(Type(Floor(x)),Types.Integer)
    evaluator.add_rule(Type(Abs(x)),Types.Natural)
    evaluator.add_rule(Type(Ceil(x)),Types.Integer)
    evaluator.add_rule(Type(Round(x)),Types.Integer)
    evaluator.add_rule(Type(Mod(x,y)),Types.Integer)

    evaluator.add_rule(Type(Real(x)),Types.Real)
    evaluator.add_rule(Type(Imag(x)),Types.Real)
    evaluator.add_rule(Type(Conjugate(x)),Types.Complex)

    evaluator.add_rule(Type(Indicator(x)),Types.Natural)
    evaluator.add_rule(Type(Piecewise(a,b)),DominantType(Type(a),Type(b)))
    evaluator.add_rule(Type(PiecewisePart(a,b)),Type(a))

    evaluator.add_rule(Type(derivative(x,y)),Type(x))
    evaluator.add_rule(Type(evaluated_at(x,y,z)),DominantType(Type(x),Type(z)))

    evaluator.add_rule(Type(tmp(x)),Type(x))
    evaluator.add_rule(Type(sqrt(x)),Type(x**(1/S(2))))

    evaluator.add_rule(Type(atan2(x,y)),DominantType(Type(x),Type(x),Types.Rational))

    for f in [exp,log,sin,cos,asin ,acos,tan,atan,cot,acot,sinh,cosh,asinh,acosh,tanh,atanh,coth,acoth]:
        evaluator.add_rule(Type(f(x)),DominantType(Type(x),Types.Rational))


def add_final_evaluator_rules(evaluator):
    
    x,y,z,a,b,c,n,m = wildcard_symbols('x,y,z,a,b,c,n,m')

    evaluator.add_rule(x**-1,1/x)
    evaluator.add_rule((1/x)**y,1/x**y)
    
    evaluator.add_rule((1/x)*(1/y),1/(x*y))
    
    evaluator.add_rule((a/match_int(b))**x,a**x/b**x)

    evaluator.add_rule(x**-(y),1/x**y)
    
    #evaluator.add_rule(x**m*y**m,(x*y)**(m))
    evaluator.add_rule(e**x,exp(x))
    evaluator.add_rule(x**(1/S(2)),sqrt(x))
    
    evaluator.add_rule(PiecewisePart(x,y),x*Indicator(y))


add_prepare_evaluator_rules(prepare_evaluator)

add_numeric_evaluation_rules(main_evaluator,{'sum','product','power','logic'})
add_basic_simplification_rules(main_evaluator)  
add_logic_rules(main_evaluator)
add_main_evaluator_rules(main_evaluator)
add_fraction_reduction_rules(main_evaluator)
add_basic_derivative_rules(main_evaluator)

add_type_evaluator_rules(type_evaluator)

add_fraction_reduction_rules(primefactor_evaluator)
add_numeric_evaluation_rules(primefactor_evaluator,{'sum','primes'})

add_intermediate_evaluator_rules(intermediate_evaluator)

add_basic_simplification_rules(expand_evaluator)  
add_numeric_evaluation_rules(expand_evaluator,{'sum','product','power','logic'})
add_expand_rules(expand_evaluator)

add_basic_simplification_rules(final_evaluator)  
add_logic_rules(final_evaluator)  
add_final_evaluator_rules(final_evaluator)
add_numeric_evaluation_rules(final_evaluator,{'sum','product','power','logic'})


def evaluate(expr,context = global_context):
    main = MultiEvaluator(recursive=True)
    main.add_evaluator(context)
    main.add_evaluator(main_evaluator)
    types = MultiEvaluator(recursive=True)
    types.add_evaluator(context)
    types.add_evaluator(type_evaluator)
    return final_evaluator(intermediate_evaluator(types(main(primefactor_evaluator(prepare_evaluator(expr))))))
    
def expand(expr):
    return final_evaluator(intermediate_evaluator(expand_evaluator(prepare_evaluator(expr))))




