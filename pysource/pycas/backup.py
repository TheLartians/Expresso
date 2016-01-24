derivative = Function('derivative')
log = Function('log')
sin = Function('sin')
cos = Function('cos')
e = S('e')

def exp(x):
    return e**x

RightArrow = BinaryOperator(" -> ",0)

def __latex_print_rightarrow(printer,expr):
    parg = printer._printed_operator_arguments(expr)
    return r'\; \rightarrow \; '.join(['{%s}' % arg for arg in parg])

latex.register_printer(RightArrow,__latex_print_rightarrow)

def __latex_print_derivative(printer,expr):
    #return printer(UnaryOperator("\partial_{%s}" % printer(expr.args[1]),ex.core.prefix,-100)(expr.args[0]))
    return printer._function_format() % (r"\partial_{%s}" % printer(expr.args[1]),printer(expr.args[0]))

latex.register_printer(derivative,__latex_print_derivative)


prepare_evaluator = RuleEvaluator()
evaluator = RuleEvaluator(recusive=True)
final_evaluator = RuleEvaluator(recusive=True)

def primesfrom2to(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    import numpy as np
    sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in xrange(int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)/3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))/3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

primes = [int(x) for x in primesfrom2to(1000000)]

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

def add_rules():
    
    x = W('x')
    y = W('y')
    z = W('z')
    a = W('a')
    b = W('b')
    c = W('c')
    n = W('n')
    m = W('m')

    def match_int(expr):
        return isinstance(expr.value,int)
    match_int = MatchCondition('match_int',match_int)
    
    def match_atomic(expr):
        return expr.is_atomic
    match_atomic = MatchCondition('match_atomic',match_atomic)

    def do_not_match_x(expr):
        return expr != x
    do_not_match_x = MatchCondition('do_not_match_x',do_not_match_x)

    def calculate_sum(m):
        m[z] = m[x].value + m[y].value

    def calculate_difference(m):
        m[z] = m[x].value - m[y].value

    def calculate_product(m):
        m[z] = m[x].value * m[y].value
        
    def calculate_power(m):
        res = m[x].value **  m[y].value
        if not isinstance(res,int):
            return False
        m[z] = res

    def extract_primes(m):
        primes = primefactor(m[a].value)
        if len(primes) <= 1:
            return False
        m[b] = Multiplication(*primes)
    
    '''
    def extract_intersection(m):
                
        ma = MulplicityList(m[x],MultiplicationGroup,Exponentiation,RealField)
        mb = MulplicityList(m[y],MultiplicationGroup,Exponentiation,RealField)
        
        common = ma.intersection(mb)
        if len(common) == 0:
            return False
        m[a] = (ma-common).as_expression()
        m[b] = (mb-common).as_expression()
        m[c] = common.as_expression()
    '''
    
    def extract_complete_intersection(m):
                
        ma = MulplicityList(m[x],MultiplicationGroup,Exponentiation,RealField)
        mb = MulplicityList(m[y],MultiplicationGroup,Exponentiation,RealField)
        
        def get_inner(a,b):
            return a if a<b else b
            
            va = a.value
            vb = b.value
            if va is not None and vb is not None:
                return a if va<vb else b
            
            aargs = MulplicityList(a, AdditionGroup, Multiplication, RealField)
            bargs = MulplicityList(b, AdditionGroup, Multiplication, RealField)
            
            def get_inner_2(a,b):
                return a if a<b else b
            
            intersection = aargs.intersection(bargs,get_inner_2)
            if len(intersection) == 0:
                return None
            
            return intersection.as_expression();            
        
        common = ma.intersection(mb,get_inner)
        if len(common) == 0:
            return False
        m[a] = (ma-common).as_expression()
        m[b] = (mb-common).as_expression()
        m[c] = common.as_expression()

    def extract_intersection(m):
                
        ma = MulplicityList(m[x],MultiplicationGroup,Exponentiation,RealField)
        mb = MulplicityList(m[y],MultiplicationGroup,Exponentiation,RealField)
        
        common = ma.intersection(mb)
        if len(common) == 0:
            return False
        m[a] = (ma-common).as_expression()
        m[b] = (mb-common).as_expression()
        m[c] = common.as_expression()

    
    def prepare(m):
        mx = MulplicityList(m[x],MultiplicationGroup,Exponentiation,RealField)
        m[y] = mx.as_expression()
    
    prepare_evaluator.add_rule(x,y,prepare)
    
    evaluator.add_rule(x+0,x)
    evaluator.add_rule(x-x,0)
    evaluator.add_rule(x*1,x)
    evaluator.add_rule(x*0,0)
    evaluator.add_rule(x/x,1)
    
    evaluator.add_rule(match_int(x)+match_int(y),z,calculate_sum)
    evaluator.add_rule(-match_int(x)-match_int(y),-z,calculate_sum)
    evaluator.add_rule(match_int(x)-match_int(y),z,calculate_difference)
    #evaluator.add_rule(match_int(x)*match_int(y),z,calculate_product)
    #evaluator.add_rule(match_int(x)**match_int(y),z,calculate_power)

    evaluator.add_rule(match_int(a),b,extract_primes)

    evaluator.add_rule(x**1,x)
    evaluator.add_rule(x**0,1)
    evaluator.add_rule(1**x,1)
    evaluator.add_rule(0**x,0)

    evaluator.add_rule(x*x,x**2)
    evaluator.add_rule(x*x**-1,1)

    evaluator.add_rule(x**n/x,x**(n-1))
    evaluator.add_rule(1/x,x**-1)
    evaluator.add_rule(x*x**n,x**(n+1))
    evaluator.add_rule(1/x**n,x**(-n))
    evaluator.add_rule((1/x)**n,x**(-n))
    evaluator.add_rule(x**m*x**n,x**(m+n))
    evaluator.add_rule(x**m*y**m,(x*y)**(m))
    evaluator.add_rule((x**m)**n,x**(m*n))
    
    evaluator.add_rule(x+x,2*x)
    evaluator.add_rule(-(x+y),-x-y)
    evaluator.add_rule(x*-1,-x)
    evaluator.add_rule(-(-x),x)
    evaluator.add_rule((-x)*y,-(x*y))

    
    evaluator.add_rule(x+y,c*(a+b),extract_intersection)
    evaluator.add_rule(x-y,c*(a-b),extract_intersection)
    
    evaluator.add_rule(x*y**-1,a*b**-1,extract_complete_intersection)
    evaluator.add_rule(x**n*y**-n,a**n*b**-n,extract_complete_intersection)
    
    evaluator.add_rule(log(e),1)
    
    evaluator.add_rule(derivative(x,x),1)
    evaluator.add_rule(derivative(do_not_match_x(match_atomic(a)),x),0)
    evaluator.add_rule(derivative(match_int(n),x),0)
    evaluator.add_rule(derivative(a+b,x),derivative(a,x)+derivative(b,x))
    evaluator.add_rule(derivative(a*b,x),derivative(a,x)*b+derivative(b,x)*a)
    evaluator.add_rule(derivative(-x,x),-1)
    evaluator.add_rule(derivative(1/x,x),-x**-2)
    evaluator.add_rule(derivative(log(x),x),1/x)

    evaluator.add_rule(derivative(sin(x),x),cos(x))
    evaluator.add_rule(derivative(cos(x),x),-sin(x))
    
    evaluator.add_rule(derivative(x**match_int(n),x),n*x**(n-1));
    evaluator.add_rule(derivative(a**b,x),derivative(b*log(a),x)*a**b);
    
    def contains_no_x(m):
        vx = m[x]
        for e in postorder_traversal(m[y]):
            if e == vx:
                return False
        return True
    
    evaluator.add_rule(derivative(y,x),0,contains_no_x);

    
    f = WF("f")
    g = WF("g")
    
    evaluator.add_rule( derivative(f(g(x)),x) , derivative(f(g(x)),g(x))*derivative(g(x),x) );

    final_evaluator.add_rule(x**-1,1/x)
    final_evaluator.add_rule(x**1,x)
    final_evaluator.add_rule(x**-(y),1/x**y)
    final_evaluator.add_rule(1/x*(1/y),1/(x*y))
    final_evaluator.add_rule(x*-y,-(x*y))
    final_evaluator.add_rule(-a-b,-(a+b))
    
    final_evaluator.add_rule(match_int(x)+match_int(y),z,calculate_sum)
    final_evaluator.add_rule(-match_int(x)-match_int(y),-z,calculate_sum)
    final_evaluator.add_rule(match_int(x)-match_int(y),z,calculate_difference)
    final_evaluator.add_rule(match_int(x)*match_int(y),z,calculate_product)
    final_evaluator.add_rule(match_int(x)**match_int(y),z,calculate_power)

add_rules()

def print_steps(r,m):
    from IPython.display import display
    print "Apply rule %s: %s -> %s" % (r,r.search.replace(m),r.replacement.replace(m))
    #display(RightArrow(r.search.replace(m),r.replacement.replace(m)))
    
evaluator.set_apply_callback(None)
final_evaluator.set_apply_callback(None)



def evaluate(expr):
    return RightArrow(expr,final_evaluator(evaluator(prepare_evaluator(expr))))
