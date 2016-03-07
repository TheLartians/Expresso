
import expresso.pycas as pc
import rule_symbols as s

logic_evaluator = pc.RewriteEvaluator(recursive=True,split_binary=True)
evaluator = logic_evaluator

is_atomic = pc.Function('is atomic')
is_symbol = pc.Function('is symbol')
is_explicit_natural = pc.Function('is explicit natural')
is_pure_numeric = pc.Function('is pure numeric')
is_numeric = pc.Function('is numeric')
is_mpmath= pc.Function('is mpmath')

is_function = pc.Function('is function')
contains_atomic = pc.Function('contains atomic')

def is_explicit_natural_evaluator(m):
    for expr in m:
        if not isinstance(expr[1].value, pc.Number):
            m[s.y] = False
            return
    m[s.y] = True

logic_evaluator.add_rule(is_explicit_natural(s.x),s.y,is_explicit_natural_evaluator)

def is_function_type(expr,function):
    return is_function(expr,pc.expresso.create_object(function))

def comp_function_evaluator(m):
    f =  m[s.x].function
    if f is None or (isinstance(m[s.y].value,pc.Function) and f != m[s.y].value):
        m[s.z] = False
        return
    m[s.z] = True

logic_evaluator.add_rule(is_function(s.x,s.y),s.z,comp_function_evaluator)

def is_atomic_evaluator(m):
    if not m[s.x].is_atomic:
        m[s.y] = False
        return
    m[s.y] = True

logic_evaluator.add_rule(is_atomic(s.x),s.y,is_atomic_evaluator)
logic_evaluator.add_rule(is_function(s.x),pc.Not(is_atomic(s.x)))

def is_symbol_evaluator(m):
    if not m[s.x].is_symbol:
        m[s.y] = False
        return
    m[s.y] = True

logic_evaluator.add_rule(is_symbol(s.x),s.y,is_symbol_evaluator)


def is_pure_numeric_evaluator(m):
    m[s.y] = pc.And(*[is_pure_numeric(arg) for arg in m[s.x].args])

logic_evaluator.add_rule(is_pure_numeric(s.x),True,condition=is_explicit_natural(s.x))
logic_evaluator.add_rule(is_pure_numeric(s.x),False,condition=is_symbol(s.x))
logic_evaluator.add_rule(is_pure_numeric(s.x),s.y,is_pure_numeric_evaluator,condition=is_function(s.x))

from mpmath import mp

def is_mpmath_evaluator(m):
    v = m[s.x].value
    if isinstance(v,(mp.mpf,mp.mpc)):
        m[s.y] = True
    else:
        m[s.y] = False

logic_evaluator.add_rule(is_mpmath(s.x),s.y,is_mpmath_evaluator,condition=is_atomic(s.x))

def is_numeric_evaluator(m):
    m[s.y] = pc.And(*[is_numeric(arg) for arg in m[s.x].args])

logic_evaluator.add_rule(is_numeric(s.x),True,condition=is_explicit_natural(s.x))
logic_evaluator.add_rule(is_numeric(pc.pi),True)
logic_evaluator.add_rule(is_numeric(pc.e),True)
logic_evaluator.add_rule(is_numeric(pc.I),True)
logic_evaluator.add_rule(is_numeric(s.x),False,condition=is_symbol(s.x))
logic_evaluator.add_rule(is_numeric(s.x),s.y,is_numeric_evaluator,condition=is_function(s.x))


def contains_evaluator(m):
    search = m[s.y]
    m[s.z] = pc.Or(*[contains_atomic(arg, search) for arg in m[s.x].args])

logic_evaluator.add_rule(contains_atomic(s.x, s.x), True)
logic_evaluator.add_rule(contains_atomic(s.x, s.y), False, condition=is_atomic(s.x))
logic_evaluator.add_rule(contains_atomic(s.x, s.y), s.z, contains_evaluator, condition=is_function(s.x))


logic_evaluator.add_rule(pc.Xor(False,False),False);
logic_evaluator.add_rule(pc.Xor(True,False),True);
logic_evaluator.add_rule(pc.Xor(False,True),True);
logic_evaluator.add_rule(pc.Xor(True,True),False);

logic_evaluator.add_rule(pc.Not(True),False);
logic_evaluator.add_rule(pc.Not(False),True);

logic_evaluator.add_rule(pc.And(s.x,True),s.x);
logic_evaluator.add_rule(pc.And(s.x,False),False);
logic_evaluator.add_rule(pc.Or(s.x,True),True);
logic_evaluator.add_rule(pc.Or(s.x,False),s.x);

logic_evaluator.add_rule(pc.Not(pc.Not(s.x)),s.x);
logic_evaluator.add_rule(pc.And(s.x,pc.Not(s.x)),False);
logic_evaluator.add_rule(pc.Or(s.x,pc.Not(s.x)),True);

logic_evaluator.add_rule(pc.And(s.x,s.x),s.x);
logic_evaluator.add_rule(pc.Or(s.x,s.x),s.x);

logic_evaluator.add_rule(pc.equal(s.x,s.x),True);


logic_evaluator.add_rule(-s.x<-s.y,s.x<s.y)

'''
logic_evaluator.add_rule(s.x<s.x,False);
logic_evaluator.add_rule(-s.x<-s.y,s.y<s.x);
logic_evaluator.add_rule(s.x<-s.x,False);
logic_evaluator.add_rule(-s.x<s.x,False);
logic_evaluator.add_rule(s.x<=s.x,True);

logic_evaluator.add_rule(s.x>s.y,s.y<s.x);
logic_evaluator.add_rule(s.x>=s.y,s.y<=s.x);
logic_evaluator.add_rule(s.x<=s.y,pc.Or(s.x<s.y,equal(s.x,s.y)));

logic_evaluator.add_rule(And(x<y,y<x),False);

logic_evaluator.add_rule(x<oo,True);
logic_evaluator.add_rule(-oo<x,True);
logic_evaluator.add_rule(oo<x,False);
logic_evaluator.add_rule(x<-oo,False);

logic_evaluator.add_rule(And(equal(x,y),f(x)),And(equal(x,y),f(y)));
logic_evaluator.add_rule(And(Or(x,y),z),Or(And(x,z),And(y,z)));

logic_evaluator.add_rule(abs(x)<y,And(x<y,-x<y));
logic_evaluator.add_rule(x<abs(y),Or(x<y,x<-y));
logic_evaluator.add_rule(equal(abs(x),y),Or(equal(x,y),equal(x,-y)));
'''