
from expression import *
from functions import *
from expresso.visitor import add_target,add_target_obj

@add_target(printer,addition)
@add_target(latex,addition)
def visit(printer,expr):
    neg_args = [arg for arg in expr.args if arg.function == negative]
    covered = set(neg_args)
    rest = [arg for  arg in expr.args if arg not in covered]
    rest_str = '+'.join(printer.printed_operator_arguments(expr,rest))
    if len(neg_args) == 0:
        return rest_str
    neg_str = '-'.join(printer.printed_operator_arguments(expr,[arg.args[0] for arg in neg_args]))
    return rest_str + '-' + neg_str

@add_target(latex,multiplication)
def visit(printer,expr):
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
            rest_str = ''
        elif len(rest) == 1:
            rest_str = printer(rest[0])
            if printer.needs_brackets_in(rest[0],expr):
                rest_str = printer.bracket_format() % rest_str
        else:
            rest = multiplication(*rest)
            rest_str =  printer(rest)

        return r'\frac{%s}{%s} \, %s ' % (num_str,denom_str,rest_str)

    needs_dot = lambda x: type(x.value)==Number

    need_dot = [x for x in expr.args if needs_dot(x)]
    rest     = [x for x in expr.args if not needs_dot(x)]

    res  = '\cdot '.join(printer.printed_operator_arguments(expr,need_dot))
    if len(need_dot) > 1:
        res += '\cdot '
    elif len(need_dot) == 1:
        res += '\, '
    res += '\, '.join(printer.printed_operator_arguments(expr,rest))

    return res

@add_target_obj(printer,SymbolicConstant)
@add_target_obj(latex,SymbolicConstant)
def visit(printer,expr):
    return expr.value.name

@add_target(printer,multiplication)
def visit(printer,expr):
    res = printer.print_operator_argument(expr.args[0],expr)
    for arg in expr.args[1:]:
        if arg.function == fraction:
            res += '/' + printer.print_operator_argument(arg.args[0],expr)
        else:
            res += '*' + printer.print_operator_argument(arg,expr)
    return res

@add_target(latex,fraction)
def visit(printer,expr):
    return r'\frac{1}{%s}' % printer(expr.args[0])

@add_target(latex,exponentiation)
def visit(printer,expr):
    parg = printer.printed_operator_arguments(expr,end=-1)
    parg += [printer(expr.args[-1])]
    return '^'.join(['{%s}' % arg for arg in parg])

@latex.register_target(derivative)
def visit(printer,expr):

    def flatten(expr,arguments=[]):
        arguments.append(expr.args[1])
        inner = expr.args[0]
        if inner.function == derivative:
            return flatten(inner,arguments)
        return inner,arguments
    
    inner,arguments = flatten(expr)
    
    argc = len(arguments)
    mul_args = [(arguments[0],1)]
    for arg in arguments[1:]:
        if mul_args[-1][0] == arg:
            mul_args[-1] = (arg,mul_args[-1][1]+1)
        else:
            mul_args.append((arg,1))
        
    formatted_arguments = [r'{\partial %s}' % printer(arg[0]) if arg[1] == 1 else r'{\partial{%s}^{%s}}' % (printer(arg[0]),arg[1]) for arg in mul_args]
    
    if argc == 1:
        formated_inner = '\partial %s' % printer(inner)
    else:
        formated_inner = '\partial^{%s} %s' % (argc,printer(inner))
    
    return r'\frac{%s}{%s}' % (formated_inner,'\,'.join(formatted_arguments))

@add_target(latex,evaluated_at)
def visit(printer,expr):
    return r'\left[ %s \right]_{%s = %s}' % tuple(printer(arg) for arg in expr.args)


latex.register_printer(pi,lambda p,e:r'\pi ')
latex.register_printer(e,lambda p,e:r'e ')
latex.register_printer(oo,lambda p,e:r'\infty ')

latex.register_printer(I,lambda p,e:r'i ')
printer.register_printer(I,lambda p,e:r'i')


latex.register_printer(Abs,lambda p,e: r"\left| %s \right|" % p(e.args[0]))

latex.register_printer(conjugate,lambda p,e: r"\overline{ %s }" % p(e.args[0]))


latex.register_printer(tmp,lambda p,e: p.print_postfix_operator(e,r"'"))
latex.register_printer(sqrt,lambda p,e:r"\sqrt{%s} " % p(e.args[0]))

latex.register_printer(real,lambda p,e: p.print_function(e,r"\Re "))
latex.register_printer(imag,lambda p,e: p.print_function(e,r"\Im "))

latex.register_printer(Indicator,lambda p,e: p.print_function(e,r"\, \mathbb{1} "))
latex.register_printer(Not,lambda p,e: p.print_unary_operator(e,r"\neg "))

latex.register_printer(LessEqual,lambda p,e: p.print_binary_operator(e,r"\le "))
latex.register_printer(GreaterEqual,lambda p,e: p.print_binary_operator(e,r"\ge "))
latex.register_printer(Or,lambda p,e: p.print_binary_operator(e,r"\lor "))
latex.register_printer(And,lambda p,e: p.print_binary_operator(e,r"\land "))
latex.register_printer(Xor,lambda p,e: p.print_binary_operator(e,r"\veebar "))


latex.register_printer(Tuple,lambda p,e: p.print_function(e,r""))

@add_target(latex, InnerPiecewise)
def visit(printer,expr):   
    for arg in expr.args:
        if arg.function != Tuple:
            return printer.print_function(expr,name=r"\text{piecewise} ")

    outer = r"\begin{cases} %s \end{cases}"

    s = len(expr.args)-1
    inner_list = [r"%s & \text{if } %s " % (printer(e.args[0]),printer(e.args[1]))
                         for e in expr.args[:-1]]
    e = expr.args[-1]

    if e.args[1] == S(True):
        inner_list += [ r"%s & \text{otherwise } " % printer(e.args[0])]
    else:
        inner_list += [ r"%s & \text{if } %s " % (printer(e.args[0]),printer(e.args[1])) ]

    inner = r"\\ ".join(inner_list)
    return outer % inner

@add_target(printer, OuterPiecewise )
@add_target(latex, OuterPiecewise )
def visit(printer,expr):
    return printer(expr.args[0])


@add_target(printer, InnerPiecewise)
def visit(printer,expr):
    return printer.print_function(expr,name="piecewise")

@add_target(latex,unequal)
def visit(printer,expr):
    return printer.print_binary_operator(expr,r" \neq ")

@add_target(latex,CustomFunction)
def visit(printer,expr):
    name = expr.args[0].name.replace('<',r'< ').replace('>',r' >')
    return printer(Function(name)(*expr.args[1:]))

@add_target(latex,ArrayAccess)
def visit(printer,expr):
    name = printer.format_name(expr.args[0].name.split('__id')[0])
    return r'%s \mathopen{} \left[ %s \right] \mathclose{} ' % (name,','.join([printer(arg) for arg in expr.args[1:]]))

@add_target(printer,ArrayAccess)
def visit(printer,expr):
    name = printer.format_name(expr.args[0].name.split('__id')[0])
    return r'%s[%s]' % (name,','.join([printer(arg) for arg in expr.args[1:]]))

@add_target_obj(printer, Number)
def visit(printer,expr):
    v = expr.value

    if v == 0:
        return '0'

    o = v
    exp = 0

    while (v % 10) == 0:
        v = v/10
        exp = exp + 1
    if exp > 1:
        if v != 1:
            return r"%se%s" % (v,exp)
        if v == 1:
            return r"1e%s" % exp

    return str(o)

@add_target_obj(latex, Number)
def visit(printer,expr):
    v = expr.value

    if v == 0:
        return '0'

    o = v
    exp = 0

    while (v % 10) == 0:
        v = v/10
        exp = exp + 1
    if exp > 1:
        if v != 1:
            return r"%s \cdot 10^{%s}" % (v,exp)
        if v == 1:
            return r"10^{%s}" % exp

    return str(o)

@add_target_obj(printer, TypeInfo)
@add_target_obj(latex, TypeInfo)
def visit(printer,expr):
    return printer.format_name(expr.value.name)

from mpmath import mp
@add_target_obj(printer, bool)
@add_target_obj(latex, bool)
def visit(printer,expr):
    return printer.format_name(str(expr.value))

@add_target_obj(latex, mp.mpc)
@add_target_obj(latex, mp.mpf)
def visit(printer,expr):
    return r"%s \; " % str(expr.value)

@add_target_obj(printer, mp.mpc)
@add_target_obj(printer, mp.mpf)
def visit(printer,expr):
    return r"%s" % str(expr.value)













