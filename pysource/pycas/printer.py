
from expression import *
from functions import *
from pysymbols.visitor import add_target


@add_target(latex,Addition)
def visit(printer,expr):
    neg_args = [arg for arg in expr.args if arg.function == Negative]
    covered = set(neg_args)
    rest = [arg for  arg in expr.args if arg not in covered]
    rest_str = '+'.join(printer.printed_operator_arguments(expr,rest))
    if len(neg_args) == 0:
        return rest_str
    neg_str = '-'.join(printer.printed_operator_arguments(expr,[arg.args[0] for arg in neg_args]))
    return rest_str + '-' + neg_str

@add_target(latex,Multiplication)
def visit(printer,expr):
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
            if printer.needs_brackets_in(rest[0],expr):
                rest_str = printer.bracket_format() % rest_str
        else:
            rest = Multiplication(*rest)
            rest_str =  printer(rest)

        return r'\frac{%s}{%s} \, %s ' % (num_str,denom_str,rest_str)

    is_numeric = lambda x: x.value != None or (x.function == Exponentiation and x.args[0].value != None)

    numeric = [x for x in expr.args if is_numeric(x)]
    non_numeric = [x for x in expr.args if not is_numeric(x)]

    res  = '\cdot '.join(printer.printed_operator_arguments(expr,numeric)) 
    res += '\, '.join(printer.printed_operator_arguments(expr,non_numeric))

    return res

@add_target(latex,Fraction)
def visit(printer,expr):
    return r'\frac{1}{%s}' % printer(expr.args[0])

@add_target(latex,Exponentiation)
def visit(printer,expr):
    parg = printer.printed_operator_arguments(expr,end=-1)
    parg += [printer(expr.args[-1])]
    return '^'.join(['{%s}' % arg for arg in parg])

@add_target(latex,derivative)
def visit(printer,expr):
    return printer.function_format() % (r"\partial_{%s}" % printer(expr.args[1]),printer(expr.args[0]))

@add_target(latex,evaluated_at)
def visit(printer,expr):
    return r'\left[ %s \right]_{%s = %s}' % tuple(printer(arg) for arg in expr.args)


latex.register_printer(pi,lambda p,e:r'\pi ')
latex.register_printer(oo,lambda p,e:r'\infty ')

latex.register_printer(I,lambda p,e:r'i ')
printer.register_printer(I,lambda p,e:r'i')


latex.register_printer(Abs,lambda p,e: r"\left| %s \right|" % p(e.args[0]))


latex.register_printer(tmp,lambda p,e: p.print_postfix_operator(e,r"'"))
latex.register_printer(sqrt,lambda p,e:r"\sqrt{%s} '" % p(e.args[0]))

latex.register_printer(Real,lambda p,e: p.print_function(e,r"\Re "))
latex.register_printer(Imag,lambda p,e: p.print_function(e,r"\Im "))

latex.register_printer(Indicator,lambda p,e: p.print_function(e,r"\, \mathbb{1} "))
latex.register_printer(Not,lambda p,e: p.print_unary_operator(e,r"\neg "))

latex.register_printer(LessEqual,lambda p,e: p.print_binary_operator(e,r"\le "))
latex.register_printer(GreaterEqual,lambda p,e: p.print_binary_operator(e,r"\ge "))
latex.register_printer(Or,lambda p,e: p.print_binary_operator(e,r"\lor "))
latex.register_printer(And,lambda p,e: p.print_binary_operator(e,r"\land "))
latex.register_printer(Xor,lambda p,e: p.print_binary_operator(e,r"\veebar "))


latex.register_printer(Tuple,lambda p,e: p.print_function(e,r""))

@add_target(latex,Piecewise)
def visit(printer,expr):   
    for arg in expr.args:
        if arg.function != Tuple :
            return printer.print_function(expr,name="Piecewise")

    outer = r"\begin{cases} %s \end{cases}"
    args = [(printer(e.args[0]),printer(e.args[1])) for e in expr.args if e.args[1] != otherwise]
    otherwise_parts = [printer(e.args[0]) for e in expr.args if e.args[1] == otherwise]

    inner = r"\\ ".join([r"%s & \text{if } %s " % arg for arg in args])

    if len(otherwise_parts)>0:
        inner += r"\\ ".join([r"\\ %s & \text{otherwise}" % e for e in otherwise_parts])

    return outer % inner

@add_target(printer,Piecewise)
def visit(printer,expr):
    return printer.print_function(expr,name="Piecewise")


for g in ['alpha', 'theta', 'tauXbeta', 'vartheta', 'pi', 'upsilonXgamma', 'gamma', 'varpi', 'phiXdelta', 'kappa', 'rho', 'varphiXepsilon', 'lambda', 'varrho', 'chiXvarepsilon', 'mu', 'sigma', 'psiXzeta', 'nu', 'varsigma', 'omegaXeta', 'xiXGamma', 'Lambda', 'Sigma', 'PsiXDelta', 'Xi', 'Upsilon', 'OmegaXTheta', 'Pi', 'Phi', 'phi', 'varphi']:
    s = '\\' + g
    latex.register_printer(Symbol(g),lambda p,e,v=s:v)

    
@add_target(latex,CustomFunction)
def visit(printer,expr):
    name = expr.args[0].name.replace('<',r'< ').replace('>',r' >')
    return printer(Function(name)(*expr.args[1:]))


