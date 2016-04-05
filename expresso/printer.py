
from expression import *
import visitor

__all__ = ["Printer","LatexPrinter"]

class Printer(visitor.visitor_class()):

    def __init__(self,S):
        self.S = S

    def format_name(self,name):
        return name

    def register_printer(self,expr,printer):
        self.dispatcher.register_target(expr,printer)
    
    def needs_brackets_in(self,expr,parent):
        if expr.is_atomic:
            return False
        if not expr.function.is_operator and parent.function.is_operator:
            return False
        return parent.function.precedence <= expr.function.precedence
    
    def print_symbol(self,expr):
        return self.format_name(expr.name)
    
    def print_wildcard_symbol(self,expr):
        return self.print_symbol(expr)

    def print_wildcard_function(self,expr):
        return self.print_function(expr)
    
    def bracket_format(self):
        return "(%s)"
    
    def function_format(self):
        return "%s(%s)"

    def print_operator_argument(self,expr,parent):
        if self.needs_brackets_in(expr,parent):
            return self.bracket_format() % self(expr)
        return self(expr)
    
    def print_prefix_operator(self,expr,symbol):
        return symbol + self.print_operator_argument(expr.args[0],expr)

    def print_postfix_operator(self,expr,symbol):
        return self.print_operator_argument(expr.args[0],expr) + symbol
    
    def print_unary_operator(self,expr,symbol = None):
        f = expr.function
        if symbol == None:
            symbol = f.symbol
        if f.is_prefix:
            return self.print_prefix_operator(expr,symbol)
        else:
            return self.print_postfix_operator(expr,symbol)

    def printed_operator_arguments(self,parent,args=None,begin=0,end=None):
        if args == None:
            args = parent.args
        printed_args = [self.print_operator_argument(arg,parent) for arg in args[begin:end]]
        if parent.function.is_commutative:
            return sorted(printed_args)
        return printed_args

    def print_binary_operator(self,expr,symbol = None):
        if symbol == None:
            symbol = expr.function.symbol
        return symbol.join(self.printed_operator_arguments(expr))
    
    def print_function(self,expr,name = None):
        if name == None:
            name = self.format_name(expr.function.name)
        return self.function_format() % (name,','.join([ self(e) for e in expr.args ]))
        
    @visitor.on('expr')
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr.name)

    @visitor.function
    def visit(self,expr):
        return self.print_function(expr)

    @visitor.atomic
    def visit(self,expr):
        return self.print_symbol(expr)

    @visitor.obj
    def visit(self,expr):
        return str(expr.value)

    @visitor.wildcard_symbol
    def visit(self,expr):
        return self.print_wildcard_symbol(expr)

    @visitor.wildcard_function
    def visit(self,expr):
        return self.print_wildcard_function(expr)

    @visitor.binary_operator
    def visit(self,expr):
        if len(expr.args) >= 2:
            return self.print_binary_operator(expr)
        else:
            return self.print_function(expr)

    @visitor.unary_operator
    def visit(self,expr):
        return self.print_unary_operator(expr)

    def __call__(self,expr):
        return self.dispatcher(self,self.S(expr))
    
class LatexPrinter(Printer):

    def __init__(self,*args,**kwargs):
        super(LatexPrinter,self).__init__(*args,**kwargs)

        self.latex_replacements = {}
        self.latex_replacements.update({w:r'\%s ' % w for w in ['Delta', 'Gamma', 'Lambda', 'Omega', 'Phi', 'Pi', 'Psi', 'Sigma', 'Theta', 'Upsilon', 'Xi', 'alpha', 'beta', 'chi', 'delta', 'epsilon', 'eta', 'gamma', 'kappa', 'lambda', 'mu', 'nabla', 'nu', 'omega', 'phi', 'pi', 'psi', 'rho', 'sigma', 'tau', 'theta', 'upsilon', 'varepsilon', 'varphi', 'varpi', 'varrho', 'varsigma', 'vartheta', 'xi', 'zeta', 'hbar']})

    @visitor.on('expr',parent = Printer)
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr.name)
        
    def print_binary_operator(self,expr,symbol = None):
        if symbol == None:
            symbol = " " + expr.function.symbol + " "
        return symbol.join(self.printed_operator_arguments(expr))
        
    def bracket_format(self):
        return r"\left( %s \right) "
    
    def print_wildcard_symbol(self,expr):
        return '\mathbf{%s}' % self.format_name(expr.name[1:])

    def function_format(self):
        return r"%s \mathopen{} \left(%s \right) \mathclose{} "

    def format_name(self,name):

        underscore_parts = name.split('_')

        for i,p in enumerate(underscore_parts):
            space_parts = p.split(' ')
            for i2,p2 in enumerate(space_parts):
                if p2 in self.latex_replacements:
                    space_parts[i2] = self.latex_replacements[p2]
                elif len(p2) > 1:
                    space_parts[i2] = r'\text{%s} ' % p2
            underscore_parts[i] = '\; '.join(space_parts)

        return reduce(lambda p1,p2:'{%s}_{%s} ' % (p2,p1),underscore_parts[::-1])

    def print_wildcard_function(self,expr):
        f = expr.function
        name = self.format_name(f.name[1:])
        name = '\mathbf{%s} ' % name
        return self.print_function(expr,name=name)

    @visitor.obj
    def visit(self,expr):
        return self.format_name(expr.name)
