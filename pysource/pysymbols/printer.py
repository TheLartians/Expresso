
from expression import *
import visitor

__all__ = ["Printer","LatexPrinter"]

class Printer(visitor.visitor_class()):

    def __init__(self,S):
        self.S = S

    def bracket_format(self):
        return "(%s)"
    
    def register_printer(self,expr,printer):
        self.dispatcher.register_target(expr,printer)
    
    def needs_brackets_in(self,expr,parent):
        if expr.is_atomic:
            return False
        if not expr.function.is_operator and parent.function.is_operator:
            return False
        return parent.function.precedence <= expr.function.precedence
    
    def print_symbol(self,expr):
        return expr.name
    
    def print_wildcard_symbol(self,expr):
        return self.print_symbol(expr)

    def print_wildcard_function(self,expr):
        return self.print_function(expr)
    
    def bracket_format(self):
        return "(%s)"
    
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
        return [self.print_operator_argument(arg,parent) for arg in args[begin:end]]
    
    def print_binary_operator(self,expr,symbol = None):
        if symbol == None:
            symbol = expr.function.symbol
        return symbol.join(self.printed_operator_arguments(expr))
    
    def print_function(self,expr,name = None):
        if name == None:
            name = expr.function.name
        return "%s(%s)" % (name,','.join([ self(e) for e in expr.args ]))
        
    @visitor.on('expr')
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr.name)

    @visitor.function
    def visit(self,expr):
        return self.print_function(expr)
        
    @visitor.atomic
    def visit(self,expr):
        return self.print_symbol(expr)

    @visitor.wildcard_symbol
    def visit(self,expr):
        return self.print_wildcard_symbol(expr)

    @visitor.wildcard_function
    def visit(self,expr):
        return self.print_wildcard_function(expr)

    @visitor.binary_operator
    def visit(self,expr):
        return self.print_binary_operator(expr)

    @visitor.unary_operator
    def visit(self,expr):
        return self.print_unary_operator(expr)

    def __call__(self,expr):
        return self.dispatcher(self,self.S(expr))
    
class LatexPrinter(Printer):
        
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
        return '\mathbf{%s}' % expr.name[1:]

    def print_symbol(self,expr):
        if len(expr.name) > 1:
            return r'\text{%s} ' % expr.name
        return expr.name

    def function_format(self):
        return r"%s \mathopen{} \left(%s \right) \mathclose{} "
    
    def print_function(self,expr,name = None):
        if name == None:
            f = expr.function
            name = f.name
            if len(name) > 1:
                name = r'\text{%s} ' % name
        return self.function_format() % (name,','.join([ self(e) for e in expr.args ]))
    
    def print_wildcard_function(self,expr):
        f = expr.function
        name = f.name[1:]
        if len(name) > 1:
            name = r'\text{%s} ' % name
        name = '\mathbf{%s} ' % name
        return self.function_format() % (name,','.join([ self(e) for e in expr.args ]))


