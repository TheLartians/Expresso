
import expresso.visitor as visitor
from expresso.printer import Printer
from mpmath import mp

import functions as f
import expression as e

class CodePrinter(Printer):

    def __init__(self):
        super(CodePrinter,self).__init__(e.expression_converter)

    @visitor.on('expr',parent = Printer)
    def visit(self,expr):
        raise ValueError('cannot compile expression %s' % expr)
    
    def print_wildcard_symbol(self,expr):
        raise ValueError('cannot compile wildcard %s' % expr)

    def print_wildcard_function(self,expr):
        raise ValueError('cannot compile wildcard %s' % expr)

    @visitor.function(f.CustomFunction)
    def visit(self,expr):
        raise ValueError('cannot compile custom function %s' % expr)

    @visitor.function(f.ArrayAccess)
    def visit(self,expr):
        raise ValueError('cannot compile expression %s' % expr)

    @visitor.function(f.Tuple)
    def visit(self,expr):
        raise ValueError('cannot compile expression %s' % expr)
