
import pysymbols.visitor as visitor
from pysymbols.printer import Printer

from .functions import *
from .expression import *

class CodePrinter(Printer):

    def __init__(self):
        super(CodePrinter,self).__init__(expression_converter)

    @visitor.on('expr',parent = Printer)
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr)
    
    def print_wildcard_symbol(self,expr):
        raise ValueError('cannot compile wildcard %s' % expr)

    def print_wildcard_function(self,expr):
        raise ValueError('cannot compile wildcard %s' % expr)

    @visitor.function(CustomFunction)
    def visit(self,expr):
        raise ValueError('cannot compile python function %s' % expr)

class FunctionDefinition(object):

    def __init__(self,name,args,expr,return_type = None,arg_types = None):
        self.name = name
        self.expr = expr
        self.args = args
        self.return_type = return_type
        self.arg_types = arg_types


import ctypes

class c_complex(ctypes.Structure):
    _fields_ = [('real',ctypes.c_double),('imag',ctypes.c_double)]
    def __str__(self):
        return str(self.real) + '+' + str(self.imag) + 'j'
    def __repr__(self):
        return '(' + str(self.real) + ',' + str(self.imag) + ')'
    def __complex__(self):
        return complex(self.real,self.imag)

class CCodePrinter(CodePrinter):

    def __init__(self):
        super(CCodePrinter,self).__init__()
        self.includes = ['cmath','complex']
        self.namespaces = ['std']
        self.typenames = {
          Types.Boolean:'bool',
          Types.Natural:'unsigned',
          Types.Integer:'int',
          Types.Rational:'double',
          Types.Real:'double',
          Types.Complex:'complex<double>',
          None:'complex<double>'
        }

        self.type_converters = {}
        self.need_conversion = {}
        self.auxiliary_code = set()

        complex_operators = '''
inline complex<double> operator{0}(complex<double> lhs, const double & rhs){{
    return lhs {0} rhs;
}}
inline complex<double> operator{0}(const double & lhs,complex<double> rhs){{
    return lhs {0} rhs;
}}
        '''

        self.auxiliary_code.update(set([complex_operators.format(op) for op in ['+','-','*','/']]))

        self.ctype_map = {
            'bool':ctypes.c_bool,
            'unsigned':ctypes.c_uint,
            'int':ctypes.c_int,
            'double':ctypes.c_double,
            'complex<double>':c_complex,
        }

    def needs_brackets_in(self,expr,parent):
        if expr.is_atomic:
            return False
        return expr.function.is_operator

    @visitor.on('expr',parent = CodePrinter)
    def visit(self,expr):
        raise ValueError('cannot print expression %s' % expr)

    @visitor.function(CustomFunction)
    def visit(self,expr):
        f = expr.args[0].value
        if hasattr(f,'ccode'):
            self.auxiliary_code.add(f.ccode)
        else:
            raise ValueError('cannot compile custom function %s' % expr)
        return "%s(%s)" % (f.name,','.join([self(arg) for arg in expr.args[1:]]))

    @visitor.function(Exponentiation)
    def visit(self,expr):
        return 'pow(%s,%s)' % (self(expr.args[0]),self(expr.args[1]))

    @visitor.atomic(I)
    def visit(self,expr):
        self.auxiliary_code.add("std::complex<double> __I(0,1);")
        return "__I"

    @visitor.function(Xor)
    def visit(self,expr):
        return self.print_binary_operator(expr,symbol='^')

    @visitor.function(Not)
    def visit(self,expr):
        return "!(%s)" % self(expr.args[0])

    @visitor.function(Equal)
    def visit(self,expr):
        return self.print_binary_operator(expr,'==')

    @visitor.function(Fraction)
    def visit(self,expr):
        return "1./(%s)" % self(expr.args[0])

    @visitor.function(Piecewise)
    def visit(self,expr):
        parts = ['(%s)?(%s):' % (self(arg.args[1]),self(arg.args[0])) for arg in expr.args]
        return '(%s%s)' % (''.join(parts),self(S(0)))

    @visitor.symbol
    def visit(self,expr):
        if expr in self.need_conversion:
            return '%s(%s)' % (self.need_conversion[expr],expr.name)
        return expr.name

    @visitor.atomic(S(True))
    def visit(self,expr):
        return 'true'

    @visitor.atomic(S(False))
    def visit(self,expr):
        return 'false'

    def print_includes(self):
        return '\n'.join(['#include <%s>' % name for name in self.includes])

    def print_namespaces(self):
        return '\n'.join(['using namespace %s;' % namespace for namespace in self.namespaces])

    def print_auxiliary_code(self):
        return '\n'.join(self.auxiliary_code)

    def print_file(self,*function_definitions):

        function_code = [self.generate_function(f) for f in function_definitions]

        return "\n\n".join([self.print_includes(),
                            self.print_namespaces(),
                            self.print_auxiliary_code()] + function_code )

    def print_typename(self,expr):
        return self.typenames.get(Type(expr).evaluate(),self.typenames[None])

    def generate_function(self,definition):

        if definition.return_type == None:
             return_type = self.print_typename(definition.expr)
        else:
            return_type = self.print_typename(definition.return_type)

        args = definition.args

        if definition.arg_types == None:
            argument_types = [self.print_typename(arg) for arg in args]
        else:
            argument_types = [self.print_typename(arg) for arg in definition.arg_types]

        self.need_conversion = {arg:self.type_converters[t]
                                for arg,t in zip(args,argument_types)
                                if t in self.type_converters}

        f_code = self(definition.expr)
        if return_type in self.type_converters:
            f_code = "%s(%s)" % (self.type_converters[return_type],f_code)

        formatted = (return_type, definition.name,
                    ','.join(['%s %s' % (type,arg.name) for arg,type in zip(args,argument_types)]),
                    f_code)

        definition.c_return_type = self.ctype_map[return_type]
        definition.c_arg_types = [self.ctype_map[arg_type] for arg_type in argument_types]

        return 'extern "C"{\n%s %s(%s){\n\treturn %s;\n}\n}' % formatted



