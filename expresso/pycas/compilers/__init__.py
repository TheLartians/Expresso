
from lambda_compiler import lambdify,mpmathify,N
from numpy_compiler import numpyfy,ncompile
from c_compiler import ccompile,c_complex


class FunctionDefinition(object):

    def __init__(self, name, args, expr, return_type = None, arg_types = None, parallel=True):
        from ..expression import S
        self.name = name
        self.expr = S(expr)
        self.args = args if isinstance(args,(list,tuple)) else (args,)
        self.return_type = return_type
        self.arg_types = arg_types
        self.parallel = parallel

    def __str__(self):
        return '%s(%s) = %s' % (self.name,','.join([str(arg) for arg in self.args]),self.expr)


