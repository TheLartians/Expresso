import _expresso as core
from _expresso import associative,left_associative,right_associative,non_associative,commutative,non_commutative,postfix,prefix

class Expression(core.Expression):

    def __init__(self,expr,S):
        if isinstance(expr,(core.Expression,Expression)):
            super(Expression,self).__init__(expr)
        else:
            super(Expression,self).__init__(S(expr))
        self.S = S
    
    @property
    def function(self):
        if not hasattr(self,'_function'):
            f = super(Expression,self).function()
            if f is not None:
                F = Function(f,S = self.S)
            else:
                F = None
        return F
    
    @property
    def is_function(self):
        return super(Expression,self).is_function()    

    @property
    def is_symbol(self):
        return super(Expression,self).is_symbol()

    @property
    def is_atomic(self):
        return super(Expression,self).is_atomic()

    @property
    def is_wildcard_symbol(self):
        return super(Expression,self).is_wildcard_symbol()

    @property
    def is_wildcard_function(self):
        return super(Expression,self).is_wildcard_function()

    @property
    def is_wildcard(self):
        if self.is_function:
            return self.is_wildcard_function
        return self.is_wildcard_symbol

    @property
    def value(self):
        return self.get_value()

    @property
    def args(self):
        args = self.get_arguments()
        if args == None:
            return []
        return [self.S(arg) for arg in args]

    @property
    def name(self):
        if self.is_function:
            return self.function.name
        return super(Expression,self).__repr__()

    def get(self,type):
        v = self.value
        if isinstance(v,type):
            return v
        return None

    def replace(self,*args):
        if len(args) == 1:
            if isinstance(args[0],ReplacementMap):
                rep = args[0]._replacement_map
            else:
                rep = ReplacementMap(args[0],self.S)._replacement_map
        elif isinstance(args[0],(list,tuple)):
            if len(args) == 1 and isinstance(args[0][0],(list,tuple)):
                rep = ReplacementMap(args[0],self.S)._replacement_map
            else:
                rep = ReplacementMap(args,self.S)._replacement_map
        elif len(args) == 2:
            rep = core.replacement_map()
            rep[self.S(args[0])] = self.S(args[1])
        else:
            raise ValueError('invalid substitution arguments')

        #from evaluator import ReplaceEvaluator
        #return ReplaceEvaluator(rep,S=self.S)(self)

        return self.S(core.replace(self,rep))


    def match(self,search):
        for expr in core.commutative_permutations(search):
            res = core.match(self,expr)
            if res:
                return ReplacementMap(core.match(self,expr),self.S)
        return None

def WrappedType(T,**parameters):
    
    class Wrapped(T):
        
        @staticmethod
        def _get_wrapped_parameter(name):
            return parameters.get(name)
        
        def __init__(self,*args,**kwargs):
            kwargs.update(parameters)
            super(Wrapped,self).__init__(*args,**kwargs)
    
    Wrapped.__name__ = "Wrapped" + T.__name__
    
    return Wrapped

WrappedExpression = lambda S:WrappedType(Expression,S=S)

class Function(object):

    def __init__(self,F,argc = None,S = None):
        if S == None:
            raise ValueError('no parameter S provided')
        self.S = S
        if argc is not None:
            if isinstance(argc,int):
                argc = [argc]
            self.argc = argc
        self._function = F

    def __call__(self,*args):
        if hasattr(self,'argc') and len(args) not in self.argc:
            raise ValueError("%s takes %s arguments" % (str(self),' or '.join([str(v) for v in self.argc])))
        return self.S(core.create_call(self._function,[self.S(arg) for arg in args]))

    def __repr__(self):
        return self.name

    @property
    def name(self):
        return self._function.get_name()

    @property
    def symbol(self):
        return self._function.get_symbol()

    @property
    def is_commutative(self):
        return self._function.is_commutative()

    @property
    def is_associative(self):
        return self._function.is_associative()

    @property
    def is_prefix(self):
        return self._function.is_prefix()

    @property
    def is_postfix(self):
        return self._function.is_postfix()

    @property
    def is_operator(self):
        return isinstance(self._function,core.Operator)    

    @property
    def is_unary_operator(self):
        return isinstance(self._function,core.UnaryOperator)    

    @property
    def is_binary_operator(self):
        return isinstance(self._function,core.BinaryOperator)    

    @property
    def precedence(self):
        return self._function.get_precedence()

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self,other):
        if not isinstance(other,Function):
            return False
        return self.name == other.name

    def __ne__(self,other):
        return not self == other

    
def WrappedFunction(F,S,argc = None):
    
    if argc == None:
        class WrappedFunction(Function):
            def __init__(self,*args,**kwargs):
                argc = kwargs.pop('argc',None)
                super(WrappedFunction,self).__init__(F(*args,**kwargs),argc,S)
    else:
        class WrappedFunction(Function):
            def __init__(self,*args,**kwargs):
                super(WrappedFunction,self).__init__(F(*args,**kwargs),argc,S)

    WrappedFunction.__name__ = "Wrapped" + F.__name__
                
    return WrappedFunction


class ReplacementMap(object):

    def __init__(self,rep = None,S = None):
        
        if isinstance(rep,core.replacement_map):
            self._replacement_map = rep
        elif rep != None:
            self._replacement_map = core.replacement_map()
            for key, value in dict(rep).iteritems():
                self._replacement_map[S(key)] = S(value)
        else:
            self._replacement_map = core.replacement_map()

        self.S = S
        
    def __str__(self):
        return str(dict(self))
    
    def __iter__(self):
        def generator():
            for v in self._replacement_map:
                yield (self.S(v.key()),self.S(v.data()))
        return generator()
    
    def __getitem__(self,key):
        return self.S(self._replacement_map[self.S(key)])

    def __setitem__(self,key,value):
        self._replacement_map[self.S(key)] = self.S(value)



WrappedReplacementMap = lambda S:WrappedType(ReplacementMap,S=S)

class Group(object):

    def __init__(self,operation,inverse = None,neutral = None,S = None):
        if isinstance(operation,core.Group):
            self._group = operation
        else:
            self._group = core.Group(operation._function,inverse._function,S(neutral))
        self.S = S
        
    @property
    def operation(self):
        return Function(self._group.get_operation(),self.S)
    @property
    def inverse(self):
        return Function(self._group.get_inverse(),self.S)
    @property
    def neutral(self):
        return self.S(self._group.neutral)
    
    def __repr__(self):
        return "Group" + str((self.operation,self.inverse,self.neutral))
    
WrappedGroup = lambda S:WrappedType(Group,S=S)

class Field(object):

    def __init__(self,additive_group,multiplicative_group,S):
        self._field = core.Field(additive_group._group,multiplicative_group._group)
        self.S = S
        
    @property
    def additive_group(self):
        return Group(self._field.additive_group,self.S)
    
    @property
    def multiplicative_group(self):
        return Group(self._field.multiplicative_group,self.S)

        
WrappedField = lambda S:WrappedType(Field,S=S)

class MulplicityList(object):

    def __init__(self,arg,operation_group = None,mulplicity_function = None,field = None,S = None):
        if S == None:
            raise ValueError("Parameter S undefined")
        if arg == None:
            self._mlist = core.MulplicityList(operation_group._group,mulplicity_function._function,field._field)
        elif isinstance(arg,core.MulplicityList):
            self._mlist = arg
        else:
            self._mlist = core.MulplicityList(S(arg),operation_group._group,mulplicity_function._function,field._field)
        self.S = S
    
    def __len__(self):
        return len(self._mlist)
    
    def __iter__(self):
        def generator():
            for v in self._mlist:
                yield (self.S(v.value),self.S(v.mulplicity))
        return generator()
    
    def __getitem__(self,index):
        if index >= len(self._mlist):
            raise ValueError("index out of range")
        v = self._mlist[index]
        return (self.S(v.value),self.S(v.mulplicity))

    def __repr__(self):
        return str(list(self))
    
    def as_expression(self):
        return self.S(self._mlist.as_expression())
    
    def _wrap(self,mlist):
        return MulplicityList(mlist,S = self.S)
    
    def intersection(self,other,get_inner = None):
        if get_inner:
            return self._wrap(self._mlist.intersection(other._mlist,lambda a,b:get_inner(self.S(a),self.S(b))))
        return self._wrap(self._mlist.intersection(other._mlist))
    
    def sub(self,other):
        return self._wrap(self._mlist.difference(other._mlist))

    def sum(self,other):
        return self._wrap(self._mlist.sum(other._mlist))

    def pow(self,expr):
        return self._wrap(self._mlist.power(self.S(expr)))

    def __sub__(self,other):
        return self.sub(other)

    def __add__(self,other):
        return self.sum(other)

    def __pow__(self,other):
        return self.pow(other)

WrappedMulplicityList = lambda S:WrappedType(MulplicityList,S=S)
    
def wrapped_postorder_traversal(S):
    def postorder_traversal(expr):
        for expr in core.postorder_traversal(expr):
            yield S(expr)
    return postorder_traversal


def wrapped_preorder_traversal(S):
    def preorder_traversal(expr):
        for expr in core.preorder_traversal(expr):
            yield S(expr)
    return preorder_traversal


def wrapped_commutative_permutations(S):
    def commutative_permutations(expr):
        for expr in core.commutative_permutations(expr):
            yield S(expr)
    return commutative_permutations


