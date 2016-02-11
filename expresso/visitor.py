
# Original copyright notice:

# The MIT License (MIT)
#
# Copyright (c) 2013 Curtis Schlak
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import inspect
import expression

class Dispatcher(object):
    
    def __init__(self, param_name, fn):
        self.param_index = inspect.getargspec(fn).args.index(param_name)
        self.param_name = param_name
    
        self.default = fn
        self.targets = {}

    def _copy_from(self,other):
        self.targets.update(other.targets)

    def get_target(self,arg):
        
        d = self.targets.get(arg)
        if d: return d

        if isinstance(arg,expression.Expression):
            
            if arg.is_function:
                f  = arg.function
                d = self.targets.get(f)
                if d: return d
                if arg.is_wildcard_function:
                    d = self.targets.get(wildcard_function)
                    if d: return d
                else:
                    if f.is_binary_operator:
                        d = self.targets.get(binary_operator)
                        if d: return d
                    if f.is_unary_operator:
                        d = self.targets.get(unary_operator)
                        if d: return d
                d = self.targets.get(function)
                if d: return d
            else:
                v = arg.value
                if v is not None:
                    d = self.targets.get((obj,type(v)))
                    if d: return d
                    d = self.targets.get(obj)
                    if d: return d
                else:
                    if arg.is_wildcard_symbol:
                        d = self.targets.get(wildcard_symbol)
                        if d: return d
                    if arg.is_symbol:
                        d = self.targets.get(symbol)
                        if d: return d
                d = self.targets.get(atomic)
                if d: return d
        else:
            typ = arg.__class__
            d = self.targets.get(typ)
            if d: return d
        
        return self.default

    def __call__(self, *args, **kw):
        arg = args[self.param_index]
        d = self.get_target(arg)
        if hasattr(d,'__call__'):
            return d(*args, **kw)
        return d

    def register_binary_operator(self, target):
        self.targets[binary_operator] = target
        
    def register_unary_operator(self, target):
        self.targets[unary_operator] = target

    def register_wildcard_symbol(self, target):
        self.targets[wildcard_symbol] = target

    def register_wildcard_function(self, target):
        self.targets[wildcard_function] = target

    def register_function(self, target):
        self.targets[function] = target

    def register_symbol(self, target):
        self.targets[symbol] = target

    def register_atomic(self, target):
        self.targets[atomic] = target

    def register_object(self, target):
        self.targets[obj] = target

    def register_target(self, arg, target):
        self.targets[arg] = target
        
def on(param_name,parent = None):

    def f(fn):
        dispatcher = Dispatcher(param_name, fn)
        if parent != None:
            func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
            other_dispatcher = getattr(parent,func_name)
            if not isinstance(other_dispatcher, Dispatcher):
                other_dispatcher = other_dispatcher.dispatcher
            dispatcher._copy_from(other_dispatcher)
        return dispatcher

    return f

def when(param_type):
    def f(fn):
        frame = inspect.currentframe().f_back
        func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
        dispatcher = frame.f_locals[func_name]
        if not isinstance(dispatcher, Dispatcher):
            dispatcher = dispatcher.dispatcher
        dispatcher.register_target(param_type, fn)
        def ff(*args, **kw):
            return dispatcher(*args, **kw)
        ff.dispatcher = dispatcher
        return ff
    return f

def function(arg):
    F = None
    
    def f(fn):
        
        frame = inspect.currentframe().f_back
        if F is None:
            frame = frame.f_back
        func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
        dispatcher = frame.f_locals[func_name]
        if not isinstance(dispatcher, Dispatcher):
            dispatcher = dispatcher.dispatcher
        if F is None:
            dispatcher.register_function(fn)
        else:
            dispatcher.register_target(F, fn)
        def ff(*args, **kw):
            return dispatcher(*args, **kw)
        ff.dispatcher = dispatcher
        return ff
    
    if isinstance(arg,expression.Function):
        F = arg
        return f
    
    return f(arg)
    
def obj(arg):
    typ = None
    
    def f(fn):
        frame = inspect.currentframe().f_back
        if typ is None:
            frame = frame.f_back
        func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
        dispatcher = frame.f_locals[func_name]
        if not isinstance(dispatcher, Dispatcher):
            dispatcher = dispatcher.dispatcher
        if typ is None:
            dispatcher.register_object(fn)
        else:
            dispatcher.register_target((obj,typ), fn)
        def ff(*args, **kw):
            return dispatcher(*args, **kw)
        ff.dispatcher = dispatcher
        return ff
    
    if isinstance(arg,type):
        typ = arg
        return f
    
    return f(arg)

def atomic(arg):
    symbol = None
    
    def f(fn):
        frame = inspect.currentframe().f_back
        if symbol is None:
            frame = frame.f_back
        func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
        dispatcher = frame.f_locals[func_name]
        if not isinstance(dispatcher, Dispatcher):
            dispatcher = dispatcher.dispatcher
        if symbol is None:
            dispatcher.register_atomic(fn)
        else:
            dispatcher.register_target(symbol, fn)
        def ff(*args, **kw):
            return dispatcher(*args, **kw)
        ff.dispatcher = dispatcher
        return ff
    
    if isinstance(arg,expression.Expression):
        symbol = arg
        return f
    
    return f(arg)

def symbol(fn):

    frame = inspect.currentframe().f_back
    func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
    dispatcher = frame.f_locals[func_name]
    if not isinstance(dispatcher, Dispatcher):
        dispatcher = dispatcher.dispatcher
    dispatcher.register_symbol(fn)
    def ff(*args, **kw):
        return dispatcher(*args, **kw)
    ff.dispatcher = dispatcher
    return ff

def binary_operator(fn):
    frame = inspect.currentframe().f_back
    func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
    dispatcher = frame.f_locals[func_name]
    if not isinstance(dispatcher, Dispatcher):
        dispatcher = dispatcher.dispatcher
    dispatcher.register_binary_operator(fn)
    def ff(*args, **kw):
        return dispatcher(*args, **kw)
    ff.dispatcher = dispatcher
    return ff

def unary_operator(fn):
    frame = inspect.currentframe().f_back
    func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
    dispatcher = frame.f_locals[func_name]
    if not isinstance(dispatcher, Dispatcher):
        dispatcher = dispatcher.dispatcher
    dispatcher.register_unary_operator(fn)
    def ff(*args, **kw):
        return dispatcher(*args, **kw)
    ff.dispatcher = dispatcher
    return ff

def wildcard_symbol(fn):
    frame = inspect.currentframe().f_back
    func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
    dispatcher = frame.f_locals[func_name]
    if not isinstance(dispatcher, Dispatcher):
        dispatcher = dispatcher.dispatcher
    dispatcher.register_wildcard_symbol(fn)
    def ff(*args, **kw):
        return dispatcher(*args, **kw)
    ff.dispatcher = dispatcher
    return ff

def wildcard_function(fn):
    frame = inspect.currentframe().f_back
    func_name = fn.func_name if 'func_name' in dir(fn) else fn.__name__
    dispatcher = frame.f_locals[func_name]
    if not isinstance(dispatcher, Dispatcher):
        dispatcher = dispatcher.dispatcher
    dispatcher.register_wildcard_function(fn)
    def ff(*args, **kw):
        return dispatcher(*args, **kw)
    ff.dispatcher = dispatcher
    return ff


def visitor_class(visit_name = 'visit'):
    
    class Visitor(object):
        
        @property
        def dispatcher(self):
            if isinstance(getattr(self,visit_name),Dispatcher):
                return self.visit
            else:
                return self.visit.dispatcher

        def register_target(self,target):
            def register(f):
                self.dispatcher.register_target(target,f)
                return f
            return register

        def register_targets(self,*targets):
            def register(f):
                for target in targets:
                    self.dispatcher.register_target(target,f)
                return f
            return register


    return Visitor


def add_target(visitor,target):
    
    def decorator(fn):
        visitor.dispatcher.register_target(target,fn)
        return fn

    return decorator

def add_target_obj(visitor,target):

    def decorator(fn):
        visitor.dispatcher.register_target((obj,target),fn)
        return fn

    return decorator





