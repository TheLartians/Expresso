from expression import core,Function,WrappedType,ReplacementMap

class Evaluator(object):

    def __init__(self,evaluator,recursive = False,split_binary = True, split_binary_size = 2,S = None):

        if S == None:
            raise ValueError('missing argument S')

        self._evaluator = evaluator
        self._evaluator.settings.recursive = recursive
        self._evaluator.settings.split_binary = split_binary
        self._evaluator.settings.split_binary_size = split_binary_size

        self.S = S

    def __call__(self,expr,cache = None):
        if cache is not None:
            if not isinstance(cache,dict):
                raise ValueError('cache error: expected dict, got %s' % type(cache))
            c = cache.get(self)
            if not c:
                c = ReplacementMap()
                cache[self] = c
            return self.S(self._evaluator(self.S(expr),c._replacement_map))
        else:
            return self.S(self._evaluator(self.S(expr)))

class Rule(object):
    
    def __init__(self,search,replacement = None,evaluator=None,condition=None,S=None):
        if S == None:
            raise ValueError('missing argument S')
        
        if evaluator is not None:
            if condition is not None:
                self._rule = core.Rule(S(search),S(replacement),S(condition),S(True),lambda m:evaluator(ReplacementMap(m,S=S)))
            else:
                self._rule = core.Rule(S(search),S(replacement),lambda m:evaluator(ReplacementMap(m,S=S)))
        elif replacement is not None:
            if condition is not None:
                self._rule = core.Rule(S(search),S(replacement),S(condition),S(True))
            else:
                self._rule = core.Rule(S(search),S(replacement))
        else:
            self._rule = search    
        self.S = S
    
    @property
    def search(self):
        return self.S(self._rule.search)

    @property
    def has_evaluator(self):
        return self._rule.has_evaluator()

    @property
    def replacement(self):
        return self.S(self._rule.replacement)

    @property
    def condition(self):
        c = self._rule.get_condition()
        if c:
            return self.S(c)

    def __repr__(self):
        l = str(self.search) + ' -> ' + str(self.replacement)
        if self.has_evaluator:
            l += ' ...'
        return l
    
    def _repr_latex_(self):
        l = self.search._repr_latex_()[2:-2] + r' \rightarrow ' + self.replacement._repr_latex_()[2:-2]
        if self.has_evaluator:
            l += '\; \dots '
        return "$$%s$$" % l

WrappedRule = lambda S:WrappedType(Rule,S=S)

class MatchCondition(Function):
    
    def __init__(self,name,F,S):
        super(MatchCondition,self).__init__(core.MatchCondition(name,lambda e:F(S(e))),S=S)

WrappedMatchCondition = lambda S:WrappedType(MatchCondition,S=S)

class ReplaceEvaluator(Evaluator):

    def __init__(self,replacement_map = None,**kwargs):
        if replacement_map is not None:
            super(ReplaceEvaluator,self).__init__(core.ReplaceEvaluator(replacement_map),**kwargs)
        else:
            super(ReplaceEvaluator,self).__init__(core.ReplaceEvaluator(),**kwargs)

    def add_replacement(self,search,replace):
        self._evaluator.add_replacement(self.S(search),self.S(replace))

WrappedReplaceEvaluator = lambda S:WrappedType(ReplaceEvaluator,S=S)

class RewriteEvaluator(Evaluator):

    def __init__(self,**kwargs):
        super(RewriteEvaluator,self).__init__(core.RuleEvaluator(),**kwargs)

    def __len__(self):
        return len(self._evaluator)
    
    def __getitem__(self,idx):
        return Rule(self._evaluator.get_rule(idx),S=self.S)
    
    def __iter__(self):
        def generator():
            for i in range(len(self)):
                yield self[i]
        return generator()
    
    def add_rule(self,search,replace = None,evaluator=None,priority = 0,**kwargs):
        self._evaluator.add_rule(Rule(search,replace,evaluator,S=self.S,**kwargs)._rule,priority)
    
    def set_rule_callback(self,f):
        if f is not None:
            self._evaluator.set_apply_callback(lambda r,m:f(Rule(r,S=self.S),ReplacementMap(m,S=self.S)))
        else:
            self._evaluator.set_apply_callback(None)


WrappedRewriteEvaluator = lambda S:WrappedType(RewriteEvaluator,S=S)
    
class MultiEvaluator(Evaluator):
    
    def __init__(self,S = None,**kwargs):
        super(MultiEvaluator,self).__init__(core.MultiEvaluator(),S=S,**kwargs)
        self._inner_evaluators = []
    
    def add_evaluator(self,evaluator):
        self._inner_evaluators.append(evaluator)
        self._evaluator.add_evaluator(evaluator._evaluator)

    def set_rule_callback(self,callback):
        for e in self._inner_evaluators:
            if isinstance(e,(RewriteEvaluator,MultiEvaluator)):
                e.set_rule_callback(callback)

WrappedMultiEvaluator = lambda S:WrappedType(MultiEvaluator,S=S)


class StepEvaluator(Evaluator):

    def __init__(self,S = None,**kwargs):
        super(StepEvaluator,self).__init__(core.StepEvaluator(),S=S,**kwargs)
        self._inner_evaluators = []

    def add_evaluator(self,evaluator):
        self._inner_evaluators.append(evaluator)
        self._evaluator.add_evaluator(evaluator._evaluator)

    def set_rule_callback(self,callback):
        for e in self._inner_evaluators:
            if isinstance(e,(RewriteEvaluator,MultiEvaluator)):
                e.set_rule_callback(callback)

WrappedStepEvaluator = lambda S:WrappedType(StepEvaluator,S=S)


    
    
    
