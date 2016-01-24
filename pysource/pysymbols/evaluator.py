from expression import core,Function,WrappedType,ReplacementMap

class Rule(object):
    
    def __init__(self,search,replacement = None,evaluator=None,S=None):
        if S == None:
            raise ValueError('missing argument S')
        
        if evaluator is not None: 
            self._rule = core.Rule(S(search),S(replacement),lambda m:evaluator(ReplacementMap(m,S=S)))
        elif replacement is not None:
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

class Evaluator(object):
    
    def __init__(self,evaluator,S):
        
        if S == None:
            raise ValueError('missing argument S')

        self._evaluator = evaluator
        self.S = S

    def __call__(self,expr):
        return self.S(self._evaluator.__call__(self.S(expr)))
                
            
class RewriteEvaluator(Evaluator):

    def __init__(self,recursive = False,S = None):
        super(RewriteEvaluator,self).__init__(core.RuleEvaluator(),S)
        self._evaluator.recursive = recursive
        
    def __len__(self):
        return len(self._evaluator)
    
    def __getitem__(self,idx):
        return Rule(self._evaluator.get_rule(idx),S=self.S)
    
    def __iter__(self):
        def generator():
            for i in range(len(self)):
                yield self[i]
        return generator()
    
    def add_rule(self,search,replace = None,evaluator=None,priority = 0):
        self._evaluator.add_rule(Rule(search,replace,evaluator,S=self.S)._rule,priority)
    
    def set_apply_callback(self,f):
        if f is not None:
            self._evaluator.set_apply_callback(lambda r,m:f(Rule(r,S=self.S),ReplacementMap(m,S=self.S)))
        else:
            self._evaluator.set_apply_callback(None)


    
WrappedRewriteEvaluator = lambda S:WrappedType(RewriteEvaluator,S=S)
    
class MultiEvaluator(Evaluator):
    
    def __init__(self,S):
        super(MultiEvaluator,self).__init__(core.MultiEvaluator(),S)
        self._inner_evaluators = []
    
    def add_evaluator(self,evaluator):
        self._inner_evaluators.append(evaluator)
        self._evaluator.add_evaluator(evaluator._evaluator)
    
WrappedMultiEvaluator = lambda S:WrappedType(MultiEvaluator,S=S)

    
    
    
    
    
