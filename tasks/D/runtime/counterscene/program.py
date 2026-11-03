"""An interpreted subset of Python, never eval/exec or an AST security filter.

There is no host object access, import, attribute lookup, filesystem or network.
Callable builtins are bounded collection helpers and pure scalar spatial math. Fuel and structure
limits bound execution. This is intentionally not arbitrary CPython support.
"""
import ast
import copy
import json
import math
import operator


class ProgramError(ValueError):
    pass


CONTRACT='Exactly one build(public_params) function, 32KiB source/64KiB JSON output. Supported: literals, assignments, arithmetic, indexing, if/else, one-generator list comprehensions (<=100 items). No for/while loops, nested functions, imports, attributes, methods, files or network. Bare callable names: range, len, min, max, abs, round, sin, cos, tan, asin, acos, atan2, sqrt, degrees, radians; pi is a fixed constant. Trigonometric inputs/outputs use radians except degrees/radians conversion. public_params contains room only.'


class Interpreter:
    def __init__(self):
        self.fuel=20000
        self.builtins={'len':len,'abs':abs,'min':min,'max':max,'round':round,
                       'range':lambda *a:list(range(*a)) if all(type(v)==int and abs(v)<=100 for v in a) else self.bad('range limit')}
        self.builtins.update({name:getattr(math,name) for name in ('sin','cos','tan','asin','acos','atan2','sqrt','degrees','radians')})

    def bad(self,message):
        raise ProgramError(message)

    def expr(self,n,env):
        self.fuel-=1
        if self.fuel<=0:
            self.bad('instruction budget exceeded')
        if isinstance(n,ast.Constant) and type(n.value) in (str,int,float,bool,type(None)):
            if isinstance(n.value,str) and len(n.value)>10000: self.bad('string limit')
            return n.value
        if isinstance(n,ast.Name) and n.id in env: return copy.deepcopy(env[n.id])
        if isinstance(n,(ast.List,ast.Tuple)): return [self.expr(e,env) for e in n.elts]
        if isinstance(n,ast.Dict): return {self.expr(k,env):self.expr(v,env) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.Subscript):
            value,index=self.expr(n.value,env),self.expr(n.slice,env)
            if type(value) not in (dict,list,str): self.bad('unsupported subscription')
            return copy.deepcopy(value[index])
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd,ast.Not)):
            value=self.expr(n.operand,env)
            return {ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_}[type(n.op)](value)
        ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
             ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}
        if isinstance(n,ast.BinOp) and type(n.op) in ops:
            a,b=self.expr(n.left,env),self.expr(n.right,env)
            if type(a) not in (int,float) or type(b) not in (int,float): self.bad('arithmetic is numeric only')
            value=ops[type(n.op)](a,b)
            if abs(value)>1e9: self.bad('numeric limit')
            return value
        if isinstance(n,ast.IfExp): return self.expr(n.body if self.expr(n.test,env) else n.orelse,env)
        if isinstance(n,ast.Compare) and len(n.ops)==1:
            ops={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda a,b:a in b}
            if type(n.ops[0]) not in ops:self.bad('unsupported comparison')
            return ops[type(n.ops[0])](self.expr(n.left,env),self.expr(n.comparators[0],env))
        if isinstance(n,ast.BoolOp):
            values=[bool(self.expr(v,env)) for v in n.values]
            return all(values) if isinstance(n.op,ast.And) else any(values)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in self.builtins and not n.keywords:
            return self.builtins[n.func.id](*[self.expr(a,env) for a in n.args])
        if isinstance(n,ast.ListComp) and len(n.generators)==1:
            g=n.generators[0]
            if not isinstance(g.target,ast.Name) or g.is_async:self.bad('comprehension target')
            values=self.expr(g.iter,env)
            if not isinstance(values,list) or len(values)>100:self.bad('iteration limit')
            out=[]
            for v in values:
                sub=dict(env,**{g.target.id:v})
                if all(self.expr(i,sub) for i in g.ifs):out.append(self.expr(n.elt,sub))
            return out
        self.bad(f'unsupported expression: {type(n).__name__}')

    def statements(self,body,env):
        for n in body:
            self.fuel-=1
            if self.fuel<=0:self.bad('instruction budget exceeded')
            if isinstance(n,ast.Return):return True,self.expr(n.value,env)
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
                env[n.targets[0].id]=self.expr(n.value,env)
            elif isinstance(n,ast.If):
                done,value=self.statements(n.body if self.expr(n.test,env) else n.orelse,env)
                if done:return done,value
            elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):pass
            else:self.bad(f'unsupported statement: {type(n).__name__}')
        return False,None


def execute(source,public_params):
    if not isinstance(source,str) or len(source.encode())>32768:
        raise ProgramError('source limit: 32 KiB')
    try:
        tree=ast.parse(source)
        if sum(1 for _ in ast.walk(tree))>5000:raise ProgramError('AST node limit')
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
            raise ProgramError('provide exactly def build(public_params): ...')
        fn=tree.body[0]
        if fn.name!='build' or len(fn.args.args)!=1 or fn.args.args[0].arg!='public_params' or fn.decorator_list or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
            raise ProgramError('expected build(public_params) without decorators/defaults')
        done,result=Interpreter().statements(fn.body,{'public_params':copy.deepcopy(public_params),'pi':math.pi})
        if not done:raise ProgramError('build must return SceneSpec')
        encoded=json.dumps(result,allow_nan=False)
        if len(encoded)>65536:raise ProgramError('output limit: 64 KiB')
        return result
    except (SyntaxError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError,RecursionError,ValueError) as e:
        raise ProgramError(str(e)) from e


def literal_program(scene):
    return 'def build(public_params):\n    return '+repr(scene)+'\n'
