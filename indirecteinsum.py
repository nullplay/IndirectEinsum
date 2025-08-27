import re
import torch
from collections import defaultdict
from sympy import sympify, preorder_traversal, Function, Symbol, Mul, Eq
from torch._inductor.lowering import (
    lowerings, 
    make_reduction, 
    make_pointwise,
    transform_args,
    index_impl_helper,
    index_put_impl_,
    clone
)
from torch._inductor.virtualized import ops
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    fwd_only,
    joint_fwd_bwd,
)

from torch._inductor.ir import Pointwise

from torch._inductor.fx_passes.post_grad import (
    register_lowering_pattern,
    pass_patterns
)

from torch._inductor.utils import print_performance
inductor = torch._inductor
aten = torch.ops.aten
torch.set_default_device("cuda")
torch.random.manual_seed(52)




# === Base AST Node Class ===
class ASTNode:
    def __init__(self, func, args=None):
        self.func = func
        self.args = args if args is not None else []
    def to_sympy(self):
        raise NotImplementedError("to_sympy must be implemented in subclass")
    def to_string(self):
        raise NotImplementedError("to_string must be implemented in subclass")
    def run(self, localenv, shapeenv):
        raise NotImplementedError("run must be implemented in subclass")
    
    def extract_unique_indices(self):
        unique = []
        # If this node is an IndexNode, add its function name if not already in the list.
        if isinstance(self, IndexNode):
            if self.func not in unique:
                unique.append(self.func)
        # Recurse on all children.
        for arg in self.args:
            for idx in arg.extract_unique_indices():
                if idx not in unique:
                    unique.append(idx)
        return unique

    def extract_unique_accesses(self):
        unique = []
        if isinstance(self, AccessNode):
            unique.append(self)
        for arg in self.args:
            for access in arg.extract_unique_accesses():
                if access not in unique:
                    unique.append(access)
        return unique

    def extract_unique_tensornames(self):
        unique = []
        if isinstance(self, AccessNode):
            if self.func not in unique:
                unique.append(self.func)
        for arg in self.args:
            for tname in arg.extract_unique_tensornames():
                if tname not in unique:
                    unique.append(tname)
        return unique


# === AST Node Subclasses ===

class IndexNode(ASTNode):
    # Instead of Identifier, represents a simple index.
    def __init__(self, name):
        super().__init__(name, [])
    def to_sympy(self):
        return Symbol(self.func)
    def to_string(self):
        return str(self.func)
    def run(self, localenv, shapeenv):
        return localenv[self.func]

class AccessNode(ASTNode):
    # Instead of TensorCall, represents a tensor access with indices.
    def __init__(self, name, args):
        super().__init__(name, args)
    
    def to_sympy(self):
        f = Function(self.func)
        return f(*[arg.to_sympy() for arg in self.args])
    
    def to_string(self):
        arg_strings = []
        for arg in self.args:
            arg_strings.append(arg.to_string())
        return f"{self.func}[{', '.join(arg_strings)}]"

    def run(self, localenv, shapeenv):
        eval_gather_src = localenv[self.func]
        eval_gather_indices = []
        for gather_index in self.args:
            eval_gather_index = gather_index.run(localenv, shapeenv)
            eval_gather_indices.append(eval_gather_index)
        return torch.ops.aten._unsafe_index(eval_gather_src, eval_gather_indices)
        #return eval_gather_src[eval_gather_indices] 


class MulNode(ASTNode):
    # Represents multiplication of factors.
    def __init__(self, factors):
        super().__init__("Mul", factors)

class EinsumNode(ASTNode):
    # Represents multiplication of factors.
    def __init__(self, factors, pure_vars):
        super().__init__("Einsum", factors)
        self.pure_vars = pure_vars
    def to_sympy(self):
        product = 1
        for factor in self.args:
            product *= factor.to_sympy()
        return product
    def to_string(self):
        arg_strings = []
        for arg in self.args:
            arg_strings.append(arg.to_string())
        return f"{' * '.join(arg_strings)}"

    def run(self, localenv, shapeenv):
        einsum_output_indices = self.pure_vars
        einsum_input_indices = []
        einsum_input_tensors = []
        for factor in self.args:
            meshgrid_args = []
            factor_indices = factor.extract_unique_indices()
            for index in factor_indices :
                tensorname, dim = shapeenv[index][0]
                tensor = localenv[tensorname]
                index_size = tensor.shape[dim]
                aranged_index = torch.arange(index_size, device=tensor.device)
                meshgrid_args.append(aranged_index)
            meshgrid_indices = torch.meshgrid(*meshgrid_args, indexing='ij')

            new_localenv = localenv.copy()
            new_localenv.update(dict(zip(factor_indices, meshgrid_indices)))
            eval_factor = factor.run(new_localenv, shapeenv)
            einsum_input_indices.append("".join(factor_indices))
            einsum_input_tensors.append(eval_factor)
        
        einsum_str = f"{','.join(einsum_input_indices)}->{''.join(einsum_output_indices)}"
        return torch.einsum(einsum_str, *einsum_input_tensors)
            

class AssignmentNode(ASTNode):
    # Instead of Assignment, represents an equation (using 'Eq').
    def __init__(self, lhs, rhs, scatter_add=False):
        self.scatter_add = scatter_add
        pure_vars = lhs.extract_unique_indices()
        if isinstance(rhs, MulNode):
            new_rhs = EinsumNode(rhs.args, pure_vars)
        else:
            new_rhs = EinsumNode([rhs], pure_vars)
        super().__init__("Eq", [lhs, new_rhs])
    def to_sympy(self):
        return Eq(self.args[0].to_sympy(), self.args[1].to_sympy())
    def to_string(self):
        return f"{self.args[0].to_string()} += {self.args[1].to_string()}"
    def run(self, localenv, shapeenv):
        lhs = self.args[0]
        rhs = self.args[1]
        
        meshgrid_args = []
        lhs_indices = lhs.extract_unique_indices()
        for index in lhs_indices :
            tensorname, dim = shapeenv[index][0]
            tensor = localenv[tensorname]
            index_size = tensor.size(dim)
            aranged_index = torch.arange(index_size, device=tensor.device)
            meshgrid_args.append(aranged_index)
        meshgrid_indices = torch.meshgrid(*meshgrid_args, indexing='ij')

        new_localenv = localenv.copy()
        new_localenv.update(dict(zip(lhs_indices, meshgrid_indices)))
        
        eval_scatter_dst = localenv[lhs.func]
        eval_scatter_src = rhs.run(localenv, shapeenv)
        eval_scatter_indices = []
        for scatter_index in lhs.args:
            eval_scatter_index = scatter_index.run(new_localenv, shapeenv)
            eval_scatter_indices.append(eval_scatter_index)
         
        if lhs_indices == lhs.args :
            return eval_scatter_src

        final_output = eval_scatter_dst.index_put_(
            tuple(eval_scatter_indices), 
            eval_scatter_src, 
            accumulate=self.scatter_add
        )

        return final_output

# === Tokenizer ===
#def tokenize(s):
#    # Matches tokens: "=", "+=", "*", "[", "]", ",", or an identifier (letters, digits, underscore)
#    token_pattern = re.compile(r'\s*(\=|\+=|\*|\[|\]|,|[A-Za-z_]\w*)\s*')
#    tokens = token_pattern.findall(s)
#    return tokens
def tokenize(s):
    """
    Tokenize a string into individual tokens without using regular expressions.
    
    Recognizes the following tokens:
    - Assignment operators: "=", "+="
    - Arithmetic operators: "*"
    - Brackets: "[", "]"
    - Comma: ","
    - Identifiers: Letters, digits, and underscores (must start with letter or underscore)
    
    Args:
        s (str): The input string to tokenize
        
    Returns:
        list: A list of tokens extracted from the input string
    """
    tokens = []
    i = 0
    
    # Skip leading whitespace
    while i < len(s) and s[i].isspace():
        i += 1
    
    while i < len(s):
        # Check for operators and special characters
        if s[i] == '=':
            tokens.append('=')
            i += 1
        elif i < len(s) - 1 and s[i:i+2] == '+=':
            tokens.append('+=')
            i += 2
        elif s[i] == '*':
            tokens.append('*')
            i += 1
        elif s[i] == '[':
            tokens.append('[')
            i += 1
        elif s[i] == ']':
            tokens.append(']')
            i += 1
        elif s[i] == ',':
            tokens.append(',')
            i += 1
        # Check for identifiers
        elif s[i].isalpha() or s[i] == '_':
            start = i
            # Continue until we hit a non-identifier character
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                i += 1
            tokens.append(s[start:i])
        # Skip whitespace
        elif s[i].isspace():
            i += 1
        else:
            # Skip unrecognized characters
            i += 1
            
        # Skip whitespace between tokens
        while i < len(s) and s[i].isspace():
            i += 1
            
    return tokens



# === Recursive Descent Parser ===
def is_identifier(token):
    # Check if token is empty
    if not token:
        return False
        
    # Check first character (must be letter or underscore)
    if not (token[0].isalpha() or token[0] == '_'):
        return False
        
    # Check remaining characters (can be letters, digits, or underscores)
    for char in token[1:]:
        if not (char.isalnum() or char == '_'):
            return False
            
    return True

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, token=None):
        cur = self.current()
        if token and cur != token:
            raise ValueError(f"Expected token '{token}' but found '{cur}'")
        self.pos += 1
        return cur

    def parse_assignment(self):
        # <assignment> ::= <tensor_access> "+=" <expr>
        #                | <tensor_access> "=" <expr>
        lhs = self.parse_access()  # Left-hand side must be an access (e.g., Out2[i, P2[j]])
        if self.current() == "=":
            self.consume("=")
            rhs = self.parse_expr()
            return AssignmentNode(lhs, rhs, scatter_add=False)
        elif self.current() == "+=":
            self.consume("+=")
            rhs = self.parse_expr()
            return AssignmentNode(lhs, rhs, scatter_add=True)
        else :
            raise ValueError(f"Expected '+=' but found '{self.current()}'")

    def parse_expr(self):
        # <expr> ::= <term> ( "*" <term> )*
        node = self.parse_term()
        factors = [node]
        while self.current() == "*":
            self.consume("*")
            factors.append(self.parse_term())
        if len(factors) == 1:
            return factors[0]
        else:
            return MulNode(factors)

    def parse_term(self):
        # <term> ::= <access> | <index>
        # Look ahead: if the current token is an identifier and the next token is "[", it's an access.
        if not self.current() or not is_identifier(self.current()):
            raise ValueError(f"Expected identifier, found '{self.current()}'")
        token = self.consume()  # identifier
        if self.current() == "[":
            return self.parse_access_with_id(token)
        else:
            return IndexNode(token)

    def parse_access(self):
        # <access> ::= <identifier> "[" <indices> "]"
        if not self.current() or not is_identifier(self.current()):
            raise ValueError("Expected tensor name")
        id_token = self.consume()
        return self.parse_access_with_id(id_token)

    def parse_access_with_id(self, id_token):
        if self.current() != "[":
            raise ValueError(f"Expected '[' after tensor name '{id_token}'")
        self.consume("[")  # consume '['
        args = self.parse_indices()
        if self.current() != "]":
            raise ValueError(f"Expected ']' after indices in access '{id_token}'")
        self.consume("]")  # consume ']'
        return AccessNode(id_token, args)

    def parse_indices(self):
        # <indices> ::= <index> ( "," <index> )*
        args = [self.parse_index()]
        while self.current() == ",":
            self.consume(",")
            args.append(self.parse_index())
        return args

    def parse_index(self):
        # <index> ::= <term>
        return self.parse_term()

def parse_einsum(einsum_str):
    tokens = tokenize(einsum_str)
    parser = Parser(tokens)
    ast_tree = parser.parse_assignment()
    return ast_tree

def parse_einsum_expression(expression, **inputs):
    """
    Parses an einsum-like expression with gather-scatter semantics.
    
    Example: "Out[i,P[j]] += A[X[p,Y[q]], i, i] * B[p,j,Y[q]]"
    
    Returns:
    - output_tensor: str
    - output_indices: list
    - input_tensors: dict {tensor_name: (indices, operation)}
    """
    
    # 0. sympify
    ast_tree = parse_einsum(expression)
    lhs = ast_tree.args[0]
    rhs = ast_tree.args[1]
    
    # 1 for each input operand, extract size of symbols.
    # A[X[p,Y[q]], i, i] -> p : X.shape[0], q : Y.shape[0], i : A.shape[1], i : A.shape[2] 
    # B[p,j,Y[q]] -> p : B.shape[0], j : B.shape[1], q : Y.shape[0]
    # Out[i,P[j]] -> i : Out.shape[0], j : P.shape[0]
    filter_access = ast_tree.extract_unique_accesses()
    shapeenv = {}
    for access in filter_access:
        tensorname = access.func
        for dim, index in enumerate(access.args) :
            if not isinstance(index, IndexNode):
                continue
            indexname = index.func
            if indexname in shapeenv:
                shapeenv[indexname].append((tensorname, dim))
            else :
                shapeenv[indexname] = [(tensorname, dim)]

    # 2 assert dim test
    # i : A.shape[1] == A.shape[2]
    # p : X.shape[0] == B.shape[0]
    # q : Y.shape[0] == Y.shape[0]
    # TODO : implement

   
    localenv = inputs
    result = ast_tree.run(localenv, shapeenv)

    return result 



def Insum(expression, **tensors):
    output = parse_einsum_expression(expression, **tensors)
    return output 

