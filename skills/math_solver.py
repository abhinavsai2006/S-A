"""
Igris AI Agent — Math Solver Skill  (Issue #1 — OpenClaw skill)

Justification:
LLMs are notoriously unreliable at arithmetic. This skill provides a safe
math evaluation environment using Python's ast module (no eval/exec) so
the agent can compute exact results for mathematical expressions.
"""

import ast
import operator
import math
from langchain.tools import tool


# Safe operators whitelist
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions whitelist
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):  # numbers, strings
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            func = _SAFE_FUNCTIONS[node.func.id]
            args = [_safe_eval(arg) for arg in node.args]
            if callable(func):
                return func(*args)
            else:
                return func  # constants like pi, e
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val  # constants like pi, e
        raise ValueError(f"Unsupported variable: {node.id}")
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, //, %, ** and functions like sqrt, sin, cos, tan, log, factorial.
    Use this when the user asks you to calculate something or do math."""
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)

        # Format result nicely
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return f"{expression} = {int(result)}"
            return f"{expression} = {result:.10g}"
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as e:
        return f"Error: {e}"
    except SyntaxError:
        return f"Error: Invalid mathematical expression: '{expression}'"
    except Exception as e:
        return f"Calculation error: {e}"


MATH_SOLVER_TOOLS = [calculate]
