from utils.countdown import compute_score
import itertools
from functools import lru_cache

import re

def simplify_parentheses(expr: str) -> str:
    expr = expr.strip()

    # Rule 1: Remove outermost parentheses if fully balanced and safe
    while expr.startswith('(') and expr.endswith(')') and is_balanced(expr[1:-1]):
        inner = expr[1:-1].strip()
        if is_balanced(inner):
            expr = inner
        else:
            break

    # Rule 2: Remove parentheses around a multiplication or division if directly followed by + or -
    # Example: (5 * (88 - 74)) - 20 => 5 * (88 - 74) - 20
    expr = re.sub(r'\((\s*[^()]+\s*[*\/]\s*\([^()]+\))\)', r'\1', expr)

    # Rule 3: Apply this transformation recursively inside the expression
    pattern = re.compile(r'([+-])\s*\(([^()]+)\)')

    def repl(match):
        op = match.group(1)
        inner = match.group(2)

        tokens = re.split(r'(\+|\-)', inner)
        tokens = [t.strip() for t in tokens if t.strip()]

        if op == '+':
            return ' + ' + ' '.join(tokens)
        elif op == '-':
            new_tokens = []
            for i in range(0, len(tokens), 2):
                sign = tokens[i - 1] if i > 0 else '+'
                val = tokens[i]
                if sign == '+':
                    new_tokens.append('- ' + val)
                elif sign == '-':
                    new_tokens.append('+ ' + val)
            return ' '.join(new_tokens)

    while True:
        new_expr = pattern.sub(repl, expr)
        if new_expr == expr:
            break
        expr = new_expr

    return expr.strip()

def is_balanced(s: str) -> bool:
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        if count < 0:
            return False
    return count == 0

def safe_eval(expr):
    try:
        val = eval(expr)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
    except ZeroDivisionError:
        return None
    except:
        return None
    return None

def find_expression(numbers, target):
    operators = ['+', '-', '*', '/']
    seen_exprs = set()

    @lru_cache(maxsize=None)
    def dfs(nums_tuple):
        if len(nums_tuple) == 1:
            return [str(nums_tuple[0])]

        expressions = []
        for i in range(1, len(nums_tuple)):
            left_parts = dfs(nums_tuple[:i])
            right_parts = dfs(nums_tuple[i:])
            for left in left_parts:
                for right in right_parts:
                    for op in operators:
                        expr = f"({left} {op} {right})"
                        if expr not in seen_exprs:
                            seen_exprs.add(expr)
                            expressions.append(expr)
        return expressions

    for perm in itertools.permutations(numbers):
        exprs = dfs(perm)
        for expr in exprs:
            score = compute_score(f"Assistant: <answer>{expr}</answer>", {"target": target, "numbers": numbers})
            if score == 1.0:
                return expr
    return None

def generate_proposal(numbers, target):
    expr = find_expression(numbers, target)
    return expr if expr else ""

def get_answer(item):
    numbers = item["nums"]
    target = item["target"]
    proposal = generate_proposal(numbers, target)
    if proposal:
        return simplify_parentheses(proposal)
    return ""

def process_item(item):
    # Must not rely on external state unless it's pickle-safe
    item["answer"] = get_answer(item)
    return item
