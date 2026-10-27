"""Lightweight capability detection, usable by any code-agent framework."""
import ast


def _uses_shared_training_entry(tree: ast.Module) -> bool:
    """Recognize the versioned library loop, not a self-declared capability flag.

    This is capability detection for trusted candidate code, not a sandbox.
    Only a direct, unshadowed canonical entry in the main guard qualifies.
    """
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "localml_scheduler.adapters.shared_training"]
    bindings = [alias.asname or alias.name for node in imports for alias in node.names
                if alias.name == "run_candidate_training"]
    if len(bindings) != 1:
        return False
    binding = bindings[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == binding:
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == binding:
            return False
        if isinstance(node, ast.arg) and node.arg == binding:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in imports:
            if any((alias.asname or alias.name.split(".")[0]) == binding for alias in node.names):
                return False
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == binding]
    if len(calls) != 1:
        return False
    call = calls[0]
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name) or call.keywords:
        return False
    guard = ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body)
    return any(isinstance(node, ast.If) and ast.dump(node.test) == guard
               and not node.orelse and len(node.body) == 1
               and isinstance(node.body[0], ast.Expr) and node.body[0].value is call
               for node in tree.body)


def supports_shared_training_source(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return _uses_shared_training_entry(tree)
