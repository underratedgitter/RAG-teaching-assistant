"""Loading the pipeline scripts' functions into a test namespace.

The scripts are top-level scripts, so the tests pull the pieces they need out
of them. Cutting the source out by hand also ran whatever module-level
statements sat between the functions — and one of those read the audios/
directory, which a fresh checkout does not have, so the tests passed on a
developer machine and failed in CI. Parsing the source and compiling only the
named definitions cannot do that: nothing unnamed ever runs.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent

# The module-level settings mp3_to_json.py derives from the environment before
# it loads a model. load_model() reads them, so they travel with it.
WHISPER_CONFIG = {
    "device", "_default_compute", "_compute", "_default_model", "_preferred",
    "_model_ladder", "_compute_ladder", "_cpu_threads",
}


def load(path, functions=(), constants=(), namespace=None):
    """Compile the named functions and constants of a script.

    `functions` may be "all" for every top-level function. `constants` names
    module-level assignments to compile as well; a constant assigned inside a
    branch brings its whole branch with it. Anything not named stays out —
    module-level code is what reads directories, loads models and talks to
    the network.
    """
    source = ROOT / path
    tree = ast.parse(source.read_text())
    wanted_functions = None if functions == "all" else set(functions)
    wanted_constants = set(constants)

    body = []
    found = set()
    found_constants = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if wanted_functions is None or node.name in wanted_functions:
                body.append(node)
                found.add(node.name)
        elif isinstance(node, ast.Assign):
            if _assigns(node, wanted_constants):
                body.append(node)
                found_constants |= _targets(node)
        elif isinstance(node, ast.If) and _assigns_anywhere(node, wanted_constants):
            body.append(node)
            found_constants |= {
                target.id
                for child in ast.walk(node)
                if isinstance(child, ast.Assign)
                for target in child.targets
                if isinstance(target, ast.Name)
            }

    missing = (set() if wanted_functions is None else set(functions)) - found
    if missing:
        raise LookupError(f"{path} no longer defines {sorted(missing)}")
    missing = wanted_constants - found_constants
    if missing:
        raise LookupError(f"{path} no longer defines {sorted(missing)}")

    ns = dict(namespace or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), ns)
    return ns


def _targets(node):
    return {target.id for target in node.targets if isinstance(target, ast.Name)}


def _assigns(node, names):
    return bool(_targets(node) & names)


def _assigns_anywhere(node, names):
    return any(
        isinstance(child, ast.Assign) and _assigns(child, names)
        for child in ast.walk(node)
    )
