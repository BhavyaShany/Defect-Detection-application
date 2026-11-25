"""Static wiring check: internal imports resolve, and no obviously undefined names.

Useful when the full runtime stack (FastAPI, SQLAlchemy, TensorFlow) is not installed.
    python -m tools.check_wiring
"""
import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mods = {}
for p in (ROOT / "backend").rglob("*.py"):
    rel = p.relative_to(ROOT).with_suffix("")
    name = ".".join(rel.parts)
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    mods[name] = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def defined(tree):
    names = set(dir(builtins)) | {"__file__", "__name__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            if not isinstance(n, ast.ClassDef):
                a = n.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
                    names.add(arg.arg)
        elif isinstance(n, ast.Lambda):
            for arg in n.args.args:
                names.add(arg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


errors = []
for name, tree in mods.items():
    defs = defined(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("backend"):
            target = mods.get(n.module)
            if target is None:
                errors.append(f"{name}: imports missing module {n.module}")
                continue
            top = defined(target)
            for a in n.names:
                if a.name not in top and f"{n.module}.{a.name}" not in mods:
                    errors.append(f"{name}: '{a.name}' not defined in {n.module}")
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defs:
            errors.append(f"{name}:{n.lineno}: undefined name '{n.id}'")

print("\n".join(errors) if errors else f"wiring OK ({len(mods)} modules)")
sys.exit(1 if errors else 0)
