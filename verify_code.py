"""Check that the published scripts contain the same code as the originals.

Before publication, comments and docstrings were rewritten in English, and string
literals of four or more words that cited internal review steps (notes written into
the JSON outputs, log lines, error messages) were translated. This script computes,
for each script under audit/, the sha256 of its syntax tree with those parts masked:
docstrings are dropped, every plain string of four or more words becomes a fixed
placeholder, and an f-string whose literal text has four or more words keeps only its
interpolated expressions, in order. Everything else (names, calls, operators, numbers,
short strings such as keys, labels and file paths, and the expressions inside every
f-string) enters the digest, so any change to executable code fails the check.

CODE_AST_SHA256 lists the digest of each ORIGINAL script, computed before publication
with Python 3.11. The tree layout differs between Python versions, so run this with
Python 3.11.

Usage (from the repository root):  python3.11 verify_code.py
"""
import ast
import hashlib
import pathlib
import sys

PLACEHOLDER = "<text>"


def _is_docstring_holder(node):
    return isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


class _Mask(ast.NodeTransformer):
    def visit_JoinedStr(self, node):
        text = "".join(v.value if isinstance(v, ast.Constant) else "X" for v in node.values)
        if len(text.split()) >= 4:
            node.values = [v for v in node.values if not isinstance(v, ast.Constant)]
        return self.generic_visit(node)

    def visit_FormattedValue(self, node):
        node.value = self.visit(node.value)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value.split()) >= 4:
            return ast.Constant(value=PLACEHOLDER)
        return node


def code_sha256(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if _is_docstring_holder(n) and n.body:
            b = n.body[0]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) \
                    and isinstance(b.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    tree = _Mask().visit(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def main():
    if sys.version_info[:2] != (3, 11):
        print(f"warning: digests were computed with Python 3.11; this is {sys.version.split()[0]}")
    root = pathlib.Path(__file__).resolve().parent
    expected = {}
    for line in (root / "CODE_AST_SHA256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    bad = 0
    for rel, digest in sorted(expected.items()):
        ok = code_sha256(root / rel) == digest
        bad += not ok
        if not ok:
            print(f"MISMATCH {rel}")
    print(f"{len(expected) - bad}/{len(expected)} scripts: code identical to the original")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
