import lark

GRAMMAR = r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""

parser = lark.Lark(GRAMMAR)

# Test simple scheduling
tree = parser.parse("[cat:dog:0.5]")
print("=== [cat:dog:0.5] ===")
print(tree.pretty())

# Find the scheduled node
for node in tree.iter_subtrees():
    if node.data == "scheduled":
        print(f"scheduled children count: {len(node.children)}")
        for i, child in enumerate(node.children):
            print(f"  [{i}]: {type(child).__name__} = {repr(child)[:60]}")
