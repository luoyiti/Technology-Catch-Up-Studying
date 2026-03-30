import json
from pathlib import Path


nb = json.loads(Path("/backup/cluster_fuzzy.ipynb").read_text())
errors = []

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    try:
        compile(src, f"cell_{i}", "exec")
    except Exception as exc:
        errors.append((i, type(exc).__name__, str(exc)))

print("n_errors", len(errors))
for item in errors:
    print(item)
