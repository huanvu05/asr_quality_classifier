import json

nb_path = "/Users/admin/Documents/AI_ThucChien/asr_quality_classifier/notebook2d72167614 (1).ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells: {len(cells)}")
for idx in range(40, len(cells)):
    cell = cells[idx]
    cell_type = cell.get("cell_type", "")
    source = "".join(cell.get("source", ""))
    
    outputs = cell.get("outputs", [])
    out_text = ""
    for out in outputs:
        if out.get("output_type") == "stream":
            out_text += "".join(out.get("text", ""))
        elif out.get("output_type") == "execute_result":
            out_text += "".join(out.get("data", {}).get("text/plain", ""))
            
    print(f"\n=========================================")
    print(f"Cell {idx} ({cell_type}): {source.strip()[:150]}")
    print("=========================================")
    if out_text:
        print("Output:")
        print(out_text.strip()[:1000])
    else:
        print("No output")
