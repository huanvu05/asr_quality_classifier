import json

nb_path = "/Users/admin/Documents/AI_ThucChien/asr_quality_classifier/notebook2d72167614 (1).ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type", "")
    source = "".join(cell.get("source", ""))
    
    outputs = cell.get("outputs", [])
    out_text = ""
    for out in outputs:
        if out.get("output_type") == "stream":
            out_text += "".join(out.get("text", ""))
        elif out.get("output_type") == "execute_result":
            out_text += "".join(out.get("data", {}).get("text/plain", ""))
            
    if cell_type == "code" and any(cmd in source for cmd in ["python ", "train_", "run_"]):
        print(f"\n=========================================")
        print(f"Cell {idx} (code): {source.strip()[:150]}")
        print("=========================================")
        lines = out_text.strip().split("\n")
        if lines:
            print(f"Output (showing up to 25 lines):")
            # print up to 25 lines of output, especially the end of output
            for line in lines[-25:]:
                print(line)
        else:
            print("No output")
