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
            
    if "run_kaggle.py" in source:
        print(f"\n=========================================")
        print(f"Cell {idx} (code): {source.strip()[:100]}")
        print("=========================================")
        # Print the last 15 lines of output if available
        lines = out_text.strip().split("\n")
        if lines:
            print("Output (last 20 lines):")
            for line in lines[-20:]:
                print(line)
        else:
            print("No output")
