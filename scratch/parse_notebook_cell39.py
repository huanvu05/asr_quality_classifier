import json

nb_path = "/Users/admin/Documents/AI_ThucChien/asr_quality_classifier/notebook2d72167614 (1).ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb.get("cells", [])[39]
outputs = cell.get("outputs", [])
out_text = ""
for out in outputs:
    if out.get("output_type") == "stream":
        out_text += "".join(out.get("text", ""))
    elif out.get("output_type") == "execute_result":
        out_text += "".join(out.get("data", {}).get("text/plain", ""))

print(f"--- Cell 39 Output ---")
print(out_text)
