import os
import zipfile
import shutil

team_name = "Enterprise_Resolution_AI"
zip_filename = f"{team_name}_submission.zip"

print(f"Creating submission package: {zip_filename}...")

# Check files
output_dir = "output"
matching_file = os.path.join(output_dir, "matching_results.tsv")
candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
doc_file = "Documentation_template.md"
code_dir = os.path.join("code", "business_entity_resolution")

assert os.path.isfile(matching_file), f"Missing {matching_file}"
assert os.path.isfile(candidate_file), f"Missing {candidate_file}"
assert os.path.isfile(doc_file), f"Missing {doc_file}"
assert os.path.isdir(code_dir), f"Missing {code_dir}"

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. output/
    zipf.write(matching_file, arcname="output/matching_results.tsv")
    zipf.write(candidate_file, arcname="output/candidate_pairs.tsv")
    
    # 2. Documentation_template.md
    zipf.write(doc_file, arcname="Documentation_template.md")
    
    # 3. code/business_entity_resolution/
    for root, dirs, files in os.walk(code_dir):
        # Exclude __pycache__
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".pyc") or f == ".DS_Store":
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, "code")
            arcname = os.path.join("code", rel_path)
            zipf.write(full_path, arcname=arcname)

print(f"Successfully created {zip_filename}!")
# List zip contents
with zipfile.ZipFile(zip_filename, "r") as zipf:
    print("\nZip package contents:")
    for info in zipf.infolist():
        print(f"  {info.filename} ({info.file_size:,} bytes)")
