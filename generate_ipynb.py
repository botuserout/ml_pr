import json
import os

py_file = "food_nutrition_ml_project_multisource_complete.py"
ipynb_file = "food_nutrition_ml_project.ipynb"

with open(py_file, "r", encoding="utf-8") as f:
    code = f.read()

# Split code into sections by header comments
sections = code.split("# ==============================================================================")

cells = []

# Add initial pip install cell for Google Colab
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install required dependencies in Google Colab environment\n",
        "!pip -q install requests beautifulsoup4 scikit-learn seaborn matplotlib pandas numpy joblib pytest\n"
    ]
})

for sec in sections:
    sec = sec.strip()
    if not sec:
        continue
    
    lines = sec.split("\n")
    md_lines = []
    code_lines = []
    
    in_md = True
    for line in lines:
        if line.startswith("# SECTION") or line.startswith("# ##") or line.startswith("# #") or line.startswith("# **") or line.startswith("# ---") or (in_md and line.startswith("#")):
            # Remove leading '#' and space for markdown
            clean_md = line[1:].strip() if line.startswith("#") else line
            md_lines.append(clean_md + "\n")
        else:
            in_md = False
            code_lines.append(line + "\n")
    
    if md_lines:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines
        })
        
    if code_lines and "".join(code_lines).strip():
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        })

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(ipynb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated {ipynb_file} with {len(cells)} cells.")
