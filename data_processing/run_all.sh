#!/usr/bin/env bash
# Full DynaCDR GDSC data-processing pipeline.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. cell line table =="
python 01_build_cell_line_table.py

echo "== 2. filter cell lines =="
python 02_filter_cell_lines.py

echo "== 3. drug PubChem fetch =="
python 03_fetch_drug_pubchem.py --resume

echo "== 4. merge IC50 labels =="
python 04_merge_ic50_labels.py

echo "== 5. omics (expression / mutation / methylation / copynumber) =="
python 05_integrate_omics.py all

echo "== 6. drug Morgan fingerprint + physicochemical =="
python 06_build_drug_fingerprints.py

echo "== 7. drug Graph2Vec embedding =="
python 07_build_drug_graph_embedding.py

echo "== done. Copy work/*.csv into DATA/GDSC_data/ =="
