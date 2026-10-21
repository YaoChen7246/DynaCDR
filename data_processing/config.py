"""Centralised paths and constants for the data-processing pipeline."""
from pathlib import Path

# --- raw data ---------------------------------------------------------------
RAW_ROOT = Path('./dataset')
GDSC_DIR = RAW_ROOT / 'GDSC'
CCLE_DIR = RAW_ROOT / 'CCLE'
TCGA_DIR = RAW_ROOT / 'TCGA'
Tahoe_DIR = RAW_ROOT / 'Tahoe'

GDSC1_XLSX = GDSC_DIR / 'GDSC1_fitted_dose_response_27Oct23.xlsx'
GDSC2_XLSX = GDSC_DIR / 'GDSC2_fitted_dose_response_27Oct23.xlsx'
EXPR_TXT = GDSC_DIR / 'Cell_line_RMA_proc_basalExpression.txt'
METH_GZ = GDSC_DIR / 'GSE68379_Matrix.processed.txt.gz'
METH_SOFT = GDSC_DIR / 'GSE68379_family.soft'
MUT1_CSV = GDSC_DIR / 'Mutation_GDSC1.csv'
MUT2_CSV = GDSC_DIR / 'Mutation_GDSC2.csv'
CNV_CSV = GDSC_DIR / 'WES_pureCN_CNV_genes_20221213.csv'


# --- intermediate / output --------------------------------------------------
WORK_DIR = Path('./data_processing/work')
WORK_DIR.mkdir(parents=True, exist_ok=True)

CELL_LINES_ALL = WORK_DIR / 'cell_lines_all.txt'
CELL_LINES_FILTERED = WORK_DIR / 'final_cell_lines_name.txt'
DRUG_INFORMATION = WORK_DIR / 'drug_information.csv'
CELL_DRUG_LABEL = WORK_DIR / 'cell_drug_label.csv'

# 4 omics outputs (unified format: cell_line_name + features)
OUT_EXPRESSION = WORK_DIR / 'raw_expression_data.csv'
OUT_MUTATION = WORK_DIR / 'raw_mutation_data.csv'
OUT_METHYLATION = WORK_DIR / 'raw_methylation_data.csv'
OUT_COPYNUMBER = WORK_DIR / 'raw_copynumber_data.csv'

# drug descriptors
OUT_DRUG_SMILE = WORK_DIR / 'drug_smile.csv'
OUT_MORGAN = WORK_DIR / 'drug_MorganFingerprint(2048).csv'
OUT_PHYSCHEM = WORK_DIR / 'drug_physicochemical_property.csv'
OUT_DRUG_3D = WORK_DIR / 'drug_3D_embedding.csv'
