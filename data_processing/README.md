# Data Processing Pipeline

Scripts that turn raw GDSC downloads into the feature CSVs consumed by `data_loader.py` — i.e. the same files that are distributed on Zenodo and
described in [`../DATA/README.md`](../DATA/README.md).

If you only want to run DynaCDR, you do not need this folder — just download the archive from Zenodo as described in the main README.

Use this pipeline if you want to:

- rebuild the features from scratch,
- verify the Zenodo CSVs bit-for-bit,
- extend the pipeline to a new cohort.

See the main [README](../README.md) for how these CSVs are used during training and evaluation.

---

## Prerequisites

```bash
pip install pandas openpyxl rdkit-pypi pubchempy networkx karateclub
```

---

## Raw Data Layout

Place the raw files under `./dataset/`:

```
dataset/
├── GDSC/
│   ├── GDSC1_fitted_dose_response_27Oct23.xlsx
│   ├── GDSC2_fitted_dose_response_27Oct23.xlsx
│   ├── Cell_line_RMA_proc_basalExpression.txt
│   ├── GSE68379_Matrix.processed.txt.gz
│   ├── GSE68379_family.soft
│   ├── Mutation_GDSC1.csv
│   └── Mutation_GDSC2.csv
├── CCLE/
├── TCGA/
└── Tahoe/
```

All paths are declared in [`config.py`](config.py); override them on the command line if your layout differs.

---

## Output CSV Format

All four omics matrices share the **same shape convention**:

- exactly **one header line**;
- **first column** is `cell_line_name` (unquoted);
- remaining columns are features;
- rows are indexed by cell line.

---

## Running

### Run the whole pipeline

```bash
bash data_processing/run_all.sh
```

### Run a single step

```bash
python data_processing/05_integrate_omics.py expression
python data_processing/05_integrate_omics.py mutation
python data_processing/05_integrate_omics.py methylation
python data_processing/05_integrate_omics.py copynumber
```

---

## Notes

Raw data is **not** included in the repository. Download it from the [GDSC](https://cellmodelpassports.sanger.ac.uk/) portals.
