# DynaCDR: Dynamic Graph Learning for Accurate Cancer Drug Response Prediction

--- 

This directory is **not tracked by git**. All contents are distributed via the project's Zenodo archive and must be downloaded before running any training or evaluation script.

> **Zenodo DOI:** [`10.5281/zenodo.23522084`](https://doi.org/10.5281/zenodo.23522084) 

---

## 1. Download & extract

### From the Zenodo web page

1. Open the Zenodo record.

2. Download **`DynaCDR_data_v1.0.0.zip`** .

3. Extract it into the repository root:
   
   ```bash
   unzip DynaCDR_data_v1.0.0.zip -d .
   
   # -> creates ./DATA/ with all cohort subdirectories
   ```

### From the command line

```bash
wget -O DynaCDR_data_v1.0.0.zip \
     https://zenodo.org/record/23522084/files/DynaCDR_data_v1.0.0.zip
unzip DynaCDR_data_v1.0.0.zip -d .
```

### Verify integrity

```bash
cd DATA
md5sum -c MANIFEST.md5
cd ..
```

If any file fails the check, re-download the archive.

---

## 2. Directory layout

```
DATA/
├── README.md                 # this file
├── MANIFEST.md5              # checksums for every file below
│
├── GDSC_data/                # training cohort
├── CCLE_data/                # regression external validation
├── TCGA_data/                # classification external validation
└── Tahoe_data/               # regression external validation
```

---

## 3. File manifest

### 3.1 `GDSC_data/` — training cohort

| File                                | Description                                          |
| ----------------------------------- | ---------------------------------------------------- |
| `raw_expression_data.csv`           | Gene expression profile                              |
| `raw_copynumber_data.csv`           | Copy number variation profile                        |
| `raw_methylation_data.csv`          | DNA methylation profile                              |
| `raw_mutation_data.csv`             | Mutation profile                                     |
| `cell_lines_name.csv`               | Ordered cell line names                              |
| `cell_drug_pair_splited.csv`        | `[cell, drug, LN_IC50, binary, cross1,..., cross10]` |
| `drug_MorganFingerprint(2048).csv`  | 2048-bit Morgan fingerprint (radius 2)               |
| `drug_3D_embedding.csv`             | Graph2Vec embedding of the molecular graph           |
| `drug_physicochemical_property.csv` | RDKit `CalcMolDescriptors` output                    |

### 3.2 `CCLE_data/` — regression external validation

Same schema as `GDSC_data/` with the `CCLE_*` prefix, **except** the label
file:

| File                       | Description                  |
| -------------------------- | ---------------------------- |
| `CCLE_cell_drug_label.csv` | `[cell, drug, ..., LN_IC50]` |

The other CSVs are named `CCLE_raw_expression_data.csv`, `CCLE_raw_copynumber_data.csv`, `CCLE_raw_methylation_data.csv`, `CCLE_raw_mutation_data.csv`, `CCLE_cell_lines_name.csv`, `CCLE_drug_Fingerprint.csv`, `CCLE_drug_3D_embedding.csv`, `CCLE_drug_physicochemical_property.csv`.

### 3.3 `TCGA_data/` — classification external validation

Same schema as GDSC, **except** the label file:

| File                    | Description                    |
| ----------------------- | ------------------------------ |
| `sample_drug_label.csv` | `[patient, ..., binary_label]` |

### 3.4 `Tahoe_data/` — regression external validation

Tahoe reuses **GDSC omics**; only the following files are specific to Tahoe:

| File                                      | Description                      |
| ----------------------------------------- | -------------------------------- |
| `Tahoe_cell_lines_name.csv`               | cell lines overlapping with GDSC |
| `Tahoe_cell_drug_label_binary.csv`        | `[cell, ..., LN_IC50]`           |
| `Tahoe_drug_MorganFingerprint(2048).csv`  | drug descriptor                  |
| `Tahoe_drug_3D_embedding.csv`             | drug descriptor                  |
| `Tahoe_drug_physicochemical_property.csv` | drug descriptor                  |
