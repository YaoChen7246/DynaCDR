# DynaCDR: Dynamic Graph Learning for Accurate Cancer Drug Response Prediction

---

## 1. Overview

This repository contains Python codes and datasets necessary to run the DynaCDR model. DynaCDR is a Python implementation of a multi-modal deep learning framework for cancer drug response (CDR) prediction, built upon a dynamically constructed batch-local heterogeneous graph strategy. Unlike static graph-based models that construct a global graph over the whole dataset, which risks information leakage by mixing training and test instances through fixed topological connections. DynaCDR constructs local subgraphs exclusively from samples within each training batch, ensuring strict separation between training and test instances while still capturing context-dependent drug–cell line interactions.

---

## 2. Architecture

```
      Cell multi-omics                          Drug descriptors
 ┌──────────┬──────────┬──────────┬──────────┐   ┌──────────┬──────────┬──────────┐
 │  Exp     │  CNV     │  Meth    │  Mut     │   │ Finger   │ Graph2Vec│ Physchem │
 │  MLP     │  MLP     │  MLP     │  MLP     │   │  MLP     │  raw     │  MLP     │
 └────┬─────┴────┬─────┴────┬─────┴────┬─────┘   └────┬─────┴────┬─────┴────┬─────┘
      │          │          │          │              │          │          │
      └──────┬───┴──────────┴──────────┘              └────┬─────┴──────────┘
             │                                            │
        cell_fusion (128-D)                      drug_fusion (128-D)
             │                                            │
             └─────────────── concatenate ────────────────┘
                                 │
                    Batch-local heterogeneous graph
                    (edges built only within the batch)
                                 │
                        Multi-head GAT (×nheads)
                                 │
                        LayerNorm + FC projection
                                 │
                           Prediction MLP
                                 │
                       ┌─────────┴─────────┐
                  regression            classification
                  (raw scalar)        (Sigmoid → prob)
```

Key components live in `model_helper.py`:

- `GraphAttentionLayer` — single-head GAT layer.
- `MLP` — BatchNorm + ReLU stack used by all modality encoders.
- `DrugEmbedding` / `CellEmbedding` — modality-specific encoders.
- `PredictionMLP` — final prediction head.

---

## 3. Requirements

DynaCDR depends on the following packages:

- Python 3.8.5
- torch 2.4.1
- numpy 1.24.4
- pandas 2.0.3
- scikit-learn 1.3.2
- scipy 1.10.1
- openpyxl 3.1.2
- rdkit-pypi 2023.9.5
- pubchempy 1.0.4
- networkx 3.2.1
- karateclub 1.3.3

---

## 4. Data

### 4.1 Download from Zenodo

All pre-processed feature CSVs, drug descriptors, label files are archived on Zenodo under a single DOI:

> **Zenodo DOI:** [`10.5281/zenodo.XXXXXXX`](https://doi.org/10.5281/zenodo.XXXXXXX) **Version:** v1.0.0 (~X.X GB compressed)

Download the archive and extract it into the repo root:

```bash
# from the Zenodo web page
# download  DynaCDR_data_v1.0.0.zip  manually, then:
unzip DynaCDR_data_v1.0.0.zip -d .
# -> creates ./DATA/ with all cohort subdirectories

# from the command line (requires wget)
wget -O DynaCDR_data_v1.0.0.zip \
     https://zenodo.org/record/XXXXXXX/files/DynaCDR_data_v1.0.0.zip
unzip DynaCDR_data_v1.0.0.zip -d .
```

After extraction the layout is:

```
DATA/
├── GDSC_data/         # training cohort
├── CCLE_data/         # regression external validation
├── TCGA_data/         # classification external validation
└── Tahoe_data/        # regression external validation
```

### 4.2 Rebuilding from raw data (optional)

If you prefer to regenerate the features from raw GDSC / CCLE/ TCGA downloads instead of using the Zenodo archive, follow [`data_processing/README.md`](data_processing/README.md).

---

## 5. Usage

### 5.1 CV10 for DynaCDR

Train DynaCDR under 10-fold cross-validation on GDSC.

**Regression task (LN_IC50)**

```bash
python train.py --task regression
```

**Classification task (binary sensitivity)**

```bash
python train.py --task classification
```

### 5.2 External validation

| Task           | Script                    | Checkpoint                          |
| -------------- | ------------------------- | ----------------------------------- |
| Regression     | `python test_on_CCLE.py`  | `DynaCDR_regression_fold_*.pkl`     |
| Regression     | `python test_on_Tahoe.py` | `DynaCDR_regression_fold_*.pkl`     |
| Classification | `python test_on_TCGA.py`  | `DynaCDR_classification_fold_*.pkl` |

Each script accepts `--ckpt PATH`:

```bash
python test_on_CCLE.py  --ckpt ./output/DynaCDR_regression_fold_1.pkl
python test_on_Tahoe.py --ckpt ./output/DynaCDR_regression_fold_1.pkl
python test_on_TCGA.py  --ckpt ./output/DynaCDR_classification_fold_1.pkl
```

`test_on_Tahoe.py` additionally writes:

- `Tahoe_predicted.csv` — per-pair prediction, ground truth and index.
- `cell_drug_information.csv` — mapping between local indices and raw names.

## 6. Data Availability

The pre-processed features, labels and splits are archived at:

- **Zenodo:** [`10.5281/zenodo.XXXXXXX`](https://doi.org/10.5281/zenodo.XXXXXXX) (v1.0.0, CC-BY-4.0)
- **Manifest & checksums:** [`DATA/README.md`](DATA/README.md)

The raw upstream data can be obtained from the original sources:

- **GDSC:** [https://cellmodelpassports.sanger.ac.uk/](https://cellmodelpassports.sanger.ac.uk/)
- **CCLE:** [https://sites.broadinstitute.org/ccle/](https://sites.broadinstitute.org/ccle/)
- **TCGA:** [https://www.cancer.gov/ccg/research/genome-sequencing/tcga](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)

We do not redistribute the raw files; only the derived feature matrices
are hosted on Zenodo. 
