"""
All dataset-specific behaviours are parameterised via DatasetConfig
"""

import csv
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import scale


# --------------------------------------------------------------------------- #
#  Shared primitives
# --------------------------------------------------------------------------- #
def _read_csv(path: str, skip_header: int = 0) -> np.ndarray:
    return np.genfromtxt(path, delimiter=',', skip_header=skip_header,
                         dtype=np.dtype(str))


def _read_label_csv(path: str, skip_header: int) -> np.ndarray:
    with open(path, 'r') as f:
        rows = list(csv.reader(f))
    return np.array(rows[skip_header:])


def _strip_quotes(arr: np.ndarray) -> np.ndarray:
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = arr[i, j].replace('"', '')
    return arr


def _data_fusion(omics: np.ndarray, cell_line,
                 fill_empty: Optional[str] = None) -> np.ndarray:
    cell_index = omics[:, 0].tolist()
    rows = []
    for cell in cell_line:
        idx = cell_index.index(cell)
        row = omics[idx, 1:]
        if fill_empty is not None:
            row = np.where(row == '', fill_empty, row)
        rows.append(np.asarray(row, dtype=float))
    return np.asarray(rows)


def _normalize_gaussian(x):
    x = np.asarray(x, dtype=float)
    std = x.std()
    return x - x.mean() if std == 0 else (x - x.mean()) / std


_FEATURE_SIZES = {'exp': 17419, 'copy': 16383, 'meth': 37677, 'mut': 6017}


def align_features(input_data, data_type: str) -> torch.Tensor:
    """Zero-pad / truncate omics features to a fixed length."""
    size = _FEATURE_SIZES[data_type]
    aligned = np.zeros((input_data.shape[0], size), dtype=float)
    if input_data.shape[1] <= size:
        aligned[:, :input_data.shape[1]] = input_data
    else:
        aligned = np.asarray(input_data)[:, :size]
    return torch.FloatTensor(aligned)


# --------------------------------------------------------------------------- #
#  Graph utilities
# --------------------------------------------------------------------------- #
def build_cell_drug_adj(cell_drug_pair, binary_label,
                        train_index, n_cell, n_drug) -> torch.Tensor:
    """Bipartite adjacency built from the given (training) pairs only.

        [[I_cell,  R  ],
         [ R^T  , I_drug]]
    """
    cell_drug = np.zeros((n_cell, n_drug), dtype=float)
    for i in np.asarray(train_index):
        c, d = int(cell_drug_pair[i, 0]), int(cell_drug_pair[i, 1])
        cell_drug[c, d] = binary_label[i]

    eye_c, eye_d = np.eye(n_cell), np.eye(n_drug)
    top = np.concatenate((eye_c, cell_drug), axis=1)
    bot = np.concatenate((cell_drug.T, eye_d), axis=1)
    return torch.tensor(np.concatenate((top, bot), axis=0), dtype=torch.float32)


def build_full_adj(cell_drug_pair, binary_label, n_cell, n_drug) -> torch.Tensor:
    """Adjacency built from *all* pairs; used for external validation."""
    return build_cell_drug_adj(cell_drug_pair, binary_label,
                               np.arange(len(cell_drug_pair)), n_cell, n_drug)


def build_identity_adj(n_cell: int, n_drug: int) -> torch.Tensor:
    """Adjacency containing only identity blocks (no cell-drug edges).
    """
    eye_c, eye_d = np.eye(n_cell, dtype=float), np.eye(n_drug, dtype=float)
    zero_cd = np.zeros((n_cell, n_drug), dtype=float)
    top = np.concatenate((eye_c, zero_cd), axis=1)
    bot = np.concatenate((zero_cd.T, eye_d), axis=1)
    return torch.tensor(np.concatenate((top, bot), axis=0), dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Label builders
# --------------------------------------------------------------------------- #
def _label_gdsc_regression(cell_line, drug_cid, label):
    reg = np.array(label[:, 2], dtype=float)
    binary = np.array(label[:, 3], dtype=float)
    pairs, bins = [], []
    for i in range(len(reg)):
        pairs.append([cell_line.index(label[i, 0]),
                      drug_cid.index(label[i, 1]),
                      reg[i]])
        bins.append(binary[i])
    return np.asarray(pairs, dtype=float), bins


def _label_gdsc_classification(cell_line, drug_cid, label):
    binary = np.array(label[:, 3], dtype=float)
    pairs, bins = [], []
    for i in range(len(binary)):
        pairs.append([cell_line.index(label[i, 0]),
                      drug_cid.index(label[i, 1]),
                      1.0 - binary[i]])   # keep (1 - x) convention
        bins.append(binary[i])
    return np.asarray(pairs, dtype=float), bins


def _label_ccle(cell_line, drug_cid, label):
    """IQR outlier filter + Gaussian normalization on column 4."""
    reg = np.array(label[:, 4], dtype=float)
    norm = _normalize_gaussian(reg)
    p25, _, p75 = np.percentile(norm, [25, 50, 75])
    iqr = p75 - p25
    up, down = p75 + 1.5 * iqr, p25 - 1.5 * iqr
    pairs, bins = [], []
    for i in range(len(norm)):
        if down < norm[i] < up and label[i, 0] in cell_line and label[i, 2] in drug_cid:
            pairs.append([cell_line.index(label[i, 0]),
                          drug_cid.index(label[i, 2]),
                          norm[i]])
            bins.append(0 if norm[i] > -0.8 else 1)
    return np.asarray(pairs, dtype=float), bins


def _label_tcga(cell_line, drug_cid, label):
    reg = np.array(label[:, 5], dtype=float)
    pairs, bins = [], []
    for i in range(len(reg)):
        if label[i, 0] in cell_line and label[i, 2] in drug_cid:
            pairs.append([cell_line.index(label[i, 0]),
                          drug_cid.index(label[i, 2]),
                          reg[i]])
            bins.append(reg[i])
    return np.asarray(pairs, dtype=float), bins


def _label_tahoe(cell_line, drug_cid, label):
    """Tahoe binary label file, using column 4 as target.

    Returns a 3-tuple (pairs, bins, info) where `info` carries the raw
    cell/drug names for downstream CSV export. `bins` is all-zeros because
    the Tahoe adjacency matrix uses identity blocks only.
    """
    reg = np.array(label[:, 4], dtype=float)
    pairs, info = [], []
    for i in range(len(reg)):
        if label[i, 0] in cell_line and label[i, 2] in drug_cid:
            c_idx = cell_line.index(label[i, 0])
            d_idx = drug_cid.index(label[i, 2])
            pairs.append([c_idx, d_idx, reg[i]])
            info.append([c_idx, d_idx, label[i, 0], label[i, 1], label[i, 2]])
    bins = [0.0] * len(pairs)
    return np.asarray(pairs, dtype=float), bins, info


# --------------------------------------------------------------------------- #
#  Dataset configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    name: str
    path: str
    omics: Dict[str, Dict]
    fingerprint_file: str
    drug_graph_file: str
    physicochemical_file: str
    label_file: str
    label_builder: Callable
    label_skip_header: int = 1
    cell_file: Optional[str] = None
    cell_skip_header: int = 1
    do_scale: Dict[str, bool] = field(
        default_factory=lambda: {'exp': True, 'copy': True,
                                 'meth': True, 'mut': False})
    align: bool = False
    fill_empty: Optional[str] = None
    omics_path: Optional[str] = None    # defaults to `path` when None


# --------------------------------------------------------------------------- #
#  Preset configs
# --------------------------------------------------------------------------- #
GDSC = DatasetConfig(
    name='GDSC',
    path='./DATA/GDSC_data/',
    omics={
        'exp':  {'file': 'raw_expression_data.csv',  'skip_header': 2},
        'copy': {'file': 'raw_copynumber_data.csv',  'skip_header': 1},
        'meth': {'file': 'raw_methylation_data.csv', 'skip_header': 1},
        'mut':  {'file': 'raw_mutation_data.csv',    'skip_header': 2},
    },
    fingerprint_file='drug_MorganFingerprint(2048).csv',
    drug_graph_file='drug_3D_embedding.csv',
    physicochemical_file='drug_physicochemical_property.csv',
    label_file='cell_drug_pair_splited.csv',
    label_skip_header=1,
    cell_file='cell_lines_name.csv',
    cell_skip_header=1,
    label_builder=_label_gdsc_regression,
    align=False,
)

CCLE = DatasetConfig(
    name='CCLE',
    path='./DATA/CCLE_data/',
    omics={
        'exp':  {'file': 'CCLE_raw_expression_data.csv',  'skip_header': 2},
        'copy': {'file': 'CCLE_raw_copynumber_data.csv',  'skip_header': 1},
        'meth': {'file': 'CCLE_raw_methylation_data.csv', 'skip_header': 1},
        'mut':  {'file': 'CCLE_raw_mutation_data.csv',    'skip_header': 2},
    },
    fingerprint_file='CCLE_drug_Fingerprint.csv',
    drug_graph_file='CCLE_drug_3D_embedding.csv',
    physicochemical_file='CCLE_drug_physicochemical_property.csv',
    label_file='CCLE_cell_drug_label.csv',
    label_skip_header=1,
    cell_file='CCLE_cell_lines_name.csv',
    cell_skip_header=1,
    label_builder=_label_ccle,
    align=True,
)

TCGA = DatasetConfig(
    name='TCGA',
    path='./DATA/TCGA_data/',
    omics={
        'exp':  {'file': 'TCGA_raw_expression_data.csv',  'skip_header': 0},
        'copy': {'file': 'TCGA_raw_copynumber_data.csv',  'skip_header': 0},
        'meth': {'file': 'TCGA_raw_methylation_data.csv', 'skip_header': 0},
        'mut':  {'file': 'TCGA_raw_mutation_data.csv',    'skip_header': 0},
    },
    fingerprint_file='TCGA_drug_MorganFingerprint(2048).csv',
    drug_graph_file='TCGA_drug_3D_embedding.csv',
    physicochemical_file='TCGA_drug_physicochemical_property.csv',
    label_file='sample_drug_label.csv',
    label_skip_header=1,
    cell_file=None,               # derived from exp column 0
    cell_skip_header=0,
    label_builder=_label_tcga,
    align=True,
    fill_empty='0',
)

# Tahoe: uses GDSC omics + Tahoe cell lines / drugs / labels.
Tahoe = DatasetConfig(
    name='Tahoe',
    path='./DATA/Tahoe_data/',
    omics_path='./DATA/GDSC_data/',         # omics come from GDSC
    omics={
        'exp':  {'file': 'raw_expression_data.csv',  'skip_header': 2},
        'copy': {'file': 'raw_copynumber_data.csv',  'skip_header': 1},
        'meth': {'file': 'raw_methylation_data.csv', 'skip_header': 1},
        'mut':  {'file': 'raw_mutation_data.csv',    'skip_header': 2},
    },
    fingerprint_file='Tahoe_drug_MorganFingerprint(2048).csv',
    drug_graph_file='Tahoe_drug_3D_embedding.csv',
    physicochemical_file='Tahoe_drug_physicochemical_property.csv',
    label_file='Tahoe_cell_drug_label.csv',
    label_skip_header=1,
    cell_file='Tahoe_cell_lines_name.csv',
    cell_skip_header=1,
    label_builder=_label_tahoe,
    align=False,
)

_REGISTRY = {'GDSC': GDSC, 'CCLE': CCLE, 'TCGA': TCGA, 'Tahoe': Tahoe}


# --------------------------------------------------------------------------- #
#  Internal loader
# --------------------------------------------------------------------------- #
def _load(cfg: DatasetConfig) -> Dict:
    p = cfg.path
    omics_p = cfg.omics_path if cfg.omics_path is not None else p

    # ---- 1. omics matrices ----
    omics_raw = {
        k: _strip_quotes(_read_csv(omics_p + v['file'], v['skip_header']))
        for k, v in cfg.omics.items()
    }

    # ---- 2. cell line names ----
    if cfg.cell_file is None:
        cell_line = list(omics_raw['exp'][:, 0])
    else:
        cell_line = _read_csv(p + cfg.cell_file, cfg.cell_skip_header).tolist()
    print(f"[{cfg.name}] #cell lines: {len(cell_line)}")

    # ---- 3. fusion + normalization ----
    omics_fused = {}
    for k, raw in omics_raw.items():
        fused = _data_fusion(raw, cell_line, fill_empty=cfg.fill_empty)
        if cfg.do_scale[k]:
            fused = scale(fused)
        omics_fused[k] = fused

    # ---- 4. drug features ----
    fp_raw = _read_csv(p + cfg.fingerprint_file)
    drug_cid = list(fp_raw[:, 0])
    fp = scale(np.asarray(fp_raw[:, 1:], dtype=float))
    dg = scale(np.asarray(_read_csv(p + cfg.drug_graph_file)[:, 1:], dtype=float))
    pc = np.nan_to_num(scale(np.asarray(
        _read_csv(p + cfg.physicochemical_file)[:, 1:], dtype=float)))

    # ---- 5. labels ----
    label_arr = _read_label_csv(p + cfg.label_file, cfg.label_skip_header)
    result = cfg.label_builder(cell_line, drug_cid, label_arr)
    if len(result) == 3:
        pairs, bins, extra_info = result
    else:
        pairs, bins = result
        extra_info = None

    print(f"[{cfg.name}] #pairs: {pairs.shape}, "
          f"#cells: {len(set(pairs[:, 0].astype(int)))}, "
          f"#drugs: {len(set(pairs[:, 1].astype(int)))}")

    # ---- 6. to tensors (with optional feature alignment) ----
    exp = torch.FloatTensor(omics_fused['exp'])
    copy = torch.FloatTensor(omics_fused['copy'])
    meth = torch.FloatTensor(omics_fused['meth'])
    mut = torch.FloatTensor(omics_fused['mut'])

    if cfg.align:
        exp = align_features(exp, 'exp')
        copy = align_features(copy, 'copy')
        meth = align_features(meth, 'meth')
        mut = align_features(mut, 'mut')

    return {
        'expression':      exp,
        'copynumber':      copy,
        'methylation':     meth,
        'mutation':        mut,
        'fingerprint':     torch.FloatTensor(fp),
        'drug_graph':      torch.FloatTensor(dg),
        'physicochemical': torch.FloatTensor(pc),
        'cell_drug_pair':  torch.tensor(pairs, dtype=torch.float32),
        'binary_label':    np.asarray(bins, dtype=float),
        'n_cell':          len(cell_line),
        'n_drug':          len(drug_cid),
        'cell_names':      cell_line,
        'drug_cid':        drug_cid,
        'extras':          {'cell_drug_information': extra_info}
                           if extra_info is not None else {},
    }


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def load_dataset(name: str, task: str = 'regression') -> Dict:
    """Unified data loading entry point.

    Parameters
    ----------
    name : {'GDSC', 'CCLE', 'TCGA', 'Tahoe'}
    task : {'regression', 'classification'}
        Only affects GDSC.

    Returns
    -------
    dict with keys:
        expression, copynumber, methylation, mutation,
        fingerprint, drug_graph, physicochemical,
        cell_drug_pair, binary_label, n_cell, n_drug,
        cell_names, drug_cid, extras
    """
    name = name.upper()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(_REGISTRY.keys())}")

    cfg = _REGISTRY[name]
    if name == 'GDSC':
        builder = (_label_gdsc_classification if task == 'classification'
                   else _label_gdsc_regression)
        cfg = replace(cfg, label_builder=builder)

    return _load(cfg)