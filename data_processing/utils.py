"""Small helpers shared across data-processing scripts."""
import csv
import gzip
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S', level=level)


def keep_columns(seq: Sequence, idx: Iterable[int]) -> List:
    """Return `seq[i]` for every i in `idx` (list comprehension wrapper)."""
    return [seq[i] for i in idx]


def read_lines(path: Path) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def read_tsv(path: Path) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def read_gzip_lines(path: Path) -> List[str]:
    """Read a gzipped file and return decoded UTF-8 lines."""
    with gzip.open(path, 'rb') as f:
        return [line.decode('utf-8', errors='replace') for line in f.readlines()]


def write_csv(path: Path, header: Sequence, rows: Iterable[Sequence], mode: str = 'w'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        w.writerows(rows)


def transpose_csv(in_path: Path, out_path: Path, header: bool = False):
    """Transpose a CSV (rows <-> columns). `header=False` drops the index."""
    df = pd.read_csv(in_path)
    data = df.values.T
    out = pd.DataFrame(data, index=list(df.keys()))
    out.to_csv(out_path, header=header)