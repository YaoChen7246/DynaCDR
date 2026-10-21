"""Select cell lines that are present in expression ∩ mutation ∩ methylation.

Output:  `final_cell_lines_name.txt`  (tab-separated COSMIC_ID \\t NAME)

Usage
-----
python data_processing/02_filter_cell_lines.py
"""
import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from config import (CELL_LINES_ALL, CELL_LINES_FILTERED,
                    EXPR_TXT, METH_GZ, MUT1_CSV, MUT2_CSV)
from utils import read_gzip_lines, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cell-lines', type=Path, default=CELL_LINES_ALL)
    p.add_argument('--expression', type=Path, default=EXPR_TXT)
    p.add_argument('--methylation', type=Path, default=METH_GZ)
    p.add_argument('--mutation-1', type=Path, default=MUT1_CSV)
    p.add_argument('--mutation-2', type=Path, default=MUT2_CSV)
    p.add_argument('--out', type=Path, default=CELL_LINES_FILTERED)
    return p.parse_args()


def load_cosmic_map(path: Path):
    """Return {COSMIC_ID: CELL_LINE_NAME} from the master table."""
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['COSMIC_ID', 'CELL_LINE_NAME'], dtype=str)
    return dict(zip(df['COSMIC_ID'], df['CELL_LINE_NAME']))


def expression_cosmic_ids(txt: Path):
    """Header line of the RMA expression file: `X241.COSMIC_ID`."""
    with open(txt, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
    ids = []
    for col in header[2:]:
        parts = col.strip().split('.')
        if len(parts) == 2:
            ids.append(parts[1])
    return set(ids)


def methylation_cell_names(gz_path: Path):
    """Header line of the methylation matrix: `NAME_<suffix>`."""
    lines = read_gzip_lines(gz_path)
    header = lines[0].rstrip('\n').split('\t')
    names = set()
    for col in header[1:]:
        names.add(col.strip().split('_')[0])
    return names


def mutation_cell_names(csv1: Path, csv2: Path):
    df = pd.concat([pd.read_csv(csv1, usecols=['Cell Line Name']),
                    pd.read_csv(csv2, usecols=['Cell Line Name'])])
    return set(df['Cell Line Name'].unique())


def main():
    setup_logging()
    args = parse_args()

    cosmic_to_name = load_cosmic_map(args.cell_lines)
    logger.info('Master table: %d cells', len(cosmic_to_name))

    exp_ids = expression_cosmic_ids(args.expression)
    mut_names = mutation_cell_names(args.mutation_1, args.mutation_2)
    meth_names = methylation_cell_names(args.methylation)

    logger.info('expression: %d | mutation: %d | methylation: %d',
                len(exp_ids), len(mut_names), len(meth_names))

    # intersect: name must appear in meth & mut, and COSMIC id in expression
    common = {cid for cid, name in cosmic_to_name.items()
              if cid in exp_ids and name in mut_names and name in meth_names}

    logger.info('Intersection: %d cell lines', len(common))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for cid in sorted(common):
            f.write(f'{cid}\t{cosmic_to_name[cid]}\n')
    logger.info('Wrote %s', args.out)


if __name__ == '__main__':
    main()
