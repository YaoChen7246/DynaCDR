"""Merge GDSC1+GDSC2 dose-response tables into cell_drug_label.csv.

For duplicated (cell, drug) pairs the mean of LN_IC50 is used, matching
the original 4.1_CDR_process.py behaviour.

Usage
-----
python data_processing/04_merge_ic50_labels.py
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from config import (GDSC1_XLSX, GDSC2_XLSX, CELL_LINES_FILTERED,
                    DRUG_INFORMATION, CELL_DRUG_LABEL)
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gdsc1', type=Path, default=GDSC1_XLSX)
    p.add_argument('--gdsc2', type=Path, default=GDSC2_XLSX)
    p.add_argument('--cell-lines', type=Path, default=CELL_LINES_FILTERED)
    p.add_argument('--drug-info', type=Path, default=DRUG_INFORMATION)
    p.add_argument('--out', type=Path, default=CELL_DRUG_LABEL)
    return p.parse_args()


def read_gdsc(path: Path):
    df = pd.read_excel(path,
                       usecols=['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50'])
    return df


def main():
    setup_logging()
    args = parse_args()

    # --- 1. read & concat the two GDSC releases ---
    raw = pd.concat([read_gdsc(args.gdsc1), read_gdsc(args.gdsc2)],
                    ignore_index=True)
    logger.info('Raw pairs: %d', len(raw))

    # --- 2. keep only the filtered cell lines ---
    cells = pd.read_csv(args.cell_lines, sep='\t', header=None,
                        names=['COSMIC_ID', 'CELL_LINE_NAME'], dtype=str)
    keep = set(cells['CELL_LINE_NAME'])
    raw = raw[raw['CELL_LINE_NAME'].isin(keep)]
    logger.info('After cell-line filter: %d pairs, %d cells',
                len(raw), raw['CELL_LINE_NAME'].nunique())

    # --- 3. join with drug metadata (drug name -> cid, smiles) ---
    drugs = pd.read_csv(args.drug_info,
                        usecols=['drug_name', 'drug_cid', 'smiles'])
    drugs = drugs.rename(columns={'drug_name': 'DRUG_NAME'})

    merged = raw.merge(drugs, on='DRUG_NAME', how='inner')
    logger.info('After drug-info join: %d pairs, %d drugs',
                len(merged), merged['DRUG_NAME'].nunique())

    # --- 4. average duplicated (cell, drug) pairs ---
    grouped = (merged.groupby(['CELL_LINE_NAME', 'DRUG_NAME',
                               'drug_cid', 'smiles'], as_index=False)
                     ['LN_IC50'].mean())
    logger.info('After dedup: %d pairs', len(grouped))

    grouped = grouped.rename(columns={
        'CELL_LINE_NAME': 'cell_lines_name',
        'drug_cid':       'drug_pubchem_cid',
        'smiles':         'drug_SMILES',
    })[['cell_lines_name', 'DRUG_NAME', 'drug_pubchem_cid',
        'drug_SMILES', 'LN_IC50']]
    grouped = grouped.rename(columns={'DRUG_NAME': 'drug_name'})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.out, index=False)
    logger.info('Wrote %s', args.out)


if __name__ == '__main__':
    main()
