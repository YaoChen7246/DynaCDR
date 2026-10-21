"""Collect all (COSMIC_ID, CELL_LINE_NAME) pairs used by GDSC1+GDSC2.

Usage
-----
python data_processing/01_build_cell_line_table.py \
    --gdsc1 ./dataset/GDSC/GDSC1_fitted_dose_response_27Oct23.xlsx \
    --gdsc2 ./dataset/GDSC/GDSC2_fitted_dose_response_27Oct23.xlsx \
    --out   ./data_processing/work/cell_lines_all.txt
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from config import GDSC1_XLSX, GDSC2_XLSX, CELL_LINES_ALL
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gdsc1', type=Path, default=GDSC1_XLSX)
    p.add_argument('--gdsc2', type=Path, default=GDSC2_XLSX)
    p.add_argument('--out',   type=Path, default=CELL_LINES_ALL)
    return p.parse_args()


def extract(xlsx: Path):
    df = pd.read_excel(xlsx, usecols=['CELL_LINE_NAME', 'COSMIC_ID'])
    logger.info('%s : %d rows, %d unique cell lines',
                xlsx.name, df.shape[0], df['CELL_LINE_NAME'].nunique())
    return df[['COSMIC_ID', 'CELL_LINE_NAME']]


def main():
    setup_logging()
    args = parse_args()

    df = pd.concat([extract(args.gdsc1), extract(args.gdsc2)], ignore_index=True)
    df = df.dropna().drop_duplicates()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False, header=False)
    logger.info('Wrote %d unique (COSMIC_ID, CELL_LINE) pairs to %s',
                len(df), args.out)


if __name__ == '__main__':
    main()
