"""Query PubChem for each drug name / CID and dump drug_information.csv.

Usage
-----
python data_processing/03_fetch_drug_pubchem.py \
    --drug-names-file ./dataset/GDSC/drug_name.txt \
    --out             ./data_processing/work/drug_information.csv
"""
import argparse
import csv
import logging
import time
from pathlib import Path

import pandas as pd
import pubchempy

from config import GDSC1_XLSX, GDSC2_XLSX, WORK_DIR
from utils import setup_logging

logger = logging.getLogger(__name__)

FIELDS = ['drug_name', 'drug_cid', 'molecular_formula',
          'molecular_weight', 'smiles', 'synonyms']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gdsc1', type=Path, default=GDSC1_XLSX)
    p.add_argument('--gdsc2', type=Path, default=GDSC2_XLSX)
    p.add_argument('--out',   type=Path, default=WORK_DIR / 'drug_information.csv')
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--sleep',   type=float, default=1.0)
    p.add_argument('--resume',  action='store_true',
                   help='Skip rows already present in --out.')
    return p.parse_args()


def collect_drug_names(x1: Path, x2: Path):
    n1 = pd.read_excel(x1, usecols=['DRUG_NAME'])['DRUG_NAME']
    n2 = pd.read_excel(x2, usecols=['DRUG_NAME'])['DRUG_NAME']
    return sorted(set(n1).union(n2))


def lookup(name, retries, sleep):
    """Return a dict or None. `name` may be a numeric CID or a text name."""
    for attempt in range(retries):
        try:
            if str(name).isdigit():
                c = pubchempy.Compound.from_cid(int(name))
            else:
                hits = pubchempy.get_compounds(name, 'name')
                if not hits:
                    return None
                c = hits[0]
            return {
                'drug_name': str(name),
                'drug_cid': c.cid,
                'molecular_formula': c.molecular_formula,
                'molecular_weight': c.molecular_weight,
                'smiles': c.isomeric_smiles,
                'synonyms': c.synonyms,
            }
        except Exception as e:               # network / rate-limit / decode
            logger.warning('PubChem lookup failed for %s (try %d/%d): %s',
                           name, attempt + 1, retries, e)
            time.sleep(sleep * (attempt + 1))
    return None


def load_done(out: Path):
    if not out.exists():
        return set()
    df = pd.read_csv(out)
    return set(df['drug_name'].astype(str))


def main():
    setup_logging()
    args = parse_args()

    names = collect_drug_names(args.gdsc1, args.gdsc2)
    logger.info('Total unique drug names across GDSC1+GDSC2: %d', len(names))

    done = load_done(args.out) if args.resume else set()
    if done:
        logger.info('Resuming; %d already fetched', len(done))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if done else 'w'
    header = None if done else FIELDS
    n_ok, n_miss = 0, 0

    with open(args.out, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if header:
            w.writeheader()
        for name in names:
            if name in done:
                continue
            info = lookup(name, args.retries, args.sleep)
            if info is None:
                n_miss += 1
                logger.warning('No PubChem record for %s', name)
                continue
            w.writerow(info)
            f.flush()                        # survive an interrupt
            n_ok += 1
            if n_ok % 50 == 0:
                logger.info('Fetched %d / %d', n_ok, len(names))

    logger.info('Done. Fetched %d, missed %d. Saved to %s',
                n_ok, n_miss, args.out)


if __name__ == '__main__':
    main()
