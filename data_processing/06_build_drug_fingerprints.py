"""Build Morgan fingerprint + physicochemical descriptors from SMILES.

Usage
-----
python data_processing/06_build_drug_fingerprints.py \
    --labels ./data_processing/work/cell_drug_label.csv \
    --drug-info ./data_processing/work/drug_information.csv
"""
import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from config import CELL_DRUG_LABEL, DRUG_INFORMATION, OUT_MORGAN, OUT_PHYSCHEM, OUT_DRUG_SMILE
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labels',     type=Path, default=CELL_DRUG_LABEL)
    p.add_argument('--drug-info',  type=Path, default=DRUG_INFORMATION)
    p.add_argument('--out-smile',  type=Path, default=OUT_DRUG_SMILE)
    p.add_argument('--out-morgan', type=Path, default=OUT_MORGAN)
    p.add_argument('--out-phys',   type=Path, default=OUT_PHYSCHEM)
    return p.parse_args()


def collect_smiles(labels: Path, info: Path):
    wanted = set(pd.read_csv(labels)['drug_pubchem_cid'].astype(str))
    df = pd.read_csv(info, dtype={'drug_cid': str})
    df = df.drop_duplicates('drug_cid')
    return [(row.drug_cid, row.smiles)
            for row in df.itertuples()
            if row.drug_cid in wanted and isinstance(row.smiles, str)]


def main():
    setup_logging()
    args = parse_args()

    pairs = collect_smiles(args.labels, args.drug_info)
    logger.info('Drugs to process: %d', len(pairs))

    with open(args.out_smile, 'w', newline='') as f_smi, \
         open(args.out_morgan, 'w', newline='') as f_mg, \
         open(args.out_phys, 'w', newline='') as f_ph:

        w_smi, w_mg, w_ph = csv.writer(f_smi), csv.writer(f_mg), csv.writer(f_ph)
        n_ok = 0
        for cid, smiles in pairs:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning('Invalid SMILES for cid=%s', cid)
                continue

            w_smi.writerow([cid, smiles])

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            w_mg.writerow([cid] + list(fp.ToBitString()))

            desc = Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
            w_ph.writerow([cid] + list(desc.values()))

            n_ok += 1
            if n_ok % 50 == 0:
                logger.info('Processed %d / %d', n_ok, len(pairs))

    logger.info('Done. %d valid molecules.', n_ok)


if __name__ == '__main__':
    main()
