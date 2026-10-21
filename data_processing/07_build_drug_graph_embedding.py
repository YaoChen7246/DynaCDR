"""Compute Graph2Vec embeddings of the molecular graphs.

Usage
-----
python data_processing/07_build_drug_graph_embedding.py \
    --smiles ./data_processing/work/drug_smile.csv \
    --out    ./data_processing/work/drug_3D_embedding.csv
"""
import argparse
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from karateclub import Graph2Vec
from rdkit import Chem

from config import OUT_DRUG_SMILE, OUT_DRUG_3D
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smiles', type=Path, default=OUT_DRUG_SMILE)
    p.add_argument('--out', type=Path, default=OUT_DRUG_3D)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--wl', type=int, default=2)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--workers', type=int, default=4)
    return p.parse_args()


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx(),
                   label=atom.GetSymbol(),
                   atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   label=bond.GetBondType().name)
    return g


def main():
    setup_logging()
    args = parse_args()

    df = pd.read_csv(args.smiles, header=None, names=['cid', 'smiles'],
                     dtype={'cid': str})
    logger.info('Loaded %d SMILES', len(df))

    graphs, valid_cids = [], []
    for cid, smi in zip(df['cid'], df['smiles']):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_cids.append(cid)
        else:
            logger.warning('Invalid SMILES for cid=%s', cid)

    logger.info('Valid graphs: %d / %d', len(graphs), len(df))

    model = Graph2Vec(dimensions=args.dim, wl_iterations=args.wl,
                      workers=args.workers, min_count=5, epochs=args.epochs)
    model.fit(graphs)
    emb = model.get_embedding()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(emb, index=valid_cids).to_csv(args.out, header=False)
    logger.info('Wrote %s (shape=%s)', args.out, emb.shape)


if __name__ == '__main__':
    main()
