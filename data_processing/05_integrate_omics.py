"""Build omics matrices for DynaCDR.

Outputs (all in `WORK_DIR`) share the same shape convention:
    rows    = cell lines
    columns = features
    first column = cell_line_name (unquoted string)
    header occupies exactly one line

Supported modalities
--------------------
expression   raw_expression_data.csv     Cell_line_RMA_proc_basalExpression.txt
mutation     raw_mutation_data.csv       Mutation_GDSC1.csv + Mutation_GDSC2.csv
methylation  raw_methylation_data.csv    GSE68379_Matrix.processed.txt.gz + .soft
copynumber   raw_copynumber_data.csv     WES_pureCN_CNV_genes_20221213.csv

Usage
-----
python data_processing/05_integrate_omics.py expression
python data_processing/05_integrate_omics.py mutation
python data_processing/05_integrate_omics.py methylation
python data_processing/05_integrate_omics.py copynumber
python data_processing/05_integrate_omics.py all
"""
import argparse
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (CELL_LINES_FILTERED, EXPR_TXT, METH_GZ, METH_SOFT,
                    MUT1_CSV, MUT2_CSV, CNV_CSV,
                    OUT_EXPRESSION, OUT_MUTATION, OUT_METHYLATION,
                    OUT_COPYNUMBER)
from utils import setup_logging

logger = logging.getLogger(__name__)

# Methylation QC thresholds (mirror the original pipeline)
PVAL_MAX = 0.01


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #
def _load_cell_map(path: Path) -> Dict[str, str]:
    """Return {COSMIC_ID: CELL_LINE_NAME} from `final_cell_lines_name.txt`."""
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['COSMIC_ID', 'CELL_LINE_NAME'], dtype=str)
    return dict(zip(df['COSMIC_ID'], df['CELL_LINE_NAME']))


def _write_wide(df_t: pd.DataFrame, out: Path):
    """df_t: rows = cell lines, columns = features, index already named."""
    out.parent.mkdir(parents=True, exist_ok=True)
    df_t.to_csv(out)
    logger.info('Wrote %s  (shape=%s)', out, df_t.shape)


# --------------------------------------------------------------------------- #
#  1. Expression
# --------------------------------------------------------------------------- #
def build_expression(cell_txt: Path, expr_txt: Path, out: Path):
    cosmic_to_name = _load_cell_map(cell_txt)

    with open(expr_txt, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')

    # locate the columns we want (header cols look like "X1.<COSMIC_ID>")
    keep_idx, keep_names = [], []
    for i, col in enumerate(header):
        parts = col.strip().split('.')
        if len(parts) == 2 and parts[1] in cosmic_to_name:
            keep_idx.append(i)
            keep_names.append(cosmic_to_name[parts[1]])

    if not keep_idx:
        raise RuntimeError('No overlapping cell lines in expression header.')

    gene_symbols, rows = [], []
    with open(expr_txt, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            # original code drops rows without GENE_SYMBOLS / GENE_title
            if parts[1] == '' or parts[2] == '':
                continue
            gene_symbols.append(parts[1])
            rows.append([parts[i] for i in keep_idx])

    df = pd.DataFrame(rows, columns=keep_names, dtype=float)
    df.insert(0, 'gene_symbol', gene_symbols)

    # collapse probes per gene (mean) -> genes × cells
    df = df.groupby('gene_symbol', as_index=False).mean()

    # transpose -> cells × genes
    df_t = df.set_index('gene_symbol').T
    df_t.index.name = 'cell_line_name'
    _write_wide(df_t, out)


# --------------------------------------------------------------------------- #
#  2. Mutation
# --------------------------------------------------------------------------- #
def _mut_pair_list(ref_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return deduplicated [(Genetic Feature, gene)] from one reference cell."""
    seen, ordered = set(), []
    for feat, genes in zip(ref_df['Genetic Feature'], ref_df['Genes in Segment']):
        for g in str(genes).split(','):
            g = g.strip()
            if not g:
                continue
            key = (feat, g)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def build_mutation(cell_txt: Path, csv1: Path, csv2: Path, out: Path):
    cells = pd.read_csv(cell_txt, sep='\t', header=None,
                        names=['COSMIC_ID', 'CELL_LINE_NAME'], dtype=str)
    wanted = set(cells['CELL_LINE_NAME'])

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Reference feature list comes from GDSC1's C-4-I cell line
    ref = df1[df1['Cell Line Name'] == 'C-4-I']
    if ref.empty:
        raise RuntimeError("Reference cell line 'C-4-I' not found in GDSC1.")
    pairs = _mut_pair_list(ref)
    features = [f for f, _ in pairs]
    genes = [g for _, g in pairs]

    def rows_from(df: pd.DataFrame, skip_names=set()):
        name_col = 'Cell Line Name'
        for name in df[name_col].unique():
            if name not in wanted or name in skip_names:
                continue
            sub = df[df[name_col] == name]
            fmap = dict(zip(sub['Genetic Feature'], sub['IS Mutated']))
            yield [name] + [fmap.get(f, 0) for f in features]

    rows, seen_names = [], set()
    for r in rows_from(df1):
        seen_names.add(r[0])
        rows.append(r)
    for r in rows_from(df2, skip_names=seen_names):
        rows.append(r)

    header = ['cell_line_name'] + genes
    df_out = pd.DataFrame(rows, columns=header)
    df_out = df_out.set_index('cell_line_name')
    _write_wide(df_out, out)


# --------------------------------------------------------------------------- #
#  3. Methylation
# --------------------------------------------------------------------------- #
def _parse_soft(soft_path: Path) -> Dict[str, List[str]]:
    """Parse GSE68379 family.soft -> {gene_name: [cpg_id, ...]}."""
    with open(soft_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    start = end = None
    for i, ln in enumerate(lines):
        if '!platform_table_begin' in ln:
            start = i + 1
        if 'rs10796216' in ln:
            end = i - 1
            break
    if start is None or end is None:
        raise RuntimeError('Failed to locate platform table in .soft file.')

    header = lines[start].rstrip('\n').split('\t')
    gene_col = header.index('UCSC_RefGene_Name')

    gene_to_cpg: Dict[str, List[str]] = {}
    for ln in lines[start + 1:end]:
        parts = ln.rstrip('\n').split('\t')
        if len(parts) <= gene_col:
            continue
        cpg, gene = parts[0], parts[gene_col]
        gene_to_cpg.setdefault(gene, []).append(cpg)
    return gene_to_cpg


def build_methylation(cell_txt: Path, gz_path: Path,
                      soft_path: Path, out: Path, drop_zero_cols: bool = True):
    cosmic_to_name = _load_cell_map(cell_txt)
    wanted = set(cosmic_to_name.values())

    # --- header: split PVal vs AVG columns ---
    with gzip.open(gz_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip('\n').split('\t')

    pval_idx: List[int] = []
    avg_idx: List[int] = []
    avg_names: List[str] = []
    for i, col in enumerate(header):
        if 'Detection.PVal' in col:
            pval_idx.append(i)
        elif '_AVG' in col:
            name = col.split('_AVG')[0]
            if name in wanted:
                avg_idx.append(i)
                avg_names.append(name)

    if not avg_idx:
        raise RuntimeError('No matching cell lines in methylation header.')

    # --- pass 1: QC filter per CpG ---
    cpg_ids, cpg_rows = [], []
    with gzip.open(gz_path, 'rb') as f:
        f.readline()
        for line in f:
            parts = line.decode('utf-8').rstrip('\n').split('\t')

            # reject if any p-value > 0.01 (or non-numeric)
            bad = False
            for i in pval_idx:
                v = parts[i]
                try:
                    if float(v) > PVAL_MAX:
                        bad = True
                        break
                except ValueError:
                    bad = True
                    break
            if bad:
                continue

            avgs = [parts[i] for i in avg_idx]
            if any(a in ('NA', '') for a in avgs):
                continue

            cpg_ids.append(parts[1])
            cpg_rows.append([float(a) for a in avgs])

    logger.info('Methylation: %d CpGs pass QC', len(cpg_ids))
    if not cpg_ids:
        raise RuntimeError('No CpG passed the QC filter.')

    # --- pass 2: aggregate CpGs per gene ---
    gene_to_cpg = _parse_soft(soft_path)
    cpg_index = {c: i for i, c in enumerate(cpg_ids)}

    genes, mats = [], []
    for gene, cpgs in gene_to_cpg.items():
        if not gene:
            continue
        idx = [cpg_index[c] for c in cpgs if c in cpg_index]
        if not idx:
            continue
        mat = np.asarray([cpg_rows[i] for i in idx], dtype=float)
        genes.append(gene)
        mats.append(mat.mean(axis=0))

    df = pd.DataFrame(np.asarray(mats), index=genes, columns=avg_names)
    df_t = df.T
    df_t.index.name = 'cell_line_name'

    # --- pass 3 (optional): drop genes that are 0.0 for any cell ---
    if drop_zero_cols:
        keep = [c for c in df_t.columns if not np.any(df_t[c].values == 0.0)]
        df_t = df_t[keep]
        logger.info('Methylation: %d genes kept after zero-filter', len(keep))

    _write_wide(df_t, out)


# --------------------------------------------------------------------------- #
#  4. Copy number
# --------------------------------------------------------------------------- #
def build_copynumber(cell_txt: Path, cnv_csv: Path, out: Path):
    cells = pd.read_csv(cell_txt, sep='\t', header=None,
                        names=['COSMIC_ID', 'CELL_LINE_NAME'], dtype=str)
    wanted = set(cells['CELL_LINE_NAME'])

    df = pd.read_csv(cnv_csv, usecols=['model_name', 'symbol', 'gene_mean'])
    df = df.dropna(subset=['gene_mean'])
    df = df[df['model_name'].isin(wanted)]

    pivot = (df.pivot_table(index='model_name', columns='symbol',
                            values='gene_mean', aggfunc='mean'))
    pivot.index.name = 'cell_line_name'
    _write_wide(pivot, out)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('modality',
                   choices=['expression', 'mutation', 'methylation',
                            'copynumber', 'all'])
    p.add_argument('--cell-lines', type=Path, default=CELL_LINES_FILTERED)
    p.add_argument('--expr', type=Path, default=EXPR_TXT)
    p.add_argument('--mut1', type=Path, default=MUT1_CSV)
    p.add_argument('--mut2', type=Path, default=MUT2_CSV)
    p.add_argument('--meth-gz', type=Path, default=METH_GZ)
    p.add_argument('--meth-soft', type=Path, default=METH_SOFT)
    p.add_argument('--cnv', type=Path, default=CNV_CSV)
    p.add_argument('--out-expr', type=Path, default=OUT_EXPRESSION)
    p.add_argument('--out-mut', type=Path, default=OUT_MUTATION)
    p.add_argument('--out-meth', type=Path, default=OUT_METHYLATION)
    p.add_argument('--out-copy', type=Path, default=OUT_COPYNUMBER)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if args.modality in ('expression', 'all'):
        build_expression(args.cell_lines, args.expr, args.out_expr)
    if args.modality in ('mutation', 'all'):
        build_mutation(args.cell_lines, args.mut1, args.mut2, args.out_mut)
    if args.modality in ('methylation', 'all'):
        build_methylation(args.cell_lines, args.meth_gz,
                          args.meth_soft, args.out_meth)
    if args.modality in ('copynumber', 'all'):
        build_copynumber(args.cell_lines, args.cnv, args.out_copy)


if __name__ == '__main__':
    main()
