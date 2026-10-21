import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Dataset
from torch.autograd import Variable
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             roc_auc_score, average_precision_score,
                             f1_score, log_loss)
from scipy.stats import pearsonr, spearmanr

from data_loader import load_dataset, build_cell_drug_adj
from DynaCDR import DynaCDR


# ---------------------------------------------------------------- arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train DynaCDR on GDSC.')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='./output')
    return parser.parse_args()


# ---------------------------------------------------------------- helpers
def make_loss(task):
    """BCELoss on Sigmoid output for classification, MSELoss otherwise."""
    return nn.BCELoss() if task == 'classification' else nn.MSELoss()


def make_loader(index, y, batch, shuffle):
    return Dataset.DataLoader(Dataset.TensorDataset(index, y),
                              batch_size=batch, shuffle=shuffle)


def train_one_epoch(model, optimizer, cell_drug_adj, index_train, y_train,
                    data_bundle, args, device, loss_func):
    model.train()
    loader = make_loader(index_train, y_train, args.batch, shuffle=True)
    total_loss, n_seen = 0.0, 0
    for index_batch, y_batch in loader:
        y_batch = y_batch.to(device)
        y_pred = model(cell_drug_adj, *data_bundle, index_batch.numpy().astype(int), device)
        loss = loss_func(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        n_seen += y_batch.size(0)
    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate(model, cell_drug_adj, index_eval, y_eval,
             data_bundle, args, device, task):
    model.eval()
    loader = make_loader(index_eval, y_eval, args.batch, shuffle=False)
    pred_list, true_list = [], []
    for index_batch, y_batch in loader:
        y_batch = y_batch.to(device)
        y_pred = model(cell_drug_adj, *data_bundle,
                       index_batch.numpy().astype(int), device)
        pred_list.extend(y_pred.cpu().numpy().ravel())
        true_list.extend(y_batch.cpu().numpy().ravel())

    pred = np.asarray(pred_list)
    true = np.asarray(true_list)

    if task == 'classification':
        pred_bin = (pred >= 0.5).astype(int)
        return {
            'loss': log_loss(true, pred, labels=[0, 1]),
            'auroc': roc_auc_score(true, pred),
            'auprc': average_precision_score(true, pred),
            'f1': f1_score(true, pred_bin),
        }
    return {
        'loss': mean_squared_error(true, pred),
        'rmse': np.sqrt(mean_squared_error(true, pred)),
        'mae': mean_absolute_error(true, pred),
        'pcc': pearsonr(true, pred)[0],
        'scc': spearmanr(true, pred)[0],
        'r2': r2_score(true, pred),
    }


# ---------------------------------------------------------------- one fold
def train_fold(fold, train_idx, valid_idx, test_idx,
               cell_drug_pair, binary_label, n_cell, n_drug,
               data_bundle, args, device):
    print(f"\n========== Fold {fold + 1}/{args.n_folds} ==========")

    # Rebuild the adjacency from *training* pairs only -> no leakage.
    cell_drug_adj = build_cell_drug_adj(cell_drug_pair, binary_label, train_idx, n_cell, n_drug).to(device)

    index_train = cell_drug_pair[train_idx, :2].numpy().astype(int)
    index_valid = cell_drug_pair[valid_idx, :2].numpy().astype(int)
    index_test = cell_drug_pair[test_idx, :2].numpy().astype(int)
    y_train = cell_drug_pair[train_idx, 2]
    y_valid = cell_drug_pair[valid_idx, 2]
    y_test = cell_drug_pair[test_idx,  2]

    model = DynaCDR(
        ncell=data_bundle[0].shape[0],
        cell_exp_dim=data_bundle[0].shape[1],
        cell_copy_dim=data_bundle[1].shape[1],
        cell_meth_dim=data_bundle[2].shape[1],
        cell_mut_dim=data_bundle[3].shape[1],
        nhid=args.hidden, nheads=args.nb_heads,
        dropout=args.dropout, alpha=args.alpha,
        task=args.task,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = make_loss(args.task)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir,
                             f'DynaCDR_{args.task}_fold_{fold + 1}.pkl')

    monitor_key = 'loss' if args.task == 'classification' else 'rmse'
    best_metric, best_epoch, bad_counter = float('inf'), 0, 0

    for epoch in range(args.epochs):
        t = time.time()
        train_loss = train_one_epoch(model, optimizer, cell_drug_adj,
                                     index_train, y_train,
                                     data_bundle, args, device, loss_func)
        tr = evaluate(model, cell_drug_adj, index_train, y_train,
                      data_bundle, args, device, args.task)
        va = evaluate(model, cell_drug_adj, index_valid, y_valid,
                      data_bundle, args, device, args.task)

        if args.task == 'classification':
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(train_loss),
                  'AUROC_train: {:.4f}'.format(tr['auroc']),
                  'AUPRC_train: {:.4f}'.format(tr['auprc']),
                  'F1_train: {:.4f}'.format(tr['f1']),
                  '\t AUROC_valid: {:.4f}'.format(va['auroc']),
                  'AUPRC_valid: {:.4f}'.format(va['auprc']),
                  'F1_valid: {:.4f}'.format(va['f1']),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(train_loss),
                  'RMSE_train: {:.4f}'.format(tr['rmse']),
                  'MAE_train: {:.4f}'.format(tr['mae']),
                  'PCC_train: {:.4f}'.format(tr['pcc']),
                  'SCC_train: {:.4f}'.format(tr['scc']),
                  'R2_train: {:.4f}'.format(tr['r2']),
                  '\t RMSE_valid: {:.4f}'.format(va['rmse']),
                  'MAE_valid: {:.4f}'.format(va['mae']),
                  'PCC_valid: {:.4f}'.format(va['pcc']),
                  'SCC_valid: {:.4f}'.format(va['scc']),
                  'R2_valid: {:.4f}'.format(va['r2']),
                  'time: {:.4f}s'.format(time.time() - t))

        if va[monitor_key] < best_metric:
            best_metric, best_epoch, bad_counter = va[monitor_key], epoch + 1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_counter += 1
            if bad_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f'Loading best epoch {best_epoch} for fold {fold + 1}')
    model.load_state_dict(torch.load(ckpt_path))

    te = evaluate(model, cell_drug_adj, index_test, y_test,
                  data_bundle, args, device, args.task)
    if args.task == 'classification':
        print("Test set results:",
              "loss= {:.4f}".format(te['loss']),
              "AUROC= {:.4f}".format(te['auroc']),
              "AUPRC= {:.4f}".format(te['auprc']),
              "F1= {:.4f}\n".format(te['f1']))
    else:
        print("Test set results:",
              "loss= {:.4f}".format(te['loss']),
              "RMSE= {:.4f}".format(te['rmse']),
              "MAE= {:.4f}".format(te['mae']),
              "PCC= {:.4f}".format(te['pcc']),
              "SCC= {:.4f}".format(te['scc']),
              "R2= {:.4f}\n".format(te['r2']))
    return te


# ---------------------------------------------------------------- main
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- unified loading ----------------
    data = load_dataset('GDSC', task=args.task)
    expression = data['expression']
    copynumber = data['copynumber']
    methylation = data['methylation']
    mutation = data['mutation']
    fingerprint = data['fingerprint']
    drug_graph = data['drug_graph']
    physicochemical = data['physicochemical']
    cell_drug_pair = data['cell_drug_pair']
    binary_label = data['binary_label']
    n_cell, n_drug = data['n_cell'], data['n_drug']

    data_bundle = (
        Variable(expression).to(device),
        Variable(copynumber).to(device),
        Variable(methylation).to(device),
        Variable(mutation).to(device),
        fingerprint.to(device),
        drug_graph.to(device),
        physicochemical.to(device),
    )

    # ---------------- 10-fold CV ----------------
    n_samples = cell_drug_pair.shape[0]
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    t_total = time.time()
    fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        train_idx, valid_idx = train_test_split(
            train_val_idx, test_size=args.val_ratio,
            random_state=args.seed + fold, shuffle=True)

        result = train_fold(
            fold, train_idx, valid_idx, test_idx,
            cell_drug_pair, binary_label, n_cell, n_drug,
            data_bundle, args, device)
        fold_results.append(result)

    # ---------------- summary ----------------
    print(f"\n=============== {args.task.upper()} CV Summary ===============")
    keys = (['loss', 'auroc', 'auprc', 'f1'] if args.task == 'classification'
            else ['loss', 'rmse', 'mae', 'pcc', 'scc', 'r2'])
    for key in keys:
        vals = np.array([r[key] for r in fold_results])
        print(f"{key.upper():>5s}: {vals.mean():.4f} ± {vals.std():.4f}")

    print("\nOptimization Finished. Total time: {:.4f}s".format(time.time() - t_total))


if __name__ == "__main__":
    main()
