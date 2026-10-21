import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import torch.utils.data as Dataset

from data_loader import load_dataset, build_identity_adj
from DynaCDR import DynaCDR


def parse_args():
    parser = argparse.ArgumentParser(
        description='External validation of the regression model on Tahoe.')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--ckpt', type=str,
                        default='./output/DynaCDR_regression_fold_1.pkl')
    parser.add_argument('--save_pred', type=str,
                        default='Tahoe_predicted.csv')
    parser.add_argument('--save_info', type=str,
                        default='cell_drug_information.csv')
    return parser.parse_args()


@torch.no_grad()
def compute_test(model, cell_drug_adj, index_test, y_test,
                 data_bundle, batch, device,
                 save_pred=None, save_info=None, extras=None):
    model.eval()
    loader = Dataset.DataLoader(Dataset.TensorDataset(index_test, y_test),
                                batch_size=batch, shuffle=False)
    pred_list, true_list, index_list = [], [], []

    for idx_batch, y_batch in loader:
        y_batch = y_batch.to(device)
        y_pred = model(cell_drug_adj, *data_bundle,
                       idx_batch.numpy().astype(int), device)
        pred_list.extend(y_pred.cpu().numpy().ravel())
        true_list.extend(y_batch.cpu().numpy().ravel())
        index_list.extend(idx_batch.numpy().astype(int).tolist())

    pred = np.asarray(pred_list)
    true = np.asarray(true_list)

    print("Test set results:",
          "loss= {:.4f}".format(mean_squared_error(true, pred)),
          "RMSE= {:.4f}".format(np.sqrt(mean_squared_error(true, pred))),
          "MAE= {:.4f}".format(mean_absolute_error(true, pred)),
          "PCC= {:.4f}".format(pearsonr(true, pred)[0]),
          "SCC= {:.4f}".format(spearmanr(true, pred)[0]),
          "R2= {:.4f}".format(r2_score(true, pred)))

    if save_pred:
        idx_str = ['[{} {}]'.format(int(c), int(d)) for c, d in index_list]
        pd.DataFrame({'pred_test': pred,
                      'true_test': true,
                      'cell_drug_index': idx_str}).to_csv(save_pred, index=False)
        print(f"Saved predictions to {save_pred}")

    if save_info and extras and 'cell_drug_information' in extras:
        pd.DataFrame(extras['cell_drug_information'],
                     columns=['cell_index', 'drug_index',
                              'cell_name', 'drug_name', 'drug_cid']
                     ).to_csv(save_info, index=False)
        print(f"Saved cell-drug info to {save_info}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- unified loading ----------------
    data = load_dataset('Tahoe', task='regression')

    data_bundle = (
        Variable(data['expression']).to(device),
        Variable(data['copynumber']).to(device),
        Variable(data['methylation']).to(device),
        Variable(data['mutation']).to(device),
        data['fingerprint'].to(device),
        data['drug_graph'].to(device),
        data['physicochemical'].to(device),
    )

    cell_drug_adj = build_identity_adj(
        data['n_cell'], data['n_drug']).to(device)

    model = DynaCDR(
        ncell=data['expression'].shape[0],
        cell_exp_dim=data['expression'].shape[1],
        cell_copy_dim=data['copynumber'].shape[1],
        cell_meth_dim=data['methylation'].shape[1],
        cell_mut_dim=data['mutation'].shape[1],
        nhid=args.hidden, nheads=args.nb_heads,
        dropout=args.dropout, alpha=args.alpha,
        task='regression',
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    index_test = data['cell_drug_pair'][:, :2]
    y_test = data['cell_drug_pair'][:, 2]

    compute_test(model, cell_drug_adj, index_test, y_test,
                 data_bundle, args.batch, device,
                 save_pred=args.save_pred,
                 save_info=args.save_info,
                 extras=data['extras'])


if __name__ == "__main__":
    main()
