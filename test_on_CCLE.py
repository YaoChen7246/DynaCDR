import argparse
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import torch.utils.data as Dataset

from data_loader import load_dataset, build_identity_adj
from DynaCDR import DynaCDR


def parse_args():
    parser = argparse.ArgumentParser(description='External validation of the '
                                                 'regression model on CCLE.')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--ckpt', type=str,
                        default='./output/DynaCDR_regression_fold_1.pkl')
    return parser.parse_args()


@torch.no_grad()
def compute_test(model, cell_drug_adj, index_test, y_test,
                 data_bundle, batch, device):
    model.eval()
    loader = Dataset.DataLoader(Dataset.TensorDataset(index_test, y_test),
                                batch_size=batch, shuffle=False)
    pred_list, true_list = [], []
    for idx, y in loader:
        y = y.to(device)
        y_pred = model(cell_drug_adj, *data_bundle,
                       idx.numpy().astype(int), device)
        pred_list.extend(y_pred.cpu().numpy().ravel())
        true_list.extend(y.cpu().numpy().ravel())

    pred, true = np.asarray(pred_list), np.asarray(true_list)
    print("Test set results:",
          "loss= {:.4f}".format(mean_squared_error(true, pred)),
          "RMSE= {:.4f}".format(np.sqrt(mean_squared_error(true, pred))),
          "MAE= {:.4f}".format(mean_absolute_error(true, pred)),
          "PCC= {:.4f}".format(pearsonr(true, pred)[0]),
          "SCC= {:.4f}".format(spearmanr(true, pred)[0]),
          "R2= {:.4f}".format(r2_score(true, pred)))


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- unified loading ----------------
    data = load_dataset('CCLE', task='regression')

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
                 data_bundle, args.batch, device)


if __name__ == "__main__":
    main()
