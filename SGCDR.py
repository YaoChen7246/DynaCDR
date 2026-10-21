import torch
import torch.nn as nn
import torch.nn.functional as F
from model_helper import *


class SGCDR(nn.Module):
    def __init__(self, ncell, cell_exp_dim, cell_copy_dim, cell_meth_dim, cell_mut_dim,
                 nhid, nheads, dropout, alpha, task='regression'):
        """
        Parameters
        ----------
        task : {'regression', 'classification'}
            - regression     : output is a raw score (used for IC50 regression)
            - classification : output is passed through a Sigmoid to produce P(y=1)
        """
        super(SGCDR, self).__init__()

        assert task in ('regression', 'classification'), \
            f"task must be 'regression' or 'classification', got '{task}'"
        self.task = task

        drug_out_dim = 128
        cell_out_dim = 128
        self.ncell = ncell

        self.drug_embedding = DrugEmbedding(drug_out_dim)
        self.cell_embedding = CellEmbedding(cell_exp_dim, cell_copy_dim, cell_meth_dim,
                                            cell_mut_dim, cell_out_dim)

        self.Graph_attentions = nn.ModuleList(
            [GraphAttentionLayer(cell_out_dim, nhid, dropout=dropout,
                                 alpha=alpha, concat=True)
             for _ in range(nheads)])
        for i, attention in enumerate(self.Graph_attentions):
            self.add_module('attentioncell_{}'.format(i), attention)

        self.Prolayer = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.LNlayer = nn.LayerNorm(nhid * nheads)

        in_dim = nhid * nheads * 2
        self.fc = nn.Linear(256, 256)
        self.FClayer = nn.Linear(in_dim, in_dim)
        mlp_hidden_dim = [1024, 512, 128]
        self.prediction = PredictionMLP(in_dim, mlp_hidden_dim, output_size=1)
        # Sigmoid is only activated when task == 'classification'
        self.sigmoid = nn.Sigmoid()

        self.cell_cat = nn.Linear(cell_out_dim * 4, cell_out_dim, bias=False)
        self.drug_cat = nn.Linear(drug_out_dim * 3, drug_out_dim, bias=False)

    def forward(self, cell_drug_adj, expression, copynumber, methylation, mutation,
                fingerprint, drug_graph, physicochemical, idx_cell_drug, device):
        fingerprint_f, druggraph_f, physicochemical_f = self.drug_embedding(
            fingerprint, drug_graph, physicochemical)
        x_copy, x_mut, x_exp, x_meth = self.cell_embedding(
            expression, copynumber, methylation, mutation)

        cell_cat = torch.cat([x_exp, x_copy, x_meth, x_mut], dim=1)
        drug_cat = torch.cat([fingerprint_f, druggraph_f, physicochemical_f], dim=1)

        cell_fusion = self.cell_cat(cell_cat)
        drug_fusion = self.drug_cat(drug_cat)

        cell_drug_feature = torch.cat([cell_fusion, drug_fusion], dim=0)
        cell_drug = cell_drug_adj

        cell_drug_x = torch.cat([att(cell_drug_feature, cell_drug) for att in self.Graph_attentions], dim=1)
        cell_drug_x = self.Prolayer(cell_drug_x)
        cell_drug_x = self.LNlayer(cell_drug_x)

        cell_drug_x = self.fc(cell_drug_x)
        cell_x = cell_drug_x[idx_cell_drug[:, 0]]
        drug_x = cell_drug_x[idx_cell_drug[:, 1]]

        cell_drug_fusion = torch.cat([cell_x, drug_x], dim=-1)
        cell_drug_fusion = self.FClayer(cell_drug_fusion)
        predict = torch.squeeze(self.prediction(cell_drug_fusion))

        # Task-specific output activation
        if self.task == 'classification':
            predict = self.sigmoid(predict)
        return predict
