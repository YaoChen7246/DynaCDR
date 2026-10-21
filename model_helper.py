from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # the trainable parameters, namely W and a in the paper
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier initialization
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier initialization

        # leakyReLU activation function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Adjacency matrix of the graph of dimension [N,N]
        h.shape: (N, in_features), self.W.shape:(in_features,out_features)
        Wh.shape: (N, out_features)
        """

        Wh = torch.mm(h, self.W)  # Calculation formula for eij
        e = self._prepare_attentional_mechanism_input(Wh)  # Calculation formula for LeakyReLU(eij)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N,N]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        h_prime = torch.matmul(attention, Wh)  # [N,N].[N,out_features]=>[N,out_features]

        # obtain a representation updated by the surrounding nodes using attention weights
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SelfAttention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.query = nn.Linear(sample_size, d_k)  # linear transformation function
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(0, 1)) / sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                break
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.hidden = nn.Sequential(*layers)
        self.fc2 = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.hidden(x)
        x = self.fc2(x)
        return x


class DrugEmbedding(nn.Module):
    def __init__(self, out_dim):
        super(DrugEmbedding, self).__init__()
        self.fingerprint_fp = MLP(2048, [1024, 512, 256], out_dim)
        self.physicochemical_fp = MLP(210, [1024, 512, 256], out_dim)

    def forward(self, fingerprint, drug_graph, physicochemical):
        fingerprint_f = self.fingerprint_fp(fingerprint)
        druggraph_f = drug_graph
        physicochemical_f = self.physicochemical_fp(physicochemical)
        return fingerprint_f, druggraph_f, physicochemical_f


class CellEmbedding(nn.Module):
    def __init__(self, exp_in_dim, copy_in_dim, meth_in_dim, mut_in_dim, out_dim):
        super(CellEmbedding, self).__init__()
        mlp_hidden_dim = [1024, 512, 256]
        self.exp_embedding = MLP(exp_in_dim, mlp_hidden_dim, out_dim)
        self.copy_embedding = MLP(copy_in_dim, mlp_hidden_dim, out_dim)
        self.meth_embedding = MLP(meth_in_dim, mlp_hidden_dim, out_dim)
        self.mut_embedding = MLP(mut_in_dim, mlp_hidden_dim, out_dim)

    def forward(self, expression_data, copynumber_data, methylation_data, mutation_data):
        x_exp = self.exp_embedding(expression_data)
        x_meth = self.meth_embedding(methylation_data)
        x_copy = self.copy_embedding(copynumber_data)
        x_mut = self.mut_embedding(mutation_data)

        return x_copy, x_mut, x_exp, x_meth


class PredictionMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(PredictionMLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


