import torch
import utils as u
import torch.nn as nn
# from torch_geometric.nn import MassagePassing
# from torch_geometric.utils import degree

class EGNNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim, bias=False)
        self.ELU = nn.ELU()

    def forward(self, X, E):
        X = self.lin1(X)
        X = torch.einsum('ijp,jk->ipk', E, X)
        X = X.view(X.size(0), -1)
        newX = self.ELU(X)
        return newX


class EGNNC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, num_node, normalize=False, dropout=0.1):
        super().__init__()
        self.Embed = nn.Embedding(num_node, in_dim)
        self.Conv1 = EGNNConv(in_dim=in_dim, out_dim=hidden_dim)
        self.Dropout = nn.Dropout(dropout)
        self.Conv2 = EGNNConv(in_dim=hidden_dim*edge_dim, out_dim=hidden_dim)
        self.lin = nn.Linear(hidden_dim*edge_dim, out_dim)
        self.normalize = normalize

    def forward(self, E, node_ids, node_feature=None):
        if node_feature != None:
            X = node_feature
        else:
            X = self.Embed(node_ids)
        if self.normalize:
            E = u.DS_normalize(E)
        X = self.Dropout(self.Conv1(X, E))
        X = self.Dropout(self.Conv2(X, E))
        return self.lin(X)


class Classifier(nn.Module):
    def __init__(self,args, in_features, out_features=2):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = in_features,
                                                       out_features =args.gcn_parameters['cls_feats']),
                                       activation,
                                       torch.nn.Linear(in_features = args.gcn_parameters['cls_feats'],
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)