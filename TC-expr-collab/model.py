from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from utils import *
from source_code import SAGEConv
from torch.nn import Parameter
from torch import nn
from torch_sparse.matmul import spmm_add

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dp):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=True, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=True, normalize=False))

        self.dropout = dp

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dp):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dp

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)

        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dp):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dp

    def forward(self, x):

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x
    
    def score(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
        
        x = self.lins[-1](x)

        return torch.sigmoid(x)




class DropAdj(nn.Module):
    def __init__(self, dp: float = 0.0, doscale=True):
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1 - dp)))
        self.doscale = doscale

    def forward(self, adj):
        if self.dp < 1e-6 or not self.training:
            return adj

        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")

        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)

        return adj




class GraphConv_NCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.lin = nn.Linear(in_channels, out_channels, bias = False)
        self.bias = Parameter(torch.zeros(out_channels))
        self.dp = nn.Dropout(dropout, inplace=True)

    def forward(self, x, adj_t, aggr):
        if aggr == 'collab-gcn':
            x = self.lin(x)
            x = self.dp(x)
            x = spmm_add(adj_t, x)

            x = x + self.bias

        elif aggr == 'planetoid-gcn-residual':
            x = self.lin(x) + self.bias
            x = self.dp(x)

            norm = torch.rsqrt_((1 + adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x

        return x
    


class GCN_NCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dp, x_dp, adj_dp, t_dp, nl):
        super(GCN_NCN, self).__init__()

        self.dp = dp
        self.x_dp = x_dp

        self.convs = torch.nn.ModuleList()

        self.adjdrop = DropAdj(adj_dp)

        if num_layers == 1:
            hidden_channels = out_channels
        self.convs.append(GraphConv_NCN(in_channels, hidden_channels, self.dp))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv_NCN(hidden_channels, hidden_channels, self.dp))

        if num_layers > 1:
            self.convs.append(GraphConv_NCN(hidden_channels, out_channels, self.dp))


        self.tail_dropout = nn.Dropout(t_dp, inplace = True) if t_dp > 0 else nn.Identity()
        self.nl = nn.ReLU() if nl else nn.Identity()

    def forward(self, x, adj_t, aggr):
        x = F.dropout(x, p = self.x_dp, training = self.training)

        for conv in self.convs[:-1]:
            x = conv(x, self.adjdrop(adj_t), aggr)
            x = self.nl(x)

        x = self.convs[-1](x, adj_t, aggr)
        x = self.tail_dropout(x)

        return x
    

class CNLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dp = 0.0,
                 adj_dp=0.0,
                 beta=1.0):
        super(CNLP, self).__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(adj_dp)

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dp, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))

        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dp, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj, edge):
        adj = self.dropadj(adj)

        xi = x[edge[:, 0]]
        xj = x[edge[:, 1]]

        cn = adjoverlap(adj, adj, edge)

        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)

        xs = torch.cat([self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns], dim=-1)

        return xs
    
    def score(self, x, adj, edge):
        adj = self.dropadj(adj)

        xi = x[edge[:, 0]]
        xj = x[edge[:, 1]]

        cn = adjoverlap(adj, adj, edge)

        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)

        xs = torch.cat([self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns], dim=-1)

        return torch.sigmoid(xs)