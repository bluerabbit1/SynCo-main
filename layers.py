import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_sparse import matmul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_sparse import matmul
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_sparse import matmul
import math


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, config):
        super(GCNEncoder, self).__init__()
        hidden_channels = config['cl_hdim']
        self.K = config.get('preprop_K', 2)
        self.beta = config.get('beta', 0.5)
        self.dropout = config['dropout']
        self.config = config
        
        # Like MHEncoder, we process multi-hop features
        self.reduce_weight = Parameter(torch.randn([self.K+1, in_channels, hidden_channels]))
        self.enc_out = nn.Linear((self.K+1)*hidden_channels, hidden_channels)
        
        self.reset_parameters()
    
    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.reduce_weight.data)
        self.enc_out.reset_parameters()
    
    @torch.no_grad()
    def prop(self, x, adj):
        """Multi-hop propagation - exactly like MHEncoder"""
        x_L = x.clone()  # (n, dim)
        out_L = []
        for _ in range(self.K):
            x_L = self.beta * x_L + matmul(adj, x_L)
            out_L.append(x_L)
        return torch.stack([x] + out_L, dim=0)  # (K+1, n, dim)
    
    def forward(self, out_prop, adj=None):
        """
        out_prop: (K+1, n, dim) if adj is None else (n, dim)
        """
        num_nodes = out_prop.shape[-2]
        
        if adj is not None:
            out_prop = self.prop(out_prop, adj)  # (K+1, n, dim)
        
        # GCN-style processing of multi-hop features
        out_prop = torch.bmm(out_prop, self.reduce_weight)  # (K+1, n, hdim)
        out_prop = F.normalize(out_prop, p=2, dim=-1)
        out_prop = F.relu(out_prop)
        out_prop = out_prop.permute(1, 0, 2).reshape(num_nodes, -1)  # (n, (K+1)*hdim)
        out_prop = F.dropout(out_prop, self.dropout, self.training)
        return F.relu(self.enc_out(out_prop))  # (n, hdim)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, config):
        super(GATEncoder, self).__init__()
        hidden_channels = config['cl_hdim']
        self.K = config.get('preprop_K', 2)
        self.beta = config.get('beta', 0.5)
        self.dropout = config['dropout']
        self.num_heads = config.get('num_heads', 4)
        self.config = config
        
        # Multi-hop attention weights
        self.attention_weights = nn.ModuleList()
        for _ in range(self.K + 1):
            self.attention_weights.append(
                nn.Linear(in_channels, hidden_channels // self.num_heads * self.num_heads)
            )
        
        # Attention parameters
        self.a = Parameter(torch.zeros(size=(2 * hidden_channels // self.num_heads, 1)))
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # Output projection
        self.enc_out = nn.Linear((self.K+1) * hidden_channels, hidden_channels)
        
        self.reset_parameters()
    
    @torch.no_grad()
    def reset_parameters(self):
        for att in self.attention_weights:
            att.reset_parameters()
        nn.init.xavier_uniform_(self.a.data)
        self.enc_out.reset_parameters()
    
    @torch.no_grad()
    def prop(self, x, adj):
        """Multi-hop propagation - exactly like MHEncoder"""
        x_L = x.clone()  # (n, dim)
        out_L = []
        for _ in range(self.K):
            x_L = self.beta * x_L + matmul(adj, x_L)
            out_L.append(x_L)
        return torch.stack([x] + out_L, dim=0)  # (K+1, n, dim)
    
    def forward(self, out_prop, adj=None):
        """
        out_prop: (K+1, n, dim) if adj is None else (n, dim)
        """
        num_nodes = out_prop.shape[-2]
        
        if adj is not None:
            out_prop = self.prop(out_prop, adj)  # (K+1, n, dim)
        
        # Apply attention to each hop
        attended_features = []
        for k in range(self.K + 1):
            h = self.attention_weights[k](out_prop[k])  # (n, hdim)
            h = F.normalize(h, p=2, dim=-1)
            h = F.relu(h)
            attended_features.append(h)
        
        # Concatenate all hop features
        out_prop = torch.cat(attended_features, dim=-1)  # (n, (K+1)*hdim)
        out_prop = F.dropout(out_prop, self.dropout, self.training)
        return F.relu(self.enc_out(out_prop))  # (n, hdim)


class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, config):
        super(SAGEEncoder, self).__init__()
        hidden_channels = config['cl_hdim']
        self.K = config.get('preprop_K', 2)
        self.beta = config.get('beta', 0.5)
        self.dropout = config['dropout']
        self.aggr = config.get('sage_aggr', 'mean')
        self.config = config
        
        # SAGE-style transformation for each hop
        self.hop_transforms = nn.ModuleList()
        for _ in range(self.K + 1):
            self.hop_transforms.append(nn.Linear(in_channels, hidden_channels))
        
        # Aggregation weights
        if self.aggr == 'concat':
            self.enc_out = nn.Linear((self.K+1)*hidden_channels, hidden_channels)
        else:
            self.enc_out = nn.Linear(hidden_channels, hidden_channels)
        
        self.reset_parameters()
    
    @torch.no_grad()
    def reset_parameters(self):
        for transform in self.hop_transforms:
            transform.reset_parameters()
        self.enc_out.reset_parameters()
    
    @torch.no_grad()
    def prop(self, x, adj):
        """Multi-hop propagation - exactly like MHEncoder"""
        x_L = x.clone()  # (n, dim)
        out_L = []
        for _ in range(self.K):
            x_L = self.beta * x_L + matmul(adj, x_L)
            out_L.append(x_L)
        return torch.stack([x] + out_L, dim=0)  # (K+1, n, dim)
    
    def forward(self, out_prop, adj=None):
        """
        out_prop: (K+1, n, dim) if adj is None else (n, dim)
        """
        num_nodes = out_prop.shape[-2]
        
        if adj is not None:
            out_prop = self.prop(out_prop, adj)  # (K+1, n, dim)
        
        # Apply SAGE transformation to each hop
        hop_features = []
        for k in range(self.K + 1):
            h = self.hop_transforms[k](out_prop[k])  # (n, hdim)
            h = F.normalize(h, p=2, dim=-1)
            h = F.relu(h)
            hop_features.append(h)
        
        # Aggregate hop features
        if self.aggr == 'mean':
            out = torch.stack(hop_features).mean(dim=0)  # (n, hdim)
        elif self.aggr == 'max':
            out = torch.stack(hop_features).max(dim=0)[0]  # (n, hdim)
        else:  # concat
            out = torch.cat(hop_features, dim=-1)  # (n, (K+1)*hdim)
        
        out = F.dropout(out, self.dropout, self.training)
        return F.relu(self.enc_out(out))  # (n, hdim)



class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()

        self.w = Parameter(torch.zeros([in_channels, out_channels]))
        self.b = Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.w)

    def forward(self, x, adj):
        x = torch.mm(x, self.w)
        x = matmul(adj, x)
        x = x + self.b
        return x


class SimpleGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGraphConvolution, self).__init__()

        self.w = Parameter(torch.zeros([in_channels, out_channels]))
        self.b = Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.w)

    def forward(self, x, adj=None):
        x = torch.mm(x, self.w)
        if adj is not None:
            x = matmul(adj, x)
        x = x + self.b
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hdim, out_dim, num_layers=2, act='relu', batch_norm=False):
        super(MLP, self).__init__()
        assert num_layers >= 2, 'num_layers should be larger or equal than 2.'

        if num_layers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hdim)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hdim, hdim))
            layers.append(nn.Linear(hdim, out_dim))
        self.layers = nn.ModuleList(layers)

        if batch_norm:
            self.norm = nn.BatchNorm2d(hdim)
        else:
            self.norm = lambda x: x

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'silu':
            self.act = nn.SiLU()

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.norm(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x


class MHEncoder(nn.Module):
    def __init__(self, in_channels, config):
        super(MHEncoder, self).__init__()
        hidden_channels = config['cl_hdim']
        self.K = config['preprop_K']
        self.beta = config['beta']
        self.dropout = config['dropout']
        self.config = config

        self.reduce_weight = Parameter(torch.randn([self.K+1, in_channels, hidden_channels]))

        self.enc_out = nn.Linear((self.K+1)*hidden_channels, hidden_channels)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.reduce_weight.data)
        self.enc_out.reset_parameters()

    @torch.no_grad()
    def prop(self, x, adj):
        x_L = x.clone()  # (n, dim)
        out_L = []
        for _ in range(self.K):
            x_L = self.beta * x_L + matmul(adj, x_L)  # X' = (βI+DAD)X = βX+DADX
            out_L.append(x_L)
        return torch.stack([x] + out_L, dim=0)  # (K+1, n, dim)

    def forward(self, out_prop, adj=None):
        """
        out_prop: (K+1, n, dim) if adj is None else (n, dim)
        """
        num_nodes = out_prop.shape[-2]

        if adj is not None:
            out_prop = self.prop(out_prop, adj)  # (K+1, n, dim)

        out_prop = torch.bmm(out_prop, self.reduce_weight)  # (K+1, n, hdim)
        out_prop = F.normalize(out_prop, p=2, dim=-1)
        out_prop = F.relu(out_prop)
        out_prop = out_prop.permute(1, 0, 2).reshape(num_nodes, -1)  # (n, (K+1)*hdim)
        out_prop = F.dropout(out_prop, self.dropout, self.training)
        return F.relu(self.enc_out(out_prop))  # (n, hdim)