


from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GINConv
# from torch_geometric.nn import HeteroConv
from torch_geometric.nn import radius_graph
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
# from torch_geometric.loader import DataLoader
from typing import Callable, List, Optional
import collections
import pdb
from copy import deepcopy
from random import choice
from itertools import repeat, product
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.nn import GINConv, GINEConv#,GATv2Conv
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
import shutil
import torch
# import networkx as nx
import matplotlib.pyplot as plt
import torch.optim as optim
import glob
import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
import torch.nn.functional as F
# from ogb.graphproppred.mol_encoder import AtomEncoder
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.nn import SAGEConv
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
# from torch_sparse import SparseTensor

from torch_geometric.nn import TGNMemory, TransformerConv
import torch
from torch_geometric.nn import TransformerConv, Linear
from scipy.ndimage import zoom

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
# from ogb.graphproppred.mol_encoder import BondEncoder

from features import d_angle_emb, d_theta_phi_emb
def compute_diherals( v1, v2, v3):
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)
    a = (n1 * n2).sum(dim=-1)
    b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
    torsion = torch.nan_to_num(torch.atan2(b, a))
    return torsion
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def bb_embs_1( X):   
    # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
    # N coords: X[:,0,:]
    # CA coords: X[:,1,:]
    # C coords: X[:,2,:]
    # return num_residues x 6 
    # From https://github.com/jingraham/neurips19-graph-protein-design
    
    X = torch.reshape(X, [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u0 = U[:-2]
    u1 = U[1:-1]
    u2 = U[2:]

    angle = compute_diherals(u0, u1, u2)
        
    # add phi[0], psi[-1], omega[-1] with value 0
    angle = F.pad(angle, [1, 2]) 
    angle = torch.reshape(angle, [-1, 3])
    angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
    return angle_features

def compute_diherals(v1, v2, v3):
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)
    a = (n1 * n2).sum(dim=-1)
    b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
    torsion = torch.nan_to_num(torch.atan2(b, a))
    return torsion
def side_chain_embs_1( pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
    v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

    # five side chain torsion angles
    # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
    angle1 = torch.unsqueeze(compute_diherals(v1, v2, v3),1)
    angle2 = torch.unsqueeze(compute_diherals(v2, v3, v4),1)
    angle3 = torch.unsqueeze(compute_diherals(v3, v4, v5),1)
    angle4 = torch.unsqueeze(compute_diherals(v4, v5, v6),1)
    angle5 = torch.unsqueeze(compute_diherals(v5, v6, v7),1)

    side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
    side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
    
    return side_chain_embs

class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels , out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        # print("edge_weight",edge_weight[0].shape,x.shape)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        # print("x_j",edge_weight.shape,x_j.shape)
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class EdgeGraphConv_wo(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv_wo, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels , out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        # print("edge_weight",edge_weight[0].shape,x.shape)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        # print("x_j",edge_weight.shape,x_j.shape)
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.7, JK="last", residual=True):
        """GIN Node Embedding Module"""

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        mmm = 6
        for layer in range(num_layers):
            # print("range(num_layers)",range(num_layers))
            self.convs.append(GINEConv(
            Sequential(Linear(81, emb_dim),
                       BatchNorm1d(emb_dim, track_running_stats=False), ReLU(),
                       Linear(emb_dim, emb_dim), ReLU()), train_eps=True, edge_dim=mmm))
            print(GINConv1)
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        # computing input node embedding
        # print()
        x = x.to(torch.int64)
        h_list = [x]  # 先将类别型原子属性转化为原子表征
        h = x
        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation
 

class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=True, drop_ratio=0.5, JK="last", graph_pooling="sum"):
        """GIN Graph Pooling Module
        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表征的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            # 因为预测目标的取值范围就在 (0, 50] 内
            return torch.clamp(output, min=0, max=50)


def swish(x):
    return x * torch.sigmoid(x)



mmm = 6
class GIN_Attribute(torch.nn.Module):
    def __init__(self, dataset, dim_h, class_num, args):
        super(GIN_Attribute, self).__init__()
        #####mmm to alter the num_of_attr
        mmm = 24
        self.relu = nn.ReLU()
        DO = 0.4
        self.m =  BatchNorm1d(dim_h, track_running_stats=False)
        dim_x =15

        self.l_1 = Linear(dim_h, dim_h)

        self.l_2 = Linear(dim_h, dim_h)

        self.l_3 = Linear(dim_h, dim_h)

        self.l_4 = Linear(dim_h, dim_h)

        self.l_5 = Linear(dim_h, dim_h)
        self.l_6 = Linear(dim_h, dim_h)

        self.l_7 = Linear(dim_h, dim_h)
        self.feature0 = d_theta_phi_emb(num_radial=6, num_spherical=2, cutoff=8)
        self.feature1 = d_angle_emb(num_radial=6, num_spherical=2, cutoff=8)
        self.relu = nn.ReLU()
   # self.mean_aggr = aggr.MeanAggregation()
        if args.version == 'their':
            self.x1 = Linear(15,dim_h)
            self.x2 = Linear(30,dim_h)
        elif args.version == 'mine':
            if args.bond == 'true':
                self.x1 = Linear(121,dim_h)
                self.x2 = Linear(6,dim_h)
            if args.bond == 'false':
                self.x1 = Linear(121,dim_h)
                self.x2 = Linear(242,dim_h)
        self.lin1 = Linear(dim_h, dim_h*5)
        self.lin2 = Linear(dim_h*5, class_num)
        self.lin3 = Linear(dim_h, mmm)
        self.x = Linear(dim_h, dim_h)
        self.lins_out = torch.nn.ModuleList()
        self.lins_out.append(Linear(dim_h*6, dim_h))
        for _ in range(2-1):
            self.lins_out.append(Linear(dim_h, dim_h))
        self.lin_out = Linear(dim_h, dim_h*6)
        # self.m = nn.BatchNorm1d(63)
        self.lin_1 = Linear(54, dim_h)
        self.lin_2 = Linear(54, dim_h)
        self.emb = nn.Linear(6, 32)
        self.z = nn.Linear(54, 81)
        self.emb_1 = nn.Linear(32, 32)
        self.z_1 = nn.Linear(81, 81)
        self.emb_2 = nn.Linear(32, 32)
        self.z_2 = nn.Linear(81, 81)
        self.emb_3 = nn.Linear(32, 32)
        self.z_3 = nn.Linear(81, 81)
        self.emb_4 = nn.Linear(32, 32)
        self.z_4 = nn.Linear(81, 81)
        self.norm = nn.BatchNorm1d(81)
        self.act = swish
        if args.type =='MPNN':
            if args.edge == 'true':
                self.conv0 = EdgeGraphConv(dim_h, dim_h)
                self.conv1 = EdgeGraphConv(dim_h, dim_h)
                self.conv2 = EdgeGraphConv(dim_h, dim_h)
                self.conv3 = EdgeGraphConv(dim_h, dim_h)
                self.conv4 = EdgeGraphConv(dim_h, dim_h)
                self.conv5 = EdgeGraphConv(dim_h, dim_h)
                self.conv6 = EdgeGraphConv(dim_h, dim_h)
                self.conv7 = EdgeGraphConv(dim_h, dim_h)
            elif args.edge == 'false':      
                self.conv0 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv1 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv2 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv3 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv4 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv5 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv6 = EdgeGraphConv_wo(dim_h, dim_h)
                self.conv7 = EdgeGraphConv_wo(dim_h, dim_h)
        elif args.type == 'GCN':
            # print("self.type",self.type)
            self.conv1 = GCNConv(dim_h, dim_h)
            self.conv2 = GCNConv(dim_h, dim_h)
            self.conv3 = GCNConv(dim_h, dim_h)
            self.conv4 = GCNConv(dim_h, dim_h)
            self.conv5 = GCNConv(dim_h, dim_h)

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
# (x, edge_index, batch1, edge_attr,batch2)
    def forward(self, x, edge_index, batch, edge_attr,batch2,args):
        DO = 0.2
        # print("args.version",args.version, args.bond)
        device = x.device
        if args.version == 'their':
            z, pos, batch1 = torch.squeeze(batch.x.long()), batch.coords_ca, batch.batch
            pos_n = batch.coords_n
            pos_c = batch.coords_c
            CA = pos
            bb_embs = batch.bb_embs
            side_chain_embs = batch.side_chain_embs
            x = x.to('cuda')
            bb_embs =bb_embs.to('cuda')
        if args.version == 'mine':
            # print("mine")
            CA = batch.x[:,60:63]
            C = batch.x[:,54:57]
            N = batch.x[:,57:60]
            # z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
            z, pos, batch1 = torch.squeeze(batch.x[:,:54].long()), CA, batch.batch
            pos_n = N
            pos_c = C
            pos_ca = CA
            pos_cb = batch.x[:,63:66]
            pos_g = batch.x[:,66:69]
            pos_d = batch.x[:,69:72]
            pos_e = batch.x[:,72:75]
            pos_z = batch.x[:,75:78]
            pos_h = batch.x[:,78:81]
            bb_embs = bb_embs_1(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
            # bb_embs = batch_data.bb_embs
            side_chain_embs = side_chain_embs_1(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
            side_chain_embs[torch.isnan(side_chain_embs)] = 0
            # print(side_chain_embs.shape)
            side_chain_embs = side_chain_embs.to('cuda')
            device = z.device
            bb_embs = self.emb(bb_embs)
            
            # print("bb_embs",bb_embs)
            z = self.z(z.float())
            x =z

        side_chain_embs = side_chain_embs.to('cuda')
        x = torch.cat((x,bb_embs, side_chain_embs),dim=1)

        edge_index = radius_graph(CA, r=9, batch=batch.batch, max_num_neighbors=15)
        edge_index =edge_index.to('cuda')

        src_index = edge_index[0]
        dst_index = edge_index[1]

        src_features = x[src_index]
        dst_features = x[dst_index]
        # tt = dst_features
        # print("src_features",src_features.shape)
        # print("dis_features",dst_features.shape)
        edge_attr = dst_features
        dst_features = torch.cat((src_features, dst_features), dim=1)
        # print("dis_features222",dst_features.shape)
        tt = dst_features.to(device)

        
        if args.version == 'mine':
            if args.edge == 'true':
                x = self.x1(x)
                tt = self.relu(self.x2(tt))
                # print(x.to(torch.float32).shape,edge_index.shape)
                h1 = self.conv1(x.to(torch.float32), edge_index,tt)
                h1 = self.l_1(h1)
                h1 = self.m(h1)

                h1 = self.act(h1)
                h1 = F.dropout(h1, p=DO, training=self.training)

                h2 = self.conv2(h1, edge_index,tt)
                h2 = self.l_2(h2)
                # h2 = self.relu(h2)
                h2 = self.act(h2)
                h2 = F.dropout(h2, p=DO, training=self.training)
                h2 = self.m(h2)
                h3 = self.conv3(h2, edge_index,tt)
                h3 = self.l_3(h3)
                h3 = self.m(h3)
                h3 = self.act(h3)
                h3 = F.dropout(h3, p=DO, training=self.training)
                
                h4 = self.conv4(h3, edge_index,tt)
                h4 = self.l_4(h4)
                h4 = self.m(h4)
                h4 = self.act(h4)
                h4 = F.dropout(h4, p=DO, training=self.training)
                h5 = self.conv5(h4, edge_index,tt)
                h5 = self.l_5(h5)
                h5 = self.m(h5)
                h5 = self.act(h5)
                h5 = F.dropout(h5, p=DO, training=self.training)

                # h6 = self.conv6(h5, edge_index,tt)
                # h6 = self.l_6(h6)
                # h6 = self.m(h6)
                # h6 = self.act(h6)
                # h6 = F.dropout(h6, p=DO, training=self.training)

                # h7 = self.conv7(h6, edge_index,tt)
                # h7 = self.l_7(h7)
                # h7 = self.m(h7)
                # h7 = self.act(h7)
                # h7 = F.dropout(h7, p=DO, training=self.training)
                x = self.x(x.to(torch.float32))     
            
            else:
                if args.bond == 'true':
                    x = self.x1(x)
                    batch.edge_attr = self.x2(batch.edge_attr)
                    batch.edge_attr = self.relu(batch.edge_attr)
                    h1 = self.conv1(x.to(torch.float32), batch.edge_index,batch.edge_attr)
                    h1 = self.l_1(h1)
                    h1 = self.m(h1)

                    h1 = self.act(h1)
                    # print(h1)
                    h1 = F.dropout(h1, p=DO, training=self.training)

                    h2 = self.conv2(h1, batch.edge_index,batch.edge_attr)
                    h2 = self.l_2(h2)
                    # h2 = self.relu(h2)
                    h2 = self.act(h2)
                    h2 = F.dropout(h2, p=DO, training=self.training)
                    h2 = self.m(h2)
                    h3 = self.conv3(h2, batch.edge_index,batch.edge_attr)
                    h3 = self.l_3(h3)
                    h3 = self.m(h3)
                    h3 = self.act(h3)
                    h3 = F.dropout(h3, p=DO, training=self.training)
                    
                    h4 = self.conv4(h3, batch.edge_index,batch.edge_attr)
                    h4 = self.l_4(h4)
                    h4 = self.m(h4)
                    h4 = self.act(h4)
                    h4 = F.dropout(h4, p=DO, training=self.training)
                    h5 = self.conv5(h4, batch.edge_index,batch.edge_attr)
                    h5 = self.l_5(h5)
                    h5 = self.m(h5)
                    h5 = self.act(h5)
                    h5 = F.dropout(h5, p=DO, training=self.training)

                    h6 = self.conv6(h5, batch.edge_index,batch.edge_attr)
                    h6 = self.l_6(h6)
                    h6 = self.m(h6)
                    h6 = self.act(h6)
                    h6 = F.dropout(h6, p=DO, training=self.training)

                    h7 = self.conv7(h6, batch.edge_index,batch.edge_attr)
                    h7 = self.l_7(h7)
                    h7 = self.m(h7)
                    h7 = self.act(h7)
                    h7 = F.dropout(h7, p=DO, training=self.training)
                    x = self.x(x.to(torch.float32))
            if args.bond == 'false' and args.edge == 'false':
                x = self.x1(x)
                # print("args.bond",args.bond)
                tt = self.x2(tt)
                tt = self.relu(tt)
                h1 = self.conv1(x.to(torch.float32), edge_index)
                h1 = self.l_1(h1)
                h1 = self.m(h1)

                h1 = self.act(h1)
                # print(h1)
                h1 = F.dropout(h1, p=DO, training=self.training)

                h2 = self.conv2(h1, edge_index)
                h2 = self.l_2(h2)
                # h2 = self.relu(h2)
                h2 = self.act(h2)
                h2 = F.dropout(h2, p=DO, training=self.training)
                h2 = self.m(h2)
                h3 = self.conv3(h2, edge_index)
                h3 = self.l_3(h3)
                h3 = self.m(h3)
                h3 = self.act(h3)
                h3 = F.dropout(h3, p=DO, training=self.training)
                
                h4 = self.conv4(h3, edge_index)
                h4 = self.l_4(h4)
                h4 = self.m(h4)
                h4 = self.act(h4)
                h4 = F.dropout(h4, p=DO, training=self.training)
                h5 = self.conv5(h4, edge_index)
                h5 = self.l_5(h5)
                h5 = self.m(h5)
                h5 = self.act(h5)
                h5 = F.dropout(h5, p=DO, training=self.training)

                # h6 = self.conv6(h5,edge_index,tt)
                # h6 = self.l_6(h6)
                # h6 = self.m(h6)
                # h6 = self.act(h6)
                # h6 = F.dropout(h6, p=DO, training=self.training)

                # h7 = self.conv7(h6,edge_index,tt)
                # h7 = self.l_7(h7)
                # h7 = self.m(h7)
                # h7 = self.act(h7)
                # h7 = F.dropout(h7, p=DO, training=self.training)
                x = self.x(x.to(torch.float32))
        if args.version == 'their':
            if args.edge == 'true':
                x = self.x1(x)
                tt = self.relu(self.x2(tt))
                # print(x.to(torch.float32).shape,edge_index.shape)
                h1 = self.conv1(x.to(torch.float32), edge_index,tt)
                h1 = self.l_1(h1)
                h1 = self.m(h1)

                h1 = self.act(h1)
                h1 = F.dropout(h1, p=DO, training=self.training)

                h2 = self.conv2(h1, edge_index,tt)
                h2 = self.l_2(h2)
                # h2 = self.relu(h2)
                h2 = self.act(h2)
                h2 = F.dropout(h2, p=DO, training=self.training)
                h2 = self.m(h2)
                h3 = self.conv3(h2, edge_index,tt)
                h3 = self.l_3(h3)
                h3 = self.m(h3)
                h3 = self.act(h3)
                h3 = F.dropout(h3, p=DO, training=self.training)
                
                h4 = self.conv4(h3, edge_index,tt)
                h4 = self.l_4(h4)
                h4 = self.m(h4)
                h4 = self.act(h4)
                h4 = F.dropout(h4, p=DO, training=self.training)
                h5 = self.conv5(h4, edge_index,tt)
                h5 = self.l_5(h5)
                h5 = self.m(h5)
                h5 = self.act(h5)
                h5 = F.dropout(h5, p=DO, training=self.training)

                # h6 = self.conv6(h5, edge_index,tt)
                # h6 = self.l_6(h6)
                # h6 = self.m(h6)
                # h6 = self.act(h6)
                # h6 = F.dropout(h6, p=DO, training=self.training)

                # h7 = self.conv7(h6, edge_index,tt)
                # h7 = self.l_7(h7)
                # h7 = self.m(h7)
                # h7 = self.act(h7)
                # h7 = F.dropout(h7, p=DO, training=self.training)
                x = self.x(x.to(torch.float32))
            if args.edge == 'false':
                x = self.x1(x)
                # tt = self.x2(tt)
                # print(x.to(torch.float32).shape,edge_index.shape)
                h1 = self.conv1(x.to(torch.float32), edge_index)
                h1 = self.l_1(h1)
                h1 = self.m(h1)

                h1 = self.act(h1)
                h1 = F.dropout(h1, p=DO, training=self.training)

                h2 = self.conv2(h1, edge_index)
                h2 = self.l_2(h2)
                # h2 = self.relu(h2)
                h2 = self.act(h2)
                h2 = F.dropout(h2, p=DO, training=self.training)
                h2 = self.m(h2)
                h3 = self.conv3(h2, edge_index)
                h3 = self.l_3(h3)
                h3 = self.m(h3)
                h3 = self.act(h3)
                h3 = F.dropout(h3, p=DO, training=self.training)
                
                h4 = self.conv4(h3, edge_index)
                h4 = self.l_4(h4)
                h4 = self.m(h4)
                h4 = self.act(h4)
                h4 = F.dropout(h4, p=DO, training=self.training)
                h5 = self.conv5(h4, edge_index)
                h5 = self.l_5(h5)
                h5 = self.m(h5)
                h5 = self.act(h5)
                h5 = F.dropout(h5, p=DO, training=self.training)

                h6 = self.conv6(h5, edge_index)
                h6 = self.l_6(h6)
                h6 = self.m(h6)
                h6 = self.act(h6)
                h6 = F.dropout(h6, p=DO, training=self.training)

                h7 = self.conv7(h6, edge_index)
                h7 = self.l_7(h7)
                h7 = self.m(h7)
                h7 = self.act(h7)
                h7 = F.dropout(h7, p=DO, training=self.training)
                x = self.x(x.to(torch.float32))

        h1 = global_add_pool(h1, batch.batch.to('cuda'))
        h2 = global_add_pool(h2, batch.batch.to('cuda'))
        h3 = global_add_pool(h3, batch.batch.to('cuda'))
        h4 = global_add_pool(h4, batch.batch.to('cuda'))
        h5 = global_add_pool(h5, batch.batch.to('cuda'))


        x = self.m(x)
        x = global_add_pool(x, batch.batch.to('cuda'))
        h = torch.cat((x,h1, h2, h3, h4, h5), dim=1)
        # print("h^^^^^^####",h,h.shape)
        # Classifier
        # print(h)
        # h = h + x
        for lin in self.lins_out:
            h = self.relu(lin(h))
            h = F.dropout(h, p=0.2, training=self.training)  
        h = self.lin1(h)
 
        # print(h
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin2(h)
        # print("h1^^^^****####",F.log_softmax(h, dim=1),F.log_softmax(h, dim=1).shape)
        return h, F.log_softmax(h, dim=1)
    





