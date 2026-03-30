





import torch
from dig.threedgraph.dataset import ECdataset,FOLDdataset
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GINConv, GINEConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.loader import DataLoader
from typing import Callable, List, Optional
import collections
import pdb
from copy import deepcopy
from random import choice
from torch_geometric.utils import degree
from itertools import repeat, product
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import shutil
import networkx as nx
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
from backbone_GIN_backup import *
# from backbone_model import *
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.nn import SAGEConv
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score
import time
names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes', 'node_attributes_2',
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes', '7000011_C','7000011_N','7000011_CA','7000011_C1','7000011_C2','7000011_C3','7000011_C4','7000011_C5','7000011_C6'
]

# to log the output of the experiments to a file


class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    elements_count = {}

    node_attributes = torch.empty((batch.size(0), 1))

    xx = np.bincount(batch)
    # print(xx)
    list = []
    cc = 0
    # for i in xx:
    #     # print(i)
    #     z = i
    #     for j in range(i):
    #         node_attributes[j+cc][0] = z
    #         # print(z)
    #     cc = cc + i
        
    node_attributes = torch.empty((batch.size(0), 0))
    # node_attributes1 = torch.empty((batch.size(0), 0))
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
        if node_attributes.dim() == 1:
            node_attributes = node_attributes.unsqueeze(-1)
    if 'node_attributes_2' in names:
        node_attributes_2 = read_file(folder, prefix, 'node_attributes_2')
        if node_attributes_2.dim() == 1:
            node_attributes_2 = node_attributes_2.unsqueeze(-1)
    if '7000011_C' in names:
        coor_C = read_file(folder, prefix, '7000011_C')
        if coor_C.dim() == 1:
            coor_C = coor_C.unsqueeze(-1)
            node_attributes_2 = node_attributes_2.unsqueeze(-1)
    if '7000011_N' in names:
        coor_N = read_file(folder, prefix, '7000011_N')
        if coor_N.dim() == 1:
            coor_N = coor_N.unsqueeze(-1)
    if '7000011_CA' in names:
        coor_CA = read_file(folder, prefix, '7000011_CA')
        if coor_CA.dim() == 1:
            coor_CA = coor_CA.unsqueeze(-1)
    if '7000011_C1' in names:
        coor_C1 = read_file(folder, prefix, '7000011_C1')
        if coor_C1.dim() == 1:
            coor_C1 = coor_C1.unsqueeze(-1)
    if '7000011_C2' in names:
        coor_C2 = read_file(folder, prefix, '7000011_C2')
        if coor_C2.dim() == 1:
            coor_C2 = coor_C2.unsqueeze(-1)
    if '7000011_C3' in names:
        coor_C3 = read_file(folder, prefix, '7000011_C3')
        if coor_C3.dim() == 1:
            coor_C3 = coor_C3.unsqueeze(-1)
    if '7000011_C4' in names:
        coor_C4 = read_file(folder, prefix, '7000011_C4')
        if coor_C4.dim() == 1:
            coor_C4 = coor_C4.unsqueeze(-1)
    if '7000011_C5' in names:
        coor_C5 = read_file(folder, prefix, '7000011_C5')
        if coor_C5.dim() == 1:
            coor_C5 = coor_C5.unsqueeze(-1)
    if '7000011_C6' in names:
        coor_C6 = read_file(folder, prefix, '7000011_C6')
        if coor_C6.dim() == 1:
            coor_C6 = coor_C6.unsqueeze(-1)
    # if 'node_attributes1' in names:
    #     node_attributes1 = read_file(folder, prefix, 'node_attributes1')
    #     if node_attributes1.dim() == 1:
    #         node_attributes1 = node_attributes1.unsqueeze(-1)

    print('ok')

    # node_labels = torch.empty((batch.size(0), 0))
    # if 'node_labels' in names:
    #     node_labels = read_file(folder, prefix, 'node_labels', torch.long)
    #     if node_labels.dim() == 1:
    #         node_labels = node_labels.unsqueeze(-1)
    #     node_labels = node_labels - node_labels.min(dim=0)[0]
    #     node_labels = node_labels.unbind(dim=-1)
    #     node_labels = [F.one_hot(x) for x in node_labels]
    #     if len(node_labels) == 1:
    #         node_labels = node_labels[0]
    #     else:
    #         print("ok")
    #         node_labels = torch.cat(node_labels, dim=-1)

    print("ok")
    node_labels = torch.empty((batch.size(0), 0))
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        print('123')
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
        print('1234')
    print("edge_index.size(1)",edge_index.size)
    print("x",node_labels.shape,node_attributes.shape)
    edge_attributes = torch.empty((edge_index.size(1), 0))
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
        if edge_attributes.dim() == 1:
            edge_attributes = edge_attributes.unsqueeze(-1)

    edge_labels = torch.empty((edge_index.size(1), 0))
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    if 'sequence' in names:
        sequence = read_file(folder, prefix, 'sequence', torch.long)
        
    print("#######",node_labels.shape,node_attributes.shape,node_attributes_2.shape)    
    # print(node_attributes.shape,node_labels.shape,node_attributes1.shape)   
    x = cat([node_attributes_2,node_attributes, node_labels,coor_C,coor_N,coor_CA,coor_C1,coor_C2,coor_C3,coor_C4,coor_C5,coor_C6]) ##,node_attributes1
    # print("x",x)
    # print("x",x.shape)
    print("#######",x.shape,node_labels.shape,node_attributes.shape)
    print("***",x,node_labels,"&&&*&*",node_attributes)
    edge_attr = cat([edge_attributes, edge_labels])
    print("edge_attr",edge_attr.shape)
    print("edge_attr",edge_attr,edge_attributes,edge_labels)
    ###########mask_x
    # print("***mask_x")
    # size_x = (x.shape[0], x.shape[1])
    # x = torch.zeros(size_x)
    # print("***after",x)
    ###########mask edge_attr
    # print("***mask, edge_attr")
    # size_edge_attr = (edge_attr.shape[0], edge_attr.shape[1])
    # edge_attr = torch.zeros(size_edge_attr)
    # print("edge_attr222",edge_attr.shape)
    # print("edge_attr222",edge_attr)
    ############
    # size_edge_attr = (edge_attr.shape[0], edge_attr.shape[1])
    # mask = torch.zeros(1, 1, 4, dtype=torch.bool)
    # mask[:, :, 2] = True

    # 使用遮罩对张量进行索引操作，保留第三维度的信息
    # edge_attr = edge_attr[mask]

    # 打印结果的形状
    print(edge_attr.shape)
    print("x",x)
    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        # _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)
    # print(num_nodes)
    # print(node_attributes.size(-1))
    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_node_labels': node_labels.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_labels': edge_labels.size(-1),
    }

    return data, slices, sizes


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


class TUDataset(InMemoryDataset):

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        # model.load_state_dict(torch.load(self.processed_paths[0]))
        # print(out)
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        self.data, self.slices, self.sizes = out

        # if self._data.x is not None and not use_node_attr:
        #     num_node_attributes = self.num_node_attributes
        #     self._data.x = self._data.x[:, num_node_attributes:]
        # if self._data.edge_attr is not None and not use_edge_attr:
        #     num_edge_attrs = self.num_edge_attributes
        #     self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
        # print(sizes)
        if self.pre_filter is not None or self.pre_transform is not None:
            # print("#(*(*(*(*(*)))))",get)
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self._data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def train(model, train_set,train_f, device='cuda', args=None, tensor_board_path=None, model_path=None, loader = None, loader2= None):

    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    if (args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    epochs = args.epochs

    accu = 0
    if args.dataset == 'fold':
        best_val_acc = 0
        test_fold_at_best_val_acc = 0
        test_super_at_best_val_acc = 0
        test_family_at_best_val_acc = 0
    elif args.dataset == 'EC':
        best_val_acc = 0
        test_at_best_val_acc = 0
    for epoch in range(epochs+1):
        t_start = time.perf_counter()
        model.train()
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        pass_flag = False
        loss_list = []
        cur_lr = get_lr(optimizer)

        for batch1 in tqdm(loader):

            batch1 = batch1.to(device)
            batch2 = batch1
            model.train()
            optimizer.zero_grad()
            x = batch1.x.to(device)
            # edge_attr = batch1.edge_attr.to(device)
            edge_attr =123

            edge_index = 123
            batch = batch1.batch.to(device)
            y = batch1.y.to(device)

            if (args.model == 'GCN'):
                _, out = model(x, edge_index, batch1)
            elif (args.model == 'GIN'):
                _, out = model(x, edge_index, batch1)
            elif (args.model == 'GIN_Attribute'):
                _, out = model(x, edge_index, batch1, edge_attr,batch2,args)
            elif (args.model == 'GIN_Attribute_1'):
                out = model(batch1)
            elif (args.model == 'GraphSAGE'):
                out = model(x, edge_index, batch,edge_attr)
            elif(args.model == 'GraphAttentionEmbedding'):
                _, out = model(batch,batch)
            elif(args.model == 'Net'):
                # indices = indices.long()
                # (self, x, edge_index, edge_attr, batch):
                _, out = model(x, edge_index, batch, edge_attr,)

    
            # if out.size(0) != y.size(0):
            #     y = y[:out.size(0)]
            loss = criterion(out, y)
            loss_list.append(loss)
            acc += accuracy(out.argmax(dim=1), y, device) / len(loader)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # scheduler.step()
        if args.dataset == 'fold':

            t_end_train = time.perf_counter()
            val_loss, val_acc = test(model, test_fold_loader,test_fold_loader)
            t_start_test = time.perf_counter()
            test_fold_loss, test_fold_acc = test(model, test_fold_loader,test_fold_loader)
            test_super_loss, test_super_acc = test(model, test_super_loader,test_super_loader)
            test_family_loss, test_family_acc = test(model, test_family_loader,test_family_loader)
            t_end_test = time.perf_counter()

            if val_acc > best_val_acc:
                # print('Saving best val checkpoint ...')
                # # print(save_dir)
                # checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
                # torch.save(checkpoint, save_dir + '/best_val.pt')
                best_val_acc = val_acc    
                test_fold_at_best_val_acc = test_fold_acc
                test_super_at_best_val_acc = test_super_acc
                test_family_at_best_val_acc = test_family_acc       
            t_end = time.perf_counter()
            print('Train:{} Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, '\
                'Test_fold: Loss:{:.6f} Acc:{:.4f}, Test_super: Loss:{:.6f} Acc:{:.4f}, Test_family: Loss:{:.6f} Acc:{:.4f}, '\
                'test_fold_acc@best_val:{:.4f}, test_super_acc@best_val:{:.4f}, test_family_acc@best_val:{:.4f}, '\
                'time:{}, train_time:{}, test_time:{}'.format(epoch,
                loss, acc, val_loss, val_acc, 
                test_fold_loss, test_fold_acc, test_super_loss, test_super_acc, test_family_loss, test_family_acc, 
                test_fold_at_best_val_acc, test_super_at_best_val_acc, test_family_at_best_val_acc, 
                t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))
        elif args.dataset == 'EC':
            val_loss, val_acc = test(model, val_loader,test_loader)
            val_loss, val_acc = test(model, test_loader,test_loader)
         

        

        total_loss = sum(loss_list) / len(loss_list)
        # print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
        #       f'| Train Acc: {acc*100:>5.2f}% '
        #       f'| Val Loss: {val_loss:.2f} '
        #       f'| Val Acc: {val_acc*100:.2f}%')
        # print("accu", accu)
        # if (epoch % 10 == 0):
        #     print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
        #           f'| Train Acc: {acc*100:>5.2f}% '
        #           f'| Val Loss: {val_loss:.2f} '
        #           f'| Val Acc: {val_acc*100:.2f}%')
        # writer.add_scalar('Total Loss:', total_loss, epoch)
        # writer.add_scalar("Lr/train", cur_lr, epoch)
        if (epoch % 10 == 0):
            torch.save(model.state_dict(), '{}/{}.pth'.format(model_path, epoch))
            # torch.save(model, '{}/{}.pth'.format(model_path, epoch))

    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    # print(LR)
    return model

criterion = torch.nn.CrossEntropyLoss()
def test(model, loader,loader2, device='cuda'):
    
    model.eval()
    loss = 0
    acc = 0

    # for data in loader:
    for batch1 in tqdm(loader):

            # batch1 = batch1.to(device)
            # model.train()
            # optimizer.zero_grad()
            # x = batch1.x.to(device)
            # edge_attr = batch1.edge_attr.to(device)
            # # print(edge_attr)
            # # batch = batch.to(device)

            # edge_index = batch1.edge_index.to(device)
            # batch = batch1.batch.to(device)
            # y = batch1.y.to(device)


        # print()
        batch2 = batch1.to(device)
        x = batch1.x.to(device)
        # edge_index = batch1.edge_index.to(device)
        batch = batch1.batch.to(device)
        y = batch1.y.to(device)
        # edge_attr = batch1.edge_attr.to(device)
        edge_attr = 123
        edge_index = 123
        # _, out = model(x, edge_index, batch)
        if (args.model == 'GCN'):
            _, out = model(x, edge_index, batch)
        elif (args.model == 'GIN'):
            _, out = model(x, edge_index, batch)
        elif (args.model == 'GraphSAGE'):
            out = model(x, edge_index, batch,edge_attr)
        elif (args.model == 'GIN_Attribute'):
            # print('x',x.shape)
            # print(edge_attr.shape)
            try:
                _, out = model(x, edge_index, batch1, edge_attr,batch2,args)
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): 
                    print('\n forward error \n')
                    raise(e)
                else:
                    print('evaluation OOM')
                torch.cuda.empty_cache()
                continue
            # _, out = model(x, edge_index, batch, edge_attr)
        elif (args.model == 'GIN_Attribute_1'):
            batch1 = batch1.to('cuda')
            out = model(batch1)
        elif (args.model == 'GraphAttentionEmbedding'):
            _, out = model(data,batch)
        elif (args.model == 'Net'):
            _, out = model(x, edge_index, batch, edge_attr)
        # out = model(x, edge_index, batch)
        # if out.size(0) != y.size(0):
        #     y = y[:out.size(0)]
        # try:
        #     loss = criterion(out, y)
        # except:
        #     print('123')
        #     continue
        loss = criterion(out, y)
        loss += criterion(out, y) / len(loader)
        # print(out,y,out.shape,y.shape)
        acc += accuracy(out.argmax(dim=1), y, device) / len(loader)

    return loss, acc

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
# def Average(lst):
#     return sum(lst) / len(lst)
f1_score_= []
recall_score_=[]
precision_score_=[]
consolidated_frequency_table = {}
def accuracy(pred_y, y, device):
    """Calculate accuracy."""
    # f1_score(y_true, y_pred, average='macro')
    # print(classification_report(pred_y.cpu(), y.cpu()))
    # print('f1-score',f1_score(pred_y.cpu(), y.cpu(), average='weighted'))
    # print('recall_score',recall_score(pred_y.cpu(), y.cpu(), average='weighted'))
    # # print('precision_score',precision_score(pred_y.cpu(), y.cpu(), average='macro'))
    # print('precision_score',precision_score(pred_y.cpu(), y.cpu(), average='macro'))
    # f1_score_.append(f1_score(pred_y.cpu(), y.cpu(), average='macro'))
    # recall_score_.append(recall_score(pred_y.cpu(), y.cpu(), average='macro'))
    # precision_score_.append(precision_score(pred_y.cpu(), y.cpu(), average='macro'))
    # print(sum(f1_score_) / len(f1_score_))
    # print(sum(recall_score_) / len(recall_score_))
    # print(sum(precision_score_) / len(precision_score_))
    
    # 找到与真实值相等的元素
    
    equal_elements = pred_y[pred_y == y]

    # 计算每个元素的频率
    unique_elements, counts = np.unique(equal_elements.cpu(), return_counts=True)

    # 创建频率表
    frequency_tables = dict(zip(unique_elements, counts))
    for key, value in frequency_tables.items():
        if key in consolidated_frequency_table:
            consolidated_frequency_table[key] += value
        else:
            consolidated_frequency_table[key] = value

    
    return ((pred_y == y).sum() / len(y)).item()

from typing import Tuple

import torch


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score


def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train_gnn')
    parser.add_argument('--dataset', type=str, default='func', help='Func or fold')

    parser.add_argument('--training_title', type=str, default='',
                        help='')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='')
    parser.add_argument('--dim_h', type=int, default=64,
                        help='')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='')
    parser.add_argument('--model', type=str, default='GIN',
                        help='')
    parser.add_argument('--pretrain_weight', type=str, default='',
                        help='')
    parser.add_argument('--eval_batch_size', type=int, default=60,
                        help='')
    # parser.add_argument('--eval_batch_size', type=int, default=60,
    #                     help='')
    parser.add_argument('--edge', type=str, default='',
                        help='')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='')
    parser.add_argument('--dataset_path', type=str, default=4,
                        help='')
    parser.add_argument('--bond', type=str, default='',
                        help='')

    parser.add_argument('--version', type=str, default='',
                        help='')
    parser.add_argument('--type', type=str, default='',
                        help='')
    args = parser.parse_args()
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'

    training_title = args.training_title
    create_path('log_outputs/hao/{}'.format(training_title))
    set_logger('log_outputs/hao/{}/train_models{}version{}bond{}modelbond{}edge{}'.format(training_title,args.version,args.bond,args.bond,args.edge,123))


    ###datasetrandomshuffle
    # dataset = TUDataset(root='/hdd/yishan0128/', name='test1w_node1')
    
    # dataset = dataset[:29880]
    # train_dataset =dataset[0:25880]
    # test_dataset = dataset[25881:27880]
    # val_dataset =dataset[27881:29880]
    ######deeep

    # train_dataset = TUDataset(root="/hdd/yishan0128/", name='train1') ##
    # test_dataset = TUDataset(root="/hdd/yishan0128/", name='test1')
    # print("www")
    # www = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='700001w')
    # print(www[0].x.shape)
    # train_f = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='train2')
    # val_set_2 = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='val2')
    # test_set_2 = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='test2')
    if args.version == 'mine':
        if args.dataset == 'fold':
            train_set = TUDataset(root='/hdd/yishan0128/Fold_class',name='7000011')
            test_family = TUDataset(root='/hdd/yishan0128/Fold_class',name='test_fam')
            test_fold = TUDataset(root='/hdd/yishan0128/Fold_class',name='test_fold')
            test_super = TUDataset(root='/hdd/yishan0128/Fold_class',name='test_super') 
            train_set = train_set[0:12300]
            test_family = test_family[0:1270]
            test_fold = test_fold[0:716]
            test_super = test_super[0:1252]
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_fold_loader = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_super_loader = DataLoader(test_super, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_family_loader = DataLoader(test_family, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

        if args.dataset == 'EC':
            train_set = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='train4')
            # val_set = TUDataset(root='/hdd/yishan0128/Fold_class',name='fold')
            val_set = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='val4')
            test_set = TUDataset(root='/hdd/yishan0128/ProteinFunc',name='test4')


            train_set = train_set[:29210]     
            test_set = test_set[:5645]
            val_set = val_set[:2560]
            test_set = test_set[:2881]+test_set[2882:2883]+test_set[2884:]
            train_set = train_set[:4242]+train_set[4243:4462]+train_set[4463:12809]+train_set[12810:14134]+train_set[14135:14200]+train_set[14200:14250]+train_set[14250:14275]+train_set[14300:14400]+train_set[14400:14600]+ train_set[14600:15777]+ train_set[15778:18045]+train_set[18046:20998]+train_set[20999:27098]+train_set[27099:]
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        if args.dataset == 'fold':
            train_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='training')
            val_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='validation')
            test_fold = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_fold')
            test_super = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_superfamily')
            test_family = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_family')
            # train_set = train_set[0:12300]
            # test_family = test_family[0:1270]
            # test_fold = test_fold[0:716]
            # test_super = test_super[0:1252]
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_fold_loader = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_super_loader = DataLoader(test_super, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_family_loader = DataLoader(test_family, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
             

    # dataset = dataset[0:20340] ##701
    # dataset = dataset[0:17420]
    # print(train_f,val_set_2)

 
    torch.manual_seed(12345)
    # dataset = dataset.shuffle() ###
    # train_dataset = dataset[1:15000]
    # val_dataset = dataset[15001:16300]
    # test_dataset = dataset[16301:17420]

    # ######701
    # train_dataset = dataset[1:18340]
    # val_dataset = dataset[18340:19340]
    # test_dataset = dataset[19341:20339]
    ######501
    # train_dataset = dataset[1:14400]
    # val_dataset = dataset[14400:15200]
    # test_dataset = dataset[15201:16078]

    # train_dataset = dataset[1:60000]
    # val_dataset = dataset[60001:62000]
    # test_dataset = dataset[62001:65700]  

    # train_dataset = dataset[1:2300]
    # val_dataset = dataset[2301:2450]
    # test_dataset = dataset[2301:2450]
    # train_dataset = dataset[1:2500]
    # val_dataset = dataset[2501:2700]
    # test_dataset = dataset[2701:2900]
    # train_dataset = dataset[1:14000]
    # val_dataset = dataset[14001:15001]
    # test_dataset = dataset[15001:16000]  
    # train_dataset = dataset[1:1500]
    # val_dataset = dataset[1500:1650]
    # test_dataset = dataset_t[11750:13000]  


    if (args.model == 'GCN'):
        model = GCN(dataset=dataset, dim_h=args.dim_h,
                    class_num=dataset.num_classes)
    elif (args.model == 'GIN'):
        model = GIN(dataset=dataset, dim_h=args.dim_h,
                    class_num=dataset.num_classes)
    elif (args.model == 'GIN_Attribute'):
        model = GIN_Attribute(dataset=train_set, dim_h=args.dim_h,
                    class_num=train_set.num_classes, args = args)
    elif (args.model == 'GIN_Attribute_1'):
        model = GINGraphPooling(num_tasks=train_f.num_classes, num_layers=2, emb_dim=81, residual=False, drop_ratio=0, JK="last", graph_pooling="sum")
        print("model",model)
    elif (args.model == 'GraphSAGE'):
        model = GraphSAGE(dataset=dataset, hidden_dim=args.dim_h, num_layers=15)
    elif (args.model == 'GraphAttentionEmbedding'):
        model = GCN_1(dataset).to(device)
    elif (args.model == 'Net'):
        model = MyGIN(22, dataset.num_classes).to(device)


    # model.to(device)
    if (args.pretrain_weight != ''):
        # model = torch.load(args.pretrain_weight)
        model.load_state_dict(torch.load(args.pretrain_weight))
        
    model.to(device)

    model_path = 'weight/{}'.format(training_title)
    create_path('weight/{}'.format(training_title))

    tensor_board_path = 'runs/{}/train_models121'.format(training_title)

    print("Arguments: ")
    argument_list = ""
    for arg in vars(args):
        argument_list += " --{} {}".format(arg, getattr(args, arg))
    print(argument_list)
    # print(f'Dataset: {dataset[0]}')
    # print(f'Dataset: {dataset}')
    # print('-------------------')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of edge_index: {dataset}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of node features: {dataset.num_node_features}')
    # print(f'Number of classes: {dataset.num_classes}')


    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)
    model = train(model=model, train_set=train_set, train_f =train_set,device=device, args=args,
                  tensor_board_path=tensor_board_path, model_path=model_path, loader= train_loader, loader2 = train_loader)
    test_loss, test_acc = test(model, test_loader,test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

    print(consolidated_frequency_table)
    # print(frequency_table)
    labels = list(consolidated_frequency_table.keys())
    counts = list(consolidated_frequency_table.values())

    # 绘制条形图
    plt.bar(labels, counts)
    plt.xlabel('Label')
    plt.ylabel('Predictions Correct Count')
    plt.title('Predictions Correct Count vs Label')
    plt.xticks(rotation=45)  # 如果标签过多，可以旋转标签以防止重叠

    # 保存图表为图像文件（PNG、JPEG、SVG等格式都支持）
    plt.savefig(f'predictions_correct_count{args.training_title}.png', bbox_inches='tight')  # 修改文件名和路径为您希望的保存位置

    # 不显示图表在界面上
    plt.close()

