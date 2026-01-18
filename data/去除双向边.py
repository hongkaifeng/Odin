
import os
import glob
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.loader import NeighborLoader
import networkx as nx
from typing import List, Tuple, Dict
import time
def is_bidirectional(data: Data, strict: bool = False) -> bool:
    """
    检查PyG图数据对象是否为双向图

    参数:
    data (Data): PyG图数据对象
    strict (bool): 是否严格检查（每个边都必须有反向边且仅出现一次）

    返回:
    bool: 如果是双向图返回True，否则返回False
    """
    edge_index = data.edge_index

    # 创建边的集合 (i, j)
    edges = set()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edges.add((src, dst))

    # 检查每条边是否都有反向边
    for src, dst in edges:
        if (dst, src) not in edges:
            return False

    # 严格模式下，检查反向边的数量是否与原始边相同
    if strict:
        return len(edges) == edge_index.size(1) // 2

    return True







def filter_edges_by_direction(edge_index):
    """
    筛选边索引，只保留标号较大的节点到标号较小的节点的边

    参数:
    edge_index: 形状为 [2, num_edges] 的边索引张量

    返回:
    filtered_edge_index: 筛选后的边索引张量
    """
    # 获取每条边的源节点和目标节点
    src = edge_index[0]
    dst = edge_index[1]

    # 创建掩码，保留 src > dst 的边
    mask = src > dst

    # 应用掩码筛选边
    filtered_edge_index = edge_index[:, mask]

    return filtered_edge_index


def make_edge_index_bidirectional(edge_index):
    reversed_edge_index = edge_index.flip(0)
    bidirectional_edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    bidirectional_edge_index = torch.unique(bidirectional_edge_index, dim=1)

    # 确保张量是连续的
    return bidirectional_edge_index.contiguous()



#file_path = "obgn-3000.pt"  # 请替换为实际文件路径citeseer_random_sbert "D:\mymodel\8-5maindataset\wikics_fixed_sbert.pt"
#file_path = "citeseer_random_sbert.pt"
#file_path = "raw_cora_data.pt"
file_path ="test.pt"
data = torch.load(file_path)
print(data.edge_index.shape)
bio_edge_index=make_edge_index_bidirectional(data.edge_index)
print(bio_edge_index.shape)
fit_edge=filter_edges_by_direction(bio_edge_index)
print(fit_edge.shape)
data.edge_index=fit_edge
torch.save(data,"test0.pt")