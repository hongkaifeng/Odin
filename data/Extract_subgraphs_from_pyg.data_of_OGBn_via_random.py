import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.data import Data

# 加载数据（添加 weights_only=False 忽略安全警告）
data = torch.load("D:/mymodel/truedataset/ogbn_arxiv_processed/processed_data.pt", weights_only=False)
# data = torch.load("D:/mymodel/truedataset/amazon_sports_token_graph6.pt", weights_only=False)
np.random.seed(422)  # 固定 numpy 随机数
torch.manual_seed(422)  # 固定 torch 随机数

# 转换为NetworkX图（仅用于后续类别统计，随机采样无需游走）
G = to_networkx(data, to_undirected=True)

# 为每个节点添加类别属性
for node in G.nodes():
    G.nodes[node]['category'] = int(data.y[node])

# ===================== 核心修改：随机采样节点 =====================
target_size = 3000  # 目标采样节点数
# 从所有节点中随机选取 target_size 个节点（无放回采样）
all_nodes = np.arange(data.num_nodes)  # 所有节点的索引数组
sampled_nodes = np.random.choice(all_nodes, size=target_size, replace=False)
visited = set(sampled_nodes)  # 采样的节点集合

# ===================== 以下逻辑与原代码一致 =====================
# 记录类别分布
category_count = {i: 0 for i in range(int(data.y.max()) + 1)}
for node in visited:
    category = G.nodes[node]['category']
    category_count[category] += 1

# 将采样的节点列表转换为PyTorch张量
subgraph_nodes = torch.tensor(list(visited), dtype=torch.long)

# 使用PyG的subgraph函数正确提取子图的边索引
subgraph_edge_index, _ = subgraph(subgraph_nodes, data.edge_index, relabel_nodes=True)

# 提取子图的文本和标签
subgraph_text = [data.text[i] for i in subgraph_nodes.tolist()]  # 转为列表索引
subgraph_y = data.y[subgraph_nodes]

# 创建正确的子图Data对象
subgraph_data = Data(
    edge_index=subgraph_edge_index,
    text=subgraph_text,
    y=subgraph_y,
    num_nodes=len(subgraph_nodes)
)

# 保存子图数据
# torch.save(subgraph_data, 'D:/mymodel/truedataset/obgn_arxiv_NO1.pt')
torch.save(subgraph_data, 'test-test.pt')

# 打印统计信息
print(subgraph_data)
print(f"子图节点数: {len(subgraph_nodes)}")
print(f"子图边数: {subgraph_edge_index.shape[1]}")
print(f"类别分布: {category_count}")
print(f"包含的类别数量: {len([c for c in category_count.values() if c > 0])}")
print(f"子图边索引示例: {subgraph_edge_index[:, :5]}")  # 验证索引范围