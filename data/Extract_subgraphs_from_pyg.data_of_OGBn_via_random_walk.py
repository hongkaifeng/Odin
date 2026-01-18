import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.data import Data

# 加载数据（添加 weights_only=False 忽略安全警告）
data = torch.load("D:/mymodel/truedataset/ogbn_arxiv_processed/processed_data.pt", weights_only=False)
# data = torch.load("D:/mymodel/truedataset/amazon_sports_token_graph6.pt", weights_only=False)
np.random.seed(22)  # 固定 numpy 随机数
torch.manual_seed(22)  # 固定 torch 随机数

# 转换为NetworkX图（无向图便于随机游走）
G = to_networkx(data, to_undirected=True)

# 为每个节点添加类别属性
for node in G.nodes():
    G.nodes[node]['category'] = int(data.y[node])

# 随机选择起始节点
start_node = np.random.randint(0, data.num_nodes)

# 随机游走参数
target_size = 3000
walk_length = 5000  # 增加游走长度以确保能访问足够多的节点
visited = set()
path = []

# 从起始节点开始随机游走
current_node = start_node
visited.add(current_node)
path.append(current_node)

while len(visited) < target_size and len(path) < walk_length:
    neighbors = list(G.neighbors(current_node))

    if not neighbors:
        # 如果当前节点没有邻居，随机选择一个已访问的节点继续
        current_node = np.random.choice(list(visited))
        continue

    # 随机选择一个邻居节点
    next_node = np.random.choice(neighbors)

    if next_node not in visited:
        visited.add(next_node)

    path.append(next_node)
    current_node = next_node

# 如果访问的节点数量不足目标大小，补充随机节点
if len(visited) < target_size:
#if 0:
    remaining_nodes = [n for n in range(data.num_nodes) if n not in visited]
    additional_nodes = np.random.choice(remaining_nodes, size=min(target_size - len(visited), len(remaining_nodes)),
                                        replace=False)
    visited.update(additional_nodes)

# 记录类别分布
category_count = {i: 0 for i in range(int(data.y.max()) + 1)}
for node in visited:
    category = G.nodes[node]['category']
    category_count[category] += 1

# 将访问的节点列表转换为PyTorch张量
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
