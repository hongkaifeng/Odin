import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.data import Data

# 加载数据（添加 weights_only=False 忽略安全警告）
data = torch.load("D:/mymodel/truedataset/ogbn_arxiv_processed/processed_data.pt", weights_only=False)
#data = torch.load("D:/mymodel/truedataset/amazon_sports_token_graph6.pt", weights_only=False)
np.random.seed (43) # 固定 numpy 随机数
torch.manual_seed (43) # 固定 torch 随机数
#标签42->1 43->2
# 转换为NetworkX图（无向图便于BFS）
G = to_networkx(data, to_undirected=True)

# 为每个节点添加类别属性
for node in G.nodes():
    G.nodes[node]['category'] = int(data.y[node])
    #G.nodes[node]['category'] = data.y[node]
    #G.nodes[node]['categories'] = data.y[node]
# 随机选择起始节点
start_node = np.random.randint(0, data.num_nodes)

# BFS队列和已访问节点集合
queue = [start_node]
visited = set([start_node])

# 记录已访问类别的节点数
category_count = {i: 0 for i in range(int(data.y.max()) + 1)}
category_count[G.nodes[start_node]['category']] += 1

# BFS扩展节点，优先选择能增加类别多样性的节点
target_size = 3000
while len(visited) < target_size and queue:
    current_node = queue.pop(0)

    # 获取未访问的邻居节点
    unvisited_neighbors = [n for n in G.neighbors(current_node) if n not in visited]


    # 按类别多样性排序
    def diversity_score(node):
        category = G.nodes[node]['category']
        return category_count[category]


    unvisited_neighbors.sort(key=diversity_score)

    # 添加邻居节点
    for neighbor in unvisited_neighbors:
        if len(visited) >= target_size:
            break

        visited.add(neighbor)
        queue.append(neighbor)
        category_count[G.nodes[neighbor]['category']] += 1

# 将访问的节点列表转换为PyTorch张量（关键修改！）
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
#torch.save(subgraph_data, 'D:/mymodel/truedataset/obgn_arxiv_NO1.pt')
torch.save(subgraph_data, 'test.pt')
# 打印统计信息
print(subgraph_data)
print(f"子图节点数: {len(subgraph_nodes)}")
print(f"子图边数: {subgraph_edge_index.shape[1]}")
print(f"类别分布: {category_count}")
print(f"包含的类别数量: {len([c for c in category_count.values() if c > 0])}")
print(f"子图边索引示例: {subgraph_edge_index[:, :5]}")  # 验证索引范围

#Data(edge_index=[2, 7940], y=[3000], text=[3000], num_nodes=3000)

#下一步：为pyg小图划分训练边验证边测试边
