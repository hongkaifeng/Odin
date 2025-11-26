# 示例 1：使用内置数据集（Cora）
import torch
from torch_geometric.data import Data


def count_bidirectional_edges(data: Data, count_pairs: bool = True) -> int:
    """
    统计 PyG 图中双向边的数量

    Args:
        data: PyG 图数据对象（需包含 edge_index）
        count_pairs: 统计口径开关（True=统计双向边对数量，False=统计双向边总条数）

    Returns:
        双向边数量（按指定口径）
    """
    edge_index = data.edge_index  # (2, E)，E 为总边数
    num_edges = edge_index.shape[1]

    # 1. 将边转换为元组形式（u, v），并去重（避免重复处理同一对边）
    # 注意：边 (u, v) 和 (v, u) 去重后仍会保留两条，后续需判断是否成对
    edges = set()
    for i in range(num_edges):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        edges.add((u, v))  # 去重后的边集合（无重复边）

    # 2. 统计双向边（核心逻辑）
    bidirectional_pairs = set()  # 存储不重复的双向边对（如 (u,v) 和 (v,u) 只存一次）
    self_loops = set()  # 存储自环边（u, u）

    for (u, v) in edges:
        if u == v:
            # 自环边：本身就是双向的，单独统计
            self_loops.add((u, v))
        else:
            # 非自环边：判断反向边是否存在，且避免重复统计（如先处理 (u,v) 就不处理 (v,u)）
            if (v, u) in edges and (u, v) not in bidirectional_pairs and (v, u) not in bidirectional_pairs:
                bidirectional_pairs.add((u, v))

    # 3. 按口径返回结果
    if count_pairs:
        # 口径 1：双向边对数量 = 双向边对组数 + 自环边数（每组/每条算1个）
        return len(bidirectional_pairs) + len(self_loops)
    else:
        # 口径 2：双向边总条数 = 双向边对组数×2 + 自环边数（每组算2条，自环算1条）
        return len(bidirectional_pairs) * 2 + len(self_loops)

# 加载数据集（Cora 是无向图，所有边都是双向的）
file_path ="test.pt"
data = torch.load(file_path)

# 统计双向边
bidirectional_pairs = count_bidirectional_edges(data, count_pairs=True)
bidirectional_total = count_bidirectional_edges(data, count_pairs=False)

print(f"图的总边数（去重后）：{len(set(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy())))}")
print(f"双向边对数量（口径1）：{bidirectional_pairs}")
print(f"双向边总条数（口径2）：{bidirectional_total}")

