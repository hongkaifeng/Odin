import torch
import json
import csv
import time
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def generate_node_text(data, output_path: str):
    """生成Node_text.tsv（参考文档图3格式，对应节点文本存储逻辑）"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # 按节点编号顺序写入，与data.text一一对应（参考类中self.text的存储方式）
        for node_id in range(data.num_nodes):
            node_text = data.text[node_id]
            writer.writerow([node_id, node_text])
    print(f"Node_text.tsv已生成，保存至：{output_path}")


def generate_node_classification(data, output_path: str, neighbor_config: list = [5, 5, 5]):
    """生成Node_classification.jsonl（参考EvalNCCGraphDataset的子图采样逻辑）"""
    # 构建图结构（参考类中self.graph的定义）
    graph = Data(
        x=torch.arange(data.num_nodes, dtype=torch.long).view(-1, 1),  # 节点特征为索引张量（避免list类型错误）
        edge_index=data.train_pos_edge_index  # 使用训练集边索引
    )

    # 训练节点索引（参考类中self.train_indices，若无需筛选可直接使用所有节点）
    train_indices = range(data.num_nodes)  # 此处使用所有节点作为中心节点

    def process_single_node(node_idx):
        """处理单个中心节点，参考类中process_fn逻辑"""
        # 1. 获取中心节点标签（对应类中self.labels）
        label = data.y[node_idx].item()  # 文档中标签存储在data.label
        label_name = data.ytext[node_idx]  # 标签文本取自ytext

        # 2. 邻居子图采样（完全复用类中NeighborLoader配置）
        loader = NeighborLoader(
            graph,
            num_neighbors=neighbor_config,  # 每层采样邻居数，与类中self.neibor一致
            batch_size=1,
            input_nodes=torch.tensor([node_idx]),  # 目标节点
        )

        # 获取子图（参考类中subgraph_batch处理）
        subgraph_batch = next(iter(loader))
        subgraph_batch.x = subgraph_batch.n_id  # 子图节点使用原始全局ID

        # 3. 提取子图节点文本x1（对应类中node_texts提取方式）
        node_global_ids = subgraph_batch.x.tolist()  # 子图节点的全局索引
        x1 = [data.text[gid] for gid in node_global_ids]  # 从原始文本列表中取文本

        # 4. 提取子图边索引edge1（转换为列表格式，匹配文档图2）
        edge1 = [
            subgraph_batch.edge_index[0].tolist(),  # 源节点局部索引
            subgraph_batch.edge_index[1].tolist()  # 目标节点局部索引
        ]

        end = time.perf_counter()
        # 输出调试信息（可选）
        # print(f"处理节点 {node_idx} 耗时: {end - start:.4f}s，子图节点数: {len(node_global_ids)}")

        return {
            "x1": x1,
            "edge1": edge1,
            "labels": [label],  # 文档要求的标签ID列表
            "label names": [label_name]  # 文档要求的标签文本列表
        }

    # 遍历所有训练节点生成JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for node_idx in train_indices:
            subgraph_data = process_single_node(node_idx)
            f.write(json.dumps(subgraph_data, ensure_ascii=False) + '\n')

    print(f"Node_classification.jsonl已生成，保存至：{output_path}，共{len(train_indices)}条数据")


if __name__ == "__main__":
    # 加载数据（参考类中data加载方式，添加weights_only参数解决警告）
    data_path = "processed_data.pt"
    data = torch.load(data_path, weights_only=False)  # 保持与类中加载方式一致

    # 生成Node_text.tsv
    generate_node_text(data, "Node_text.tsv")

    # 生成Node_classification.jsonl（使用与EvalNCCGraphDataset相同的邻居采样配置）
    generate_node_classification(
        data,
        output_path="Node_classification.jsonl",
        neighbor_config=[5, 5, 5]  # 与类中self.neibor保持一致
    )