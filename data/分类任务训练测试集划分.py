import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
import random

# 设置随机种子确保结果可复现
random.seed(42)
torch.manual_seed(42)


def load_data(data_path):
    """加载数据集"""
    print(f"加载数据: {data_path}")
    data = torch.load(data_path)
    print(f"数据加载完成: {data}")
    return data


def generate_train_val_test_by_coarse(data, train_samples_per_coarse=8, val_samples_per_coarse=8):
    """
    按数据自带的粗类标签划分训练集、验证集和测试集，使用索引列表而非掩码
    假设数据中已有 'label' 字段作为粗类标签
    """
    required_samples = train_samples_per_coarse + val_samples_per_coarse

    # 直接使用数据自带的粗类标签
    if isinstance(data.label, torch.Tensor):
        coarse_labels = data.label.tolist()
    else:
        coarse_labels = data.label

    # 获取所有独特的粗类标签
    unique_coarse_labels = list(set(coarse_labels))
    print(f"\n数据中包含的粗类标签: {unique_coarse_labels}")

    # 收集每个粗类的样本索引
    coarse_samples = defaultdict(list)
    for idx, coarse_label in enumerate(coarse_labels):
        coarse_samples[coarse_label].append(idx)

    # 打印粗类样本分布
    print("\n粗类样本分布:")
    for coarse_label, samples in sorted(coarse_samples.items(), key=lambda x: -len(x[1])):
        print(f"  粗类 {coarse_label}: {len(samples)} 个样本")

    # 筛选出样本数足够的粗类
    valid_coarse_labels = []
    insufficient_coarse_labels = []
    for coarse_label, samples in coarse_samples.items():
        if len(samples) >= required_samples:
            valid_coarse_labels.append(coarse_label)
        else:
            insufficient_coarse_labels.append(coarse_label)

    print(f"\n有效粗类 ({len(valid_coarse_labels)}): {valid_coarse_labels}")
    print(f"样本不足的粗类 ({len(insufficient_coarse_labels)}): {insufficient_coarse_labels}")

    # 初始化结果列表
    train_indices = []
    val_indices = []
    test_indices = []

    # 为每个有效粗类抽取固定数量的训练和验证样本
    for coarse_label in valid_coarse_labels:
        samples = coarse_samples[coarse_label]
        random.shuffle(samples)

        train = samples[:train_samples_per_coarse]
        val = samples[train_samples_per_coarse:train_samples_per_coarse + val_samples_per_coarse]

        # 打印训练样本的标签
        train_labels = [coarse_labels[idx] for idx in train]
        print(f"粗类 {coarse_label}: 训练标签={train_labels}")

        train_indices.extend(train)
        val_indices.extend(val)

        # 测试集包含所有未被选中的节点
        test_indices.extend([idx for idx in samples if idx not in train and idx not in val])

        print(
            f"粗类 {coarse_label}: 训练集={len(train)}, 验证集={len(val)}, 测试集={len(samples) - len(train) - len(val)}")

    # 对于样本不足的粗类，所有样本都放入测试集
    for coarse_label in insufficient_coarse_labels:
        test_indices.extend(coarse_samples[coarse_label])

    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    print(f"测试集大小: {len(test_indices)}")

    # 保存索引列表
    data.trainindex = torch.tensor(train_indices, dtype=torch.long)
    data.valindex = torch.tensor(val_indices, dtype=torch.long)
    data.testindex = torch.tensor(test_indices, dtype=torch.long)

    # 创建只包含训练集标签的张量
    train_labels = torch.tensor(
        [coarse_labels[idx] for idx in train_indices],
        dtype=torch.long
    )

    return train_indices, val_indices, test_indices, valid_coarse_labels, insufficient_coarse_labels, train_labels


def save_processed_data(data, train_indices, val_indices, test_indices, valid_coarse_labels,
                        filtered_node_indices, insufficient_coarse_labels, output_path):
    """
    保存处理后的数据集，保留所有节点

    参数:
        data: 原始数据集
        train_indices: 训练节点索引列表
        val_indices: 验证节点索引列表
        test_indices: 测试节点索引列表
        valid_coarse_labels: 有效粗类列表
        filtered_node_indices: 有效节点索引列表（未使用）
        insufficient_coarse_labels: 样本不足的粗类列表
        output_path: 输出文件路径
    """
    # 创建新的数据对象
    processed_data = data.clone()

    # 保存划分索引
    processed_data.classtrain = torch.tensor(train_indices, dtype=torch.long)
    processed_data.classval = torch.tensor(val_indices, dtype=torch.long)
    processed_data.classtest = torch.tensor(test_indices, dtype=torch.long)

    # 保存粗类相关信息
    processed_data.valid_coarse_labels = valid_coarse_labels
    processed_data.insufficient_coarse_labels = insufficient_coarse_labels
    print(processed_data,"data")
    # 保存处理后的数据
    torch.save(processed_data, output_path)
    print(f"已保存处理后的数据到 {output_path}")
    print(f"训练样本数: {len(train_indices)}")
    print(f"验证样本数: {len(val_indices)}")
    print(f"测试样本数: {len(test_indices)}")
    print(f"总节点数: {data.num_nodes}")
    print(f"有效粗类: {valid_coarse_labels}")
    print(f"样本不足的粗类: {insufficient_coarse_labels}")


def print_coarse_class_names(valid_coarse_labels):
    """打印所有有效粗类的名称"""
    print("\n所有有效粗类的标签:")
    for idx, label in enumerate(sorted(valid_coarse_labels)):
        print(f"  {idx + 1}. {label}")


def main():
    # 配置参数
    data_path = "test3.pt"  # 数据集路径
    output_dir = "processed"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置每个粗类选取的样本数
    k = 8  # 训练样本数
    v = 8  # 验证样本数

    # 加载数据
    data = load_data(data_path)
    print(data)

    # 检查数据是否包含粗类标签
    if not hasattr(data, 'label'):
        raise ValueError("数据中未找到 'label' 字段，请确认粗类标签的字段名")

    # 按粗类生成划分（使用数据自带的粗类标签）
    train_indices, val_indices, test_indices, valid_coarse_labels, insufficient_coarse_labels, trainlabel = generate_train_val_test_by_coarse(
        data,
        train_samples_per_coarse=k,
        val_samples_per_coarse=v
    )

    print(f"样本不足的粗类: {insufficient_coarse_labels}")

    # 保存处理后的数据
    output_path = os.path.join(output_dir, f"test4.pt")
    save_processed_data(data, train_indices, val_indices, test_indices, valid_coarse_labels, [],
                        insufficient_coarse_labels, output_path)

    # 打印所有有效粗类的标签
    print_coarse_class_names(valid_coarse_labels)
    print(data)
    print("训练集标签:", data.label[data.trainindex])


if __name__ == "__main__":
    main()
#Data(y=[3000], text=[3000], num_nodes=3000, train_pos_edge_index=[2, 6352], val_pos_edge_index=[2, 794], test_pos_edge_index=[2, 794], train_neg_edge_index=[2, 6352], val_neg_edge_index=[2, 794], test_neg_edge_index=[2, 794], label=[3000], coarse_classes=[9], trainindex=[64], valindex=[64], testindex=[2872], classtrain=[64], classval=[64], classtest=[2872], valid_coarse_labels=[8], insufficient_coarse_labels=[1])
#下一步：检索任务粗处理