import torch

# 加载处理后的数据集
data = torch.load("test2.pt")

# 定义完整的细类到粗类的映射（包含所有细类）
category_mapping = {
    # 人工智能与机器学习
    "arxiv cs ai": "Artificial Intelligence",  # 人工智能
    "arxiv cs lg": "Artificial Intelligence",  # 机器学习
    "arxiv cs ne": "Artificial Intelligence",  # 神经计算

    # 计算机视觉与多媒体
    "arxiv cs cv": "Computer Vision & Multimedia",  # 计算机视觉
    "arxiv cs mm": "Computer Vision & Multimedia",  # 多媒体

    # 自然语言处理
    "arxiv cs cl": "Natural Language Processing",  # 计算语言学
    "arxiv cs ir": "Natural Language Processing",  # 信息检索

    # 系统与网络
    "arxiv cs os": "Systems & Networks",  # 操作系统
    "arxiv cs dc": "Systems & Networks",  # 分布式计算
    "arxiv cs ni": "Systems & Networks",  # 网络
    "arxiv cs ar": "Systems & Networks",  # 架构

    # 数据与数据库
    "arxiv cs db": "Data & Databases",  # 数据库
    "arxiv cs ds": "Data & Databases",  # 数据结构
    "arxiv cs si": "Data & Databases",  # 社会信息网络

    # 算法与理论
    "arxiv cs cc": "Algorithms & Theory",  # 计算复杂性
    "arxiv cs dm": "Algorithms & Theory",  # 离散数学
    "arxiv cs gt": "Algorithms & Theory",  # 博弈论
    "arxiv cs ma": "Algorithms & Theory",  # 数学分析
    "arxiv cs ms": "Algorithms & Theory",  # 数学软件（新增映射）

    # 软件工程与编程语言
    "arxiv cs se": "Software Engineering",  # 软件工程
    "arxiv cs pl": "Software Engineering",  # 编程语言
    "arxiv cs fl": "Software Engineering",  # 形式化方法
    "arxiv cs cr": "Software Engineering",  # 密码学与安全（新增映射）

    # 人机交互与可视化
    "arxiv cs hc": "HCI & Visualization",  # 人机交互
    "arxiv cs cg": "HCI & Visualization",  # 计算机图形学
    "arxiv cs gr": "HCI & Visualization",  # 图形学

    # 其他领域
    "arxiv cs ce": "Other",  # 计算工程
    "arxiv cs cy": "Other",  # 计算机与社会
    "arxiv cs dl": "Other",  # 数字图书馆
    "arxiv cs gl": "Other",  # 一般文献
    "arxiv cs it": "Other",  # 信息论
    "arxiv cs lo": "Other",  # 逻辑
    "arxiv cs na": "Other",  # 数值分析
    "arxiv cs oh": "Other",  # 其他
    "arxiv cs pf": "Other",  # 性能分析
    "arxiv cs ro": "Other",  # 机器人学
    "arxiv cs sc": "Other",  # 符号计算
    "arxiv cs sd": "Other",  # 语音识别
    "arxiv cs st": "Other",  # 统计理论
    "arxiv cs sy": "Other",  # 系统理论
    "arxiv cs et": "Other"  # 教育技术（新增映射）
}

# 更正后的原始细类列表（完整40个）
original_classes = [
    "arxiv cs na", "arxiv cs mm", "arxiv cs lo", "arxiv cs cy", "arxiv cs cr",
    "arxiv cs dc", "arxiv cs hc", "arxiv cs ce", "arxiv cs ni", "arxiv cs cc",
    "arxiv cs ai", "arxiv cs ma", "arxiv cs gl", "arxiv cs ne", "arxiv cs sc",
    "arxiv cs ar", "arxiv cs cv", "arxiv cs gr", "arxiv cs et", "arxiv cs sy",
    "arxiv cs cg", "arxiv cs oh", "arxiv cs pl", "arxiv cs se", "arxiv cs lg",
    "arxiv cs sd", "arxiv cs si", "arxiv cs ro", "arxiv cs it", "arxiv cs pf",
    "arxiv cs cl", "arxiv cs ir", "arxiv cs ms", "arxiv cs fl", "arxiv cs ds",
    "arxiv cs os", "arxiv cs gt", "arxiv cs db", "arxiv cs dl", "arxiv cs dm"
]

# 创建粗类到索引的映射
coarse_classes = sorted(list(set(category_mapping.values())))  # 排序确保一致性
coarse_class_to_idx = {name: i for i, name in enumerate(coarse_classes)}

# 将原始细类标签转换为粗类标签
data.label = torch.zeros_like(data.y)  # 初始化粗类标签张量
for i, original_label_idx in enumerate(data.y):
    original_class_name = original_classes[original_label_idx.item()]
    coarse_class_name = category_mapping[original_class_name]  # 此时已确保所有细类都有映射
    data.label[i] = coarse_class_to_idx[coarse_class_name]

# 输出统计信息
print(f"原始细类数量: {len(original_classes)}")
print(f"合并后的粗类数量: {len(coarse_classes)}")
print(f"粗类名称: {coarse_classes}")
print(f"粗类标签形状: {data.label.shape}")
data.coarse_classes=coarse_classes
# 验证映射完整性（最终检查）
unmapped = [cls for cls in original_classes if cls not in category_mapping]
if unmapped:
    print(f"警告: 以下细类未被映射到粗类: {unmapped}")
else:
    print("所有细类均已成功映射到粗类")

# 保存处理后的数据
save_path = "test3.pt"
print(data.label)
print(data.coarse_classes)
torch.save(data, save_path)
print(f"含粗类标签的数据已保存至: {save_path}")

#Data(y=[3000], text=[3000], num_nodes=3000, train_pos_edge_index=[2, 6352], val_pos_edge_index=[2, 794], test_pos_edge_index=[2, 794], train_neg_edge_index=[2, 6352], val_neg_edge_index=[2, 794], test_neg_edge_index=[2, 794], label=[3000], coarse_classes=[9])
