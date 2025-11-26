import torch
import json
from torch_geometric.data import Data


def add_ytext_and_export_json(data: Data, label_map: dict) -> Data:
    """
    为PyG数据对象添加ytext属性并导出标签文本到JSON

    参数:
    data (Data): 包含y属性的PyG数据对象
    label_map (dict): 标签ID到完整文本的映射字典

    返回:
    Data: 添加了ytext属性的PyG数据对象
    """
    # 确保标签映射覆盖所有可能的标签
    unique_labels = torch.unique(data.y).tolist()
    missing_labels = [label for label in unique_labels if label not in label_map]
    if missing_labels:
        raise ValueError(f"标签映射缺少以下标签: {missing_labels}")

    # 创建ytext属性
    ytext = [label_map[int(label)] for label in data.y]
    data.ytext = ytext

    # 导出完整标签文本到JSON（新格式）
    document = [{"id": int(label_id), "contents": text} for label_id, text in label_map.items()]
    with open('documents.json', 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    return data
def load_data_from_pt(file_path: str) -> Data:
    """从.pt文件加载PyG数据对象"""
    try:
        data = torch.load(file_path)
        if not isinstance(data, Data):
            raise TypeError(f"加载的数据类型不是torch_geometric.data.Data，而是{type(data)}")
        print(f"成功从{file_path}加载数据，包含{data.num_nodes}个节点")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise
def save_data_to_pt(data: Data, file_path: str) -> None:
    """将PyG数据对象保存到.pt文件"""
    try:
        torch.save(data, file_path)
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时出错: {e}")
        raise

# 使用示例
if __name__ == "__main__":
    # 假设这是您的数据集
    file_path = "test4.pt"  # 请替换为实际文件路径
    data = load_data_from_pt(file_path)

    # 使用完整的OGBN-Arxiv标签映射（英文全写）
    label_map = {
        0: "Numerical Analysis",  # 数值分析
        1: "Multimedia",  # 多媒体
        2: "Logic in Computer Science",  # 逻辑学
        3: "Computers and Society",  # 计算机与社会
        4: "Cryptography and Security",  # 密码学与安全
        5: "Distributed Parallel and Cluster Computing",  # 分布式计算
        6: "Human Computer Interaction",  # 人机交互
        7: "Computational Engineering",  # 计算工程
        8: "Networking and Internet Architecture",  # 网络
        9: "Computational Complexity",  # 计算复杂性
        10: "Artificial Intelligence",  # 人工智能
        11: "Multiagent Systems",  # Multiagent Systems
        12: "General Literature",  # 一般文献General Literature
        13: "Neural and Evolutionary Computing",  # 神经计算Neural and Evolutionary Computing
        14: "Symbolic Computation",  # 符号计算Symbolic Computation
        15: "Hardware Architecture",  # 计算机架构Hardware Architecture
        16: "Computer Vision",  # 计算机视觉
        17: "Graphics",  # 图形学Graphics
        18: "Emerging Technologies",  # Emerging Technologies
        19: "Systems and Control",  # 系统理论Systems and Control
        20: "Computational Geometry",  # 计算机图形学Computers and Society
        21: "Other Computer Science",  # 其他计算机科学Other Computer Science
        22: "Programming Languages",  # 编程语言Programming Languages
        23: "Software Engineering",  # 软件工程
        24: "Machine Learning",  # 机器学习
        25: "Sound",  # 语音识别Sound
        26: "Social and Information Networks",  # 社会信息网络Social and Information Networks
        27: "Robotics",  # 机器人学
        28: "Information Theory",  # 信息论
        29: "Performance",  # 性能分析Performance
        30: "Computation and Language",  # 计算语言学Computation and Language
        31: "Information Retrieval",  # 信息检索
        32: "Mathematical Software",  # 数学软件
        33: "Formal Languages and Automata Theory",  # 形式化方法Formal Languages and Automata Theory
        34: "Data Structures and Algorithms",  # 数据结构 Data Structures and Algorithms
        35: "Operating Systems",  # 操作系统
        36: "Computer Science and Game Theory",  # 博弈论
        37: "Databases",  # 数据库
        38: "Digital Libraries",  # 数字图书馆Digital Libraries
        39: "Discrete Mathematics"  # 离散数学
    }

    # 添加ytext并导出JSON
    updated_data = add_ytext_and_export_json(data, label_map)
    output_file_path = "processed_data.pt"  # 输出文件路径
    save_data_to_pt(updated_data, output_file_path)

    # 验证结果
    print(f"ytext长度: {len(updated_data.ytext)}")
    print(f"前5个ytext样本: {updated_data.ytext[:5]}")
    print("Document.json已生成，格式为[{id: 0, contents: '类别名称'}, ...]")
#下一步：生成document-txt