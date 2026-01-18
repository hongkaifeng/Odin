def generate_document_txt(label_map, output_path="documents.txt"):
    """
    根据标签映射生成document.txt，格式为「编号\t标签内容」（匹配文档要求）

    参数:
        label_map: 标签ID到标签文本的映射字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 按标签ID升序写入（确保顺序与映射一致）
        for label_id in sorted(label_map.keys()):
            label_text = label_map[label_id]
            # 每行格式：编号\t标签内容（文档图1示例格式）
            f.write(f"{label_id}\t{label_text}\n")
    print(f"document.txt已生成，保存至：{output_path}")


if __name__ == "__main__":
    # 40类标签映射（使用全名形式）
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

    # 生成document.txt（文档中检索/排序任务的必要文件）
    generate_document_txt(label_map)
#下一步生成nodetext和nodeclassion