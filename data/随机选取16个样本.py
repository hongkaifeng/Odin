import json
import random
import argparse
from pathlib import Path


def select_random_samples(input_file, output_file, sample_size=16, seed=None):
    """从JSONL文件中随机选择指定数量的样本"""
    # 设置随机种子以确保结果可复现
    if seed is not None:
        random.seed(seed)

    # 读取所有样本
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    # 检查样本数量是否足够
    if len(samples) < sample_size:
        raise ValueError(f"样本数量不足：文件中只有{len(samples)}个样本，但需要{sample_size}个样本。")

    # 随机选择样本
    selected_samples = random.sample(samples, sample_size)

    # 保存到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"已从{input_file}中随机选择{sample_size}个样本并保存到{output_file}")
    return selected_samples


def main():


    input="D:/mymodel/truedataset/final/retrieve/train.text.jsonl"
    output="D:/mymodel/truedataset/final/retrieve/train16.text.jsonl"
    # 确保输出目录存在
    #output_path = Path(args.output)
    #output_path.parent.mkdir(parents=True, exist_ok=True)

    select_random_samples(input, output, 16, 42)


if __name__ == "__main__":
    main()