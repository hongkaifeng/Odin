import csv
import json
from dataclasses import dataclass
from typing import Dict

import datasets
from datasets import load_dataset
from transformers import PreTrainedTokenizer, EvalPrediction
import logging
import unicodedata
import regex
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score


def calculate_ncc_metrics(evalpred: EvalPrediction):
    '''
    This function is for coarse-grained classificaion evaluation.
    '''
    device = torch.device("cpu")
    scores, labels = evalpred.scores.to(device), evalpred.target.to(device)
    preds = np.argmax(scores, 1)

    recall_macro = recall_score(labels, preds, average='macro',zero_division=0)
    precision_macro = precision_score(labels, preds, average='macro',zero_division=0)
    F1_macro = f1_score(labels, preds, average='macro',zero_division=0)
    accuracy = accuracy_score(labels, preds)
    print("here is calu ncc")

    return {
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "F1_macro": F1_macro,
        "accuracy": accuracy,
    }


import logging
import os
import sys

from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import TestNCCGraphDataset
from OpenLP.dataset import TrainNCCCollator
from OpenLP.modeling import DenseModelforNCC
from OpenLP.trainer import DenseTrainer as Trainer
#from OpenLP.trainer import GCDenseTrainer
#from OpenLP.utils import calculate_ncc_metrics
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback

LABEL_FILE_NAME = "label_name.txt"

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    # ++++++++++++++++
    data_args.class_num = training_args.labelnum
    # ++++++++++++++++
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = training_args.labelnum
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModelforNCC.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    test_dataset = TestNCCGraphDataset(tokenizer, data_args, shuffle_seed=training_args.seed,
                                  cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    trainer_cls = Trainer
    trainer = trainer_cls(
        eval_dataset=test_dataset,
        model=model,
        args=training_args,
        data_collator=TrainNCCCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        compute_metrics=calculate_ncc_metrics,
    )
    eval_dataloader = trainer.get_eval_dataloader()

    # 加载模型并设置到评估模式
    print("eval_dataloader")
    print(eval_dataloader)
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用GPU
    else:
        device = torch.device("cpu")  # 使用CPU
    model.to(device)  # 将模型移动到正确的设备（例如'cuda'或'cpu'）

    # 初始化评估指标
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # 禁用梯度计算，因为评估阶段不需要进行反向传播
    totalaccuracy = 0.0000
    totalF1_macro = 0.0000
    allbatch = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            # print("batch")
            # print(batch)
            inputsQ, inputsK = batch  # 假设batch包含输入数据和标签
            #for key in inputsQ:
            #    print(inputsQ[key])
            #    inputsQ[key] = inputsQ[key].to(device)
            #inputsK=inputsK.to(device)
            # inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputsQ, inputsK)
            # print("outputs")
            # print(len(outputs.q_reps))
            # 计算损失（如果需要）
            #try:
            if 1:
                allbatch += 1
                loss = calculate_ncc_metrics(outputs)  # 替换为你的损失函数
                print(loss)
                totalaccuracy += loss['accuracy']  # 累加损失，考虑批次大小
                totalF1_macro += loss['F1_macro']  # 累加损失，考虑批次大小
                print("totalaccuracy", totalaccuracy)
                print("totalF1_macro", totalF1_macro)
            #except:
            #    print("error")

            # 计算评估指标（例如准确率）

    # 计算平均损失和最终评估指标
    print("totalaccuracy", totalaccuracy)
    print("totalF1_macro", totalF1_macro)

    print('totalaccuracy:', totalaccuracy / allbatch, 'totalmrr:', totalF1_macro / allbatch)
    print("allbatch", allbatch)
    result_file_path = "ending.txt"
    # 构造要写入的内容（包含累加值、平均值和批次数量，便于后续查看）
    write_content = f"分类任务 totalprc: {totalaccuracy / allbatch}, totalmrr: {totalF1_macro / allbatch}, allbatch: {allbatch}\n"
    # 以追加模式（'a'）打开文件，新内容自动写入下一行，编码指定为utf-8避免乱码
    with open(result_file_path, 'a', encoding='utf-8') as f:
        f.write(write_content)
if __name__ == "__main__":
    main()
