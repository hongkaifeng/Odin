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

import logging
import os
import sys

from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import TrainRerankCollator, EvalRerankDataset
from OpenLP.modeling import DenseRerankModel
from OpenLP.trainer import DenseTrainer as Trainer
from OpenLP.trainer import GCDenseTrainer
from OpenLP.utils import calculate_rerank_metrics
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback
from transformers.tokenization_utils_base import BatchEncoding

def mrr_rerank_score(y_true, y_score):
    y_score = y_score.cpu().numpy()
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.max(rr_score)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def dcg_score(y_true, y_score, k=10):
    #y_score = y_score.numpy()
    #print(y_score)
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def calculate_rerank_metrics(evalpred: EvalPrediction):
    '''
    This function is for reranking evaluation.
    '''

    scores, mask_labels = evalpred.scores.cpu(), evalpred.target.cpu()
    print(mask_labels)
    pos_num, neg_num = mask_labels[0][-2], mask_labels[0][-1]
    mask_labels = mask_labels[:, :-2]
    labels = np.array([1] * pos_num + [0] * neg_num)
    print(pos_num)
    print(neg_num)
    prc_1_all = []
    mrr_all = []
    ndcg_5_all = []
    ndcg_10_all = []
    for score, mask_label in zip(scores, mask_labels):
        valid_score = score[mask_label == 1]
        valid_label = labels[mask_label == 1]
        print(valid_score)
        prc_1_all.append(valid_label[np.argmax(valid_score)])
        mrr_all.append(mrr_rerank_score(valid_label, valid_score))
        ndcg_5_all.append(ndcg_score(valid_label, valid_score, 5))
        ndcg_10_all.append(ndcg_score(valid_label, valid_score, 10))

    prc = np.mean(prc_1_all)
    mrr = np.mean(mrr_all)
    ndcg_5 = np.mean(ndcg_5_all)
    ndcg_10 = np.mean(ndcg_10_all)

    return {
        "prc": prc,
        "mrr": mrr,
        "ndcg_5": ndcg_5,
        "ndcg_10": ndcg_10,
    }




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

    num_labels = 15
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
    model = DenseRerankModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    test_dataset = EvalRerankDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)

    trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        eval_dataset=test_dataset,
        model=model,
        args=training_args,
        data_collator=TrainRerankCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        compute_metrics=calculate_rerank_metrics,
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
            #3print("batch")
            #print(len(batch))
            inputsQ, inputsK = batch  # 假设batch包含输入数据和标签
            #print("inputs   K")
            #print(inputsK)
            for key in inputsQ:
                value = inputsQ[key]
                # 如果是列表或张量（可迭代）
                if isinstance(value, (list, torch.Tensor)):
                    for j in range(len(value)):
                        item = value[j]
                        if isinstance(item, BatchEncoding):
                            value[j] = item.to(device)
                        elif isinstance(item, torch.Tensor):
                            value[j] = item.to(device)
                        # 对于其他类型（如int），不做处理或根据需要转换
                # 如果是单个值
                else:
                    if isinstance(value, BatchEncoding):
                        inputsQ[key] = value.to(device)
                    elif isinstance(value, torch.Tensor):
                        inputsQ[key] = value.to(device)

            # 前向传播
            outputs = model(inputsQ, inputsK)
            #print("outputs")
            #print(outputs)
            # print(len(outputs.q_reps))
            # 计算损失（如果需要）
            #try:
            if 1:
                allbatch += 1
                loss = calculate_rerank_metrics(outputs)  # 替换为你的损失函数
                print(loss)
                totalaccuracy += loss['prc']  # 累加损失，考虑批次大小
                totalF1_macro += loss['mrr']  # 累加损失，考虑批次大小

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
    write_content = f"排序任务 totalprc: {totalaccuracy / allbatch}, totalmrr: {totalF1_macro / allbatch}, allbatch: {allbatch}\n\n\n"
    # 以追加模式（'a'）打开文件，新内容自动写入下一行，编码指定为utf-8避免乱码
    with open(result_file_path, 'a', encoding='utf-8') as f:
        f.write(write_content)
if __name__ == "__main__":
    main()
