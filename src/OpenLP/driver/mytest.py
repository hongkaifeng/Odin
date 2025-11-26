import torch
import logging
import os
import sys

from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
#from OpenLB.dataset import TrainCollator, EvalDataset #, TrainDataset
from OpenLP.dataset import TrainCollator, TestGraphDataset
from OpenLP.modeling import DenseModel
from OpenLP.trainer import DenseTrainer as Trainer
#from OpenLB.trainer import GCDenseTrainer
from OpenLP.utils import calculate_metrics
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from torch.utils.data import DataLoader
from OpenLP.trainer import DenseTrainer as Trainer
#from OpenLB.trainer import GCDenseTrainer
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

#============
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

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score


logger = logging.getLogger()

from IPython import embed

# metrics
import numpy as np
def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    #print("y_score")
    #print(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.sum(rr_score) / np.sum(y_true)

def calculate_metrics(evalpred: EvalPrediction):
    '''
    This function is for link prediction in batch evaluation.
    '''

    #scores, labels = evalpred.predictions[-2], evalpred.predictions[-1]
    scores, labels = evalpred.scores.cpu().numpy(), evalpred.target.cpu()
    predictions = np.argmax(scores, -1)
    print("predictions")
    print(scores)
    print(predictions)
    #equals = predictions == labels
    #print(equals)
    #equals_float = equals.float()
    labels=labels.numpy()
    prc = (np.sum((predictions == labels)) / labels.shape[0])
    #sum_equals = equals_float.sum().item()  # 使用.item()将单个值的Tensor转换为Python标量

    # 计算总元素数
    #total_elements = labels.numel()  # 或者使用labels.shape[0] * labels.shape[1]（如果labels是二维的）
    # 注意：如果你的labels是多维的，你需要确保numel()或shape的使用方式是正确的

    # 计算准确率
    #prc = sum_equals / total_elements

    n_labels = np.max(labels) + (labels[1] - labels[0])
    labels = np.eye(n_labels)[labels]
    #print("label")
    #print(labels)
    # auc_all = [roc_auc_score(labels[i], scores[i]) for i in tqdm(range(labels.shape[0]))]
    # auc = np.mean(auc_all)
    mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
    mrr = np.mean(mrr_all)
    ndcg_10_all = [ndcg_score(labels[i], scores[i], 10) for i in range(labels.shape[0])]
    ndcg_10 = np.mean(ndcg_10_all)
    ndcg_100_all = [ndcg_score(labels[i], scores[i], 100) for i in range(labels.shape[0])]
    ndcg_100 = np.mean(ndcg_100_all)
    return {
        "prc": prc,
        "mrr": mrr,
        "ndcg_10": ndcg_10,
        "ndcg_100": ndcg_100,
    }
#============

if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
else:
    device = torch.device("cpu")  # 使用CPU

logger = logging.getLogger(__name__)
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    print("2222222222222222222")
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    print("11111111111111111111111111")
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


set_seed(training_args.seed)

training_args.dataloader_pin_memory=False #==============================

num_labels = 1
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
model = DenseModel.build(
    model_args,
    data_args,
    training_args,
    config=config,
    cache_dir=model_args.cache_dir,
)

test_dataset = TestGraphDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)


# 假设你已经有了一个加载了评估数据的DataLoader
trainer_cls = Trainer
trainer = trainer_cls(
        train_dataset=test_dataset,
        model=model,
        args=training_args,
        data_collator=TrainCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        compute_metrics=calculate_metrics,
    )

#eval_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
eval_dataloader=trainer.get_train_dataloader()

# 加载模型并设置到评估模式

model.eval()
model.to(device)  # 将模型移动到正确的设备（例如'cuda'或'cpu'）

# 初始化评估指标
total_loss = 0.0
correct_predictions = 0
total_predictions = 0

# 禁用梯度计算，因为评估阶段不需要进行反向传播
totalprc=0.0000
totalmrr=0.0000
allbatch=0
#print("eval_dataloader")
#print(eval_dataloader)
with torch.no_grad():
    for batch in eval_dataloader:
        inputsQ, inputsK = batch  # 假设batch包含输入数据和标签
        #print("inputsQ")
        #print(inputsQ)
        #for i in inputsQ['xxx1']:
        #    print(i["edge"])
        #print("inputsK")
        #for i in inputsK['xxx2']:
        #    print(i["edge"])
        for key in inputsQ:
            for j in range(len(inputsQ[key])):
                inputsQ[key][j]=inputsQ[key][j].to(device)
            #inputsQ[key] = inputsQ[key].to(device)
        for key in inputsK:
            for j in range(len(inputsK[key])):
                inputsK[key][j]=inputsK[key][j].to(device)
            #inputsK[key] = inputsK[key].to(device)
        #inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputsQ,inputsK)
        #print(outputs)
        #print(len(outputs))
        #import error

        # 计算损失（如果需要）
        try:
            allbatch += 1
            loss = calculate_metrics(outputs)  # 替换为你的损失函数
            print(loss)
            totalprc += loss['prc']  # 累加损失，考虑批次大小
            totalmrr += loss['mrr']  # 累加损失，考虑批次大小
            print("totalprc",totalprc)
            print("totalmrr",totalmrr)
        except:
            print("1 error")

        # 计算评估指标（例如准确率）



# 计算平均损失和最终评估指标
print("totalprc",totalprc)
print("totalmrr",totalmrr)

print('totalprc:',totalprc/allbatch,'totalmrr:',totalmrr/allbatch)
print(allbatch)
result_file_path = "ending.txt"
# 构造要写入的内容（包含累加值、平均值和批次数量，便于后续查看）
write_content = f"链接预测 totalprc: {totalprc}, totalmrr: {totalmrr}, allbatch: {allbatch}, average_prc: {totalprc/allbatch:.6f}, average_mrr: {totalmrr/allbatch:.6f}\n"
# 以追加模式（'a'）打开文件，新内容自动写入下一行，编码指定为utf-8避免乱码
with open(result_file_path, 'a', encoding='utf-8') as f:
    f.write(write_content)
