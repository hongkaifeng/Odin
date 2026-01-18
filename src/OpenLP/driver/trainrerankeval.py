import logging
import os
import sys
from tqdm import tqdm
from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import TrainHnDataset, EvalHnDataset,TrainHnCollator
from OpenLP.modeling import DenseModel #,DenseRerankModel
from OpenLP.trainer import DenseTrainer as Trainer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EvalPrediction, AdamW
import torch
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)
import json
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


# 定义适配结构，确保calculate_metrics能获取到所需属性
class MetricInput:
    def __init__(self, scores, target):
        self.scores = scores  # 预测分数
        self.target = target  # 标签


def manual_validate(model, eval_dataloader, device):
    """根据已知的DenseOutput结构提取scores和target"""
    logger.info("Starting manual validation...")

    model.eval()
    model.to(device)

    # 初始化指标累加器
    total_prc = 0.0
    total_mrr = 0.0
    total_ndcg10 = 0.0
    total_loss = 0.0
    all_batch = 0  # 初始化批次计数器

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Validating"):
            inputsQ, inputsK = batch
            all_batch += 1  # 累加批次计数

            # 处理inputsQ
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
                    # 对于int等不需要移动到设备的类型，不做处理
                    # elif isinstance(value, int):
                    #     pass  # 整数不需要移动到设备

            # 处理inputsK（与inputsQ逻辑相同）
            for key in inputsK:
                value = inputsK[key]
                if isinstance(value, (list, torch.Tensor)):
                    for j in range(len(value)):
                        item = value[j]
                        if isinstance(item, BatchEncoding):
                            value[j] = item.to(device)
                        elif isinstance(item, torch.Tensor):
                            value[j] = item.to(device)
                else:
                    if isinstance(value, BatchEncoding):
                        inputsK[key] = value.to(device)
                    elif isinstance(value, torch.Tensor):
                        inputsK[key] = value.to(device)

            # 模型前向传播
            outputs = model(inputsQ, inputsK)
            metrics = calculate_metrics(outputs)
            total_prc += metrics["prc"]
            total_mrr += metrics["mrr"]
            total_ndcg10 += metrics["ndcg_10"]

    # 计算平均指标
    if all_batch == 0:
        logger.warning("No valid batches processed during validation")
        return {}

    avg_metrics = {
        "prc": total_prc / all_batch,
        "mrr": total_mrr / all_batch,
        "ndcg_10": total_ndcg10 / all_batch
    }

    if total_loss > 0:
        avg_metrics["eval_loss"] = total_loss / all_batch

    return avg_metrics



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # 日志配置
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
    device = training_args.device

    # 加载模型、配置和分词器
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
    ).to(device)

    # 固定BERT参数
    if training_args.fix_bert:
        for param in model.lm.bert.parameters():
            param.requires_grad = False

    # 初始化数据集
    train_dataset = TrainHnDataset(
        tokenizer,
        data_args,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    )
    eval_dataset = EvalHnDataset(
        tokenizer,
        data_args,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    ) if data_args.eval_path is not None else None

    # 初始化数据收集器
    data_collator = TrainHnCollator(
        tokenizer,
        max_len=data_args.max_len,
    )

    # 初始化训练器
    trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
    )

    # 关键修复：初始化trainer的epoch状态，避免为None
    if trainer.state.epoch is None:
        trainer.state.epoch = 0.0

    train_dataset.trainer = trainer

    # 准备数据加载器
    train_dataloader = trainer.get_train_dataloader()

    evaldata_collator = TrainHnCollator(
        tokenizer,
        max_len=data_args.max_len,
    )
    eval_dataloader = None
    if eval_dataset:
        trainer_eval = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=eval_dataset,
            data_collator=evaldata_collator,
            compute_metrics=calculate_metrics,
        )
        # 同样初始化评估训练器的epoch
        if trainer_eval.state.epoch is None:
            trainer_eval.state.epoch = 0.0
        eval_dataloader = trainer_eval.get_train_dataloader()

    # 手动初始化优化器和调度器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        eps=training_args.adam_epsilon
    )
    total_train_steps = len(train_dataloader) * training_args.num_train_epochs
    warmup_steps = int(total_train_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps
    )

    # 训练开始前添加初始验证
    global_step = 0
    best_metric = None
    alleval = []

    # 初始验证并保存检查点
    if eval_dataloader is not None and trainer.is_world_process_zero():
        logger.info("Starting initial validation before training...")
        initial_metrics = manual_validate(model, eval_dataloader, device)
        alleval.append(initial_metrics)
        logger.info(f"Initial validation metrics: {initial_metrics}")

        # 保存初始检查点
        initial_checkpoint_dir = os.path.join(training_args.output_dir, "initial_checkpoint")
        trainer.save_model(initial_checkpoint_dir)
        logger.info(f"Saved initial checkpoint to {initial_checkpoint_dir}")

        # 更新最佳指标
        if "mrr" in initial_metrics:
            best_metric = initial_metrics["mrr"]
            # 同时保存为最佳模型
            best_model_dir = os.path.join(training_args.output_dir, "best_model")
            trainer.save_model(training_args.output_dir)
            logger.info(f"Initial model set as best model with mrr: {best_metric}")

    # 手动训练循环
    model.train()
    for epoch in range(int(training_args.num_train_epochs)):
        # 更新训练器的epoch状态
        trainer.state.epoch = epoch

        epoch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(epoch_iterator):
            inputsQ, inputsK = batch

            # 移动数据到设备
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
                    # 对于int等不需要移动到设备的类型，不做处理
                    # elif isinstance(value, int):
                    #     pass  # 整数不需要移动到设备

            # 处理inputsK（与inputsQ逻辑相同）
            for key in inputsK:
                value = inputsK[key]
                if isinstance(value, (list, torch.Tensor)):
                    for j in range(len(value)):
                        item = value[j]
                        if isinstance(item, BatchEncoding):
                            value[j] = item.to(device)
                        elif isinstance(item, torch.Tensor):
                            value[j] = item.to(device)
                else:
                    if isinstance(value, BatchEncoding):
                        inputsK[key] = value.to(device)
                    elif isinstance(value, torch.Tensor):
                        inputsK[key] = value.to(device)

            # 添加return_loss参数
            if "return_loss" not in inputsQ:
                inputsQ["return_loss"] = True
            if "return_loss" not in inputsK:
                inputsK["return_loss"] = True

            # 前向传播
            outputs = model(inputsQ, inputsK)
            loss = outputs.loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if training_args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

            # 参数更新
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 日志输出
            global_step += 1
            epoch_iterator.set_postfix(loss=loss.item(), step=global_step)

            # 每n步验证
            if training_args.eval_steps is not None and global_step % training_args.eval_steps == 0:
                if eval_dataloader is not None and trainer.is_world_process_zero():
                    metrics = manual_validate(model, eval_dataloader, device)
                    alleval.append(metrics)
                    logger.info(f"Step {global_step} metrics: {metrics}")

                    if training_args.save_strategy == "steps" and "mrr" in metrics:
                        if best_metric is None or metrics["mrr"] > best_metric:
                            best_metric = metrics["mrr"]
                            trainer.save_model(training_args.output_dir)
                            logger.info(f"Saved best model with mrr: {best_metric}")

        # 每轮结束验证
        if training_args.eval_strategy == "epoch":
            if eval_dataloader is not None and trainer.is_world_process_zero():
                metrics = manual_validate(model, eval_dataloader, device)
                alleval.append(metrics)
                logger.info(f"Epoch {epoch + 1} metrics: {metrics}")

                if training_args.save_strategy == "epoch" and "mrr" in metrics:
                    if best_metric is None or metrics["mrr"] > best_metric:
                        best_metric = metrics["mrr"]
                        trainer.save_model(training_args.output_dir)
                        logger.info(f"Saved best model with mrr: {best_metric}")

    # 保存最终模型
    #trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

        if eval_dataloader is not None:
            final_metrics = manual_validate(model, eval_dataloader, device)
            logger.info(f"Final validation metrics: {final_metrics}")
            alleval.append(final_metrics)

            with open(os.path.join(training_args.output_dir, "final_metrics.json"), "w") as f:
                json.dump(final_metrics, f, indent=4)

    # 输出所有评估结果
    logger.info("All evaluation metrics during training:")
    for i, metrics in enumerate(alleval):
        logger.info(f"Evaluation {i + 1}: {metrics}")


if __name__ == "__main__":
    main()
