import logging
import os
import sys
import json
from datetime import datetime

import torch
from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import EvalNCCDataset, TrainNCCCollator
from OpenLP.dataset import TrainNCCGraphDataset,EvalNCCGraphDataset
from OpenLP.modeling import DenseModelforNCC
from OpenLP.trainer import LLMDenseTrainer as Trainer

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

from transformers import PreTrainedTokenizer, EvalPrediction
def calculate_ncc_metrics(evalpred: EvalPrediction):
    '''
    This function is for coarse-grained classificaion evaluation.
    '''
    device = torch.device("cpu")
    scores, labels = evalpred.scores.to(device), evalpred.target.to(device)
    preds = np.argmax(scores, 1)

    recall_macro = recall_score(labels, preds, average='macro')
    precision_macro = precision_score(labels, preds, average='macro')
    F1_macro = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    print("here is calu ncc")

    return {
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "F1_macro": F1_macro,
        "accuracy": accuracy,
    }


from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed, AdamW
from transformers.integrations import TensorBoardCallback
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm

LABEL_FILE_NAME = "label_name.txt"

logger = logging.getLogger(__name__)


class CheckpointTracker:
    """跟踪所有检查点及其性能指标的类"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoints = []
        self.metric_name = "accuracy"
        self.checkpoint_logs_dir = os.path.join(output_dir, "checkpoint_logs")
        os.makedirs(self.checkpoint_logs_dir, exist_ok=True)

    def add_checkpoint(self, checkpoint_name, metrics, step=None, epoch=None):
        checkpoint_info = {
            "checkpoint_name": checkpoint_name,
            "metrics": metrics,
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.checkpoints.append(checkpoint_info)

        log_file = os.path.join(self.checkpoint_logs_dir, f"{checkpoint_name}.json")
        with open(log_file, "w") as f:
            json.dump(checkpoint_info, f, indent=4)

    def save_all_checkpoints_summary(self):
        summary_file = os.path.join(self.output_dir, "checkpoint_summary.json")
        with open(summary_file, "w") as f:
            json.dump(self.checkpoints, f, indent=4)
        logger.info(f"Checkpoint summary saved to {summary_file}")

    def print_summary(self):
        if not self.checkpoints:
            logger.info("No checkpoints to summarize.")
            return

        logger.info("\n===== Checkpoint Performance Summary =====")

        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x["metrics"].get(self.metric_name, 0),
            reverse=True
        )

        header = f"Rank\tCheckpoint Name\tEpoch\tStep\t{self.metric_name}\tTimestamp"
        logger.info(header)
        logger.info("-" * len(header))

        for i, checkpoint in enumerate(sorted_checkpoints, 1):
            line = (f"{i}\t{checkpoint['checkpoint_name']}\t"
                    f"{checkpoint['epoch'] or 'N/A'}\t"
                    f"{checkpoint['step'] or 'N/A'}\t"
                    f"{checkpoint['metrics'].get(self.metric_name, 'N/A'):.4f}\t"
                    f"{checkpoint['timestamp']}")
            logger.info(line)

        best_checkpoint = sorted_checkpoints[0]
        logger.info("\nBest checkpoint:")
        logger.info(f"Name: {best_checkpoint['checkpoint_name']}")
        logger.info(f"{self.metric_name}: {best_checkpoint['metrics'][self.metric_name]:.4f}")
        logger.info(f"Epoch: {best_checkpoint['epoch'] or 'N/A'}")
        logger.info(f"Step: {best_checkpoint['step'] or 'N/A'}")
        logger.info("==========================================\n")


def manual_evaluate(model, eval_dataset, data_collator, training_args, device):
    """手动执行评估并返回指标"""
    logger.info("Starting manual evaluation...")

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    metrics={"accuracy":0.0}
    batchnum=0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batchnum+=1
            # 解包列表类型的batch
            inputs, labels = batch

            # 移动标签到设备
            #labels = labels.to(device)

            # 移动输入数据到设备
            for key in inputs:
                if isinstance(inputs[key], list):
                    for i in range(len(inputs[key])):
                        inputs[key][i] = inputs[key][i].to(device)
                else:
                    inputs[key] = inputs[key].to(device)

            # 模型前向传播
            outputs = model(inputs, labels)
            loss = calculate_ncc_metrics(outputs)
            metrics["accuracy"] = metrics["accuracy"]+loss["accuracy"]

    # 计算指标
    metrics["accuracy"] = metrics["accuracy"]/batchnum
    model.train()
    return metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
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

    # 初始化检查点跟踪器
    checkpoint_tracker = CheckpointTracker(training_args.output_dir)
    checkpoint_tracker.metric_name = "accuracy"

    # 加载模型、配置和分词器
    num_labels = training_args.labelnum
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModelforNCC.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    ).to(device)

    # 加载类别名称
    class_name_path = training_args.labeltxt
    #class_name_path = "D:/mymodel/8-5maindataset/cora/label.txt"
    logger.info("Read class name for task from %s" % (class_name_path))
    label_names = []
    with open(class_name_path) as f:
        for l in f:
            label_names.append(l.strip())

    # 初始化分类器权重
    with torch.no_grad():
        label_input = tokenizer(label_names, return_tensors="pt", padding=True)
        label_ipt_device = {k: v.to(device) for k, v in label_input.items()}

        G = [{'input_ids': label_ipt_device['input_ids'],
              'attention_mask': label_ipt_device['attention_mask'],
              'edge': torch.tensor([[], []]).long().to(device)}]

        out = model.lm(G=G).last_hidden_state[:, 0]
        model.classifier.linear.weight.copy_(out)
        model.classifier.linear.bias.copy_(torch.zeros((out.shape[0],)).to(out.device))

    # 固定BERT参数
    #if training_args.fix_bert:
    #    for param in model.lm.bert.parameters():
    #        param.requires_grad = False

    # 初始化数据集
    train_dataset = TrainNCCGraphDataset(
        tokenizer,
        data_args,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    eval_dataset = (
        EvalNCCGraphDataset(
            tokenizer,
            data_args,
            shuffle_seed=training_args.seed,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if data_args.eval_path is not None
        else None
    )

    # 数据收集器
    data_collator = TrainNCCCollator(
        tokenizer,
        max_len=data_args.max_len,
    )

    # 初始化训练器
    trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[TensorBoardCallback()],
        compute_metrics=calculate_ncc_metrics,
    )
    train_dataset.trainer = trainer

    # 准备数据加载器和优化器
    train_dataloader = trainer.get_train_dataloader()
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

    # 手动训练循环
    global_step = 0
    best_metric = None
    model.train()

    for epoch in range(int(training_args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(epoch_iterator):
            # 解包列表类型的batch
            inputs, labels = batch

            # 移动标签到设备
            #labels = labels.to(device)

            # 移动输入数据到设备
            for key in inputs:
                if isinstance(inputs[key], list):
                    for i in range(len(inputs[key])):
                        inputs[key][i] = inputs[key][i].to(device)
                else:
                    inputs[key] = inputs[key].to(device)

            # 前向传播
            outputs = model(inputs,labels)
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

            # 每n步评估并保存检查点
            if training_args.eval_steps is not None and global_step % training_args.eval_steps == 0:
                if eval_dataset is not None and trainer.is_world_process_zero():
                    metrics = manual_evaluate(model, eval_dataset, data_collator, training_args, device)
                    logger.info(f"Step {global_step} metrics: {metrics}")

                    # 保存检查点
                    checkpoint_name = f"checkpoint_step_{global_step}"
                    checkpoint_path = os.path.join(training_args.output_dir, checkpoint_name)
                    trainer.save_model(checkpoint_path)

                    # 跟踪检查点性能
                    checkpoint_tracker.add_checkpoint(
                        checkpoint_name=checkpoint_name,
                        metrics=metrics,
                        step=global_step,
                        epoch=epoch
                    )

                    # 更新最佳模型
                    if best_metric is None or metrics[checkpoint_tracker.metric_name] > best_metric:
                        best_metric = metrics[checkpoint_tracker.metric_name]
                        #best_model_path = os.path.join(training_args.output_dir, "best_model")
                        trainer.save_model(training_args.output_dir)
                        logger.info(f"Saved best model with {checkpoint_tracker.metric_name}: {best_metric}")

        # 每轮结束评估
        if training_args.eval_strategy == "epoch":
            if eval_dataset is not None and trainer.is_world_process_zero():
                metrics = manual_evaluate(model, eval_dataset, data_collator, training_args, device)
                logger.info(f"Epoch {epoch + 1} metrics: {metrics}")

                # 保存检查点
                checkpoint_name = f"checkpoint_epoch_{epoch + 1}"
                checkpoint_path = os.path.join(training_args.output_dir, checkpoint_name)
                trainer.save_model(checkpoint_path)

                # 跟踪检查点性能
                checkpoint_tracker.add_checkpoint(
                    checkpoint_name=checkpoint_name,
                    metrics=metrics,
                    epoch=epoch + 1
                )

                # 更新最佳模型
                if best_metric is None or metrics[checkpoint_tracker.metric_name] > best_metric:
                    best_metric = metrics[checkpoint_tracker.metric_name]
                    best_model_path = os.path.join(training_args.output_dir, "best_model")
                    trainer.save_model(training_args.output_dir)
                    logger.info(f"Saved best model with {checkpoint_tracker.metric_name}: {best_metric}")

    # 保存最终模型
    final_checkpoint_name = "final_checkpoint"
    final_checkpoint_path = os.path.join(training_args.output_dir, final_checkpoint_name)
    #trainer.save_model(final_checkpoint_path)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

        # 最终评估
        if eval_dataset is not None:
            final_metrics = manual_evaluate(model, eval_dataset, data_collator, training_args, device)
            logger.info(f"Final evaluation metrics: {final_metrics}")

            # 跟踪最终检查点
            checkpoint_tracker.add_checkpoint(
                checkpoint_name=final_checkpoint_name,
                metrics=final_metrics,
                epoch=int(training_args.num_train_epochs)
            )

            # 保存最终指标
            with open(os.path.join(training_args.output_dir, "final_metrics.json"), "w") as f:
                json.dump(final_metrics, f, indent=4)

        # 生成并打印检查点汇总报告
        checkpoint_tracker.save_all_checkpoints_summary()
        checkpoint_tracker.print_summary()


if __name__ == "__main__":
    main()
