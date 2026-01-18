import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers.file_utils import is_datasets_available
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    nested_detach,
)

from ..loss import DistributedContrastiveLoss, SimpleContrastiveLoss

from IPython import embed

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class DenseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DenseTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=False,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # def compute_loss(self):
        # import error
        #print("computeloss")
        G1, G2 = inputs
        # query, keys = inputs
        outputs = model(G1=G1, G2=G2)
        # print("compute_loss")
        # print(model)
        # print(query)
        # print(keys)
        # outputs = model(query=query, keys=keys)
        # print("loss")
        # print(outputs)
        # print("outputloss")
        # print(outputs.loss)
        # return 0
        return (outputs.loss, outputs) if return_outputs else outputs.loss
        query, keys = inputs
        outputs = model(query=query, keys=keys)


        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, *args):
        return super(DenseTrainer, self).training_step(*args) / self._dist_loss_scale_factor

class LLMDenseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(LLMDenseTrainer, self).__init__(*args, **kwargs)

        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        #print("inputs")
        #print(inputs)
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        #print("train_dataset")
        #print(train_dataset)
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=False,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
    #def compute_loss(self):
        #import error
        #print(inputs)
        #print("computeloss")
        G1,G2=inputs
        #query, keys = inputs
        outputs=model(G1=G1,G2=G2)
        #print("compute_loss")
        #print(model)
        #print(query)
        #print(keys)
        #outputs = model(query=query, keys=keys)
        #print("loss")
        #print(outputs)
        #print("outputloss")
        #print(outputs.loss)
        #return 0
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, *args):
        return super(LLMDenseTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def create_optimizer(self):
        """创建优化器，为GraphSAGE参数设置更高的学习率"""
        # 获取基础学习率（来自TrainingArguments）
        base_lr = self.args.learning_rate
        # GraphSAGE学习率 = 基础学习率 * 倍数
        #graphsage_lr = base_lr * 50
        graphsage_lr = base_lr * 100
        optimizer_grouped_parameters = []
        print("这里是自定义优化器")
        # 1. 收集BERT参数（非GraphSAGE部分）
        bert_params = []
        # 2. 收集GraphSAGE参数
        graphsage_params = []

        # 遍历所有参数，按名称分组
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "sagelayer" in name:
                    graphsage_params.append(param)
                else:
                    bert_params.append(param)
        #print("sage参数")
        #print(graphsage_params)
        #print("bert参数")
        #print(bert_params)
        # 添加BERT参数组（使用基础学习率）
        optimizer_grouped_parameters.append({
            "params": bert_params,
            "lr": base_lr,
            "weight_decay": self.args.weight_decay  # 保持原有权重衰减
        })

        # 添加GraphSAGE参数组（使用10倍学习率）
        optimizer_grouped_parameters.append({
            "params": graphsage_params,
            "lr": graphsage_lr,
            "weight_decay": self.args.weight_decay  # 保持原有权重衰减
        })
        if hasattr(self.args, 'adam_betas'):
            betas = self.args.adam_betas
        else:
            betas = (self.args.adam_beta1, self.args.adam_beta2)
        # 创建优化器
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=base_lr,  # 初始学习率会被参数组覆盖
            eps=self.args.adam_epsilon,
            betas=betas
        )

        return self.optimizer

def split_dense_inputs(model_input: dict, chunk_size: int):

    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    assert set(keys) == set(['center_input', 'neighbor_input', 'mask'])
    chunked_center_input_tensors = [arg_val['center_input'][k].split(chunk_size, dim=0) for k in arg_val['center_input']]
    chunked_neighbor_input_tensors = [arg_val['neighbor_input'][k].split(chunk_size * arg_val['mask'].shape[1], dim=0) for k in arg_val['neighbor_input']]
    chunked_mask_tensors = arg_val['mask'].split(chunk_size, dim=0)

    chunked_center_input_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(list(arg_val['center_input'].keys())), zip(*chunked_center_input_tensors))]
    chunked_neighbor_input_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(list(arg_val['neighbor_input'].keys())), zip(*chunked_neighbor_input_tensors))]
    
    return [{arg_key: {'center_input': c, 'neighbor_input': n, 'mask': m}} for c, n, m in zip(chunked_center_input_arg_val, chunked_neighbor_input_arg_val, chunked_mask_tensors)]


def get_dense_rep(x):

    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCDenseTrainer(DenseTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCDenseTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler
        )
        

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        query, keys = self._prepare_inputs(inputs)
        query, keys = {'query': query}, {'keys': keys}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(query, keys, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
