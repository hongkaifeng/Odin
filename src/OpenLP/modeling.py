import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, BatchEncoding, PreTrainedModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from typing import Optional, Dict, List

from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

#from OpenLB.models import AutoModels
from OpenLP.models import AutoModels

from IPython import embed

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None
    target: Tensor = None


class LinearClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768
    ):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, h: Tensor = None):
        if h is not None:
            return self.linear(h)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _classifier_path = os.path.join(ckpt_dir, 'classifier.pt')
            if os.path.exists(_classifier_path):
                logger.info(f'Loading Classifier from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'classifier.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Classifier from scratch")
        return

    def save_classifier(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'classifier.pt'))
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(self._config, f)


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, p: Tensor = None):
        if p is not None:
            return self.linear(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class DenseModel(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            G1,
            G2,
    ):

        G1_hidden, G1_reps = self.encode(G1["xxx1"])

        G2_hidden, G2_reps = self.encode(G2["xxx2"])


        mainnodeQ = []
        mainnodeK = []

        bias = 0
        for i in G1["xxx1"]:
            bias += len(i['input_ids'])

            mainnodeQ.append(bias)

        bias = 0
        for i in G2["xxx2"]:
            bias += len(i['input_ids'])

            mainnodeK.append(bias)

        if G1_reps is None:  # or G2_reps is None:
            return DenseOutput(
                q_reps=G1_reps.contiguous(),
                # p_reps=G2_reps.contiguous()
            )

        if self.train_args.negatives_x_device:
            G1_reps = self.dist_gather_tensor(G1_reps)
            G2_reps = self.dist_gather_tensor(G2_reps)

        scores = torch.matmul(G1_reps, G2_reps.transpose(0, 1))

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        if scores.shape[0] != scores.shape[1]:
            # otherwise in batch neg only
            target = target * (1 + self.data_args.hn_num)
        # print(scores)
        # print(target)
        loss = self.cross_entropy(scores, target)
        #print("loss1", loss)



        return DenseOutput(
            loss=loss,
            scores=scores,
            target=target,
            q_reps=G1_reps.contiguous(),
            p_reps=G2_reps.contiguous()
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        # psg = BatchEncoding(psg)
        # print("psg")
        # print(psg)
        # return error
        psg = {"G": psg}
        # print(psg['center_input']['input_ids'].shape)
        # print(psg)
        psg_out = self.lm(**psg)
        # print('psg_out')
        # print(psg_out.last_hidden_state.shape)

        if 1:
            # BERT, Graphformers
            # print("BERT, Graphformers")
            p_hidden = psg_out.last_hidden_state
            if self.pooler is not None:
                p_reps = self.pooler(p=p_hidden[:, 0])  # D * d
            else:
                p_reps = p_hidden[:, 0]
        else:
            # BERT+SAGE
            # print("BERT+SAGE")
            p_reps = psg_out
            p_hidden = None

        return p_hidden, p_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        # print(9999999999999999999999999999999999999)
        # print(model_args.model_name_or_path)
        hf_kwargs.update({"attn_implementation": "eager"})
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        # add neighbor_mask_ratio
        if model_args.model_type in ['graphformer', 'SGformer', 'GCLSformer']:
            lm.bert.neighbor_mask_ratio = model_args.neighbor_mask_ratio

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(DenseModel):
    POOLER_CLS = LinearPooler

    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm = lm
        self.pooler = pooler
        self.model_args = model_args

    @torch.no_grad()
    def encode(self, psg):
        #print(type(psg))
        #print(len(psg))
        #print(psg)
        #return error
        #psg={'input_ids':psg['center_input']['input_ids'],'attention_mask':psg['center_input']['attention_mask'],'edge':torch.tensor([[],[]]).long()}

        for i in psg:
            print(i)
        return super(DenseModelForInference, self).encode(psg['x1'])
    
    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        print(query)
        print(passage)
        if query ==None:
            q_reps=None
        else:
            q_hidden, q_reps = self.encode(query)
        if passage ==None:
            p_reps=None
        else:
            p_hidden, p_reps = self.encode(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path
        hf_kwargs.update({"attn_implementation": "eager"})
        # load local
        logger.info(f'try loading tied weight')
        logger.info(f'loading model weight from {model_name_or_path}')
        # lm = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        # add neighbor_mask_ratio
        if model_args.model_type in ['graphformer', 'SGformer', 'GCLSformer']:
            lm.bert.neighbor_mask_ratio = 0

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if not model_args.add_pooler:
            pooler = None
        elif os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER_CLS(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            model_args=model_args
        )
        return model


class DenseRerankModel(DenseModel):
    '''
    This class is only for reranking test phase.
    '''
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__(lm, pooler, model_args, data_args, train_args)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            keys: Dict[str, Tensor] = None,
    ):
        #if keys["eval"] == 1:
        if 1:
            # print("===========================")
            # print(query)
            # print("============================")
            # print(keys)
            # G1_hidden, G1_reps = self.encode(G1["xxx1"])
            # G2_hidden, G2_reps = self.encode(G2["xxx2"])
            q_hidden, q_reps = self.encode(query["xxx1"])
            p_hidden, p_reps = self.encode(keys["xxx2"])

            if q_reps is None or p_reps is None:
                return DenseOutput(
                    q_reps=q_reps,
                    p_reps=p_reps
                )

            if self.train_args.negatives_x_device:
                q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)
            bias = 0
            mainnodeQ = []
            for i in query["xxx1"]:
                # print(i["edge"])
                mainnodeQ.append(bias)
                bias += len(i['input_ids'])
            bias = 0
            # print(q_reps.size())
            # print((p_reps.size()))
            q_reps = q_reps
            # scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
            scores = torch.matmul(q_reps.unsqueeze(1),
                                  p_reps.view(q_reps.shape[0], -1, q_reps.shape[1]).transpose(1, 2)).squeeze(1)

            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            if scores.shape[0] != scores.shape[1]:
                # otherwise in batch neg only
                target = target * (1 + self.data_args.hn_num)
            #print(target)
            #print(scores)
            #loss = self.cross_entropy(scores, target)
            # print("target")
            # print(keys)

            #print("验证moding")
            target = keys['label_mask']
            loss = torch.FloatTensor([0])

            # print(keys['label_mask'])
            return DenseOutput(
                loss=loss,
                scores=scores,
                target=target,
                q_reps=q_reps,
                p_reps=p_reps
            )
        else:
            G1_hidden, G1_reps = self.encode(query["xxx1"])
            G2_hidden, G2_reps = self.encode(keys["xxx2"])

            edges = []
            mainnodeQ = []
            mainnodeK = []
            num_edge = 0
            num_nodes = 0
            bias = 0
            for i in query["xxx1"]:
                # print(i["edge"])
                mainnodeQ.append(bias)
                bias += len(i['input_ids'])
            bias = 0
            # import error
            for i in keys["xxx2"]:
                mainnodeK.append(bias)
                bias += len(i['input_ids'])
            # G1_mainreps = G1_reps[mainnodeQ]
            # G2_mainreps = G2_reps[mainnodeK]
            G1_mainreps = G1_reps
            G2_mainreps = G2_reps
            # print(len(G1_mainreps),len(G2_mainreps))
            if G1_reps is None:  # or G2_reps is None:
                return DenseOutput(
                    q_reps=G1_reps.contiguous(),
                    # p_reps=G2_reps.contiguous()
                )
            # print("neg size")
            # print(G1_mainreps.size())
            # print(G2_mainreps.size())
            if self.train_args.negatives_x_device:
                G1_reps = self.dist_gather_tensor(G1_reps)
                G2_reps = self.dist_gather_tensor(G2_reps)
            scores = torch.matmul(G1_mainreps, G2_mainreps.transpose(0, 1))
            #print(scores)
            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            # print(G1_mainreps)
            # print(G1_mainreps.shape)
            # print("scores in moding")
            # print(scores)
            # print("target")
            # print(target.device)
            # print(target.shape)
            # print(scores.shape[0], scores.shape[1])
            # raise ValueError('stop')

            if scores.shape[0] != scores.shape[1]:
                # otherwise in batch neg only
                target = target * (1 + self.data_args.hn_num)

            loss = self.cross_entropy(scores, target)
            if self.training and self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            # print("densemodel output")
            print("loss", loss)
            # print(DenseOutput(loss=loss,scores=scores,target=target,q_reps=q_reps.contiguous(),p_reps=p_reps.contiguous()))
            # return 0
            return DenseOutput(
                loss=loss,
                scores=scores,
                target=target,
                q_reps=G1_reps.contiguous(),
                p_reps=G2_reps.contiguous()
            )



    def forwarda(
            self,
            query: Dict[str, Tensor] = None,
            keys: Dict[str, Tensor] = None,
    ):
        #print("===========================")
        #print(query)
        #print("============================")
        #print(keys)
        #G1_hidden, G1_reps = self.encode(G1["xxx1"])
        #G2_hidden, G2_reps = self.encode(G2["xxx2"])
        q_hidden, q_reps = self.encode(query["xxx1"])
        p_hidden, p_reps = self.encode(keys["xxx2"])

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)
        bias = 0
        mainnodeQ=[]
        for i in query["xxx1"]:
            #print(i["edge"])
            mainnodeQ.append(bias)
            bias += len(i['input_ids'])
        bias = 0
        #print(q_reps.size())
        #print((p_reps.size()))
        q_reps = q_reps
        # scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        scores = torch.matmul(q_reps.unsqueeze(1), p_reps.view(q_reps.shape[0], -1, q_reps.shape[1]).transpose(1, 2)).squeeze(1)


        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )
        if scores.shape[0] != scores.shape[1]:
            # otherwise in batch neg only
            target = target * (1 + self.data_args.hn_num)
        print(target)
        print(scores)
        loss = self.cross_entropy(scores, target)
        #print("target")
        #print(keys)
        if keys["eval"] == 1:
            print("验证moding")
            target=keys['label_mask']
            loss=torch.FloatTensor([0])

        #print(keys['label_mask'])
        return DenseOutput(
            loss=loss,
            scores=scores,
            target=target,
            q_reps=q_reps,
            p_reps=p_reps
        )


class DenseLMModel(DenseModel):
    def __init__(
            self,
            lm: PreTrainedModel,
            mlm_head: nn.Module,
            lm_config: PretrainedConfig,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__(lm, pooler, model_args, data_args, train_args)
        self.mlm_head = mlm_head
        self.lm_config = lm_config

    def forward(
            self,
            G1,
            G2,
    ):

        G1_hidden,G1_reps=self.encode(G1["xxx1"])

        G2_hidden, G2_reps = self.encode(G2["xxx2"])

        if G1_reps is None: #or G2_reps is None:
            return DenseOutput(
                q_reps=G1_reps.contiguous(),
                p_reps=G2_reps.contiguous()
            )

        if self.train_args.negatives_x_device:
            G1_reps = self.dist_gather_tensor(G1_reps)
            G2_reps = self.dist_gather_tensor(G2_reps)

        scores = torch.matmul(G1_reps, G2_reps.transpose(0, 1))

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        if scores.shape[0] != scores.shape[1]:
            # otherwise in batch neg only
            target = target * (1 + self.data_args.hn_num)

        loss = self.cross_entropy(scores, target)
        #print("loss1",loss)
        # mlm loss
        label1=[]
        label2=[]
        for i in range(len(G1["xxx1"])):
            label1=label1+[G1["xxx1"][i]['labels']]
            label2 = label2 + [G2["xxx2"][i]['labels']]
        label1=torch.cat(label1,dim=0)
        label2 = torch.cat(label2, dim=0)


        if self.train_args.mlm_loss:

            G1_prediction_scores = self.mlm_head(G1_hidden)
            G2_prediction_scores = self.mlm_head(G2_hidden)

            q_mlm_loss = self.cross_entropy(G1_prediction_scores.view(-1, self.lm_config.vocab_size), label1.view(-1))
            k_mlm_loss = self.cross_entropy(G2_prediction_scores.view(-1, self.lm_config.vocab_size), label2.view(-1))
            loss = loss + self.train_args.mlm_weight * (q_mlm_loss + k_mlm_loss)
            #loss2=self.train_args.mlm_weight * (q_mlm_loss + k_mlm_loss)
            #print(loss)


        return DenseOutput(
            loss=loss,
            scores=scores,
            target=target,
            q_reps=G1_reps.contiguous(),
            p_reps=G2_reps.contiguous()
        )

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

        if self.train_args.mlm_loss:
            torch.save(self.mlm_head.state_dict(), os.path.join(output_dir, 'mlm_head.pt'))

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        lm_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        hf_kwargs.update({"attn_implementation": "eager"})
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        #mlm_head = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs).cls
        # mlm_head = BertOnlyMLMHead(lm_config)
        # mlm_head.load_state_dict(torch.load(f'{model_args.model_name_or_path}/mlm_head.pt'))
        from transformers.models.distilbert.modeling_distilbert import DistilBertForMaskedLM

        # 加载 DistilBertForMaskedLM 模型
        mlm_model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        # 检查是否为 DistilBertForMaskedLM，适配其 Masked LM 头结构
        if isinstance(mlm_model, DistilBertForMaskedLM):
            # 自定义类封装 DistilBERT 的 Masked LM 头（与 BERT 的 cls 功能对齐）
            class DistilBertMLMHead(nn.Module):
                def __init__(self, mlm_model):
                    super().__init__()
                    self.vocab_transform = mlm_model.vocab_transform
                    self.vocab_layer_norm = mlm_model.vocab_layer_norm
                    self.vocab_projector = mlm_model.vocab_projector
                    self.activation = mlm_model.activation  # 保留原模型的激活函数

                def forward(self, hidden_states):
                    # 复现 DistilBERT Masked LM 头的前向逻辑
                    hidden_states = self.vocab_transform(hidden_states)
                    hidden_states = self.activation(hidden_states)
                    hidden_states = self.vocab_layer_norm(hidden_states)
                    hidden_states = self.vocab_projector(hidden_states)
                    return hidden_states

            # 初始化封装后的 Masked LM 头
            mlm_head = DistilBertMLMHead(mlm_model)
        else:
            # 若为其他模型（如 BERT），仍使用原 cls 字段
            mlm_head = mlm_model.cls

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            mlm_head=mlm_head,
            lm_config=lm_config,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model


class DenseModelforNCC(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            classifier: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        self.pooler = pooler
        self.classifier = classifier

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            G1: Dict[str, Tensor] = None,
            G2: Tensor = None,
    ):
        # print("q")
        # print(query)
        q_hidden, q_reps = self.encode(G1["xxx1"])
        labels = torch.tensor(G2).to(q_reps.device)

        if q_reps is None:
            return DenseOutput(
                q_reps=q_reps,
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
        # print(q_reps)
        # print(keys)

        # mainnodeQ = []
        # bias = 0
        # for i in G1["xxx1"]:
        #    # print(i["edge"])
        #    mainnodeQ.append(bias)
        #    bias += len(i['input_ids'])
        # q_reps = q_reps[mainnodeQ]

        scores = self.classifier(q_reps)
        # print("scores")
        # print(scores)
        loss = self.cross_entropy(scores, labels)
        #print(loss)
        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction

        return DenseOutput(
            loss=loss,
            scores=scores,
            target=labels,
            q_reps=q_reps,
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        # psg = BatchEncoding(psg)
        psg = {"G": psg}
        psg_out = self.lm(**psg)

        try:
            p_hidden = psg_out.last_hidden_state
            if self.pooler is not None:
                p_reps = self.pooler(p=p_hidden)  # D * d
            else:
                p_reps = p_hidden[:, 0]
        except:
            p_reps = psg_out
            p_hidden = None

        return p_hidden, p_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def build_classifier(data_args, model_args):
        classifier = LinearClassifier(
            model_args.projection_in_dim,
            data_args.class_num
        )
        classifier.load(model_args.model_name_or_path)
        return classifier

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        hf_kwargs.update({"attn_implementation": "eager"})
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        # init classifier
        classifier = cls.build_classifier(data_args, model_args)

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            classifier=classifier,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        self.classifier.save_classifier(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
