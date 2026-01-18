import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertSelfAttention, BertEmbeddings, BertPreTrainedModel, BertAttention, BertIntermediate, BertOutput#, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from IPython import embed
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv



class SingleLayerGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        print("聚集层初始化")
        self.conv = SAGEConv(in_channels, out_channels)  # 单层聚合
        #self.conv = SAGEConv(in_channels, out_channels, aggr='max')

    def forward(self, hidden_states, graph,aggl):
        for param in self.conv.parameters():
            nowdevice=param.device
            break
        if aggl == 10000000000000:
            print("============================================")
            for name, param in self.conv.named_parameters():
                print(f"参数名称: {name}")
                print(f"参数形状: {param.shape}")
                print(f"参数设备: {param.device}")
                print(f"是否需要梯度: {param.requires_grad}")
                print(f"参数值:")
                print(param.data)
        #import torch
        #print(torch.cuda.is_available())  # 应返回 True（若有 CUDA 环境）
        #print(torch.version.cuda)
        limx = hidden_states.clone()
        #nowdevice = graph.device
        #print(nowdevice)
        limx=limx.to(nowdevice)
        graph=graph.long().to(nowdevice)
        x = self.conv(limx, graph)  # 直接聚合邻居特征
        #x = self.conv(limx, graph.long())  # 直接聚合邻居特征

        #return F.log_softmax(x, dim=1)  # 接分类器
        return x





class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        print("BertSelfAttention")
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_value = None,
        output_attentions = False,
    ):
        #print("BertSelfAttention forward")
        mixed_query_layer = self.query(hidden_states)
        #print(215)
        #print(mixed_query_layer)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # embed()
        ##### attention_probs[0][:,2,:].sum(0) ##### for attention map study
        # raise ValueError('here!')

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        #print(298)
        #print(outputs)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        print("BertSelfOutput")
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        #print("BertSelfOutput forward")
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        #print("BertAttention")
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_value = None,
        output_attentions = False,
    ):
        #print("BertAttention forward")
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# class BertIntermediate(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        print("BertOutput")
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        #print("BertOutput forward")
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        print("BertLayer")
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, 
        attention_mask= None, head_mask= None, 
        encoder_hidden_states= None, encoder_attention_mask= None, past_key_value= None, output_attentions= False):
        #print("BertLayer forward")
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )

        attention_output = self_attention_outputs[0]


        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

def remove_nodes(data, edge_index, removal_mask,current_label,attention_mask):
    """
    从图中删除指定标签的节点及其相关边，并更新删除表

    参数:
    data (torch.Tensor): 节点特征数据，形状为[节点数量, 特征维度]
    edge_index (torch.Tensor): 边索引，形状为[2, 边数量]
    removal_mask (torch.Tensor): 删除表，形状为[节点数量]，包含不同删除标签(0表示保留)
    current_label (int): 当前要删除的标签值，默认为1

    返回:
    tuple: (新的节点特征数据, 新的边索引, 新的删除表, 节点映射表)
    """
    # 确定需要保留的节点（删除表中值不为current_label的节点）
    #print(removal_mask,current_label)
    keep_mask = removal_mask < current_label
    keep_indices = torch.where(keep_mask)[0]
    #print("keepmask")
    #print(keep_mask)
    #print(len(keep_mask))
    # 检查是否有节点被保留
    if keep_indices.numel() == 0:
        print("警告：所有节点都被标记为删除，返回空图")
        return torch.empty(0, data.size(1)), torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long), {}

    # 构建旧节点索引到新节点索引的映射
    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}

    # 过滤边：仅保留两端节点都被保留的边
    #print("边q")
    #print(edge_index)
    edge_index = edge_index.to(torch.long)
    src, dst = edge_index
    #print("src", src)
    #print("dst", dst)
    edge_mask = keep_mask[src] & keep_mask[dst]
    filtered_edges = edge_index[:, edge_mask]

    # 重新映射边索引到新的节点编号
    if filtered_edges.numel() > 0:
        new_src = torch.tensor([old_to_new[idx.item()] for idx in filtered_edges[0]], dtype=torch.long)
        new_dst = torch.tensor([old_to_new[idx.item()] for idx in filtered_edges[1]], dtype=torch.long)
        new_edge_index = torch.stack([new_src, new_dst], dim=0)
    else:
        new_edge_index = torch.empty(2, 0, dtype=torch.long)

    # 过滤节点特征和删除表
    filtered_data = data[keep_indices]
    filtered_removal_mask = removal_mask[keep_indices]
    filtered_attention_mask = attention_mask[keep_indices]
    #print("返回值们",filtered_data, new_edge_index, filtered_removal_mask, old_to_new,filtered_attention_mask)
    return filtered_data, new_edge_index, filtered_removal_mask, old_to_new,filtered_attention_mask


class GraphBertEncoder(nn.Module):
    def __init__(self, config):
        print("现在是1-8号的GraphBertEncoder")
        super(GraphBertEncoder, self).__init__()
        self.aggnum=3
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(8)])
        #self.sagelayer114514 = nn.ModuleList([GraphsageAg(config=config) for _ in range(config.num_hidden_layers)])

        self.sagelayer = nn.ModuleList([SingleLayerGraphSAGE(in_channels=768, out_channels=768) for _ in range(self.aggnum)])
        #self.graph_attention = GraphAggregation(config=config)

    def forward(self,
                hidden_states,
                attention_mask,
                graph,
                all_delet_belt
                ):


        all_hidden_states = ()
        all_attentions = ()
        #print("endcoder 接收到alldelet",all_delet_belt)
        #print("graph的形状以及种类",graph)
        aggl=[1,6,11]
        #aggl=[4,8]
        #nowdelete_mask=max(all_delet_belt)
        nowdelete_mask=len(aggl)
        all_delet_belt = torch.tensor(all_delet_belt, dtype=torch.long)
        #print("大删除表")
        #print(all_delet_belt)
        #aggl=[]
        nowaggl=0
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if i >= aggl[0]:
                if i in aggl:
                    cls_emb = hidden_states[:, 1].clone()  # B SN D
                    #print("=============================",nowaggl)
                    station_emb = self.sagelayer[nowaggl](cls_emb, graph,nowaggl)

                    hidden_states[:, 0] = station_emb
                    #print("===============================================")
                    torch.set_printoptions(threshold=float('inf'))
                    #print("现在开始测试移除，移除前各个成员的变量为")
                    #print(nowdelete_mask)
                    #print(nowdelete_mask-nowaggl)
                    #print(hidden_states.shape)
                    #print(hidden_states)
                    #print(graph.shape)
                    #print(graph)
                    #print(all_delet_belt)
                    hidden_states, graph, all_delet_belt, mapping,attention_mask = remove_nodes(hidden_states, graph, all_delet_belt,max(nowdelete_mask-nowaggl,1),attention_mask)
                    station_emb = hidden_states[:, 0:1, :].squeeze(1)
                    #station_emb=hidden_states[:, 0:1, :].squeeze(1)
                    #print("移除成功，移除前各个成员的变量为")
                    #print(nowdelete_mask)
                    #print(nowdelete_mask - nowaggl)
                    #print(hidden_states.shape)
                    #print(hidden_states)
                    #print(graph.shape)
                    #print(graph)
                    #print(all_delet_belt)
                    #print("===============================================")
                    #import error
                    layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)

                    nowaggl += 1
                else:

                    hidden_states[:, 0] = station_emb
                    layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
                    #layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask)


            else:

                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:, :, :, 0] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        #import error
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphFormers(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphFormers, self).__init__(config=config)
        self.config = config
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = GraphBertEncoder(config=config)
        # print(self.base_model_prefix)
        self.neighbor_mask_ratio = 0
        print("GraphFormers")

    def forward(self,
                input_ids,
                attention_mask,
                graph,
                mainnode,
                all_delet_belt,
                neighbor_mask=None
                ):
        #print("Graphformer模块接收到all_delet",all_delet_belt)

        all_nodes_num, seq_length = input_ids.shape

        embedding_output = self.embeddings(input_ids=input_ids) #token embedding

        station_mask = torch.ones((all_nodes_num, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 1+L

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        station_placeholder = torch.zeros(all_nodes_num, 1, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 1+L D
        # stop point 1
        # embed()

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            graph = graph,
            all_delet_belt=all_delet_belt
        )

        return encoder_outputs


import torch
import networkx as nx
from torch_geometric.data import Data
def get_all_distances_from_center(edge,numnode, center_node,max):
    """获取所有节点到中心节点的距离列表（索引对应节点编号）"""
    G = nx.Graph()
    edges = edge.t().tolist()
    G.add_edges_from(edges)
    if isinstance(edge, torch.Tensor) and edge.shape == (2, 0):
        return [0]
    # 初始化距离列表（默认-1表示不可达）
    num_nodes = numnode  # 获取总节点数（此处为10）
    distance_list = [max] * num_nodes
    #print("边", edges)

        # 计算中心节点到所有可达节点的距离
    reachable_nodes = nx.single_source_shortest_path_length(G, center_node)
    for node, dist in reachable_nodes.items():
        distance_list[node] = min(dist,max)  # 按节点编号填充距离


    return distance_list


class GraphFormersForLinkPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = GraphFormers(config)
        self.init_weights()
        print("GraphFormersForLinkPredict")

    #def forward(self, center_input, neighbor_input=None, mask=None, **kwargs):
    def forward(self,G, mask=None, **kwargs):
        '''
        B: batch size, N: 1 + neighbour_num, L: max_token_len, D: hidden dimmension
        '''
        B=len(G)
        #rint("graphformer样本数",B)
        #print("=================")
        #print(G)
        #print("================================")
        L=len(G[0]['input_ids'][0])
        N=[len(i['input_ids']) for i in G]

        input_ids_tensors = [d['input_ids'] for d in G]
        #print("原始输入形状")

        all_input_ids = torch.cat(input_ids_tensors, dim=0)
        #'attention_mask'
        #print(all_input_ids.shape)
        attention_mask_tensors = [d['attention_mask'] for d in G]
        all_attention_mask = torch.cat(attention_mask_tensors, dim=0)

        edge_tensor=[d['edge'] for d in G]
        #for d in G:
            #print("边",d['edge'])
        #print("edge_tensor")
        #import error
        graph=[]
        for i in range(len(N)):
            graph.append({"nodenum":N[i],"edge":edge_tensor[i]})
        #=========================================
        all_edges = []
        offset = 0  # 用于调整边索引的偏移量
        mainnode=[]
        bias=0
        #print(G)
        for i in G:
            mainnode.append(bias)
            bias += len(i['input_ids'])
        # 遍历每个图的数据并合并边
        all_delet_belt=[]
        for data in graph:
            num_nodes = data['nodenum']
            edges = data['edge'].cpu()  # 为了简化，我们将数据移动到CPU。如果需要在GPU上处理，请保留.cuda()调用。
            # 调整边的索引以反映大图中的新节点编号
            #print("第i个图",edges)
            #print("节点数量",num_nodes)
            center_node=0
            delet_belt=get_all_distances_from_center(edges,num_nodes, center_node,3)
            #print("删除顺序表",delet_belt)
            all_delet_belt=all_delet_belt+delet_belt
            adjusted_edges = torch.stack([edges[0] + offset, edges[1] + offset], dim=0)
            all_edges.append(adjusted_edges)
            # 更新偏移量以包含当前图的节点数
            offset += num_nodes
        graph = torch.cat(all_edges, dim=1).contiguous()
        #print(graph)

        #print("中心点位置",mainnode)
        #print("最终删除数据表",all_delet_belt)
        #print(len(all_delet_belt))
        #print(all_input_ids.shape)
        #print(all_input_ids.shape)
        #import error
        D = self.config.hidden_size
        hidden_states = self.bert(all_input_ids, all_attention_mask,graph,mainnode,all_delet_belt)

        #print("graphformer输出",hidden_states.size())
        last_hidden_states = hidden_states[0]

        node_embeddings = last_hidden_states[:, 1:]
        #print("调试模式结束")
        #import error
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=node_embeddings
        )
