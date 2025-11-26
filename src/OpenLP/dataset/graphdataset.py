import os
import glob
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.loader import NeighborLoader
import networkx as nx
from typing import List, Tuple, Dict
import time
def is_bidirectional(data: Data, strict: bool = False) -> bool:
    """
    检查PyG图数据对象是否为双向图

    参数:
    data (Data): PyG图数据对象
    strict (bool): 是否严格检查（每个边都必须有反向边且仅出现一次）

    返回:
    bool: 如果是双向图返回True，否则返回False
    """
    edge_index = data.edge_index

    # 创建边的集合 (i, j)
    edges = set()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edges.add((src, dst))

    # 检查每条边是否都有反向边
    for src, dst in edges:
        if (dst, src) not in edges:
            return False

    # 严格模式下，检查反向边的数量是否与原始边相同
    if strict:
        return len(edges) == edge_index.size(1) // 2

    return True


def remove_node(data, node_id):
    """删除指定节点（保证设备一致性）"""
    # 自动匹配输入数据的设备
    device = data.x.device if hasattr(data, 'x') and data.x is not None else torch.device('cpu')
    remaining_nodes = torch.tensor(
        [i for i in range(data.num_nodes) if i != node_id],
        dtype=torch.long,
        device=device
    )

    # 子图生成（自动处理边索引）
    edge_index, _ = subgraph(
        subset=remaining_nodes,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    # 构造新Data并保证设备一致
    new_data = Data(
        x=data.x[remaining_nodes] if hasattr(data, 'x') else None,
        edge_index=edge_index,
        **{k: v for k, v in data.items() if k not in {'x', 'edge_index', 'y'}}
    )
    return new_data.to(device)


def remove_unreachable_nodes(data, center_node):
    """删除无法连通到中心节点的节点（设备兼容）"""
    # NetworkX仅支持CPU，临时转CPU处理拓扑
    data_cpu = data.cpu()
    G = to_networkx(data_cpu, to_undirected=False)
    G_reversed = G.reverse()

    # BFS找可达节点
    reachable_nodes = set()
    queue = [center_node]
    visited = set([center_node])
    while queue:
        current_node = queue.pop(0)
        reachable_nodes.add(current_node)
        for neighbor in G_reversed.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # 转回原设备
    device = data.x.device if hasattr(data, 'x') and data.x is not None else torch.device('cpu')
    reachable_nodes_tensor = torch.tensor(list(reachable_nodes), dtype=torch.long, device=device)

    # 生成子图
    edge_index, _ = subgraph(
        subset=reachable_nodes_tensor,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    new_data = Data(
        x=data.x[reachable_nodes_tensor] if hasattr(data, 'x') else None,
        edge_index=edge_index,
        **{k: v for k, v in data.items() if k not in {'x', 'edge_index', 'y'}}
    )
    return new_data


def filter_edges_by_direction(edge_index):
    """
    筛选边索引，只保留标号较大的节点到标号较小的节点的边

    参数:
    edge_index: 形状为 [2, num_edges] 的边索引张量

    返回:
    filtered_edge_index: 筛选后的边索引张量
    """
    # 获取每条边的源节点和目标节点
    src = edge_index[0]
    dst = edge_index[1]

    # 创建掩码，保留 src > dst 的边
    mask = src > dst

    # 应用掩码筛选边
    filtered_edge_index = edge_index[:, mask]

    return filtered_edge_index


def make_edge_index_bidirectional(edge_index):
    reversed_edge_index = edge_index.flip(0)
    bidirectional_edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    bidirectional_edge_index = torch.unique(bidirectional_edge_index, dim=1)

    # 确保张量是连续的
    return bidirectional_edge_index.contiguous()




class TrainGraphDataset(Dataset):
    """
    增强版训练数据集，支持从JSONL或PyG格式(.pt)加载数据

    参数:
        tokenizer: 预训练模型的分词器
        data_args: 数据参数配置
        trainer: 训练器实例
        shuffle_seed: 随机打乱种子
        cache_dir: 缓存目录
        use_pyg_data: 是否使用PyG格式的数据
    """

    def __init__(
            self,
            tokenizer,
            data_args,
            trainer=None,
            shuffle_seed: int = None,
            cache_dir: str = None,
            use_pyg_data: bool = False,
    ) -> None:
        super().__init__()
        print("这里是用于训练的图dataset")
        self.tokenizer=tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.use_pyg_data = use_pyg_data
        #data_path="D:/mymodel/truedataset/final/amazon_sports_NO1_811.pt"
        #print(data_args)
        #import error
        data_path = self.data_args.train_path
        #data_path="D:\mymodel\8-5maindataset\cora\cora_final\processed_data_by_coarse_k8_v8.pt"

        self.data = torch.load(data_path)
        print("data",self.data)
        self.text=self.data.text #存储文本
        self.index_to_text = {i: text for i, text in enumerate(self.text)}  # 创建索引到文本的查找表
        n_nodes=len(self.text) #节点数量
        self.nodenum=n_nodes

        biotrain_pos_edge_index=make_edge_index_bidirectional(self.data.train_pos_edge_index)

        self.train_graph = Data(x=torch.arange(n_nodes, dtype=torch.long).view(-1, 1), edge_index=biotrain_pos_edge_index) #用一个索引列表和训练边集合作为训练图
        self.neibor=[5,5,5]
        #self.neibor = [3,3,3,3]
        #print(f"宽松检查: {is_bidirectional(self.train_graph)}")
        #train_sample=self.data.train_pos_edge_index.t()
        #train_sample = self.data.val_pos_edge_index.t()
        #train_sample = self.data.train_pos_edge_index[:32].t()
        #print("数据集",self.data.val_pos_edge_index)
        train_sample = self.data.val_pos_edge_index.t()[:32]
        #shuffled_indices = torch.randperm(train_sample.size(0))
        #train_sample = train_sample[shuffled_indices]
        #print("测试中...检查各变量内容")
        testlist=train_sample.tolist()
        #print("训练:",testlist)
        nobio_train_sample=[]
        for i in testlist:  #把双向边去重变为单向边
            #if [i[1],i[0]] not in nobio_train_sample:
            if 1:
                nobio_train_sample.append(i)
            else:
                continue


        self.train_sample=torch.tensor(nobio_train_sample, dtype=torch.long)

        self.token_cache = {}#token缓存
        self._preload_all_tokens()

        #print(nobio_train_sample)
        #print(len(nobio_train_sample))
        #print(self.data)
        #print(n_nodes)
        #print(self.train_graph)
        #print(self.train_sample)
        #print(len(self.train_sample))
        print("训练", len(self.train_sample))
        #return error

    def _preload_all_tokens(self):
        """预加载所有节点的tokenization结果"""
        if hasattr(self, 'text') and self.text:
            print("开始预加载所有节点的tokenization结果...")
            start = time.perf_counter()

            # 批量处理所有文本
            all_texts = self.text
            batch_size = 1000  # 调整批量大小以适应内存

            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_encoding = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    truncation='only_first',
                    max_length=self.data_args.max_len,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                # 更新缓存 - 正确访问编码结果
                for j, text in enumerate(batch_texts):
                    self.token_cache[text] = {
                        'input_ids': batch_encoding['input_ids'][j]
                    }

            end = time.perf_counter()
            print(f"预加载完成，耗时: {end - start:.2f}秒，缓存大小: {len(self.token_cache)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.train_sample)
    def create_one_example(self, texts):
        """从缓存获取tokenization结果，必要时进行批量编码"""
        # 检查是否有未缓存的文本
        uncached_texts = [text for text in texts if text not in self.token_cache]

        if uncached_texts:
            # 批量编码未缓存的文本
            batch_encoding = self.tokenizer.batch_encode_plus(
                uncached_texts,
                truncation='only_first',
                max_length=self.data_args.max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

        # 更新缓存 - 正确访问编码结果
            for j, text in enumerate(uncached_texts):
                self.token_cache[text] = {
                    'input_ids': batch_encoding['input_ids'][j]
                }

    # 从缓存获取所有结果
        return [self.token_cache[text] for text in texts]

    from torch_geometric.loader import LinkNeighborLoader
    def process_fn(self, example):
        """处理单个样本"""
        #print("这里是processfn")
        #example=torch.tensor([6535, 3713],dtype=torch.long) #test
        #print("用时测试")
        start = time.perf_counter()
        Q_indic=example[0] #获取Q和K的索引
        K_indic=example[1]
        #print(Q_indic,K_indic)
        loaderQ = NeighborLoader(
            self.train_graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小，这里只处理一个节点
            input_nodes=torch.tensor([Q_indic]),  # 目标节点
            #subgraph_type="induced"
        )
        Qsubgraph_batch = next(iter(loaderQ))
        Qsubgraph_batch.x=Qsubgraph_batch.n_id
        start1 = time.perf_counter()
        #print("Q的minibatch采样", start1 - start)
        #print(Qsubgraph_batch)
        #print(Qsubgraph_batch.x)
        #print("Q子图",Qsubgraph_batch.edge_index)
        #print()
        if K_indic in Qsubgraph_batch.n_id:
            #print("K在Q子图中")
            index=Qsubgraph_batch.x.tolist().index(K_indic.item())
            #print("index",index)
            Qsubgraph_batch=remove_node(Qsubgraph_batch, index) #从Q子图中清除K
            Qsubgraph_batch=remove_unreachable_nodes(Qsubgraph_batch,0) #清除会因此链接不到中心节点的节点
        start2 = time.perf_counter()
        #print("Q的去除K", start2 - start1)
        Qsubgraph_batch.edge_index=filter_edges_by_direction(Qsubgraph_batch.edge_index)  #过滤双向边 #####################################################
        start3 = time.perf_counter()
        #print("过滤双向边", start3 - start2)
        q_text = [self.text[i] for i in Qsubgraph_batch.x.tolist()] #获得节点文本
        start4 = time.perf_counter()
        #print("获得节点文本", start4 - start3)
        encoded_query = self.create_one_example(q_text)
        start5 = time.perf_counter()
        #print("token化", start5 - start4)
        #print("token内容:")
        #print(encoded_query)
        #print("q图大小", len(q_text))
        #print(len(q_text))
        #print(Qsubgraph_batch)
        #print(Qsubgraph_batch.x)
        #print("Q的batch")
        #print(Qsubgraph_batch)
        #处理K
        loaderK = NeighborLoader(
            self.train_graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小，这里只处理一个节点
            input_nodes=torch.tensor([K_indic]),  # 目标节点
            #subgraph_type="induced"
            #subgraph_type = "induced"
        )
        Ksubgraph_batch = next(iter(loaderK))

        Ksubgraph_batch.x = Ksubgraph_batch.n_id

        #print("K子图", Ksubgraph_batch.edge_index)
        if Q_indic in Ksubgraph_batch.n_id:
            #print("K在Q子图中")
            index=Ksubgraph_batch.x.tolist().index(Q_indic.item())
            #print("index",index)
            Ksubgraph_batch=remove_node(Ksubgraph_batch, index)
            Ksubgraph_batch=remove_unreachable_nodes(Ksubgraph_batch,0)
        Ksubgraph_batch.edge_index = filter_edges_by_direction(Ksubgraph_batch.edge_index)#===============================================
        k_text = [self.text[i] for i in Ksubgraph_batch.x.tolist()]
        #print("k图大小", len(k_text))
        encoded_key = self.create_one_example(k_text)
        end = time.perf_counter()
        #print("dataset耗时",start-end)
        return {"x1": encoded_query, 'edge1': Qsubgraph_batch.edge_index,"x2": encoded_key, 'edge2': Ksubgraph_batch.edge_index}

    def __getitem__(self, index):
        """获取单个样本"""
        #print("这里是getitem")
        #print(index)

        example = self.train_sample[index]
        #print(example)
        #print("离开getitem")
        #import error

        return self.process_fn(example)






class EvalGraphDataset(Dataset):
    """
    增强版训练数据集，支持从JSONL或PyG格式(.pt)加载数据

    参数:
        tokenizer: 预训练模型的分词器
        data_args: 数据参数配置
        trainer: 训练器实例
        shuffle_seed: 随机打乱种子
        cache_dir: 缓存目录
        use_pyg_data: 是否使用PyG格式的数据
    """

    def __init__(
            self,
            tokenizer,
            data_args,
            trainer=None,
            shuffle_seed: int = None,
            cache_dir: str = None,
            use_pyg_data: bool = False,
    ) -> None:
        super().__init__()
        print("这里是用于训练的图evaldataset")
        self.tokenizer=tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.use_pyg_data = use_pyg_data
        #data_path="D:/mymodel/truedataset/final/amazon_sports_NO1_811.pt"
        data_path = self.data_args.eval_path
        #data_path = "D:\mymodel\8-5maindataset\cora\cora_final\processed_data_by_coarse_k8_v8.pt"
        self.data = torch.load(data_path)
        self.text=self.data.text #存储文本
        self.index_to_text = {i: text for i, text in enumerate(self.text)}  # 创建索引到文本的查找表
        n_nodes=len(self.text) #节点数量
        self.nodenum=n_nodes

        biotrain_pos_edge_index = make_edge_index_bidirectional(self.data.train_pos_edge_index)

        self.train_graph = Data(x=torch.arange(n_nodes, dtype=torch.long).view(-1, 1),edge_index=biotrain_pos_edge_index)
        #self.train_graph = Data(x=torch.arange(n_nodes, dtype=torch.long).view(-1, 1), edge_index=self.data.train_pos_edge_index) #用一个索引列表和训练边集合作为训练图
        self.neibor=[5,5,5]
        #self.neibor = [3,3,3,3]
        #print(f"宽松检查: {is_bidirectional(self.train_graph)}")
        #train_sample=self.data.test_pos_edge_index.t()
        #train_sample = self.data.train_pos_edge_index.t()
        train_sample = self.data.val_pos_edge_index.t()[-32:]
        #shuffled_indices = torch.randperm(train_sample.size(0))
        #train_sample = train_sample[shuffled_indices]
        #print("测试中...检查各变量内容")
        testlist=train_sample.tolist()
        nobio_train_sample=[]
        #print("验证:",testlist)
        #return error
        for i in testlist:  #把双向边去重变为单向边
            #if [i[1],i[0]] not in nobio_train_sample:
            if 1:
                nobio_train_sample.append(i)
            else:
                continue
        self.train_sample=torch.tensor(nobio_train_sample, dtype=torch.long)
        #print(len(self.train_sample))
        #print(nobio_train_sample)
        #print(len(nobio_train_sample))
        #print(self.data)
        #print(n_nodes)
        #print(self.train_graph)
        #print(self.train_sample)
        print("验证",len(self.train_sample))

    def create_one_example(self, text_encoding: List[int]):
        item = []
        for i in text_encoding:
            item.append(self.tokenizer.encode_plus(
                i,
                truncation='only_first',
                max_length=self.data_args.max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            ))  # ['input_ids'])
        return item

    def __len__(self):
        """返回数据集大小"""
        return len(self.train_sample)

    def process_fn(self, example):
        """处理单个样本"""
        #print("这里是processfn")
        #example=torch.tensor([6535, 3713],dtype=torch.long) #test
        #print("用时测试")

        start = time.perf_counter()
        Q_indic=example[0] #获取Q和K的索引
        K_indic=example[1]
        #print(Q_indic,K_indic)
        loaderQ = NeighborLoader(
            self.train_graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小，这里只处理一个节点
            input_nodes=torch.tensor([Q_indic]),  # 目标节点
            #subgraph_type="induced"
        )
        Qsubgraph_batch = next(iter(loaderQ))
        Qsubgraph_batch.x=Qsubgraph_batch.n_id
        #print(Qsubgraph_batch)
        #print(Qsubgraph_batch.x)
        #print(Qsubgraph_batch.edge_index)
        #print()
        if K_indic in Qsubgraph_batch.n_id:
            #print("K在Q子图中")
            index=Qsubgraph_batch.x.tolist().index(K_indic.item())
            #print("index",index)
            Qsubgraph_batch=remove_node(Qsubgraph_batch, index) #从Q子图中清除K
            Qsubgraph_batch=remove_unreachable_nodes(Qsubgraph_batch,0) #清除会因此链接不到中心节点的节点
        Qsubgraph_batch.edge_index=filter_edges_by_direction(Qsubgraph_batch.edge_index)  #过滤双向边 #============================
        q_text = [self.text[i] for i in Qsubgraph_batch.x.tolist()] #获得节点文本
        encoded_query = self.create_one_example(q_text)
        #print(q_text)
        #print("q图大小",len(q_text))

        #print(Qsubgraph_batch)
        #print(Qsubgraph_batch.x)
        #print("Q的batch")
        #print(Qsubgraph_batch)
        #处理K
        loaderK = NeighborLoader(
            self.train_graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小，这里只处理一个节点
            input_nodes=torch.tensor([K_indic]),  # 目标节点
            #subgraph_type="induced"
        )
        Ksubgraph_batch = next(iter(loaderK))

        Ksubgraph_batch.x = Ksubgraph_batch.n_id
        if Q_indic in Ksubgraph_batch.n_id:
            #print("K在Q子图中")
            index=Ksubgraph_batch.x.tolist().index(Q_indic.item())
            #print("index",index)
            Ksubgraph_batch=remove_node(Ksubgraph_batch, index)
            Ksubgraph_batch=remove_unreachable_nodes(Ksubgraph_batch,0)
        Ksubgraph_batch.edge_index = filter_edges_by_direction(Ksubgraph_batch.edge_index) #==============================================
        k_text = [self.text[i] for i in Ksubgraph_batch.x.tolist()]
        #print("k图大小", len(k_text))
        encoded_key = self.create_one_example(k_text)
        end = time.perf_counter()
        #print("dataset耗时",start-end)
        return {"x1": encoded_query, 'edge1': Qsubgraph_batch.edge_index,"x2": encoded_key, 'edge2': Ksubgraph_batch.edge_index}

    def __getitem__(self, index):
        """获取单个样本"""
        #print("这里是getitem")
        #print(index)

        example = self.train_sample[index]
        #print(example)
        #print("离开getitem")
        #import error

        return self.process_fn(example)


class TestGraphDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_args,
            trainer=None,
            shuffle_seed: int = None,
            cache_dir: str = None,
            use_pyg_data: bool = False,
    ) -> None:
        super().__init__()
        print("这里是用于测试的图dataset")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.use_pyg_data = use_pyg_data
        # 自动选择设备（GPU优先）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前设备: {self.device}")

        # 加载数据
        data_path = self.data_args.eval_path
        self.data = torch.load(data_path)
        self.text = self.data.text  # 文本列表
        self.index_to_text = {i: text for i, text in enumerate(self.text)}  # 索引-文本映射
        n_nodes = len(self.text)
        self.nodenum = n_nodes

        # 构建双向边并移至设备
        biotrain_pos_edge_index = make_edge_index_bidirectional(self.data.train_pos_edge_index)
        biotrain_pos_edge_index = biotrain_pos_edge_index.to(self.device)

        # 构建训练图并移至设备
        x = torch.arange(n_nodes, dtype=torch.long, device=self.device).view(-1, 1)
        self.train_graph = Data(x=x, edge_index=biotrain_pos_edge_index).to(self.device)

        self.neibor = [5,5,5] # 邻居采样数

        # 处理测试样本
        train_sample = self.data.test_pos_edge_index.t()
        #shuffled_indices = torch.randperm(train_sample.size(0))
        #train_sample = train_sample[shuffled_indices]
        testlist = train_sample.tolist()
        nobio_train_sample = [i for i in testlist]  # 简化双向边去重（示例保持原逻辑）
        self.train_sample = torch.tensor(nobio_train_sample, dtype=torch.long, device=self.device)
        print(f"测试样本数量: {len(self.train_sample)}")

        # 预加载Token到GPU缓存
        self.token_cache = {}
        self._preload_all_tokens()

    def _preload_all_tokens(self):
        """预加载所有文本的Tokenization结果到GPU"""
        if not hasattr(self, 'text') or not self.text:
            return
        print("开始预加载Token到GPU...")
        start = time.perf_counter()

        all_texts = self.text
        batch_size = 1000  # 批量大小，避免内存溢出
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_encoding = self.tokenizer.batch_encode_plus(
                batch_texts,
                truncation='only_first',
                max_length=self.data_args.max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            # 缓存到GPU
            for j, text in enumerate(batch_texts):
                self.token_cache[text] = {
                    'input_ids': torch.tensor(batch_encoding['input_ids'][j], device=self.device)
                }

        end = time.perf_counter()
        print(f"预加载完成，耗时: {end - start:.2f}秒，缓存大小: {len(self.token_cache)}")

    def create_one_example(self, text_indices: List[int]):
        """从缓存获取GPU上的Token结果"""
        return [self.token_cache[self.text[i]] for i in text_indices]

    def __len__(self):
        return len(self.train_sample)

    def process_fn(self, example):
        """单样本处理（核心逻辑移至GPU）"""
        start = time.perf_counter()
        Q_indic, K_indic = example[0], example[1]  # 已在device上

        # Q节点子图采样（GPU上执行）
        loaderQ = NeighborLoader(
            self.train_graph.cpu(),
            num_neighbors=self.neibor,
            batch_size=1,
            input_nodes=torch.tensor([Q_indic], device=self.device),
              # 子图数据加载到GPU
        )
        Qsubgraph_batch = next(iter(loaderQ)).to(self.device)
        Qsubgraph_batch.x = Qsubgraph_batch.n_id  # 原始节点ID

        # 移除K节点（若存在）
        if K_indic in Qsubgraph_batch.n_id:
            print("K在Q子图中")
            # GPU上高效查找索引
            index = (Qsubgraph_batch.n_id == K_indic).nonzero(as_tuple=True)[0].item()
            Qsubgraph_batch = remove_node(Qsubgraph_batch, index)
            Qsubgraph_batch = remove_unreachable_nodes(Qsubgraph_batch, 0)
        # 过滤双向边（GPU上执行）
        Qsubgraph_batch.edge_index = filter_edges_by_direction(Qsubgraph_batch.edge_index)

        # 获取文本并加载Token（已在GPU缓存）
        q_indices = Qsubgraph_batch.x.cpu().tolist()  # 临时转CPU取文本索引
        encoded_query = self.create_one_example(q_indices)

        # K节点子图采样（GPU上执行）
        loaderK = NeighborLoader(
            self.train_graph.cpu(),
            num_neighbors=self.neibor,
            batch_size=1,
            input_nodes=torch.tensor([K_indic], device=self.device),

        )
        Ksubgraph_batch = next(iter(loaderK)).to(self.device)
        Ksubgraph_batch.x = Ksubgraph_batch.n_id

        # 移除Q节点（若存在）
        if Q_indic in Ksubgraph_batch.n_id:
            print("Q在K子图中")
            index = (Ksubgraph_batch.n_id == Q_indic).nonzero(as_tuple=True)[0].item()
            Ksubgraph_batch = remove_node(Ksubgraph_batch, index)
            Ksubgraph_batch = remove_unreachable_nodes(Ksubgraph_batch, 0)
        # 过滤双向边（GPU上执行）
        Ksubgraph_batch.edge_index = filter_edges_by_direction(Ksubgraph_batch.edge_index)

        # 获取K的文本Token
        k_indices = Ksubgraph_batch.x.cpu().tolist()
        encoded_key = self.create_one_example(k_indices)

        end = time.perf_counter()
        return {"x1": encoded_query, 'edge1': Qsubgraph_batch.edge_index, "x2": encoded_key, 'edge2': Ksubgraph_batch.edge_index}

    def __getitem__(self, index):
        example = self.train_sample[index]
        return self.process_fn(example)




class TrainNCCGraphDataset(Dataset):
    """
    分类任务的训练数据集，支持从PyG格式(.pt)加载数据

    参数:
        tokenizer: 预训练模型的分词器
        data_args: 数据参数配置
        trainer: 训练器实例
        shuffle_seed: 随机打乱种子
        cache_dir: 缓存目录
        use_pyg_data: 是否使用PyG格式的数据
    """

    def __init__(
            self,
            tokenizer,
            data_args,
            trainer=None,
            shuffle_seed: int = None,
            cache_dir: str = None,
            use_pyg_data: bool = False,
    ) -> None:
        super().__init__()
        print("这里是用于分类的图dataset")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.use_pyg_data = use_pyg_data

        # 加载处理后的分类数据集
        #data_path = "D:/mymodel/truedataset/final/processed/processed_data_by_coarse_k8_v8.pt"data_path = self.data_args.train_path
        data_path = self.data_args.train_path
        #data_path = "D:\mymodel\8-5maindataset\cora\cora_final\processed_data_by_coarse_k8_v8.pt"
        self.data = torch.load(data_path)
        print(self.data)
        # 存储文本和标签
        self.text = self.data.text  # 存储文本
        self.labels = self.data.label  # 存储标签
        #self.labels = self.data.y  # 存储标签

        # 创建索引到文本的查找表
        self.index_to_text = {i: text for i, text in enumerate(self.text)}

        # 获取节点数量
        self.n_nodes = len(self.text)

        biotrain_pos_edge_index = make_edge_index_bidirectional(self.data.train_pos_edge_index)

        self.graph = Data(x=torch.arange(self.n_nodes, dtype=torch.long).view(-1, 1),
                                edge_index=biotrain_pos_edge_index)

        # 分类任务使用整个图结构
        #self.graph = Data(
        #    x=torch.arange(self.n_nodes, dtype=torch.long).view(-1, 1),
        #    edge_index=self.data.train_pos_edge_index
        #)

        # 邻居采样配置
        self.neibor = [5,5,5]  # 每层采样的邻居数
        #self.neibor = [3,3,3,3]
        # 获取训练节点索引
        self.train_indices = self.data.trainindex.tolist()

        # 预加载所有节点的tokenization结果
        self.token_cache = {}
        self._preload_all_tokens()

        print(f"数据集加载完成: {self.n_nodes} 个节点, {len(self.train_indices)} 个训练样本")

    def _preload_all_tokens(self):
        """预加载所有节点的tokenization结果"""
        if hasattr(self, 'text') and self.text:
            print("开始预加载所有节点的tokenization结果...")
            start = time.perf_counter()

            # 批量处理所有文本
            all_texts = self.text
            batch_size = 1000  # 调整批量大小以适应内存

            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_encoding = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    truncation='only_first',
                    max_length=self.data_args.max_len,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                # 更新缓存
                for j, text in enumerate(batch_texts):
                    self.token_cache[text] = {
                        'input_ids': batch_encoding['input_ids'][j]
                    }

            end = time.perf_counter()
            print(f"预加载完成，耗时: {end - start:.2f}秒，缓存大小: {len(self.token_cache)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.train_indices)

    def create_one_example(self, texts):
        """从缓存获取tokenization结果，必要时进行批量编码"""
        # 检查是否有未缓存的文本
        uncached_texts = [text for text in texts if text not in self.token_cache]

        if uncached_texts:
            # 批量编码未缓存的文本
            batch_encoding = self.tokenizer.batch_encode_plus(
                uncached_texts,
                truncation='only_first',
                max_length=self.data_args.max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            # 更新缓存
            for j, text in enumerate(uncached_texts):
                self.token_cache[text] = {
                    'input_ids': batch_encoding['input_ids'][j]
                }

        # 从缓存获取所有结果
        return [self.token_cache[text] for text in texts]

    def process_fn(self, node_idx):
        """处理单个样本"""
        start = time.perf_counter()

        # 获取节点标签
        #print(self.labels)
        label = self.labels[node_idx].item()
        #label = self.y[node_idx].item()

        # 采样节点的邻居子图
        loader = NeighborLoader(
            self.graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小
            input_nodes=torch.tensor([node_idx]),  # 目标节点
            #subgraph_type = "induced"
        )

        # 获取子图
        subgraph_batch = next(iter(loader))
        #print(subgraph_batch)
        subgraph_batch.x = subgraph_batch.n_id  # 使用原始节点ID
        #print(subgraph_batch)

        # 过滤双向边
        subgraph_batch.edge_index = filter_edges_by_direction(subgraph_batch.edge_index)

        # 获取节点文本
        node_texts = [self.text[i] for i in subgraph_batch.x.tolist()]

        # 文本tokenization
        encoded_texts = self.create_one_example(node_texts)

        end = time.perf_counter()
        # print(f"处理样本 {node_idx} 耗时: {end - start:.4f}秒")
        #print({
        #    'label': label  # 节点的分类标签
        #})
        #print(f"子图节点ID范围: {subgraph_batch.n_id.min()} 到 {subgraph_batch.n_id.max()}")
        #print("边",subgraph_batch.edge_index)
        #return error
        return {
            'x1': encoded_texts,  # 节点文本的token表示
            'edge1': subgraph_batch.edge_index,  # 子图的边索引
            'label': label  # 节点的分类标签
        }

    def __getitem__(self, index):
        """获取单个样本"""
        # 获取训练节点索引
        node_idx = self.train_indices[index]

        # 处理样本
        return self.process_fn(node_idx)



class EvalNCCGraphDataset(Dataset):
    """
    分类任务的训练数据集，支持从PyG格式(.pt)加载数据

    参数:
        tokenizer: 预训练模型的分词器
        data_args: 数据参数配置
        trainer: 训练器实例
        shuffle_seed: 随机打乱种子
        cache_dir: 缓存目录
        use_pyg_data: 是否使用PyG格式的数据
    """

    def __init__(
            self,
            tokenizer,
            data_args,
            trainer=None,
            shuffle_seed: int = None,
            cache_dir: str = None,
            use_pyg_data: bool = False,
    ) -> None:
        super().__init__()
        print("这里是用于分类的图dataset")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.use_pyg_data = use_pyg_data

        # 加载处理后的分类数据集
        #data_path = "D:/mymodel/truedataset/final/processed/processed_data_by_coarse_k8_v8.pt"
        data_path = self.data_args.eval_path
        #data_path = "D:\mymodel\8-5maindataset\cora\cora_final\processed_data_by_coarse_k8_v8.pt"
        self.data = torch.load(data_path)

        # 存储文本和标签
        self.text = self.data.text  # 存储文本
        self.labels = self.data.label  # 存储标签

        # 创建索引到文本的查找表
        self.index_to_text = {i: text for i, text in enumerate(self.text)}

        # 获取节点数量
        self.n_nodes = len(self.text)

        biotrain_pos_edge_index = make_edge_index_bidirectional(self.data.train_pos_edge_index)

        self.graph = Data(x=torch.arange(self.n_nodes, dtype=torch.long).view(-1, 1),
                          edge_index=biotrain_pos_edge_index)
        # 分类任务使用整个图结构
        #self.graph = Data(
        #    x=torch.arange(self.n_nodes, dtype=torch.long).view(-1, 1),
        #    edge_index=self.data.train_pos_edge_index
        #)

        # 邻居采样配置
        self.neibor = [5,5,5] # 每层采样的邻居数
        #self.neibor = [3, 3, 3, 3]

        # 获取训练节点索引
        self.train_indices = self.data.valindex.tolist()
        #self.train_indices = self.data.testindex.tolist()

        # 预加载所有节点的tokenization结果
        self.token_cache = {}
        self._preload_all_tokens()

        print(f"数据集加载完成: {self.n_nodes} 个节点, {len(self.train_indices)} 个训练样本")

    def _preload_all_tokens(self):
        """预加载所有节点的tokenization结果"""
        if hasattr(self, 'text') and self.text:
            print("开始预加载所有节点的tokenization结果...")
            start = time.perf_counter()

            # 批量处理所有文本
            all_texts = self.text
            batch_size = 1000  # 调整批量大小以适应内存

            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_encoding = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    truncation='only_first',
                    max_length=self.data_args.max_len,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                # 更新缓存
                for j, text in enumerate(batch_texts):
                    self.token_cache[text] = {
                        'input_ids': batch_encoding['input_ids'][j]
                    }

            end = time.perf_counter()
            print(f"预加载完成，耗时: {end - start:.2f}秒，缓存大小: {len(self.token_cache)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.train_indices)

    def create_one_example(self, texts):
        """从缓存获取tokenization结果，必要时进行批量编码"""
        # 检查是否有未缓存的文本
        uncached_texts = [text for text in texts if text not in self.token_cache]

        if uncached_texts:
            # 批量编码未缓存的文本
            batch_encoding = self.tokenizer.batch_encode_plus(
                uncached_texts,
                truncation='only_first',
                max_length=self.data_args.max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            # 更新缓存
            for j, text in enumerate(uncached_texts):
                self.token_cache[text] = {
                    'input_ids': batch_encoding['input_ids'][j]
                }

        # 从缓存获取所有结果
        return [self.token_cache[text] for text in texts]

    def process_fn(self, node_idx):
        """处理单个样本"""
        start = time.perf_counter()

        # 获取节点标签
        label = self.labels[node_idx].item()

        # 采样节点的邻居子图
        loader = NeighborLoader(
            self.graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小
            input_nodes=torch.tensor([node_idx]),  # 目标节点
            #subgraph_type="induced"
        )

        # 获取子图
        subgraph_batch = next(iter(loader))
        #print(subgraph_batch)
        subgraph_batch.x = subgraph_batch.n_id  # 使用原始节点ID
        #print(subgraph_batch)

        # 过滤双向边
        subgraph_batch.edge_index = filter_edges_by_direction(subgraph_batch.edge_index)

        # 获取节点文本
        node_texts = [self.text[i] for i in subgraph_batch.x.tolist()]

        # 文本tokenization
        encoded_texts = self.create_one_example(node_texts)

        end = time.perf_counter()
        # print(f"处理样本 {node_idx} 耗时: {end - start:.4f}秒")
        print({
            'label': label  # 节点的分类标签
        })
        print(f"子图节点ID范围: {subgraph_batch.n_id.min()} 到 {subgraph_batch.n_id.max()}")
        #return error
        return {
            'x1': encoded_texts,  # 节点文本的token表示
            'edge1': subgraph_batch.edge_index,  # 子图的边索引
            'label': label  # 节点的分类标签
        }

    def __getitem__(self, index):
        """获取单个样本"""
        # 获取训练节点索引
        node_idx = self.train_indices[index]

        # 处理样本
        return self.process_fn(node_idx)


class TestNCCGraphDataset(Dataset):
    """
    分类任务的训练数据集，支持从PyG格式(.pt)加载数据

    参数:
        tokenizer: 预训练模型的分词器
        data_args: 数据参数配置
        trainer: 训练器实例
        shuffle_seed: 随机打乱种子
        cache_dir: 缓存目录
        use_pyg_data: 是否使用PyG格式的数据
    """

    def __init__(
            self,
            tokenizer,
            data_args,
            trainer=None,
            shuffle_seed: int = None,
            cache_dir: str = None,
            use_pyg_data: bool = False,
    ) -> None:
        super().__init__()
        print(data_args)
        print("这里是用于分类的图dataset")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.use_pyg_data = use_pyg_data

        # 加载处理后的分类数据集
        # data_path = "D:/mymodel/truedataset/final/processed/processed_data_by_coarse_k8_v8.pt"
        data_path = self.data_args.eval_path
        # data_path = "D:\mymodel\8-5maindataset\cora\cora_final\processed_data_by_coarse_k8_v8.pt"
        self.data = torch.load(data_path)

        # 存储文本和标签
        self.text = self.data.text  # 存储文本
        self.labels = self.data.label  # 存储标签

        # 创建索引到文本的查找表
        self.index_to_text = {i: text for i, text in enumerate(self.text)}

        # 获取节点数量
        self.n_nodes = len(self.text)

        biotrain_pos_edge_index = make_edge_index_bidirectional(self.data.train_pos_edge_index)

        self.graph = Data(x=torch.arange(self.n_nodes, dtype=torch.long).view(-1, 1),
                          edge_index=biotrain_pos_edge_index)
        # 分类任务使用整个图结构
        # self.graph = Data(
        #    x=torch.arange(self.n_nodes, dtype=torch.long).view(-1, 1),
        #    edge_index=self.data.train_pos_edge_index
        # )

        # 邻居采样配置
        self.neibor = [5,5,5]  # 每层采样的邻居数
        # self.neibor = [3, 3, 3, 3]

        # 获取训练节点索引
        #self.train_indices = self.data.valindex.tolist()
        self.train_indices = self.data.testindex.tolist()

        # 预加载所有节点的tokenization结果
        self.token_cache = {}
        self._preload_all_tokens()

        print(f"数据集加载完成: {self.n_nodes} 个节点, {len(self.train_indices)} 个训练样本")

    def _preload_all_tokens(self):
        """预加载所有节点的tokenization结果"""
        if hasattr(self, 'text') and self.text:
            print("开始预加载所有节点的tokenization结果...")
            start = time.perf_counter()

            # 批量处理所有文本
            all_texts = self.text
            batch_size = 1000  # 调整批量大小以适应内存

            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_encoding = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    truncation='only_first',
                    max_length=self.data_args.max_len,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                # 更新缓存
                for j, text in enumerate(batch_texts):
                    self.token_cache[text] = {
                        'input_ids': batch_encoding['input_ids'][j]
                    }

            end = time.perf_counter()
            print(f"预加载完成，耗时: {end - start:.2f}秒，缓存大小: {len(self.token_cache)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.train_indices)

    def create_one_example(self, texts):
        """从缓存获取tokenization结果，必要时进行批量编码"""
        # 检查是否有未缓存的文本
        uncached_texts = [text for text in texts if text not in self.token_cache]

        if uncached_texts:
            # 批量编码未缓存的文本
            batch_encoding = self.tokenizer.batch_encode_plus(
                uncached_texts,
                truncation='only_first',
                max_length=self.data_args.max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            # 更新缓存
            for j, text in enumerate(uncached_texts):
                self.token_cache[text] = {
                    'input_ids': batch_encoding['input_ids'][j]
                }

        # 从缓存获取所有结果
        return [self.token_cache[text] for text in texts]

    def process_fn(self, node_idx):
        """处理单个样本"""
        start = time.perf_counter()

        # 获取节点标签
        label = self.labels[node_idx].item()

        # 采样节点的邻居子图
        loader = NeighborLoader(
            self.graph,
            num_neighbors=self.neibor,  # 每层采样的邻居数
            batch_size=1,  # 批处理大小
            input_nodes=torch.tensor([node_idx]),  # 目标节点
            # subgraph_type="induced"
        )

        # 获取子图
        subgraph_batch = next(iter(loader))
        # print(subgraph_batch)
        subgraph_batch.x = subgraph_batch.n_id  # 使用原始节点ID
        # print(subgraph_batch)

        # 过滤双向边
        subgraph_batch.edge_index = filter_edges_by_direction(subgraph_batch.edge_index)

        # 获取节点文本
        node_texts = [self.text[i] for i in subgraph_batch.x.tolist()]

        # 文本tokenization
        encoded_texts = self.create_one_example(node_texts)

        end = time.perf_counter()
        # print(f"处理样本 {node_idx} 耗时: {end - start:.4f}秒")
        print({
            'label': label  # 节点的分类标签
        })
        print(f"子图节点ID范围: {subgraph_batch.n_id.min()} 到 {subgraph_batch.n_id.max()}")
        # return error
        return {
            'x1': encoded_texts,  # 节点文本的token表示
            'edge1': subgraph_batch.edge_index,  # 子图的边索引
            'label': label  # 节点的分类标签
        }

    def __getitem__(self, index):
        """获取单个样本"""
        # 获取训练节点索引
        node_idx = self.train_indices[index]

        # 处理样本
        return self.process_fn(node_idx)