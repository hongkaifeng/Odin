from transformers import BertConfig


class GraphBertConfig(BertConfig):
    """GraphBERT 模型的配置类，继承自 BERT 配置"""
    model_type = "graph_bert"

    def __init__(
            self,
            aggnum=3,
            concat_strategy="damc",
            num_sage_layers=3,  # 与aggnum对应，可合并
            **kwargs
    ):
        """
        初始化 GraphBERT 配置

        Args:
            aggnum: GraphSAGE 层数
            concat_strategy: 聚合策略 ('da', 'dac', 'damc')
            num_sage_layers: GraphSAGE 层数（与aggnum一致）
            **kwargs: BERT 原生配置参数
        """
        super().__init__(**kwargs)
        self.aggnum = aggnum
        self.concat_strategy = concat_strategy
        self.num_sage_layers = num_sage_layers