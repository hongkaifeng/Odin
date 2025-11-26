import torch
from torch_geometric.data import Data
from torch_geometric import transforms as T
from torch_geometric.utils import is_undirected, negative_sampling

# 加载数据
file_path = "test.pt"
try:
    data = torch.load(file_path, weights_only=True)
except:
    data = torch.load(file_path, weights_only=False)

# 保存原始数据属性（排除边相关信息）
original_attrs = {}
for key in data.keys():
    if key not in ['edge_index', 'edge_attr']:
        original_attrs[key] = data[key]

# 定义链路预测划分参数
link_split = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=False,
    add_negative_train_samples=True,
    split_labels=True,
)

# 执行边划分
train_data, val_data, test_data = link_split(data)

# 提取正边索引
train_pos_edge_index = train_data.pos_edge_label_index
val_pos_edge_index = val_data.pos_edge_label_index
test_pos_edge_index = test_data.pos_edge_label_index

# 提取或生成负边索引
train_neg_edge_index = (train_data.neg_edge_label_index
                        if hasattr(train_data, 'neg_edge_label_index')
                        else negative_sampling(train_pos_edge_index,
                                               num_nodes=data.num_nodes,
                                               num_neg_samples=train_pos_edge_index.size(1)))

val_neg_edge_index = (val_data.neg_edge_label_index
                      if hasattr(val_data, 'neg_edge_label_index')
                      else negative_sampling(val_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=val_pos_edge_index.size(1)))

test_neg_edge_index = (test_data.neg_edge_label_index
                       if hasattr(test_data, 'neg_edge_label_index')
                       else negative_sampling(test_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=test_pos_edge_index.size(1)))

# 构建最终的PyG Data对象
result_data = Data(
    **original_attrs,
    train_pos_edge_index=train_pos_edge_index,
    val_pos_edge_index=val_pos_edge_index,
    test_pos_edge_index=test_pos_edge_index,
    train_neg_edge_index=train_neg_edge_index,
    val_neg_edge_index=val_neg_edge_index,
    test_neg_edge_index=test_neg_edge_index
)

# 输出验证信息
print("划分后的数据结构:")
print(result_data)
print("\n边数量统计:")
print(f"训练正边: {train_pos_edge_index.shape[1]}")
print(f"验证正边: {val_pos_edge_index.shape[1]}")
print(f"测试正边: {test_pos_edge_index.shape[1]}")
print(f"总正边: {train_pos_edge_index.shape[1] + val_pos_edge_index.shape[1] + test_pos_edge_index.shape[1]}")
print(f"原始边数: {data.edge_index.shape[1]}")

# 保存结果
torch.save(result_data, 'test2.pt')


#Data(y=[3000], text=[3000], num_nodes=3000, train_pos_edge_index=[2, 6352], val_pos_edge_index=[2, 794], test_pos_edge_index=[2, 794], train_neg_edge_index=[2, 6352], val_neg_edge_index=[2, 794], test_neg_edge_index=[2, 794])
#下一步：细类合并