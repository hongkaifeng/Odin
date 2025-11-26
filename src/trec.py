import pytrec_eval
import pytrec_eval
import argparse
# 加载 qrel 和 run 文件
qrel = {}
run = {}
parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--truth_path", required=True, type=str)


args = parser.parse_args()
#with open("D:/mymodel/newdatatest/test.truth.trec", "r") as f:
with open(args.truth_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        query_id, _, doc_id, rel = parts
        if query_id not in qrel:
            qrel[query_id] = {}
        qrel[query_id][doc_id] = int(rel)
#with open("D:/GraphFormers-main/end/bert-retrievetoken/retrieve", "r") as f:
with open("D:/mymodel/truedataset/final/retrieve/token/retrieve", "r") as f:
    for line in f:
        parts = line.strip().split()
        query_id, _, doc_id, rank, score, _ = parts
        if query_id not in run:
            run[query_id] = {}
        run[query_id][doc_id] = float(score)

# 初始化评估器
evaluator = pytrec_eval.RelevanceEvaluator(
    qrel,
    {"map", "ndcg", "recip_rank", "recall_10"}  # 选择需要计算的指标
)
#evaluator = pytrec_eval.RelevanceEvaluator(
#    qrel,
#    {"recall_10"}  # 选择需要计算的指标
#)
# 进行评估
results = evaluator.evaluate(run)

# 打印结果
#for query_id, metrics in results.items():
#    print(f"Query {query_id}:")
#    for metric, value in metrics.items():
#        print(f"  {metric}: {value:.4f}")
metric_sums = {metric: 0.0 for metric in ["map", "ndcg", "recip_rank", "recall_10"]}
#metric_sums = {metric: 0.0 for metric in ["recall_10"]}
query_count = len(results)

for query_id, metrics in results.items():
    for metric, value in metrics.items():
        metric_sums[metric] += value

average_metrics = {metric: total / query_count for metric, total in metric_sums.items()}
strr=""
# 打印平均值
print("Average Metrics:")
for metric, avg_value in average_metrics.items():
    strr=strr+f"  {metric}: {avg_value:.4f}"
result_file_path = "ending.txt"
# 构造要写入的内容（包含累加值、平均值和批次数量，便于后续查看）
write_content = "检索" +strr+"\n"
# 以追加模式（'a'）打开文件，新内容自动写入下一行，编码指定为utf-8避免乱码
with open(result_file_path, 'a', encoding='utf-8') as f:
    f.write(write_content)
