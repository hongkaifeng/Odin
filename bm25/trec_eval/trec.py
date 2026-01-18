import pytrec_eval
import pytrec_eval

# 加载 qrel 和 run 文件
qrel = {}
run = {}

#with open("D:/mymodel/newdatatest/test.truth.trec", "r") as f:
with open("D:/mymodel/truedataset/final/retrieve/test.truth.trec", "r") as f:
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

# 打印平均值
print("Average Metrics:")
for metric, avg_value in average_metrics.items():
    print(f"  {metric}: {avg_value:.4f}")
