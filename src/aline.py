import subprocess
import os
import sys
from typing import List


def run_commands(commands: List[str], use_shell: bool = True, encoding: str = "utf-8") -> None:
    """
    在终端连续执行一系列命令

    Args:
        commands: 命令列表，每个元素为一条终端命令（字符串）
        use_shell: 是否使用系统 Shell 执行命令（True 支持管道、通配符等高级语法）
        encoding: 命令输出的编码格式（默认 utf-8，Windows 可尝试 "gbk"）
    """
    # 记录当前工作目录（用于处理 cd 命令，避免子进程目录不继承问题）
    current_dir = os.getcwd()

    for idx, cmd in enumerate(commands, 1):
        cmd = cmd.strip()  # 去除命令前后空格
        if not cmd:  # 跳过空命令
            continue

        print(f"\n=== 开始执行第 {idx} 条命令：{cmd} ===")
        try:
            # 特殊处理 cd 命令（subprocess 执行 cd 仅影响子进程，需手动更新当前目录）
            if cmd.lower().startswith("cd "):
                # 提取目标目录（处理 "cd ./test" 或 "cd /home/user" 等格式）
                target_dir = cmd.split("cd ", 1)[1].strip()
                # 处理相对路径（基于当前工作目录）
                target_dir = os.path.abspath(os.path.join(current_dir, target_dir))

                # 验证目录是否存在
                if not os.path.exists(target_dir):
                    raise FileNotFoundError(f"目录不存在：{target_dir}")
                if not os.path.isdir(target_dir):
                    raise NotADirectoryError(f"不是目录：{target_dir}")

                # 更新当前工作目录
                os.chdir(target_dir)
                current_dir = target_dir
                print(f"✅ 目录切换成功，当前目录：{current_dir}")
                continue

            # 执行普通命令（非 cd）
            result = subprocess.run(
                cmd,
                shell=use_shell,
                cwd=current_dir,  # 基于当前工作目录执行命令
                stdout=subprocess.PIPE,  # 捕获标准输出
                stderr=subprocess.PIPE,  # 捕获标准错误
                encoding=encoding,
                timeout=None  # 命令超时时间（秒），可根据需求调整
            )

            # 打印命令输出（标准输出 + 标准错误）
            if result.stdout:
                print(f"📤 命令输出：\n{result.stdout}")
            if result.stderr:
                print(f"⚠️  命令警告/错误：\n{result.stderr}")

            # 检查命令是否执行成功（返回码为 0 表示成功）
            result.check_returncode()
            print(f"✅ 第 {idx} 条命令执行成功")

        except subprocess.TimeoutExpired:
            print(f"❌ 第 {idx} 条命令超时（超过 {300} 秒）")
            sys.exit(1)  # 超时可选择退出或继续，此处默认退出
        except subprocess.CalledProcessError as e:
            print(f"❌ 第 {idx} 条命令执行失败（返回码：{e.returncode}）")
            print(f"   错误详情：{e.stderr}")
            sys.exit(1)  # 命令失败可选择退出或继续，此处默认退出
        except Exception as e:
            print(f"❌ 第 {idx} 条命令处理异常：{str(e)}")
            sys.exit(1)

    print("\n🎉 所有命令均执行完成！")


if __name__ == "__main__":
    # --------------------------
    # 核心配置：替换为你的命令列表
    # --------------------------
    # 示例 1：Linux/macOS 环境（更新包 + 安装依赖 + 查看目录）
    # commands = [
    #     "sudo apt update && sudo apt upgrade -y",  # Ubuntu 更新系统
    #     "pip install numpy pandas",  # 安装 Python 依赖
    #     "cd ./project",  # 切换到项目目录
    #     "ls -l",  # 查看目录文件详情
    #     "python main.py --epochs 10"  # 运行 Python 脚本
    # ]

    # 示例 2：Windows 环境（查看目录 + 安装依赖 + 运行脚本）
    commands = [
        "dir",  # 查看当前目录文件（Windows CMD）
        "pip install requests",  # 安装 Python 依赖
        "cd ./test_data",  # 切换到数据目录
        "dir /s",  # 递归查看目录文件（Windows CMD）
        "python process_data.py"  # 运行数据处理脚本
    ]
    from pathlib import Path

    randomseed=[42]

    datasets=["obgn3000","obgn6000","obgn10000","cora","citeseer","GLcora"]
    #datasets = ["cora","citeseer"]
    #datasets = ["ARXIV2023", "obgn-produce"]
    #datasets=["obgn6000","obgn10000","ARXIV2023"]
    datasets=["cora"]

    labelnum={"obgn3000":9,"obgn6000":9,"obgn10000":9,"GLcora":7,"cora":7,"citeseer":6,"ARXIV2023":9,"obgn-produce":11}
    linkbias=[]
    classbias=[]
    lmdatasets=["obgn3000","obgn6000","obgn10000"]
    lmdatasets =["obgn3000"]
    #lmdatasets = ["obgn6000","obgn10000","ARXIV2023"]
    retribias=[]
    rerankbia=[]
    # --------------------------
    # 执行命令（根据系统调整编码）
    # --------------------------
    # Windows 若出现乱码，可将 encoding 改为 "gbk"
    # run_commands(commands, encoding="gbk")"allenai/scibert_scivocab_uncased"
    modelname="TYPEV"

    #modelpath="allenai/scibert_scivocab_uncased"
    datapath="D:\mymodel\8-5maindataset"
    #run_commands(commands)
    for i in datasets:
        for j in randomseed:
            if i in linkbias:
                break
            linkcom1=f"python -m OpenLP.driver.traineval  --output_dir D:/mymodel/linkp_end  --model_name_or_path D:/mymodel/formal/{i}/{modelname}  --model_type graphformer --do_train  --save_steps 160  --eval_steps 160  --logging_steps 160 --train_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --eval_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --fp16  --per_device_train_batch_size 4  --per_device_eval_batch_size 4 --learning_rate 1e-5  --max_len 32  --num_train_epochs 100  --logging_dir D:/Patton-main/logs/sports/link_prediction  --evaluation_strategy steps --remove_unused_columns False --overwrite_output_dir True --report_to tensorboard  --seed {j}"
            linktest1=f"python -m OpenLP.driver.mytest  --output_dir D:/Patton-main/data/sports/link_prediction/tmp  --model_name_or_path D:\mymodel\linkp_end  --tokenizer_name D:/mymodel/formal/{i}/{modelname} --model_type graphformer --do_eval  --train_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --eval_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --fp16  --per_device_eval_batch_size 4 --max_len 32  --evaluation_strategy steps --remove_unused_columns False --overwrite_output_dir True --dataloader_num_workers 0  --seed {j}"
            run_commands([linkcom1,linktest1], encoding="gbk")

        for j in randomseed:
            if i in classbias:
                break
            numm = labelnum[i]
            classcom1 = f"python -m OpenLP.driver.trainclasseval  --output_dir D:/mymodel/classend  --model_name_or_path D:/mymodel/formal/{i}/{modelname}  --tokenizer_name D:/mymodel/formal/{i}/{modelname}  --model_type graphformer  --do_train  --save_steps {numm * 20}  --eval_steps {numm * 20}  --logging_steps {numm * 20} --train_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --eval_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --class_num {numm}  --fp16  --per_device_train_batch_size 4  --per_device_eval_batch_size 4  --learning_rate 1e-5  --max_len 32  --num_train_epochs 50  --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  --evaluation_strategy steps  --remove_unused_columns False  --overwrite_output_dir True  --report_to tensorboard  --seed {j}  --labeltxt D:\mymodel\8-5maindataset\\{i}\labels.txt  --labelnum {numm}"
            classtest1 = f"python -m OpenLP.driver.mytest_class  --output_dir D:/Patton-main/data/sports/class_task/tmp  --model_name_or_path D:\mymodel\classend  --tokenizer_name D:\mymodel\classend  --model_type graphformer  --do_eval  --train_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --eval_path D:/mymodel/8-5maindataset/{i}/{i}.pt  --fp16  --per_device_eval_batch_size 4  --max_len 32   --evaluation_strategy steps  --remove_unused_columns False  --overwrite_output_dir True  --dataloader_num_workers 0  --seed {j}  --labeltxt D:\mymodel\8-5maindataset\\{i}\labels.txt  --labelnum {numm}"
            run_commands([classcom1,classtest1], encoding="gbk")
            #run_commands([classtest1], encoding="gbk")

    for i in lmdatasets:
        for j in randomseed:
            if i in retribias:
                break
            file_path = Path("D:/mymodel/truedataset/final/retrieve/token/embeddings.query.rank.0")  # 替换为你的文件路径
            try:
                if file_path.exists():
                    file_path.unlink()  # 删除文件
                    print(f"✅ 文件 {file_path} 已成功删除")
                else:
                    print(f"⚠️ 文件 {file_path} 不存在，无需删除")
            except FileNotFoundError:
                print(f"❌ 错误：文件 {file_path} 不存在")
            file_path = Path("D:/mymodel/truedataset/final/retrieve/token/sports.embeddings.corpus.rank.0")  # 替换为你的文件路径
            try:
                if file_path.exists():
                    file_path.unlink()  # 删除文件
                    print(f"✅ 文件 {file_path} 已成功删除")
                else:
                    print(f"⚠️ 文件 {file_path} 不存在，无需删除")
            except FileNotFoundError:
                print(f"❌ 错误：文件 {file_path} 不存在")
            file_path = Path("D:/mymodel/truedataset/final/retrieve/token/sports_sports_retrieval_dict.pkl")  # 替换为你的文件路径
            try:
                if file_path.exists():
                    file_path.unlink()  # 删除文件
                    print(f"✅ 文件 {file_path} 已成功删除")
                else:
                    print(f"⚠️ 文件 {file_path} 不存在，无需删除")
            except FileNotFoundError:
                print(f"❌ 错误：文件 {file_path} 不存在")
            retri1=f"python -m OpenLP.driver.trainrerankeval  --output_dir D:/mymodel/retrieveend  --model_name_or_path D:/mymodel/formal/{i}/{modelname}   --tokenizer_name D:/mymodel/formal/{i}/{modelname}  --model_type graphformer  --do_train  --hn_num 4  --save_steps 80  --eval_steps 80  --logging_steps 80  --train_path D:/mymodel/8-5maindataset/{i}/retrieve/train16.jsonl  --eval_path D:/mymodel/8-5maindataset/{i}/retrieve/val.jsonl  --fp16  --per_device_train_batch_size 4   --per_device_eval_batch_size 4  --learning_rate 1e-5     --max_len 32   --num_train_epochs 100   --logging_dir D:/Patton-main/logs/sports/link_prediction     --evaluation_strategy steps  --remove_unused_columns False  --overwrite_output_dir True  --report_to tensorboard  --seed {j}"
            retri2=f"python -m OpenLP.driver.infer  --output_dir D:/mymodel/truedataset/final/retrieve/token  --model_name_or_path D:/mymodel/retrieveend  --tokenizer_name D:/mymodel/formal/{i}/{modelname}  --model_type graphformer  --per_device_eval_batch_size 4  --corpus_path D:/mymodel/8-5maindataset/{i}/retrieve/documents.txt   --doc_column_names id,text  --max_len 32  --retrieve_domain sports  --dataloader_num_workers 0"
            retri3=f"python -m OpenLP.driver.search  --output_dir D:/mymodel/truedataset/final/retrieve/token  --model_name_or_path D:/mymodel/retrieveend  --tokenizer_name D:/mymodel/formal/{i}/{modelname}  --model_type graphformer  --per_device_eval_batch_size 4  --corpus_path D:/mymodel/8-5maindataset/{i}/retrieve/documents.txt  --query_path D:/mymodel/8-5maindataset/{i}/retrieve/test.node.text.jsonl  --query_column_names id,text  --max_len 32  --save_trec True  --retrieve_domain sports  --source_domain sports  --save_path D:/mymodel/truedataset/final/retrieve/token/retrieve  --dataloader_num_workers 0"
            retri4=f"python -m trec  --truth_path D:/mymodel/8-5maindataset/{i}/retrieve/test.truth.trec"
            run_commands([retri1,retri2,retri3,retri4], encoding="gbk")
            #import error
        for j in randomseed:
            if i in rerankbia:
                break
            rerank1=f"python -m OpenLP.driver.trainrerankeval  --output_dir D:/mymodel/rerank  --model_name_or_path D:/mymodel/formal/{i}/{modelname}   --tokenizer_name D:/mymodel/formal/{i}/{modelname}  --model_type graphformer  --do_train  --hn_num 4  --save_steps 48  --eval_steps 48  --logging_steps 48  --train_path D:/mymodel/8-5maindataset/{i}/rank/train32.rerank.jsonl  --eval_path D:/mymodel/8-5maindataset/{i}/rank/val.rerank.jsonl  --fp16  --per_device_train_batch_size 4   --per_device_eval_batch_size 4  --learning_rate 1e-5     --max_len 32   --num_train_epochs 30   --logging_dir D:/Patton-main/logs/sports/link_prediction     --evaluation_strategy steps  --remove_unused_columns False  --overwrite_output_dir True  --report_to tensorboard  --seed {j}"
            rerank2=f"python -m OpenLP.driver.mytest_rerank  --output_dir $TEST_DIR/tmp  --model_name_or_path D:/mymodel/rerank  --tokenizer_name D:/mymodel/formal/{i}/{modelname}  --model_type graphformer  --do_eval  --pos_rerank_num 5  --neg_rerank_num 45  --train_path D:/mymodel/8-5maindataset/{i}/rank/test.rerank.jsonl  --eval_path D:/mymodel/8-5maindataset/{i}/rank/test.rerank.jsonl  --fp16  --per_device_eval_batch_size 4  --max_len 32  --evaluation_strategy steps  --remove_unused_columns False  --overwrite_output_dir True  --dataloader_num_workers 0 --seed {j}"
            run_commands([rerank1, rerank2], encoding="gbk")





