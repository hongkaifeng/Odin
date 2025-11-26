import torch

#file_path = "obgn-3000.pt"  # 请替换为实际文件路径citeseer_random_sbert "D:\mymodel\8-5maindataset\wikics_fixed_sbert.pt"
#file_path = "citeseer_random_sbert.pt"
#file_path = "raw_cora_data.pt"
file_path ="test4.pt"
data = torch.load(file_path)
print(data)
listt=[]
for i in data.y:
    if i not in listt:
        listt.append(i)
#print(listt)
#print(list(data.label))
#print(data.text)
#for i in data.text:
#    print(" ")
#    print(i)