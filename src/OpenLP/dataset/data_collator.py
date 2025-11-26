from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding, DefaultDataCollator, PreTrainedTokenizer, DataCollatorForLanguageModeling

from IPython import embed

@dataclass
class TrainLMCollator(DataCollatorForLanguageModeling):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):
        print("datacollecter")
        # print(features)
        # for i in range(len(features)):
        # ii=[]
        # for j in features[i]['x']:
        #    print(j)
        #    ii.append(self.tokenizer.pad(j,padding='max_length',max_length=self.max_len,return_tensors="pt"))
        # features[i]['x']=ii.copy()
        # print("feature after pad")
        #print(features)
        #3print(type(features))
        xx1 = [f["x1"] for f in features]
        edge1 = [f["edge1"] for f in features]
        xx2 = [f["x2"] for f in features]
        edge2 = [f["edge2"] for f in features]
        # print("x")
        # print(xx)
        # print(len(xx))
        start = time.perf_counter()
        #==========================================================================
        xxx1 = []
        xxx2=[]
        ii = 0
        for i in range(len(xx1)):
            xx_collect1 = self.tokenizer.pad(
                xx1[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            #print("2025.6.10调试")
            #print("mask之前",xx_collect1["input_ids"])

            zerotensor, xx_collect1["labels"] = self.torch_mask_tokens(
                xx_collect1["input_ids"][0].unsqueeze(0), special_tokens_mask=None
            )  # maskkkkkkkk

            #print("mask之后",zerotensor)
            #print("此时的tensor",xx_collect1["input_ids"])
            #print("edge",edge1[ii])
            #xx_collect1["input_ids"][0]
            #import error
            xx_collect1["edge"] = edge1[ii]
            xx_collect2 = self.tokenizer.pad(
                xx2[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            zerotensor, xx_collect2["labels"] = self.torch_mask_tokens(
                xx_collect2["input_ids"][0].unsqueeze(0), special_tokens_mask=None
            )  # maskkkkkkkk
            xx_collect2["edge"] = edge2[ii]
            ii += 1
            # print("xx_collect")
            # print(xx_collect)
            xxx1.append(xx_collect1)
            xxx2.append(xx_collect2)
        #print("xxx")
        #print(xxx)
        #print("collecter")
        #print("Xx1")
        #print(xxx1)
        #print("xxx2")
        #print(xxx2)
        #import error
        #end = time.perf_counter()
        #print("datacollecter耗时",start-end)
        return {"xxx1":xxx1}, \
            {"xxx2":xxx2}



@dataclass
class TrainHnCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):
        #print(features)
        #for i in features:
        #    print(i)
        #    print(114514)
        xx1 = [f["x1"] for f in features]
        xx2 = [f["x2"] for f in features]
        edgee1 = [f["edge1"] for f in features]
        edgee2 = [f["edge2"] for f in features]
        #print(1145141919)
        #print(xx1)
        #print(xx2)
        #print(edgee1)
        #print(edgee2)
        xxx1 = []
        ii = 0
        for i in range(len(xx1)):
            xx_collect1 = self.tokenizer.pad(
                xx1[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            #print(xx_collect1)
            #return error
            #xx_collect1["edge"] = edgee1[ii]
            xx_collect1["edge"] = torch.tensor(edgee1[ii], dtype=torch.long)
            ii+=1
            xxx1.append(xx_collect1)
        #print(888888888888888888888888888888)
        #print(xxx1)
        xxx2=[]
        ii = 0
        #print(len(xx2))
        #print(xx2)
        for i in range(len(xx2)):
            for j in range(len(xx2[i])):
                xx_collect2 = self.tokenizer.pad(
                    xx2[i][j],
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                xx_collect2["edge"] = torch.tensor([[], []], dtype=torch.long)
                ii += 1
                xxx2.append(xx_collect2)
        return {"xxx1": xxx1}, \
            {"xxx2": xxx2,"eval":0}
        #q_mask = [f["query_n_mask"] for f in features]
        #k_mask = [f["key_n_mask"] for f in features]

        #if isinstance(qq[0], list):
        #    qq = sum(qq, [])
        #if isinstance(kk[0], list):
        #    kk = sum(kk, [])
        #if isinstance(q_n[0], list):
        #    q_n = sum(q_n, [])
        #if isinstance(k_n[0], list):
        #    k_n = sum(k_n, [])
        #if isinstance(k_n[0], list):
        #    k_n = sum(k_n, [])
        #if isinstance(k_mask[0], list):
        #    k_mask = sum(k_mask, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask}


@dataclass
class TrainRerankCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):
        #print(features)
        # for i in features:
        #    print(i)
        #    print(114514)
        label_mask = [f["label"] for f in features]
        xx1 = [f["x1"] for f in features]
        xx2 = [f["x2"] for f in features]
        edgee1 = [f["edge1"] for f in features]
        edgee2 = [f["edge2"] for f in features]
        # print(1145141919)
        # print(xx1)
        # print(xx2)
        # print(edgee1)
        # print(edgee2)
        xxx1 = []
        ii = 0
        for i in range(len(xx1)):
            xx_collect1 = self.tokenizer.pad(
                xx1[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            # print(xx_collect1)
            # return error
            # xx_collect1["edge"] = edgee1[ii]
            xx_collect1["edge"] = torch.tensor(edgee1[ii], dtype=torch.long)
            ii += 1
            xxx1.append(xx_collect1)
        # print(888888888888888888888888888888)
        # print(xxx1)
        xxx2 = []
        ii = 0
        #print(len(xx2))
        #print(xx2)
        for i in range(len(xx2)):
            for j in range(len(xx2[i])):
                xx_collect2 = self.tokenizer.pad(
                    xx2[i][j],
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                xx_collect2['input_ids']=xx_collect2['input_ids'].unsqueeze(0)
                xx_collect2['attention_mask'] = xx_collect2['attention_mask'].unsqueeze(0)
                #return error
                xx_collect2["edge"] = torch.tensor([[], []], dtype=torch.long)
                ii += 1
                xxx2.append(xx_collect2)
        label_mask = torch.LongTensor(label_mask)
        #print(label_mask,"这里是evaldatacoll label")
        return {"xxx1": xxx1}, \
            {"xxx2": xxx2,'label_mask':label_mask,"eval":1}
        #================================================================
        print(features)


        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]
        label_mask = [f["label_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_mask[0], list):
            k_mask = sum(k_mask, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)
        label_mask = torch.LongTensor(label_mask)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask, 'label_mask':label_mask}


@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):
        xx1 = [f["x1"] for f in features]
        edge1 = [f["edge1"] for f in features]
        xx2 = [f["x2"] for f in features]
        edge2 = [f["edge2"] for f in features]
        #print("collecter")
        #print(features)
        xxx1 = []
        xxx2 = []
        ii = 0

        for i in range(len(xx1)):
            xx_collect1 = self.tokenizer.pad(
                xx1[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            xx_collect1["edge"] = edge1[ii]
            xx_collect2 = self.tokenizer.pad(
                xx2[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            xx_collect2["edge"] = edge2[ii]
            ii += 1
            # print("xx_collect")
            # print(xx_collect)
            xxx1.append(xx_collect1)
            xxx2.append(xx_collect2)
        #print(xxx1)
        return {"xxx1": xxx1}, \
            {"xxx2": xxx2}



@dataclass

@dataclass
class TrainNCCCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):
        xx1 = [f["x1"] for f in features]
        edge1 = [f["edge1"] for f in features]
        labels = [f["label"] for f in features]
        # print("collecter")
        # print(features)
        xxx1 = []
        ii = 0
        xx_collect1 = {}
        xx_collect2 = {}
        for i in range(len(xx1)):
            xx_collect1 = self.tokenizer.pad(
                xx1[i],
                padding='max_length',
                max_length=self.max_len,
                return_tensors="pt",
            )
            xx_collect1["edge"] = edge1[ii]
            ii += 1
            # print("xx_collect")
            # print(xx_collect)
            xxx1.append(xx_collect1)
        # print(xxx1)
        # import error
        return {"xxx1": xxx1}, labels




@dataclass
class EncodeCollator(DefaultDataCollator):
    def __call__(self, features):
        #print(features)
        #print("EncodeCollator")
        #return error
        text_ids = [x["text_id"] for x in features]
        x1 = [x["x1"] for x in features]

        #print(x)

        #collated_features = super().__call__(center_inputs)

        #if 'neighbor_input' in features[0]:
        #    neighbor_inputs = [x["neighbor_input"] for x in features]
        #    masks = [x["mask"] for x in features]
        #    n_collated_features = super().__call__(neighbor_inputs)
        #    n_mask = torch.LongTensor(masks)
                        
        #    return text_ids, {'center_input': collated_features, 'neighbor_input': n_collated_features, 'mask': n_mask}
        
        return text_ids, {'x1': x1}
