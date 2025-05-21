import pandas as pd
import numpy as np
import torch
import emoji
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.optim import AdamW
import matplotlib.pyplot as plt
import warnings

# 屏蔽所有警告信息
warnings.filterwarnings("ignore")

def replace_mask(text):
    return text.replace("[MASK]", "<mask>")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(data):
    sents = ["Related:" + ', '.join(i["emoji"]) + '.' + i["text_a"] for i in data]
    labels = [i['label_text'] for i in data]
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = sents,
        truncation = True,
        padding='max_length',
        max_length=500,
        return_length=True,
        return_tensors='pt'
    )
    input_ids = data['input_ids']
    att_post = data["attention_mask"]
    attention_mask = []
    max_length = 20
    for i in sents:
        # 获取句子中所有[MASK]的索引
        mask_indices = [i for i, x in enumerate(tokenizer(i)["input_ids"]) if x == tokenizer.mask_token_id]
        # 如果mask_indices长度不足3，则用0补齐
        mask_indices += [0] * (max_length - len(mask_indices))
        # 添加到attention_mask中
        attention_mask.append(mask_indices)
    attention_mask = np.array(attention_mask).astype(int)
    attention_mask = torch.from_numpy(attention_mask)

    alabels = []
    for label in labels:
        label_ids = [tokenizer.convert_tokens_to_ids(l) for l in label]
        # 不足长度时用0补齐
        label_ids.extend([0] * (max_length - len(label_ids)))
        alabels.append(label_ids)
    alabels = np.array(alabels).astype(int)
    alabels = torch.from_numpy(alabels)
    return input_ids, att_post, attention_mask, alabels

def train_epoch(model, optimizer, train_loader, lossFunc, use_cuda):
    model.train()
    j = 0
    tot_loss = 0
    tot_len = 0
    for index, (input_ids, att_post, attention_mask, alabels) in enumerate(tqdm(train_loader,desc=f"Training...")):
        
        if use_cuda:
            input_ids = input_ids.cuda()
            att_post = att_post.cuda()
            attention_mask = attention_mask.cuda()
            alabels = alabels.cuda()

        outputs = model(input_ids,attention_mask=att_post)
        predictions = outputs[0]

        loss = 0
        for prediction, alabel, mask_pos in zip(predictions, alabels, attention_mask):
            # 将预测值展开为二维张量
            prediction = torch.unsqueeze(prediction, 0)
            # 对预测值排序并提取对应的词汇
            values, indices = torch.sort(prediction[0, mask_pos], descending=True)
            get_list = [k.item() for k in indices[:, 0]]
            # 计算正确预测数量
            j += sum(1 for pdec, ture in zip(get_list, alabel) if pdec == ture)
            tot_len += len(alabel)
            pre = prediction[0, mask_pos]
            # 提取预测值和损失
            for k in range(0,len(alabel)):
                pred = pre[k,alabel[k]].to(torch.device("cuda:0"))
                top = torch.tensor(values[k, 0]).to(torch.device("cuda:0"))
                loss += lossFunc(top, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss +=  loss.item()
    acc = j / tot_len
    return tot_loss / tot_len, acc


def dev_evaluate(model, loader, lossFunc, use_cuda):
    model.eval()
    with torch.no_grad():
        j = 0
        tot_len = 0
        for index, (input_ids, att_post, attention_mask, alabels) in enumerate(tqdm(loader, desc=f"Deving...")):
            if use_cuda:
                input_ids = input_ids.cuda()
                att_post = att_post.cuda()
                attention_mask = attention_mask.cuda()
                alabels = alabels.cuda()

            outputs = model(input_ids,attention_mask=att_post)
            predictions = outputs[0]

            for prediction, alabel, mask_pos in zip(predictions, alabels, attention_mask):
                # 将预测值展开为二维张量
                prediction = torch.unsqueeze(prediction, 0)
                # 对预测值排序并提取对应的词汇
                values, indices = torch.sort(prediction[0, mask_pos], descending=True)
                get_list = [k.item() for k in indices[:, 0]]
                tot_len += len(alabel)
                # 计算正确预测数量
                j += sum(1 for pdec, ture in zip(get_list, alabel) if pdec == ture)
                
        acc = j / tot_len
        return acc

def test_evaluate(model, loader, lossFunc, use_cuda, p):
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        answer_predictions = 0
        total_samples = 0
        allpresion = 0
        tot_len = 0
        for index, (input_ids, att_post, attention_mask, alabels) in enumerate(tqdm(loader, desc=f"Testing...")):
            if use_cuda:
                input_ids = input_ids.cuda()
                att_post = att_post.cuda()
                attention_mask = attention_mask.cuda()
                alabels = alabels.cuda()

            outputs = model(input_ids, attention_mask=att_post)
            predictions = outputs[0]

            for prediction, alabel, mask_pos in zip(predictions, alabels, attention_mask):
                # 将预测值展开为二维张量
                prediction = torch.unsqueeze(prediction, 0)
                # 对预测值排序并提取对应的词汇
                values, indices = torch.sort(prediction[0, mask_pos], descending=True)
                get_list = [k.item() for k in indices[:, 0]]
                tot_len += len(alabel)
                # 计算正确预测数量
                correct_predictions += sum(1 for pdec, ture in zip(get_list, alabel) if pdec == ture)
                # 将预测结果和索引存储到字典中
                # 转置 values 和 indices，以便高效访

                # 统计正确预测数量
                for i in range(len(alabel)):
                    # 获取当前句子的预测结果和标签
                    pred_values = values[i, :]
                    pred_indices = indices[i, :]
                    label = alabel[i]
                    # 找到预测结果中标签的位置
                    label_index = torch.where(pred_indices == label)[0]
                    if len(label_index) > 0:
                        label_index = label_index[0].item()
                        # 统计正确预测数量
                        answer_predictions += 1 if pred_values[label_index] >= p else 0
                        allpresion += torch.sum(pred_values > p).item() if pred_values[label_index] <= p else 0
            total_samples += len(input_ids)
        print(allpresion)
        accuracy = correct_predictions / tot_len
        accuracy_all = answer_predictions / tot_len
        precision = answer_predictions / allpresion
        fake_precision = correct_predictions / allpresion
        return accuracy, accuracy_all, precision, fake_precision

def train_and_evaluate(model, tokenizer, train_loader, dev_loader, test_loader, lossFunc, optimizer, num_epochs=20, p_list=[4, 4.5, 5, 5.5, 6], use_cuda=True, batch_size=8, learning_rate=1e-5, ways="Dataset_ipa", language="english", data_name="SST-2", sim=0.4, w_the=3, expock = 5):
    histary = []
    dev_list = []
    if use_cuda:
        model = model.cuda()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, lossFunc, use_cuda)
        print("Epoch {}, average loss: {}, train acc: {}".format(epoch, train_loss, train_acc), flush=True)
        histary.append([epoch, train_loss.tolist(), train_acc])

        dev_acc = dev_evaluate(model, dev_loader, lossFunc, use_cuda)
        dev_list.append([epoch,dev_acc])
        print("Dev acc: {}".format(dev_acc), flush=True)


    result_df = pd.DataFrame(histary, columns=['epoch', 'train_loss', 'train_acc'])
    result_df.to_csv(f'./results_EPLM/{language}/{data_name}/train_{data_name}_{p_list}_sim={sim}_w_the={w_the}_expock={expock}.txt', sep='|', encoding='UTF-8', index=False)

    result_df = pd.DataFrame(dev_list, columns=['epoch', 'dev_acc'])
    result_df.to_csv(f'./results_EPLM/{language}/{data_name}/dev_{data_name}_{p_list}_sim={sim}_w_the={w_the}_expock={expock}.txt', sep='|', encoding='UTF-8', index=False)

    # plt.xlabel('epoch') 
    # plt.ylabel('average loss') 
    # plt.plot(result_df['epoch'] + 1, result_df['train_loss'])
    # plt.show()

    acc_list = []
    model.eval()
    with torch.no_grad():
        for p in p_list:
            acc, accuracy_all,precision, fake_precision = test_evaluate(model, test_loader, lossFunc, use_cuda, p)
            acc_list.append([p, acc,accuracy_all, precision, fake_precision])
    result_df = pd.DataFrame(acc_list, columns=['p', 'Acc', "accuracy_all",'precision',"fake_precision"])
    result_df.to_csv(f'./results_EPLM/{language}/{data_name}/test_{data_name}_{p_list}_sim={sim}_w_the={w_the}_expock={expock}.txt',sep='|', encoding='UTF-8', index=False)

# 超参数设置

ways = "Dataset_ipa"
language = "english"
data_name = "SST-2"
# model_path = "../bert-base-cased"
# model_path = "../bert-base-chinese"
# model_path = "../bert-large-cased"
# model_path = "../bert-large-chinese"
model_path = "../roberta-base"
# model_path = "../chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
sim = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
w_the = 0.4
p_list = [4, 4.5, 5, 5.5, 6]
batch_size = 8
num_epochs = 0
learning_rate = 1e-8

# 加载数据集并处理成PyTorch Dataset
data_class = ["test", "dev", "train"]
# 定义损失函数和优化器
lossFunc = torch.nn.L1Loss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 遍历 sim 参数列表
for sim_value in sim:
    print(f"Training for sim = {sim_value}")
    
    # 加载数据集并设置 sim 参数
    datasets = {}
    if model_path == "../roberta-base":
        for class_name in data_class:
            data = pd.read_csv(f"./dataset/{ways}/{language}/{data_name}_{class_name}_{sim_value}_{w_the}.txt", sep='|')
            data['emoji_mean'] = data['emoji_mean'].apply(eval)
            data['word'] = data['word'].apply(eval)
            datasets[class_name] = MyDataset([{'text_a': f"Related:{replace_mask(row['text'])}",
                                                'emoji': row['emoji_mean'],
                                                'label': row['lables'],
                                                'label_text': row['word']}
                                                for _, row in data.iterrows()])
    else:
        for class_name in data_class:
            data = pd.read_csv(f"./dataset/{ways}/{language}/{data_name}_{class_name}_{sim_value}_{w_the}.txt", sep='|')
            data['emoji_mean'] = data['emoji_mean'].apply(eval)
            data['word'] = data['word'].apply(eval)
            datasets[class_name] = MyDataset([{'text_a': f"Related:{row['text']}",
                                                'emoji': row['emoji_mean'],
                                                'label': row['lables'],
                                                'label_text': row['word']}
                                                for _, row in data.iterrows()])
        
    # 加载数据集的 DataLoader
    train_loader = DataLoader(dataset=datasets["train"], collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dataset=datasets["dev"], collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=datasets["test"], collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 内部进行五次训练
    for i in range(5):
        print(f"Training iteration {i+1}")
        
        # 每次迭代前重新加载模型和优化器
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # 训练和评估
        train_and_evaluate(model, tokenizer, train_loader, dev_loader, test_loader, lossFunc, optimizer, num_epochs, p_list, use_cuda=True, batch_size=batch_size, learning_rate=learning_rate, ways=ways, language=language, data_name=data_name, sim=sim_value, w_the=w_the,expock = i)