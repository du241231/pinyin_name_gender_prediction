import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import copy
import random
import os
from pinyinsplit import PinyinSplit
import torch.nn.functional as F
from pypinyin import lazy_pinyin, Style

pys = PinyinSplit()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random_seed = 42


def seed_everything(seed=random_seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication, self).__init__()
        self.model_name = 'pretrain_model/chinese-bert-wwm'
        self.model_name_pre = BertModel.from_pretrained(self.model_name)
        self.model_name_gen = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc_pinyin_gender = nn.Linear(768, 2)
        self.fc_name_gender = nn.Linear(768, 2)
        self.fc_name_gen = nn.Linear(768, 21128)
        self.fc_tone = nn.Linear(768, 5)

    def forward(self, pinyin_list, name_list, pinyin_name, targets, train=True):
        loss_ce = nn.CrossEntropyLoss()
        loss_kl = nn.KLDivLoss(reduction='batchmean')
        loss_l2 = nn.MSELoss()

        pinyin_ids = []
        token_type_ids = []
        target_marks = []
        for pinyin in pinyin_list:
            yin_list = pinyin.split()
            if len(yin_list) == 2:
                pinyin_id_1 = self.tokenizer.encode(yin_list[0])
                pinyin_id_2 = self.tokenizer.encode(yin_list[1])
                pinyin_id = pinyin_id_1 + pinyin_id_2[1:]
                token_type_id = [0] * len(pinyin_id_1) + [1] * (len(pinyin_id_2) - 1)
                target_mark = [len(pinyin_id_1), len(pinyin_id)]
                target_marks.append(target_mark)
            else:
                pinyin_id = self.tokenizer.encode(yin_list[0])
                token_type_id = [0] * len(pinyin_id)
                target_mark = [len(pinyin_id), 0]
                target_marks.append(target_mark)
            while len(pinyin_id) < 16:
                pinyin_id.append(0)
                token_type_id.append(0)
            pinyin_ids.append(pinyin_id)
            token_type_ids.append(token_type_id)
        hidden_output = self.model_name_gen(input_ids=torch.tensor(pinyin_ids).to(device),
                                            token_type_ids=torch.tensor(token_type_ids).to(device)
                                            )
        pinyin_gender_feature = hidden_output[1]
        pinyin_gender_output = self.fc_pinyin_gender(pinyin_gender_feature)

        if train:
            loss = loss_ce(pinyin_gender_output, targets)

            target_ids = []
            for name in name_list:
                tone = lazy_pinyin(name, style=Style.TONE3)
                if len(name) == 2:
                    name_id = self.tokenizer.encode(name)
                    target_id = [name_id[1], name_id[2]]
                else:
                    name_id = self.tokenizer.encode(name)
                    target_id = [name_id[1], 0]
                target_ids.append(target_id)

            name_gen_hidden_output = self.fc_name_gen(hidden_output[0])
            name_gen_output = []
            for marks, feature in zip(target_marks, name_gen_hidden_output):
                name_feature = []
                for mark in marks:
                    if mark == 0:
                        name_feature.append(torch.zeros(21128).to(device))
                    else:
                        name_feature.append(feature[mark])
                name_gen_output.append(torch.stack(name_feature).to(device))
            name_gen_output = torch.stack(name_gen_output).to(device)
            loss_gen = loss_ce(name_gen_output.transpose(1, 2), torch.tensor(target_ids).to(device))

            batch_tokenized_name = self.tokenizer.batch_encode_plus(name_list, add_special_tokens=True, max_length=16, truncation=True, padding="max_length")
            input_ids_name = torch.tensor(batch_tokenized_name['input_ids']).to(device)
            hiden_outputs_name = self.model_name_pre(input_ids_name)[1]
            outputs_name = self.fc_name_gender(hiden_outputs_name)
            loss_name = loss_ce(outputs_name, targets)

            teacher = F.softmax(outputs_name, dim=1)
            student = F.log_softmax(pinyin_gender_output, dim=1)
            loss_d = loss_kl(student, teacher)

            loss_f = loss_l2(pinyin_gender_feature, hiden_outputs_name)

            final_loss = loss +loss_name +loss_d +loss_f +loss_gen
        else:
            loss = loss_ce(pinyin_gender_output, targets)
            final_loss = loss

        return pinyin_gender_output, final_loss


if __name__ == '__main__':
    train_data_df = pd.read_csv('data/9800_train_1.csv')
    train_data_df = train_data_df[['label', 'first_name', 'first_name_pinyin']]
    test_data_df = pd.read_csv('data/total_9800.csv')
    test_data_df = test_data_df[['label', 'first_name_pinyin']]
    dev_data_df = pd.read_csv('data/9800_val_1.csv')
    dev_data_df = dev_data_df[['label', 'first_name_pinyin']]

    train_pinyin = np.array(train_data_df['first_name_pinyin'].tolist())
    test_pinyin = np.array(test_data_df['first_name_pinyin'].tolist())
    dev_pinyin = np.array(dev_data_df['first_name_pinyin'].tolist())

    train_tag = list(train_data_df['label'].tolist())
    test_tag = list(test_data_df['label'].tolist())
    dev_tag = list(dev_data_df['label'].tolist())

    train_name = np.array(train_data_df['first_name'].tolist())

    
    def split_pinyin(pinyin_list):
        pinyin_split = []
        for pinyin in pinyin_list:
            try:
                temp = pys.split(pinyin)
                temp = temp[0]
                if len(temp) == 2:
                    if temp[0] == 'eng' or temp[1] == 'eng':
                        temp[1] = temp[0][-1] + temp[1]
                        temp[0] = temp[0][:-1]
                temp = ' '.join(temp)
                if 'biang ' in temp or 'vn ' in temp or 'diang ' in temp:
                    temp = temp.split()
                    temp[1] = temp[0][-1] + temp[1]
                    temp[0] = temp[0][:-1]
                    temp = ' '.join(temp)
                if 'ue' in temp:
                    temp = temp.replace('ue', 've')
                elif temp == 'diang':
                    temp = 'di ang'
                elif temp == 'biang':
                    temp = 'bi ang'
                pinyin_split.append(temp)
            except:
                print(pinyin)
                pinyin_split.append(pinyin)
        return np.array(pinyin_split)
        

    train_pinyin_split = split_pinyin(train_pinyin)
    test_pinyin_split = split_pinyin(test_pinyin)
    dev_pinyin_split = split_pinyin(dev_pinyin)

    train_pinyin_name = []
    for pinyin, name in zip(train_pinyin_split, train_name):
        train_pinyin_name.append(pinyin + " " + name)

    train_text = np.stack([np.array(train_pinyin_split), np.array(train_name), np.array(train_pinyin_name)]).T
    test_text = test_pinyin_split
    dev_text = dev_pinyin_split

    train_label = train_tag
    dev_label = dev_tag
    test_label = test_tag

    def load_data(dataset, label, batch_size):
        batch_count = int(len(dataset) / batch_size)
        print(batch_count + 1)
        batch_inputs, batch_targets = [], []
        for i in range(batch_count):
            batch_inputs.append(dataset[i * batch_size: (i + 1) * batch_size])
            batch_targets.append(label[i * batch_size: (i + 1) * batch_size])
        batch_inputs.append(dataset[batch_count * batch_size:])
        batch_targets.append(label[batch_count * batch_size:])
        return batch_inputs, batch_targets, batch_count+1


    batch_train_inputs, batch_train_targets, train_batch_count = load_data(train_text, train_label, 128)
    batch_dev_inputs, batch_dev_targets, dev_batch_count = load_data(dev_text, dev_label, 128)
    batch_test_inputs, batch_test_targets, test_batch_count = load_data(test_text, test_label, 128)

    new_model = BertClassfication().to(device)
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.000005)
    epoch = 50
    print_every_batch = 10

    new_model.train()
    dev_acc = 0
    dev_final_loss = 99
    for _ in range(epoch):
        print_avg_loss = 0
        for i in range(train_batch_count):
            inputs = np.array(batch_train_inputs[i])
            targets = torch.tensor(batch_train_targets[i]).to(device)
            optimizer.zero_grad()
            outputs, loss = new_model(inputs[:, 0], inputs[:, 1], inputs[:, 2], targets)
            loss.backward()
            optimizer.step()
            print_avg_loss += loss.item()
            if i % print_every_batch == 0:
                print("Epoch: %d, Batch: %d, Loss: %.4f" % (_ + 1, (i + 1), print_avg_loss / print_every_batch))
                print_avg_loss = 0
        dev_pre = []
        dev_all_loss = 0
        for i in range(dev_batch_count):
            if i % print_every_batch == 0:
                print("dev_batch: %d" % (i + 1))
            dev_inputs = batch_dev_inputs[i]
            dev_targets = torch.tensor(batch_dev_targets[i]).to(device)
            dev_outputs, dev_loss = new_model(dev_inputs, None, None, dev_targets, False)
            dev_all_loss += dev_loss.item()
            dev_outputs = torch.argmax(dev_outputs, dim=1).cpu().numpy().tolist()
            dev_pre.extend(dev_outputs)
        temp_acc = accuracy_score(dev_tag[:len(dev_pre)], dev_pre)
        temp_loss = dev_all_loss / len(dev_pre)
        print('dev_acc: %.4f, loss: %.4f' % (temp_acc, temp_loss))
        if temp_acc > dev_acc:
            dev_acc = temp_acc
            best_val_model_acc = copy.deepcopy(new_model.module) if hasattr(new_model, "module") else copy.deepcopy(new_model)
        if temp_loss < dev_final_loss:
            dev_final_loss = temp_loss
            best_val_model_loss = copy.deepcopy(new_model.module) if hasattr(new_model, "module") else copy.deepcopy(new_model)
    output_model_file = 'model/temp_acc.bin'
    torch.save(best_val_model_acc.state_dict(), output_model_file)
    output_model_file_loss = 'model/temp.bin'
    torch.save(best_val_model_loss.state_dict(), output_model_file_loss)

    total = len(test_label)
    m_state_dict = torch.load('model/temp_acc.bin')
    best_model = BertClassfication().to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()
    test_pre = []
    with torch.no_grad():
        for i in range(test_batch_count):
            test_inputs = batch_test_inputs[i]
            test_targets = torch.tensor(batch_test_targets[i]).to(device)
            test_outputs,_ = best_model(test_inputs,None,None,test_targets,False)
            test_outputs = torch.argmax(test_outputs, dim=1).cpu().numpy().tolist()
            test_pre.extend(test_outputs)
    print(classification_report(test_label, test_pre, digits=6))

    # We report the results of the model with the lowest loss on the validation set in the paper
    total = len(test_label)
    m_state_dict = torch.load('model/temp.bin')
    best_model = BertClassfication().to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()
    test_pre = []
    with torch.no_grad():
        for i in range(test_batch_count):
            test_inputs = batch_test_inputs[i]
            test_targets = torch.tensor(batch_test_targets[i]).to(device)
            test_outputs,_ = best_model(test_inputs,None,None,test_targets,False)
            test_outputs = torch.argmax(test_outputs, dim=1).cpu().numpy().tolist()
            test_pre.extend(test_outputs)
    print(classification_report(test_label, test_pre, digits=6))