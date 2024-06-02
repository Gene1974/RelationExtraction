import json
import random
import re
from unittest import result
import torch

class DiaKGDataset():
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class DiaKGData():
    def __init__(self, mod = 'train', shuffle = False):
        self.entity_list = []
        self.relation_list = [
            'Other',
            'Test_Disease',
            'Symptom_Disease',
            'Treatment_Disease',
            'Drug_Disease',
            'Anatomy_Disease',
            'Reason_Disease',
            'Pathogenesis_Disease',
            'Operation_Disease',
            'Class_Disease',
            'Test_items_Disease',
            'Frequency_Drug',
            'Duration_Drug',
            'Amount_Drug',
            'Method_Drug',
            'ADE_Drug'
        ]
        self.tagset_size = len(self.relation_list)
        self.NEGATIVE_TAG = 'Other'
        self.PAD_TAG = '<PAD>'
        self.OOV_TAG = '<OOV>'
        self.word_list = [self.PAD_TAG, self.OOV_TAG]
        self.max_sen_len = 128
        
        if mod == 'train':
            self.dataset = []
            self.char_set = set()
            self.sen_len = []
            self.ent_gap = []
            for i in range(1, 42):
                #dataset, char_set, sen_len = self.load_data('/data/DiaKG/0521_new_format/{}.json'.format(i))
                dataset, char_set = self.load_parse_data('/data/DiaKG/0521_new_format/{}.json'.format(i))
                self.dataset += dataset
                self.char_set |= char_set
                #self.sen_len += sen_len
            # self.max_sen_len = max(self.sen_len)
            # print(self.max_sen_len)
            # print(len(self.sen_len), sum([1 if s > 128 else 0 for s in self.sen_len]))
            num = 0
            for item in self.dataset:
                self.sen_len.append(len(item['text']))
                if self.sen_len[-1] >= 128:
                    num += 1
            print(max(self.sen_len), len(self.sen_len), num)
        
        self.word_list += list(self.char_set)
        self.word_to_ix = {self.word_list[i]: i for i in range(len(self.word_list))}
        self.char_to_ix = {list(self.char_set)[i]: i + 1 for i in range(len(self.char_set))}
        self.tag_to_ix = {self.relation_list[i]: i for i in range(len(self.relation_list))}
        self.pos_to_ix = {i: i + self.max_sen_len - 1 for i in range(-self.max_sen_len + 1, self.max_sen_len)}
        self.char_to_ix[self.PAD_TAG] = 0
        self.char_to_ix[self.OOV_TAG] = 1
        self.map_and_pad(self.dataset)

        if shuffle:
            random.shuffle(self.dataset)
        self.total_num = len(self.dataset)
        self.test_num = int(self.total_num * 0.2)
        self.train_num = int(self.total_num * 0.64)
        self.valid_num = self.total_num - self.train_num - self.test_num
        self.train_data = DiaKGDataset(self.dataset[:self.train_num])
        self.valid_data = DiaKGDataset(self.dataset[self.train_num: self.train_num + self.valid_num])
        self.test_data = DiaKGDataset(self.dataset[self.train_num + self.valid_num:])

    def parse_sentence(self, sentence):
        '''
        sentence: sen['sentence']
        '''
        max_length = self.max_sen_len // 2
        sub_sen_idx = [] # [[start_idx, end_idx], [start_idx, end_idx], ...]
        sub_sens = sentence.replace('；', '。').replace('？', '。').replace('！', '。').replace(';', '。').replace('?', '。').replace('!', '。').split('。') # 手动分句
        #sub_sens = sentence.replace('；', '。').replace('？', '。').replace('！', '。').replace(';', '。').replace('?', '。').replace('!', '。').replace('。', '，').split('，') # 手动分句
        
        begin_index = 0 # index of text
        for sen in sub_sens:
            end_index = begin_index + len(sen) + 1
            while len(sen) > max_length - 1: # 加上句号小于128
                sub_sen = sen[:max_length]
                pos = sub_sen.replace(',', '，').rfind('，')
                if pos == -1 or pos == 0:
                    pos = max_length - 1
                end_index = begin_index + pos + 1
                sub_sen_idx.append([begin_index, end_index])
                sen = sen[pos + 1:]
                begin_index = end_index
                end_index = begin_index + len(sen) + 1
            sub_sen_idx.append([begin_index, end_index])
            begin_index = end_index
            end_index = begin_index + len(sen) + 1
        return sub_sen_idx
    
    def pack_item(self, e1_id, e2_id, entity_dict, sub_sen_idx, sentence):
        # 需要处理句子长度问题
        item = {}
        item['e1'] = entity_dict[e1_id]['entity']
        item['e1_start'] = entity_dict[e1_id]['start_idx'] # 每个 sentence 中 entity 的 idx 从0开始计数
        item['e2'] = entity_dict[e2_id]['entity']
        item['e2_start'] = entity_dict[e2_id]['start_idx']
        for i in range(len(sub_sen_idx)):
            if item['e1_start'] >= sub_sen_idx[i][0] and item['e1_start'] < sub_sen_idx[i][1]:
                item['sen_start_idx'] = sub_sen_idx[i][0]
                while i < len(sub_sen_idx) - 1 and item['e2_start'] >= sub_sen_idx[i][0]:
                    i += 1
                item['sen_end_idx'] = sub_sen_idx[i][1]
                item['text'] = sentence[item['sen_start_idx']: item['sen_end_idx']]
                item['words'] = list(item['text'])
                break
        return item
    
    def load_parse_data(self, path):
        dataset = [] # [item]
        char_set = set()
        f = open(path, 'r')
        data = json.load(f)
        f.close()
        for para in data['paragraphs'][:2]:
            paragraph = para['paragraph']
            for sen in para['sentences']:
                sentence = sen['sentence']
                entity_dict = {e['entity_id']: e for e in sen['entities']} # {T1: e}
                sub_sen_idx = self.parse_sentence(sentence)
                
                for r in sen['relations']:
                    e1_id, e2_id = r['head_entity_id'], r['tail_entity_id']
                    item = self.pack_item(e1_id, e2_id, entity_dict, sub_sen_idx, sentence)
                    if r['relation_type'] not in self.relation_list:
                        item['relation'] = 'Other'
                    else:
                        item['relation'] = r['relation_type']
                    if len(item['text']) < 128:
                        dataset.append(item)
                
                entity_ids = list(entity_dict.keys())
                relation_ids = [[r['head_entity_id'], r['tail_entity_id']] for r in sen['relations']]
                for e1_id, e2_id in negative_ids(entity_ids, relation_ids): # negative samples
                    item = self.pack_item(e1_id, e2_id, entity_dict, sub_sen_idx, sentence)
                    if abs(item['e1_start'] - item['e2_start']) <= 15 and len(item['text']) < 128:
                        item['relation'] = 'Other'
                        dataset.append(item)
            char_set |= set(paragraph)
        return dataset, char_set
    
    def map_and_pad(self, dataset):
        for item in dataset:
            item['word_ids'] = [self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.OOV_TAG] for word in item['words']] + [self.word_to_ix[self.PAD_TAG]] * (self.max_sen_len - len(item['words']))
            item['word_masks'] = [1 for _ in item['words']] + [0] * (self.max_sen_len - len(item['words']))
            item['pos1_ids'] = [self.pos_to_ix[i - item['e1_start']] for i in range(self.max_sen_len)]
            item['pos2_ids'] = [self.pos_to_ix[i - item['e2_start']] for i in range(self.max_sen_len)]
            item['tag_id'] = self.tag_to_ix[item['relation']]

    # archieved
    def load_data(self, path):
        # without parse sentence
        dataset = []
        char_set = set()
        sen_len = []
        f = open(path, 'r')
        data = json.load(f)
        f.close()
        for para in data['paragraphs']:
            for sen in para['sentences']:
                entities = {e['entity_id']: e for e in sen['entities']}
                entity_ids = list(entities.keys())
                relation_ids = [[r['head_entity_id'], r['tail_entity_id']] for r in sen['relations']]
                for r in sen['relations']:
                    e1_id, e2_id = r['head_entity_id'], r['tail_entity_id']
                    item = {'text': sen['sentence'], 'relation': r['relation_type'], 'words': list(sen['sentence'])}
                    item['e1'] = entities[e1_id]['entity']
                    item['e1_start'] = entities[e1_id]['start_idx']
                    item['e2'] = entities[e2_id]['entity']
                    item['e2_start'] = entities[e2_id]['start_idx']
                    self.ent_gap.append(abs(item['e1_start'] - item['e2_start']))
                    if r['relation_type'] not in self.relation_list:
                        item['relation'] = 'Other'
                    dataset.append(item)
                for e1_id, e2_id in negative_ids(entity_ids, relation_ids): # negative samples
                    item = {'text': sen['sentence'], 'relation': 'Other', 'words': list(sen['sentence'])}
                    item['e1'] = entities[e1_id]['entity']
                    item['e1_start'] = entities[e1_id]['start_idx']
                    item['e2'] = entities[e2_id]['entity']
                    item['e2_start'] = entities[e2_id]['start_idx']
                    if abs(item['e1_start'] - item['e2_start']) <= 15:
                        dataset.append(item)
                char_set |= set(sen['sentence'])
                sen_len.append(len(sen['sentence']))
        return dataset, char_set, sen_len

def negative_ids(entity_ids, relation_ids):
    ids = []
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            if [entity_ids[i], entity_ids[j]] not in relation_ids and [entity_ids[j], entity_ids[i]] not in relation_ids:
                ids.append([entity_ids[i], entity_ids[j]])
    return ids

def diakg_collate(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = [item['words'] for item in batch]
    sen_len = max([len(sen)for sen in text])
    word_ids = torch.tensor([item['word_ids'] for item in batch], dtype = torch.long, device = device)[:, :sen_len] # (batch_size, max_sen_len)
    word_masks = torch.tensor([item['word_masks'] for item in batch], dtype = torch.bool, device = device)[:, :sen_len]
    pos1_ids = torch.tensor([item['pos1_ids'] for item in batch], dtype = torch.long, device = device)[:, :sen_len]
    pos2_ids = torch.tensor([item['pos2_ids'] for item in batch], dtype = torch.long, device = device)[:, :sen_len]
    tag_id = torch.tensor([item['tag_id'] for item in batch], dtype = torch.long, device = device)
    return text, word_ids, pos1_ids, pos2_ids, tag_id

if __name__ == '__main__':
    diakg = DiaKGData()
    # dataset, char_set, sen_len = diakg.load_data('/data/DiaKG/0521_new_format/1.json')
    # print(dataset[2])
    # print(len(diakg.ent_gap), max(diakg.ent_gap))
    # print(len(diakg.sen_len), max(diakg.sen_len))
    # cnt = 0
    # for i in diakg.sen_len:
    #     if i > 128:
    #         cnt += 1
    # print(cnt)
