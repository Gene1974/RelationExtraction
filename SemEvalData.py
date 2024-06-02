import random
import re
import torch
from torch.utils.data import Dataset, DataLoader

class SemEvalVocab():
    def __init__(self, word_list = None):
        pass

class SemEvalDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SemEvalData():
    def __init__(self, mod = 'train', shuffle = False):
        self.relation_list = [
            'Other',
            'Cause-Effect(e1,e2)','Cause-Effect(e2,e1)', # e1 is the Cause and e2 is the Effect
            'Instrument-Agency(e1,e2)','Instrument-Agency(e2,e1)',
            'Product-Producer(e1,e2)','Product-Producer(e2,e1)',
            'Content-Container(e1,e2)','Content-Container(e2,e1)',
            'Entity-Origin(e1,e2)','Entity-Origin(e2,e1)',
            'Entity-Destination(e1,e2)','Entity-Destination(e2,e1)',
            'Component-Whole(e1,e2)','Component-Whole(e2,e1)',
            'Member-Collection(e1,e2)','Member-Collection(e2,e1)',
            'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)'
        ]
        self.tagset_size = len(self.relation_list)
        self.NEGATIVE_TAG = 'Other'
        self.PAD_TAG = '<PAD>'
        self.OOV_TAG = '<OOV>'
        self.word_list = [self.PAD_TAG, self.OOV_TAG]
        
        if mod == 'train':
            dataset, char_set, sen_len = self.load_data('/data/SemEval2010/TRAIN_FILE.TXT')
            test_dataset, test_char_set, test_sen_len = self.load_data('/data/SemEval2010/TEST_FILE_FULL.TXT')
            self.char_set = char_set | test_char_set
            self.sen_len = sen_len + test_sen_len
            self.max_sen_len = max(sen_len)
        else:
            self.test_data, _, _ = self.load_data('/data/SemEval2010/TEST_FILE_FULL.TXT')
        
        self.word_to_ix = {self.word_list[i]: i for i in range(len(self.word_list))}
        self.char_to_ix = {list(self.char_set)[i]: i + 1 for i in range(len(self.char_set))}
        self.tag_to_ix = {self.relation_list[i]: i for i in range(len(self.relation_list))}
        self.pos_to_ix = {i: i + self.max_sen_len - 1 for i in range(-self.max_sen_len + 1, self.max_sen_len)}
        self.char_to_ix[self.PAD_TAG] = 0
        self.char_to_ix[self.OOV_TAG] = 1

        if shuffle:
                random.shuffle(dataset)
        self.total_num = len(dataset)
        self.train_num = int(self.total_num * 0.8)
        self.valid_num = self.total_num - self.train_num
        self.map_and_pad(dataset)
        self.map_and_pad(test_dataset)
        self.train_data = SemEvalDataset(dataset[:self.train_num])
        self.valid_data = SemEvalDataset(dataset[self.train_num:])
        self.test_data = SemEvalDataset(test_dataset)
        
    
    def load_data(self, path):
        dataset = []
        char_set = set()
        sen_len = []
        f = open(path, 'r')
        data = f.readlines()
        for i in range(0, len(data), 4):
            # index, sentence = data[i].strip().split('\t')
            # sentence = sentence[1:-1]
            sentence = data[i].strip().split('\t')[1][1:-1]
            item = self.parse_sentence(sentence)
            item['relation'] = data[i + 1].strip()
            dataset.append(item)
            char_set |= set(sentence.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', ''))
            sen_len.append(len(item['words']))
        f.close()
        return dataset, char_set, sen_len
    
    def parse_sentence(self, sentence):
        item = {'text': sentence, 'e1': [], 'e2': []}
        words = sentence.split(' ')
        e1 = re.search('<e1>(.*)</e1>', sentence)
        item['e1'] = e1.group(0)[4: -5].split(' ')
        e2 = re.search('<e2>(.*)</e2>', sentence)
        item['e2'] = e2.group(0)[4: -5].split(' ')
        i = 0
        while i < len(words):
            words[i] = words[i].strip(',').strip('.').strip('!').strip('?').strip(';').strip(':').lower()
            if not words[i]:
                continue
            if '<e1>' in words[i]:
                words[i] = words[i].replace('<e1>', '')
                item['e1_start'] = i
            if '</e1>' in words[i]:
                words[i] = words[i].replace('</e1>', '')
                item['e1_end'] = i
            if '<e2>' in words[i]:
                words[i] = words[i].replace('<e2>', '')
                item['e2_start'] = i
            if '</e2>' in words[i]:
                words[i] = words[i].replace('</e2>', '')
                item['e2_end'] = i
            if words[i].lower() not in self.word_list:
                self.word_list.append(words[i].lower())
            i += 1
        item['words'] = words
        return item

    def map_and_pad(self, dataset):
        for item in dataset:
            item['word_ids'] = [self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.OOV_TAG] for word in item['words']] + [self.word_to_ix[self.PAD_TAG]] * (self.max_sen_len - len(item['words']))
            item['word_masks'] = [1 for _ in item['words']] + [0] * (self.max_sen_len - len(item['words']))
            item['pos1_ids'] = [self.pos_to_ix[i - item['e1_start']] for i in range(self.max_sen_len)]
            item['pos2_ids'] = [self.pos_to_ix[i - item['e2_start']] for i in range(self.max_sen_len)]
            item['tag_id'] = self.tag_to_ix[item['relation']]

def semeval_collate(batch):
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
    sem_set = SemEvalData('train')
    #item = sem_set.parse_sentence('The <e1>factory</e1>\'s products have included flower pots, Finnish rooster-whistles, pans, <e2>trays</e2>, tea pots, ash trays and air moisturisers')
    #item = sem_set.parse_sentence('The <e1>lawsonite</e1> was contained in a <e2>platinum crucible</e2> and the counter-weight was a plastic crucible with metal pieces')
    # train_loader = DataLoader(sem_set.train_data, 4, collate_fn = semeval_collate)
    # for i, item in enumerate(train_loader):
    #     print(i, item)
    #     break
