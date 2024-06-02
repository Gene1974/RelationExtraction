import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DiaKGData import DiaKGData
from Pretrained import GloveEmbedding, SennaEmbedding, Word2vecEmbedding
from SemEvalData import SemEvalData, SemEvalDataset, semeval_collate


from pytorchtools import EarlyStopping
from RE_CNN import RC_CNN
from Utils import logger

torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class TrainConfig():
    def __init__(self, mod = 'train', model_time = None, epochs = 100, window_size = [3]):
        self.pretrained = 'glove'
        self.pos_emb_dim = 50
        self.dropout = 0.2
        self.epochs = epochs
        self.batch_size = 5
        self.lr = 1e-4

class Trainer():
    def __init__(self, mod = 'train', model_time = None, dataset = 'semeval', epochs = 100, window_size = [3]):
        if mod == 'train':
            self.model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
        else:
            self.model_time = model_time
        self.model_path = './results/{}'.format(self.model_time)

        self.mod = mod
        self.pretrained = 'none'
        self.dataset = dataset
        
        self.pos_emb_dim = 50
        self.dropout = 0.2
        self.epochs = epochs
        self.batch_size = 50
        self.lr = 1e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger('device = {}, batch_size = {}, lr = {}, epochs = {}'.format(self.device, self.batch_size, self.lr, epochs))
        logger('window_size = {}, dropout = {}, pos_emb = {}'.format(window_size, self.dropout, self.pos_emb_dim))
        
        if mod == 'train':
            if dataset == 'semeval':
                self.data = SemEvalData(mod, shuffle = False)
            elif dataset == 'diakg':
                self.data = DiaKGData(mod, shuffle = False)
            self.train_set = self.data.train_data
            self.valid_set = self.data.valid_data
            self.test_set = self.data.test_data
            self.train_loader = DataLoader(self.train_set, self.batch_size, collate_fn = semeval_collate)
            self.valid_loader = DataLoader(self.valid_set, self.batch_size, collate_fn = semeval_collate)
            self.test_loader = DataLoader(self.test_set, self.batch_size, collate_fn = semeval_collate)
            logger('Load data. Train data: {}, Valid data: {}, Test data: {}'.format(len(self.train_set), len(self.valid_set), len(self.test_set)))
        else:
            with open(self.model_path + '/vocab_' + self.model_time, 'rb') as f:
                self.data = pickle.load(f)
            self.test_set = self.data.test_data
            self.test_loader = DataLoader(self.test_set, self.batch_size, collate_fn = semeval_collate)
            logger('Load data. Test data: {}'.format(len(self.test_set)))

        self.word_list = self.data.word_list
        self.tagset_size = self.data.tagset_size
        self.max_sen_len = self.data.max_sen_len
        if self.pretrained == 'glove':
            self.word_emb = GloveEmbedding(word_list = self.word_list)
        elif self.pretrained == 'word2vec':
            self.word_emb = Word2vecEmbedding(word_list = self.word_list)
        elif self.pretrained == 'senna':
            self.word_emb = SennaEmbedding(word_list = self.word_list)
        else:
            self.word_emb = self.word_list
        self.model = RC_CNN(
            self.word_emb, self.pos_emb_dim, self.max_sen_len, window_size, 150, self.tagset_size, self.dropout
        ).to(self.device)

        if mod == 'train':
            self.train()
        else:
            self.model.load_state_dict(torch.load(self.model_path + '/model_' + model_time))
            self.test()

    def train(self):
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        early_stopping = EarlyStopping(patience = 20, verbose = False)
        entrophy = nn.CrossEntropyLoss()

        avg_train_losses = []
        avg_valid_losses = []
        for epoch in range(self.epochs):
            train_losses = []
            valid_losses = []
            model.train()
            for _, batch in enumerate(self.train_loader):
                text, word_ids, pos1_ids, pos2_ids, tag_ids = batch
                optimizer.zero_grad()
                output = model(text, word_ids, pos1_ids, pos2_ids) # (batch_size, tagset_size)
                loss = entrophy(output, tag_ids)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(self.valid_loader):
                    text, word_ids, pos1_ids, pos2_ids, tag_ids = batch
                    output = model(text, word_ids, pos1_ids, pos2_ids)
                    loss = entrophy(output, tag_ids)
                    valid_losses.append(loss.item())
                avg_train_loss = np.average(train_losses)
                avg_valid_loss = np.average(valid_losses)
                avg_train_losses.append(avg_train_loss)
                avg_valid_losses.append(avg_valid_loss)
                logger('[epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}'.format(epoch + 1, avg_train_loss, avg_valid_loss))
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    logger("Early stopping")
                    break

        self.model = model
        self.save_model(avg_train_losses, avg_valid_losses)
        self.test()
    
    def save_model(self, avg_train_losses = None, avg_valid_losses = None):
        os.mkdir(self.model_path)
        torch.save(self.model.state_dict(), self.model_path + '/model_' + self.model_time)
        with open(self.model_path + '/vocab_' + self.model_time, 'wb') as f:
            pickle.dump(self.dataset, f)
        if avg_train_losses is not None:
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(avg_train_losses)
            plt.plot(avg_valid_losses)
            plt.legend(['train_loss', 'valid_loss'])
            plt.savefig(self.model_path + '/{}.png'.format(self.model_time), format = 'png')
        logger('Save result {}'.format(self.model_time))

    def test(self):
        model = self.model
        model.eval()
        softmax = nn.Softmax(dim = -1)
        self.target_names = self.data.relation_list
        self.NEGATIVE_TAG = self.data.NEGATIVE_TAG
        self.NEGATIVE_ID = self.data.tag_to_ix[self.NEGATIVE_TAG]
        logger('Begin testing.')
        y_pred = []
        y_true = []
        pred = []
        gold = []
        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                text, word_ids, pos1_ids, pos2_ids, tag_id = batch
                output = model(text, word_ids, pos1_ids, pos2_ids) # (batch_size, tagset_size)
                predict = torch.argmax(softmax(output), dim = -1)
                y_pred += predict.tolist()
                y_true += tag_id.tolist()
                pred += [self.data.relation_list[i] for i in predict]
                gold += [self.data.relation_list[i] for i in tag_id]
            if self.dataset == 'semeval':
                f = open('./results/{}/answer.txt'.format(self.model_time), 'w')
                for i in range(len(pred)):
                    f.write(str(i + 8001) + '\t' + pred[i] + '\n')
                f.close()
                f = os.popen('./Eval/semeval2010_task8_scorer-v1.2.pl ./results/{}/answer.txt ./Eval/answer_key.txt'.format(self.model_time))
                f1 = f.readlines()[-1].strip().split(' ')[-2]
                logger('[Test] macro-averaged F1 = {}'.format(f1))
                f.close()
            elif self.dataset == 'diakg':
                print(classification_report(y_true, y_pred, target_names = self.target_names))

if __name__ == '__main__':
    trainer = Trainer('train', '01191455', dataset = 'diakg', epochs = 1, window_size = [2, 3, 4, 5])
