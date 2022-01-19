import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Pretrained import GloveEmbedding, SennaEmbedding, Word2vecEmbedding
from SemEvalData import SemEvalData, SemEvalDataset, semeval_collate


from pytorchtools import EarlyStopping
from RE_CNN import RC_CNN
from Utils import logger

torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Trainer():
    def __init__(self, 
        mod = 'train', model_time = None, epochs = 100, window_size = [3]
        ):
        super().__init__()
        self.model_time = model_time

        self.pretrained = 'glove'
        self.pos_emb_dim = 50
        self.dropout = 0.2
        self.epochs = epochs
        self.batch_size = 50
        self.lr = 5e-5
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger('device = {}, batch_size = {}, lr = {}, epochs = {}'.format(self.device, self.batch_size, self.lr, epochs))
        logger('window_size = {}, dropout = {}, pos_emb = {}'.format(window_size, self.dropout, self.pos_emb_dim))
        
        if mod == 'train':
            self.dataset = SemEvalData(mod, shuffle = False)
            self.train_set = self.dataset.train_data
            self.valid_set = self.dataset.valid_data
            self.test_set = self.dataset.test_data
            self.train_loader = DataLoader(self.train_set, self.batch_size, collate_fn = semeval_collate)
            self.valid_loader = DataLoader(self.valid_set, self.batch_size, collate_fn = semeval_collate)
            self.test_loader = DataLoader(self.test_set, self.batch_size, collate_fn = semeval_collate)
            logger('Load data. Train data: {}, Valid data: {}, Test data: {}'.format(len(self.train_set), len(self.valid_set), len(self.test_set)))
        else:
            model_path = './results/{}'.format(model_time)
            with open(model_path + '/vocab_' + model_time, 'rb') as f:
                self.dataset = pickle.load(f)
            self.test_set = self.dataset.test_data
            self.test_loader = DataLoader(self.test_set, self.batch_size, collate_fn = semeval_collate)
            logger('Load data. Test data: {}'.format(len(self.test_set)))

        self.word_list = self.dataset.word_list
        self.tagset_size = self.dataset.tagset_size
        self.max_sen_len = self.dataset.max_sen_len
        if self.pretrained == 'glove':
            self.word_emb = GloveEmbedding(word_list = self.word_list)
        if self.pretrained == 'word2vec':
            self.word_emb = Word2vecEmbedding(word_list = self.word_list)
        if self.pretrained == 'senna':
            self.word_emb = SennaEmbedding(word_list = self.word_list)
        self.model = RC_CNN(
            self.word_emb, self.pos_emb_dim, self.max_sen_len, window_size, 150, self.tagset_size, self.dropout
        ).to(self.device)

        if mod == 'train':
            self.train()
        else:
            self.model.load_state_dict(torch.load(model_path + '/model_' + model_time))
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
                word_ids, pos1_ids, pos2_ids, tag_ids = batch
                optimizer.zero_grad()
                output = model(word_ids, pos1_ids, pos2_ids) # (batch_size, tagset_size)
                loss = entrophy(output, tag_ids)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(self.valid_loader):
                    word_ids, pos1_ids, pos2_ids, tag_ids = batch
                    output = model(word_ids, pos1_ids, pos2_ids)
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
        model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
        self.model_time = model_time
        model_path = './results/{}'.format(model_time)
        os.mkdir(model_path)
        torch.save(model.state_dict(), model_path + '/model_' + model_time)
        with open(model_path + '/vocab_' + model_time, 'wb') as f:
            pickle.dump(self.dataset, f)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(avg_train_losses)
        plt.plot(avg_valid_losses)
        plt.legend(['train_loss', 'valid_loss'])
        plt.savefig(model_path + '/{}.png'.format(model_time), format = 'png')

        logger('Save result {}'.format(model_time))

        self.test()
            
    def test(self):
        model = self.model
        model.eval()
        softmax = nn.Softmax(dim = -1)
        self.NEGATIVE_TAG = self.dataset.NEGATIVE_TAG
        self.NEGATIVE_ID = self.dataset.tag_to_ix[self.NEGATIVE_TAG]
        logger('Begin testing.')
        pred = []
        gold = []
        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                word_ids, pos1_ids, pos2_ids, tag_id = batch
                output = model(word_ids, pos1_ids, pos2_ids) # (batch_size, tagset_size)
                predict = torch.argmax(softmax(output), dim = -1)
                pred += [self.dataset.relation_list[i] for i in predict]
                gold += [self.dataset.relation_list[i] for i in tag_id]
            f = open('./results/{}/answer.txt'.format(self.model_time), 'w')
            for i in range(len(pred)):
                f.write(str(i + 8001) + '\t' + pred[i] + '\n')
            f.close()
            f = os.popen('./Eval/semeval2010_task8_scorer-v1.2.pl ./results/{}/answer.txt ./Eval/answer_key.txt'.format(self.model_time))
            f1 = f.readlines()[-1].strip().split(' ')[-2]
            logger('[Test] macro-averaged F1 = {}'.format(f1))
            f.close()
    

if __name__ == '__main__':
    trainer = Trainer('train', '01191455', epochs = 100, window_size = [2, 3, 4, 5])
