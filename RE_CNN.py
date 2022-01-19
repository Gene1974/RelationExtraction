import torch
import torch.nn as nn


class RC_CNN(nn.Module):
    def __init__(self, 
        word_emb, pos_emb_dim, max_sen_len, cnn_windows, filter_num, tagset_size,
        dropout = 0.1
        ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.word_emb_dim = word_emb.shape[1]
        self.pos_emb_dim = pos_emb_dim
        self.max_sen_len = max_sen_len
        self.cnn_windows = cnn_windows
        self.filter_num = filter_num
        self.tagset_size = tagset_size
        self.emb_dim = self.word_emb_dim + 2 * pos_emb_dim
        self.cnn_out_dim = self.filter_num * len(self.cnn_windows)

        self.word_embeds = nn.Embedding.from_pretrained(word_emb, freeze = True)
        self.pos1_embeds = nn.Embedding(2 * max_sen_len - 1, pos_emb_dim)
        self.pos2_embeds = nn.Embedding(2 * max_sen_len - 1, pos_emb_dim)
        self.cnns = nn.ModuleList([nn.Sequential(nn.Conv1d(self.emb_dim, self.filter_num, window_size), nn.Tanh()) for window_size in self.cnn_windows])
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.cnn_out_dim, tagset_size)
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, word_ids, pos1_ids, pos2_ids):
        word_emb = self.word_embeds(word_ids)
        pos1_emb = self.pos1_embeds(pos1_ids)
        pos2_emb = self.pos2_embeds(pos2_ids)
        emb = torch.cat((word_emb, pos1_emb, pos2_emb), dim = -1) # (batch_size, max_sen_len, emb_size)
        emb = emb.permute(0, 2, 1) # (batch_size, emb_size, max_sen_len)

        cnn_outs = []
        for cnn in self.cnns:
            cnn_out = cnn(emb) # (batch_size, filter_num, max_sen_len - window_size + 1)
            cnn_out, _ = torch.max(cnn_out, dim = -1) # (batch_size, filter_num)
            cnn_outs.append(cnn_out)
        cnn_out = torch.cat(cnn_outs, dim = -1) # (batch_size, filter_num * window_num)
        cnn_out = self.dropout(cnn_out)
        out = self.dense(cnn_out) # (batch_size, tagset_size)
        #out = self.softmax(out)
        return out