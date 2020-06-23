import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader


class DependencyParser(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, word_emb_dim, pos_emb_dim, hidden_dim, mlp_size=100):
        super(DependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        self.lstm = nn.LSTM(input_size=(word_emb_dim+pos_emb_dim), hidden_size=hidden_dim, num_layers=2,
                            bidirectional=True)
        self.mlp_h = nn.Linear(hidden_dim*2, mlp_size)
        self.mlp_m = nn.Linear(hidden_dim*2, mlp_size)
        self.activation = nn.Tanh()
        self.mlp = nn.Linear(mlp_size, 1)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)   # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)   # [batch_size, seq_length, mlp_size]
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores = self.mlp(self.activation(scores)) # [batch_size, seq_length, seq_length, 1]
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores



from pytorch_model_summary import summary


net = DependencyParser(word_vocab_size=100, pos_vocab_size=100, word_emb_dim=100,
                       pos_emb_dim=100, hidden_dim=100, mlp_size=100)
net.cuda()
sentence = (torch.tensor([[1, 5, 3, 24, 74]], dtype=torch.long, requires_grad=False),
            torch.tensor([[1, 5, 3, 24, 74]], dtype=torch.long, requires_grad=False),
            torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long, requires_grad=False),
            [5])
print(summary(net, sentence, show_input=False, show_hierarchical=True))