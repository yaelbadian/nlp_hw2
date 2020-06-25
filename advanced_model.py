import torch
import torch.nn as nn
import math


class BasicDependencyParser(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, word_emb_dim=100, pos_emb_dim=25,
                 hidden_dim=125, mlp_dim=100, lstm_layers=2):
        super(BasicDependencyParser, self).__init__()
        self.use_coda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.use_coda else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        self.lstm = nn.LSTM(input_size=(word_emb_dim+pos_emb_dim), hidden_size=hidden_dim, num_layers=lstm_layers,
                            bidirectional=True)
        self.mlp_h = nn.Linear(hidden_dim*2, mlp_dim)
        self.mlp_m = nn.Linear(hidden_dim*2, mlp_dim)
        self.activation = nn.Tanh()
        self.mlp = nn.Linear(mlp_dim, 1)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)   # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)   # [batch_size, seq_length, mlp_size]
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores = self.mlp(self.activation(scores))  # [batch_size, seq_length, seq_length, 1]
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        net = torch.load(path)
        net.eval()
        return net



class DependencyParser(nn.Module):
    def __init__(self, word_embeddings, pos_vocab_size, pos_emb_dim=25, hidden_dim=125, mlp_dim=100,
                 lstm_layers=2, lstm_dropout=0):
        super(DependencyParser, self).__init__()
        self.use_coda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.use_coda else "cpu")
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings.to(self.device))
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        self.lstm = nn.LSTM(input_size=(word_embeddings.shape[1] + pos_emb_dim), hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            bidirectional=True, dropout=lstm_dropout)
        self.mlp_h = nn.Linear(hidden_dim * 2, mlp_dim)
        self.mlp_m = nn.Linear(hidden_dim * 2, mlp_dim)
        self.activation = nn.Tanh()
        self.mlp = nn.Linear(mlp_dim, 1)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores = self.mlp(self.activation(scores))  # [batch_size, seq_length, seq_length, 1]
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        net = torch.load(path)
        net.eval()
        return net


class DependencyParserAttention(nn.Module):
    def __init__(self, word_embeddings, pos_vocab_size, pos_emb_dim=25, hidden_dim=125, mlp_dim=100,
                 lstm_layers=2, lstm_dropout=0):
        super(DependencyParserAttention, self).__init__()
        self.use_coda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.use_coda else "cpu")
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings.to(self.device))
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        self.lstm = nn.LSTM(input_size=(word_embeddings.shape[1] + pos_emb_dim), hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            bidirectional=True, dropout=lstm_dropout)
        self.mlp_h = nn.Linear(hidden_dim * 2, mlp_dim)
        self.mlp_m = nn.Linear(hidden_dim * 2, mlp_dim)
        self.activation = nn.Tanh()
        self.attention = nn.Parameter(torch.Tensor(1, mlp_dim + 1, mlp_dim + 1))
        nn.init.zeros_(self.attention)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        # add bias
        h_out = torch.cat((h_out, torch.ones_like(h_out[..., :1])), -1)
        m_out = torch.cat((m_out, torch.ones_like(m_out[..., :1])), -1)
        # attention
        scores = torch.einsum('bxi,oij,byj->boxy', self.activation(h_out), self.attention,
                              self.activation(m_out))  # [batch_size, seq_length, seq_length, 1]
        scores = scores.squeeze(1)
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class DependencyParserCombinedAttention(nn.Module):
    def __init__(self, word_embeddings, pos_vocab_size, pos_emb_dim=25, hidden_dim=125, mlp_dim=100,
                 lstm_layers=2, lstm_dropout=0):
        super(DependencyParserCombinedAttention, self).__init__()
        self.use_coda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.use_coda else "cpu")
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings.to(self.device))
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        self.lstm = nn.LSTM(input_size=(word_embeddings.shape[1] + pos_emb_dim), hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            bidirectional=True, dropout=lstm_dropout)
        self.mlp_h = nn.Linear(hidden_dim * 2, mlp_dim)
        self.mlp_m = nn.Linear(hidden_dim * 2, mlp_dim)
        self.activation = nn.Tanh()
        self.attention = nn.Parameter(torch.Tensor(1, mlp_dim + 1, mlp_dim + 1))
        nn.init.zeros_(self.attention)
        self.mlp = nn.Linear(mlp_dim, 1)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        # add bias
        h_out_b = torch.cat((self.activation(h_out), torch.ones_like(h_out[..., :1])), -1)
        m_out_b = torch.cat((self.activation(m_out), torch.ones_like(m_out[..., :1])), -1)
        # attention
        scores_att = torch.einsum('bxi,oij,byj->boxy', h_out_b, self.attention,
                                  m_out_b)  # [batch_size, seq_length, seq_length, 1]
        scores_att = scores_att.squeeze(1)
        scores_att = scores_att.view(1, scores_att.shape[1], scores_att.shape[2])

        scores_mlp = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out,
                                                                 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores_mlp = self.mlp(self.activation(scores_mlp))  # [batch_size, seq_length, seq_length, 1]
        scores_mlp = scores_mlp.view(1, scores_mlp.shape[1], scores_mlp.shape[2])

        scores = scores_att + scores_mlp
        return scores

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class DependencyParserLinear(nn.Module):
    def __init__(self, word_embeddings, pos_vocab_size, pos_emb_dim=40, word_lin_dim=150, pos_lin_dim=20,
                 hidden_dim=125, mlp_dim=100, lstm_layers=2, lstm_dropout=0):
        super(DependencyParserLinear, self).__init__()
        self.use_coda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.use_coda else "cpu")
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings.to(self.device))
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)
        self.word_linear = nn.Sequential(nn.Linear(word_embeddings.shape[1], word_lin_dim), nn.ReLU())
        self.pos_linear = nn.Sequential(nn.Linear(pos_emb_dim, pos_lin_dim), nn.ReLU())
        self.lstm = nn.LSTM(input_size=(word_lin_dim + pos_lin_dim), hidden_size=hidden_dim, num_layers=lstm_layers,
                            bidirectional=True, dropout=lstm_dropout)
        self.mlp_h = nn.Linear(hidden_dim * 2, mlp_dim)
        self.mlp_m = nn.Linear(hidden_dim * 2, mlp_dim)
        self.activation = nn.Tanh()
        self.mlp = nn.Linear(mlp_dim, 1)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        word_embeds = self.word_linear(word_embeds)  # [batch_size, seq_length, lin_dim]
        pos_embeds = self.pos_linear(pos_embeds)  # [batch_size, seq_length, lin_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*lin_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)  # [batch_size, seq_length, mlp_size]
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores = self.mlp(self.activation(scores))  # [batch_size, seq_length, seq_length, 1]
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        net = torch.load(path)
        net.eval()
        return net


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DependencyParserTransformer(nn.Module):
    def __init__(self, word_embeddings, pos_vocab_size, pos_emb_dim=25, hidden_dim=125, mlp_dim=100,
                 n_layers=2, dropout=0, nhead=2):
        super(DependencyParserTransformer, self).__init__()
        self.use_coda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.use_coda else "cpu")
        self.ninp = word_embeddings.shape[1] + pos_emb_dim
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings.to(self.device))
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.ninp, nhead, hidden_dim * 2, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.mlp_h = nn.Linear(hidden_dim * 2, mlp_dim)
        self.mlp_m = nn.Linear(hidden_dim * 2, mlp_dim)
        self.activation = nn.Tanh()
        self.mlp = nn.Linear(mlp_dim, 1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, sentence_len = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(self.device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]

        if self.src_mask is None or self.src_mask.size(0) != len(embeds):
            mask = self._generate_square_subsequent_mask(len(embeds)).to(self.device)
            self.src_mask = mask

        src = embeds * 1
        src = self.pos_encoder(src)
        trans_out = self.transformer_encoder(src, self.src_mask)

        h_out = self.mlp_h(trans_out)  # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(trans_out)  # [batch_size, seq_length, mlp_size]
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores = self.mlp(self.activation(scores))  # [batch_size, seq_length, seq_length, 1]
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        net = torch.load(path)
        net.eval()
        return net
