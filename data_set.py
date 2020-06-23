import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from data_reader import clean_word, split

from data_reader import UNKNOWN_TOKEN, ROOT_TOKEN, PAD_TOKEN


class DepParserDataset(Dataset):
    def __init__(self, data_mapping, file, padding=False, alpha_dropout=0.25):
        super().__init__()
        self.data_mapping = data_mapping
        self.alpha_dropout = alpha_dropout
        self.sentences = []
        self.read_data(file)

        self.sentence_lens = [len(sentence) for sentence in self.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, header, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, header, sentence_len

    def read_data(self, file):
        cur_sentence = [(0, ROOT_TOKEN, ROOT_TOKEN, -1)]
        with open(file, 'r') as f:
            for line in f:
                if line == '\n':
                    self.sentences.append(cur_sentence)
                    cur_sentence = [(0, ROOT_TOKEN, ROOT_TOKEN, -1)]
                    continue
                splited_words = split(line, ('\t', '\n'))
                # del splited_words[-1]
                idx = splited_words[0]
                word = clean_word(splited_words[1])
                pos_tag = splited_words[3]
                header = splited_words[6]
                cur_sentence.append((idx, word, pos_tag, header))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_header_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.sentences):
            words_idx_list = []
            pos_idx_list = []
            header_list = []
            for idx, word, pos, header in sentence:
                prob = self.alpha_dropout / (self.data_mapping.word_dict[word] + self.alpha_dropout)
                if prob > np.random.rand():
                    word = UNKNOWN_TOKEN
                words_idx_list.append(self.data_mapping.word_idx_mappings.get(word, self.data_mapping.word_idx_mappings[UNKNOWN_TOKEN]))
                pos_idx_list.append(self.data_mapping.pos_idx_mappings.get(pos, self.data_mapping.pos_idx_mappings[UNKNOWN_TOKEN]))
                header_list.append(header)
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_header_list.append(torch.tensor(header_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)
        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_header_list,
                                                                     sentence_len_list))}

