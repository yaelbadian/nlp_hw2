from collections import defaultdict
from torchtext.vocab import Vocab
from collections import Counter
import torch

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]


def split(string, delimiters):
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)
    return stack


def clean_word(word):
    return word.lower()


class DataMapping:
    def __init__(self, list_of_paths, word_embeddings=None, vectors_str="glove.6B.300d"):
        self.word_dict, self.pos_dict = self.get_vocabs(list_of_paths)
        self.vocab_size = len(self.word_dict)
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.word_dict, vectors_str)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.pos_dict)
        # self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        # self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        # self.word_vector_dim = self.word_vectors.size(-1)

    @staticmethod
    def get_vocabs(list_of_paths):
        word_dict = defaultdict(int)
        pos_dict = defaultdict(int)
        for file_path in list_of_paths:
            with open(file_path) as f:
                for line in f:
                    if line == '\n':
                        continue
                    splited_words = split(line, ('\t', '\n'))
                    word = clean_word(splited_words[1])
                    pos_tag = splited_words[3]
                    word_dict[word] += 1
                    pos_dict[pos_tag] += 1

        return word_dict, pos_dict

    @staticmethod
    def init_word_embeddings(word_dict, vectors_str="glove.6B.300d"):
        glove = Vocab(Counter(word_dict), vectors=vectors_str, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = []
        pos_idx_mappings = {}
        for i, pos in enumerate(sorted(SPECIAL_TOKENS) + sorted(pos_dict.keys())):
            pos_idx_mappings[str(pos)] = i
            idx_pos_mappings.append(str(pos))
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)