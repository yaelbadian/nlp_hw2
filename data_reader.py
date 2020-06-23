from collections import defaultdict


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


class DataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.read_data()

    def read_data(self):
        cur_sentence = []
        with open(self.file, 'r') as f:
            for line in f:
                if line == '\n':
                    self.sentences.append(cur_sentence)
                    cur_sentence = []
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


