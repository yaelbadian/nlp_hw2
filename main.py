import pickle
from torch.utils.data.dataloader import DataLoader
from data_set import DepParserDataset
from data_reader import DataMapping
from model import DependencyParser
import train_model

path_train = "HW2-files/train.labeled"
path_test = "HW2-files/test.labeled"
paths_list = [path_train, path_test]

data_mapping = DataMapping(paths_list)
word_vocab_size, pos_vocab_size = len(data_mapping.word_idx_mappings), len(data_mapping.pos_idx_mappings)

train = DepParserDataset(data_mapping, path_train)
train_loader = DataLoader(train, shuffle=True)
test = DepParserDataset(data_mapping, path_test, alpha_dropout=0)
test_loader = DataLoader(test, shuffle=False)
net = DependencyParser(word_vocab_size, pos_vocab_size)

train_model.train(net, train_loader, test_loader, path='models/', epochs=10)






