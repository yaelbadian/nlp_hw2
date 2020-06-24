import pickle
from torch.utils.data.dataloader import DataLoader
from data_set import DepParserDataset
from data_reader import DataMapping
from model import DependencyParser
from advanced_model import DependencyParser as advancedDependencyParser
import train_model

path_train = "HW2-files/train.labeled"
path_test = "HW2-files/test.labeled"
paths_list = [path_train, path_test]

vectors_strs = ["glove.6B.300d", "fasttext.en.300d", "fasttext.simple.300d", "glove.42B.300d",
                "glove.840B.300d", "glove.twitter.27B.200d"]


data_mapping = DataMapping(paths_list, vectors_str="")
word_vocab_size, pos_vocab_size = len(data_mapping.word_idx_mappings), len(data_mapping.pos_idx_mappings)

train = DepParserDataset(data_mapping, path_train)
train_loader = DataLoader(train, shuffle=True)
test = DepParserDataset(data_mapping, path_test, alpha_dropout=0)
test_loader = DataLoader(test, shuffle=False)
# net = DependencyParser(word_vocab_size, pos_vocab_size)
word_embeddings = data_mapping.word_vectors

word_embedding_dim = word_embeddings.shape[1]
lstm_dropouts = [0, 0.1, 0.3]
lstm_layers = [2, 3, 4]
pos_emb_dims = [15, 25]
hidden_dims = [0.3, 0.4, 0.5]
mlp_dims = [0.3, 0.4, 0.5]

for vectors_str in vectors_strs:
    for lstm_dropout in lstm_dropouts:
        for lstm_layer in lstm_layers:
            for pos_emb_dim in pos_emb_dims:
                for hidden in hidden_dims:
                    for mlp in mlp_dims:
                        hidden_dim = int((word_embedding_dim + pos_emb_dim) * hidden)
                        mlp_dim = int(hidden_dim * 2 * mlp)
                        net = advancedDependencyParser(word_embeddings, pos_vocab_size, pos_emb_dim, hidden_dim,
                                                       mlp_dim, lstm_layers, lstm_dropout)
                        path_str = f"models/{vectors_str.replace('.', '')}_{str(lstm_dropout).replace('.', '')}_{lstm_layer}_" \
                                   f"{pos_emb_dim}_{hidden_dim}_{mlp_dim}_"
                        train_model.train(net, train_loader, test_loader, path=path_str, epochs=10)






