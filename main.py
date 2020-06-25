import pickle
from torch.utils.data.dataloader import DataLoader
from data_set import DepParserDataset
from data_reader import DataMapping
from model import *
import train_model


path_train = "HW2-files/train.labeled"
path_train2 = "HW2-files/new_data_train.labeled"
path_test = "HW2-files/test.labeled"
path_comp = "HW2-files/comp.unlabeled"


# --------- training basic model ---------- #
paths_list = [path_train, path_test]
data_mapping = DataMapping(paths_list)
data_mapping.save(f"models/data_mapping_basic.pt")
word_vocab_size, pos_vocab_size = len(data_mapping.word_idx_mappings), len(data_mapping.pos_idx_mappings)
train = DepParserDataset(data_mapping, path_train)
train_loader = DataLoader(train, shuffle=True)
test = DepParserDataset(data_mapping, path_test, alpha_dropout=0)
test_loader = DataLoader(test, shuffle=False)

for i in range(10):
  net = BasicDependencyParser(word_vocab_size, pos_vocab_size)
  train_model.train(net, train_loader, test_loader, path='models/', epochs=10)

# # --------- training advanced model ---------- #
# paths_list = [path_train, path_test, path_comp]
# vectors_strs =  ["glove.840B.300d"]
# pos_emb_dim_lst = [25]
# hidden_dim_fac_lst = [0.5]
# mlp_dim_fac_lst = [0.4]
# lstm_layers_lst = [2]
# lstm_dropout_lst = [0.15]
#
# for vectors_str in vectors_strs:
#     data_mapping = DataMapping(paths_list, vectors_str=vectors_str)
#     word_vocab_size, pos_vocab_size = len(data_mapping.word_idx_mappings), len(data_mapping.pos_idx_mappings)
#     train = DepParserDataset(data_mapping, path_train)
#     train_loader = DataLoader(train, shuffle=True)
#     test = DepParserDataset(data_mapping, path_test, alpha_dropout=0)
#     test_loader = DataLoader(test, shuffle=False)
#     # net = DependencyParser(word_vocab_size, pos_vocab_size)
#     word_embeddings = data_mapping.word_vectors
#     word_embedding_dim = word_embeddings.shape[1]
#     data_mapping.save(f"models/data_mapping_{vectors_str.replace('.', '')}.pkl")
#     for pos_emb_dim in pos_emb_dim_lst:
#         for hidden_dim_fac in hidden_dim_fac_lst:
#             for mlp_dim_fac in mlp_dim_fac_lst:
#                 for lstm_dropout in lstm_dropout_lst:
#                     for lstm_layers in lstm_layers_lst:
#                         hidden_dim = int((word_embedding_dim + pos_emb_dim) * hidden_dim_fac)
#                         mlp_dim = int(hidden_dim * 2 * mlp_dim_fac)
#                         net = DependencyParser(word_embeddings, pos_vocab_size, pos_emb_dim=pos_emb_dim,
#                                                      hidden_dim=hidden_dim, mlp_dim=mlp_dim, lstm_layers=lstm_layers,
#                                                      lstm_dropout=lstm_dropout)
#                         # net = DependencyParserAttention(word_embeddings, pos_vocab_size, pos_emb_dim=pos_emb_dim,
#                         #                              hidden_dim=hidden_dim, mlp_dim=mlp_dim, lstm_layers=lstm_layers,
#                         #                              lstm_dropout=lstm_dropout)
#                         # net = DependencyParserCombinedAttention(word_embeddings, pos_vocab_size, pos_emb_dim=pos_emb_dim,
#                         #                                 hidden_dim=hidden_dim, mlp_dim=mlp_dim, lstm_layers=lstm_layers,
#                         #                                 lstm_dropout=lstm_dropout)
#                         # net = DependencyParserLinear(word_embeddings, pos_vocab_size, pos_emb_dim=40, word_lin_dim=200, pos_lin_dim=25,
#                         #                              hidden_dim=hidden_dim, mlp_dim=mlp_dim, lstm_layers=lstm_layers,
#                         #                              lstm_dropout=lstm_dropout)
#                         # net = DependencyParserTransformer(word_embeddings, pos_vocab_size, pos_emb_dim=pos_emb_dim,
#                         #                              hidden_dim=hidden_dim, mlp_dim=mlp_dim, n_layers=1,
#                         #                              dropout=0.1, nhead=2)
#                         path_str = f"models/{vectors_str.replace('.', '')}_{pos_emb_dim}_{hidden_dim}_{mlp_dim}" \
#                                    f"_{lstm_layers}_{str(lstm_dropout).replace('.', '')}"
#                         print('\n#----------', path_str.replace('_', ' '), '----------#')
#                         train_model.train(net, train_loader, test_loader, path=path_str, epochs=100)