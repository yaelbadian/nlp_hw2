import data_reader


path_train = "HW2-files/train.labeled"
path_test = "HW2-files/test.labeled"
paths_list = [path_train, path_test]
word_dict, pos_dict = data_reader.get_vocabs(paths_list)

dataReader = data_reader.DataReader(path_train, word_dict, pos_dict)
print(dataReader.sentences)

