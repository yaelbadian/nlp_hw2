from torch.utils.data.dataloader import DataLoader
from data_set import DepParserDataset
from data_reader import split
from data_reader import DataMapping
from train_model import predict_dep
from model import *
from time import time

basic_data_mapping_path = 'models/basic_data_mapping.pt'
advanced_data_mapping_path = 'models/advanced_data_mapping.pt'
basic_model_path = 'models/basic_model_final.pt'
advanced_model_path = 'models/advanced_model_final.pt'
comp_path = 'HW2-files/comp.unlabeled'
basic_output_path = 'comp_m1_204434161.labeled'
advanced_output_path = 'comp_m2_204434161.labeled'


def predict(net, loader):
    net.eval()
    predictions_lst = []
    for i, sentence in enumerate(loader):
        scores = net(sentence)
        predictions = predict_dep(scores)[:, 1:]
        predictions_lst.append(predictions)
    return predictions_lst


def create_output_comp(path, output_path, predictions):
    i, j = 0, 0
    with open(path, 'r') as r_file:
        with open(output_path, 'w') as w_file:
            for line in r_file:
                if line == '\n':
                    i += 1
                    j = 0
                    w_file.write(line)
                    continue
                splitted_line = split(line, ('\t'))
                splitted_line[6] = predictions[i][0][j]
                line = '\t'.join([str(x) for x in splitted_line])
                w_file.write(line)
                j += 1


if __name__ == '__main__':
    basic_data_mapping = DataMapping.load(basic_data_mapping_path)
    comp = DepParserDataset(basic_data_mapping, comp_path, alpha_dropout=0)
    comp_loader = DataLoader(comp, shuffle=False)
    basic_model = BasicDependencyParser.load(basic_model_path)
    t0 = time()
    basic_predictions = predict(basic_model, comp_loader)
    print(f'Basic model prediction time: {(time() - t0):.2f}s')
    create_output_comp(comp_path, basic_output_path, basic_predictions)

    del basic_data_mapping
    del basic_model
    del basic_predictions

    advanced_data_mapping = DataMapping.load(advanced_data_mapping_path)
    comp = DepParserDataset(advanced_data_mapping, comp_path, alpha_dropout=0)
    comp_loader = DataLoader(comp, shuffle=False)
    advanced_model = DependencyParser.load(advanced_model_path)
    t0 = time()
    advanced_predictions = predict(advanced_model, comp_loader)
    print(f'Advanced model prediction time: {(time() - t0):.2f}s')
    create_output_comp(comp_path, advanced_output_path, advanced_predictions)