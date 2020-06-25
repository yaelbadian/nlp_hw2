from torch.utils.data.dataloader import DataLoader
from data_set import DepParserDataset
from data_reader import DataMapping, split
from model import DependencyParser
from advanced_model import DependencyParser as AdvancedDependencyParser
from train_model import predict_dep

data_mapping_path = 'models/'
basic_model_weights = 'models/'
advanced_model_weights = 'models/'
comp_path = 'HW2-files/comp.unlabeled'
basic_output_path = 'comp_basic.labeled'
advanced_output_path = 'comp_advanced.labeled'


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
                splitted_line[6] = predictions[i][j]
                line = '\t'.join(splitted_line)
                w_file.write(line)
                j += 1


if __name__ == '__main__':
    data_mapping = DataMapping.load(data_mapping_path)
    comp = DepParserDataset(data_mapping, comp_path, alpha_dropout=0)
    comp_loader = DataLoader(comp, shuffle=False)

    basic_model = DependencyParser()
    basic_predictions = predict(basic_model, comp_loader)
    create_output_comp(comp_path, basic_output_path, basic_predictions)
    advanced_model = AdvancedDependencyParser()
    advanced_predictions = predict(advanced_model, comp_loader)
    create_output_comp(comp_path, advanced_output_path, advanced_predictions)



