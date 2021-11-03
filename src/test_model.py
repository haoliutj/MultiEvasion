"test the performance of malware detector: Binary classification: sigmoid"

import os
os.sys.path.append('..')

import torch
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pandas as pd
import yaml

# from src.model import MalConv,FireEye
from src.model import FireEye
from secml_malware.models.malconv import MalConv
from src.util import ExeDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.model import MalConv_freezeEmbed,FireEye_freezeEmbed


class test_model:
    def __init__(self,opts):
        self.opts = opts

        self.mode = opts['mode']
        self.model_path = '../checkpoint/%s' % opts['model_name']
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL: ', torch.cuda.is_available())
        print('test model path: %s' % self.model_path)



    def binary_acc(self,preds, y):

        # round predictions to cloest integer; sigmoid -->[0,1]
        # rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (preds == y).float()
        acc = correct.sum() / len(correct)

        return acc


    def testing(self):
        "load data"
        test_label_path = self.opts['test_label_path']
        test_data_path = self.opts['test_data_path']

        # data pre-process
        test_label_table = pd.read_csv(test_label_path, header=None, index_col=0)
        test_label_table.index = test_label_table.index.str.upper()  # upper the string
        test_label_table = test_label_table.rename(columns={1: 'ground_truth'})

        # data loader
        test_loader = DataLoader(
            ExeDataset(list(test_label_table.index), test_data_path, list(test_label_table.ground_truth), self.opts['first_n_byte']),
            batch_size=self.opts['batch_size'], shuffle=False, num_workers=self.opts['use_cpu'])


        "load model"
        if self.mode == 'MalConv':
            model = MalConv(max_input_size=self.opts['first_n_byte'], window_size=self.opts['window_size']).to(self.device)
        elif self.mode == 'FireEye':
            model = FireEye_freezeEmbed().to(self.device)

        else:
            print('choose the right model between MalConv and FireEye.')


        model.load_state_dict(torch.load(self.model_path,map_location=self.device))
        model.eval()

        "testing all samples"

        # accuracy = 0
        y_pred,y_test = [],[]
        for i, data in enumerate(test_loader):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred_lab = model(test_x)

            "binary for two values"
            preds = torch.argmax(pred_lab,1)
            "for binary single value"
            # preds = torch.round(pred_lab)

            y_test += (test_y.cpu().detach().numpy().squeeze().tolist())
            y_pred += (preds.cpu().detach().numpy().tolist())
        # print(f'accuracy is {accuracy/len(test_loader)}')
        print('confusion matrix:')
        print(confusion_matrix(y_test, y_pred))
        print(f"classification report: ".format(classification_report(y_test, y_pred,labels=[0,1])))
        print(classification_report(y_test, y_pred,labels=[0,1]))
        print(f'accuracy is {accuracy_score(y_test, y_pred)}')



def main(opts):
    testing_model = test_model(opts)
    testing_model.testing()


def get_opts_malconv():

    return{
        'mode': 'MalConv',
        'model_name': 'malcov_secml_model.pth',
        'first_n_byte': 102400,
        'window_size': 500,
        'test_label_path':'../data/test_data_label.csv',
        'test_data_path': '../data/all_file/',
        'batch_size': 32,
        'use_cpu':1,
    }


def get_opts_fireeye():

    return{
        'mode': 'FireEye',
        'model_name': 'fireeye_model.pth',
        'first_n_byte': 102400,
        'window_size': 512,
        'test_label_path':'../data/test_data_label.csv',
        'test_data_path': '../data/all_file/',
        'batch_size': 1,
        'use_cpu':1,
    }

if __name__ == '__main__':

    # opts = get_opts_malconv()
    opts = get_opts_fireeye()


    main(opts)