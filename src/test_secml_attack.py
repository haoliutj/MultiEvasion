import os
os.sys.path.append('..')

import torch
from attacks.attack import FGSM,PGD
import copy,time,sys
import pandas as pd
from torch.utils.data import DataLoader
from src.util import ExeDataset, ExeDataset_Malware
from src.model import MalConv_freezeEmbed,FireEye_freezeEmbed
import yaml
import numpy as np

from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
from secml_malware.models.basee2e import End2EndModel




def binary_acc(preds,y):
    "input are list, add one dimension to compare if equal"
    preds = np.array(preds)[np.newaxis:]
    y = np.array(y)[np.newaxis:]

    corrects = (preds==y)
    acc = float(sum(corrects))/len(corrects)
    FP = len(corrects) - corrects.sum()
    return acc,FP






def binary_acc_1(preds,y):

    preds = torch.argmax(preds, 1)
    y = y.squeeze(1)
    correct = (preds == y)
    acc = float(correct.sum()) / len(correct)
    FP = len(correct) - correct.sum()

    return acc,FP


def get_adv_x_y(x,y,model,adversary):
    """
    For input x, output the corresponding adv_x and adv_y
    Note: input y here is the prediction y (i.e. y=model(x)), to prevent label leaking
    """
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False

    model_cp.eval()

    adversary.model = model_cp
    adv_y,adv_x,pert = adversary.perturbation(x,y)

    return adv_y,adv_x,pert


def test_loop(model,data_iterator,adversary,device):
    """
    for each input x, get adv_y and adv_x;
    and output the accuracy sum(adv_y==y)/len(y)
    """
    adv_labels = []
    labels = []
    perturbations = []
    start = time.time()
    for i,batch in enumerate(data_iterator):
        print(f"sample: {i} under {opts['adversary']} attack.",'\n')
        X,y = batch
        labels.append(y.item())
        X, y = X.float().to(device), y.long().to(device)    # X to long, because model has a embed layer, which is a lookup table

        # embed = model.embed
        # m = embed(torch.arange(0, 256).to(device))  # create lookup table [0,256]

        embed_x = End2EndModel.embed(input_x=X.long(),transpose=False)
        embed_x.require_grad = True

        #get the pred of X for adv_x generation to prevent label leaking
        #y_pred = torch.round(torch.sigmoid(model(X)))
        # y_pred = torch.argmax(model(embed_x),1)
        y_target = torch.tensor([0], dtype=torch.long)    # approach to the benign class
        adv_y,_,pert = get_adv_x_y(X,y_target,model,adversary)

        adv_labels.append((adv_y.item()))
        # perturbations.append(sum(pert))

    end = time.time()

    acc,FP = binary_acc(adv_labels,labels)
    print(f'The accuracy of malware detector against adv_x is {acc}')
    print(f'FP: {FP}')
    print(f'perturbation overhead is {len(pert)} bytes')
    print(f'test time is {int((end-start)/60)}m {int((end-start)%60)}s.')


def main_test(opts):
    "gpu/cpu"
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('CUDA AVAIBABEL: ', torch.cuda.is_available())

    "load data"
    test_label_table = pd.read_csv(opts['test_label_path'], header=None, index_col=0)
    test_label_table.index = test_label_table.index.str.upper()  # upper the string
    test_label_table = test_label_table.rename(columns={1: 'ground_truth'})

    # output the statistic of loading data
    print('Testing Set:')
    print('\tTotal', len(test_label_table), 'files')
    print('\tMalware Count :', test_label_table['ground_truth'].value_counts()[1])
    # print('\tGoodware Count:', test_label_table['ground_truth'].value_counts()[0])

    test_data_loader = DataLoader(ExeDataset(list(test_label_table.index), opts['test_data_path'],
                                              list(test_label_table.ground_truth), opts['first_n_byte']),
                                   batch_size=opts['batch_size'],
                                   shuffle=False, num_workers=opts['use_cpu'])

    "load model"
    if opts['mode'] == 'MalConv':
        model = MalConv_freezeEmbed(max_input_size=opts['first_n_byte'], window_size=opts['window_size']).to(device)

    elif opts['mode'] == 'FireEye':
        model = FireEye_freezeEmbed().to(device)
    else:
        print('choose the right model between MalConv and FireEye.')
    print(model)

    model.load_state_dict(torch.load(opts['model_path'], map_location=device))
    model = CClassifierEnd2EndMalware(model)
    # model.load_pretrained_model(torch.load(opts['model_path']))
    # model.eval()

    "get payload_size"
    # kernel_size = opts['window_size']
    # input_size = opts['first_n_byte']
    # payload_size = kernel_size + (kernel_size - np.mod(input_size, kernel_size))
    payload_size = int(opts['first_n_byte'] / 100)

    "load adversary"
    if opts['adversary'] == 'FGSM':
        adversary = FGSM(payload_size=payload_size,epsilon=0.7,model=model)
    elif opts['adversary'] == 'PGD':
        adversary = PGD(payload_size=payload_size,num_loop=10,epsilon=0.7,model=model)
    elif opts['adversary'] == 'secml':
        adversary = CHeaderEvasion(model,random_init=False,iterations=50,optimize_all_dos=False,threshold=0.5)
    else:
        print('please specify an adversary')
        sys.exit()

    "get adv_y, adv_x"
    test_loop(model,test_data_loader,adversary,device)


if __name__ == '__main__':
    "fix random seed"
    torch.backends.cudnn.deterministic = True

    config_path = '../config/config_secml_attack.yaml'

    opts = yaml.load(open(config_path, 'r'))

    print(f'working dir is {config_path}')
    main_test(opts)
    print(f"adversary: {opts['adversary']}, working dir: {config_path}")


