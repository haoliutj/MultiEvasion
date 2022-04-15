"""
target on two models with appending attack

method: updated final perturbation based on the mean value of two perturbations generated from two models respectively

"""

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



def mean_element_wise(array1,array2):
    """
    given two array, output the mean value element-wise
    array1=[1,2,3], array2=[4,5,6]
    --> [2.5,3.5,4.5]
    """
    out = []
    for i in range(len(array1)):
        out.append(np.mean([array1[i],array2[i]]))
    return np.array(out)


def binary_acc(preds,y):
    "input are list, add one dimension to compare if equal"
    preds = np.array(preds)[np.newaxis:]
    y = np.array(y)[np.newaxis:]

    corrects = (preds==y)
    acc = float(sum(corrects))/len(corrects)
    FP = len(corrects) - corrects.sum()
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


def test_loop(model1,model2,data_iterator,adversary1,adversary2,device,payload_size):
    """
    model1: malconv
    model2: fireEye
    """
    adv_labels1 = []
    adv_labels2 = []
    labels = []
    perturbations = []
    start = time.time()
    for i,batch in enumerate(data_iterator):
        print(f'sample: {i}','\n')
        X,y = batch
        labels.append(y.item())
        X, y = X.float().to(device), y.long().to(device)    # X to long, because model has a embed layer, which is a lookup table

        embed1 = model1.embed
        embed2 = model2.embed
        # m = embed(torch.arange(0, 256))  # create lookup table [0,256]

        # get the embed information
        embed1_x = embed1(X.long()).detach()
        embed2_x = embed2(X.long()).detach()
        embed1_x.require_grad = True
        embed2_x.rwquire_grad = True

        # prepare the target label bengin
        y_target = torch.tensor([0], dtype=torch.long)    # approach to the benign class

        # get perturbation
        _,_,pert1 = get_adv_x_y(X,y_target,model1,adversary1)
        _,_,pert2 = get_adv_x_y(X,y_target,model2,adversary2)

        # get final perturbation
        pert = mean_element_wise(pert1,pert2)
        perturbations.append(sum(pert))

        # get adv_x
        X_sqz = X.squeeze()
        adv_x = np.concatenate([X_sqz.cpu().detach().numpy()[:-payload_size], pert.astype(np.uint8)])
        adv_x = adv_x[np.newaxis, :]        # add dimension

        "get label for generated adv example"
        adv_x = torch.Tensor(adv_x).to(device)
        embed1_adv_x = embed1(adv_x.long()).detach()
        embed2_adv_x = embed2(adv_x.long()).detach()
        adv_pred1 = model1(embed1_adv_x.to(device))
        adv_pred2 = model2(embed2_adv_x.to(device))
        adv_y1 = torch.argmax(adv_pred1,1)
        adv_y2 = torch.argmax(adv_pred2,1)

        adv_labels1.append((adv_y1.item()))
        adv_labels2.append((adv_y2.item()))

        "print result of each batch"
        print('\n','-'*30)
        print(f'ensemble adv_x against Malconv: {adv_y1.item()}')
        print(f'ensemble adv_x against FireEye: {adv_y2.item()}')
        print('-' * 30)


    end = time.time()

    acc1,FP1 = binary_acc(adv_labels1,labels)
    acc2,FP2 = binary_acc(adv_labels2,labels)
    print(f'The accuracy of malware detector against adv_x is {acc1}')
    print(f'FP: {FP1}')
    print('*'*30)
    print(f'The accuracy of malware detector against adv_x is {acc2}')
    print(f'FP: {FP2}')
    print(f'Average perturbation is {sum(perturbations) / len(perturbations)}')
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
    model_malconv = MalConv_freezeEmbed(max_input_size=opts['first_n_byte'], window_size=opts['window_size']).to(device)
    model_fireye = FireEye_freezeEmbed().to(device)

    model_malconv.load_state_dict(torch.load(opts['malconv_model_path'], map_location=device))
    model_fireye.load_state_dict(torch.load(opts['fireye_model_path'], map_location=device))

    model_malconv.eval()
    model_fireye.eval()

    "get the size of perturbation (payload_size)"
    # kernel_size = opts['window_size']
    # input_size = opts['first_n_byte']
    # payload_size = kernel_size + (kernel_size - np.mod(input_size, kernel_size))
    payload_size = int(opts['first_n_byte']/100)

    "load adversary"
    if opts['adversary'] == 'FGSM':
        adversary1 = FGSM(epsilon=0.7,model=model_malconv,payload_size=payload_size)
        adversary2 = FGSM(epsilon=0.7,model=model_fireye,payload_size=payload_size)
    elif opts['adversary'] == 'PGD':
        adversary1 = PGD(payload_size=payload_size,num_loop=10,epsilon=0.7,model=model_malconv)
        adversary2 = PGD(payload_size=payload_size,num_loop=10,epsilon=0.7,model=model_fireye)
    else:
        print('please specify an existing adversary')
        sys.exit()

    "get adv_y, adv_x"
    test_loop(model_malconv,model_fireye,test_data_loader,adversary1,adversary2,device,payload_size)


if __name__ == '__main__':
    config_path = '../config/config_two_models_adv.yaml'
    opts = yaml.load(open(config_path, 'r'))

    main_test(opts)


