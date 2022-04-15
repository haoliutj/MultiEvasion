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
import torchattacks
from src import util




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
    adv_x = adversary(x,y)

    return adv_x


def test_loop(model,data_iterator,adversary,device,payload_size=1000):
    """
    for each input x, get adv_y and adv_x;
    and output the accuracy sum(adv_y==y)/len(y)
    """
    adv_labels = []
    labels = []
    perturbations = []
    start = time.time()
    for i,batch in enumerate(data_iterator):
        X,y = batch
        labels.append(y.item())
        X, y = X.float().to(device), y.long().to(device)    # X to long, because model has a embed layer, which is a lookup table

        # initial adv_x
        pert = np.random.randint(1, 257,payload_size)  # number from 1 to 256, exclude 257; keep consistent with preprocessing step
        adv_x = torch.reshape(torch.Tensor(np.concatenate([X.cpu().numpy().squeeze()[:-payload_size], pert])),X.shape)

        "use the head of input as the appending payload "

        # get embed for input
        embed = model.embed
        embed_x = embed(X.long()).detach()
        embed_adv_x = embed(adv_x.long().to(device))
        embed_adv_x.require_grad = True

        # normalize embed to 0-1
        normalizer = util.data_normalize_inverse(embed_adv_x)
        embed_adv_x_norm = normalizer.data_normalize()
        embed_adv_x_norm = torch.tensor(embed_adv_x_norm,requires_grad=True,device=device)

        adversary.normalizer = normalizer
        # embed_adv_x_norm = get_adv_x_y(embed_adv_x_norm,y,model,adversary)
        embed_adv_x_norm = adversary(embed_adv_x_norm,y)

        # inverse normalization
        embed_adv_x = normalizer.inverse_normalize(embed_adv_x_norm.detach().numpy())

        # update adv_x in embed format: only craft the last payload size bytes
        embed_adv_x = np.concatenate([embed_x.squeeze().detach().numpy()[:-payload_size],embed_adv_x.squeeze()[-payload_size:]])
        embed_adv_x = torch.tensor(np.reshape(embed_adv_x,embed_x.shape),device=device)

        adv_pred = model(embed_adv_x)
        adv_y = torch.argmax(adv_pred, 1)

        adv_labels.append((adv_y.item()))
        # perturbations.append(sum(pert))

        print(f'input {i+1}, adversarial label for input {i+1} is {adv_y.item()}')

    end = time.time()

    acc,FP = binary_acc(adv_labels,labels)
    print(f'The accuracy of malware detector against adv_x is {acc}')
    print(f'FP: {FP}')
    print(f'Average perturbation is {payload_size} bytes')
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
                                   shuffle=True, num_workers=opts['use_cpu'])

    "load model"
    if opts['mode'] == 'MalConv':
        model = MalConv_freezeEmbed(max_input_size=opts['first_n_byte'], window_size=opts['window_size']).to(device)

    elif opts['mode'] == 'FireEye':
        model = FireEye_freezeEmbed().to(device)
    else:
        print('choose the right model between MalConv and FireEye.')
    model.load_state_dict(torch.load(opts['model_path'], map_location=device))
    model.eval()

    "get payload_size"
    # kernel_size = opts['window_size']
    # input_size = opts['first_n_byte']
    # payload_size = kernel_size + (kernel_size - np.mod(input_size, kernel_size))
    payload_size = int(opts['first_n_byte'] / 100)

    "load adversary"
    if opts['adversary'] == 'FGSM':
        adversary = torchattacks.FGSM(model, eps=8 / 255)
    elif opts['adversary'] == 'PGD':
        torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
    elif opts['adversary'] == 'cw':
        adversary = torchattacks.CW_2(model, c=1, kappa=0, steps=1000, lr=0.1)
        # adversary = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=7)

        # target mode: aim to convet mal (1) to benign (0)
        target_map_function = lambda images, labels: labels.fill_(0)
        adversary.set_targeted_mode(target_map_function=target_map_function)
    else:
        print('please specify an adversary')
        sys.exit()

    "get adv_y, adv_x"
    test_loop(model,test_data_loader,adversary,device)


if __name__ == '__main__':
    # config_path = '../config/config_fireeye_adv.yaml'
    config_path = '../config/config_white_box_attack.yaml'

    opts = yaml.load(open(config_path, 'r'))

    print(f'working dir is {config_path}')
    main_test(opts)
    print(f"adversary: {opts['adversary']}, target on {opts['mode']}, working dir: {config_path}")


