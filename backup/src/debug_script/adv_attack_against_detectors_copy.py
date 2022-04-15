"""
produce generic adversarial malware that evade multiple malware detectors,
or perform adversarial attack against multiple malware detectors at same time,
different adversarial attacks can be selected (e.g., FGSM, PGD, DeepFool, etc)

200 malware test samples
label: 0 --> benign; 1 --> malware

index to perturbate:
1. DOS Header Attack: all bytes except two magic numbers "MZ" [2,0x3c) and 4 bytes values at [0x3c,0x40] (about 243 bytes for Full DOS)
2. Extended DOS Header Attack: bytes from DOS Header Attacks, plus new extended space of DOS Header
3. Content Shift Attack: new created space between PE header and first section

steps:
1. obtain index to perturb
2. initiate obtained index_to_perturb: either with bytes from benign samples or random bytes
3. applied with adversarial example generation method on the index_to_perturb
4. obtained best adversarial malware that can evade detectors or terminate until max iterations
"""
import os
os.sys.path.append('..')

import torch
import time,sys,os
import pandas as pd
from torch.utils.data import DataLoader
from src.util import ExeDataset, get_acc_FP
from src.model import MalConv_freezeEmbed,FireEye_freezeEmbed
import yaml
import numpy as np

from attacks.attackbox import FGSM
from attacks.attackbox.attack_utils import get_perturb_index_and_init_x
from src.util import forward_prediction_process




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



def get_predicted_malware(model_1,data_iterator,device,model_2=None,verbal=True):
    """
    one model: discard samples that assigned label with benign, otherwise keep for attacking
    two models: discard samples that assigned label with benign by both models, otherwise keep for attacking
    label: 1--> malware, 0--> benign

    return:
        - the group of predicted malwares (list)
        - the list of corresponding labels
    """
    forward_prediction = forward_prediction_process()

    X_mal,y_mal = [],[]
    for i,batch in enumerate(data_iterator):
        X,y = batch
        X, y = X.float().to(device), y.long().to(device)

        confidence_1,_ = forward_prediction._forward(X, model_1)
        if model_2:
            confidence_2,_ = forward_prediction._forward(X, model_2)
            if confidence_1[0] > 0.5 and confidence_2[0] > 0.5:
                continue
            else:
                X_mal.append(X.cpu().squeeze().numpy())
                y_mal.append(y.item())
        else:
            if confidence_1[0] < 0.5:
                X_mal.append(X.squeeze().numpy())
                y_mal.append(y.item())
    print(f"Add {len(X_mal)} Malwares.")

    if verbal:
        output_path = '../../result/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(output_path+'input_malware_stat.txt','w') as f:
            print(f'{len(data_iterator)} malwares in total.', file=f)
            print(f'{len(X_mal)} malwares predicted as malware at least by one detector', file=f)
            print(f'These {len(X_mal)} malwares will be used to produce adversarial malwares.', file=f)
    return X_mal,y_mal



def test_loop(model_1, data_iterator, adversary, device,model_2=None,verbal=True):
    """
    for each input x, get adv_y and adv_x;
    and output the accuracy sum(adv_y==y)/len(y)
    """
    num_pert_bytes = [] # to record the number of bytes perturbed each sample
    num_samples = 0
    adv_labels = []
    labels = []
    start = time.time()

    ## filter malwares that predicted as benign ones
    ## i.e., discard predicted benign and get predicted malware
    X_mal, y_mal = get_predicted_malware(model_1,data_iterator,device,model_2=model_2)

    not_pe_file_count = 0
    for i,(X,y) in enumerate(zip(X_mal,y_mal)):
        num_samples += 1
        print('\n',f"sample {i} under {opts['adversary']} attack.",'\n')
        labels.append(y)

        ## convert list of integer (range must in [0,255]) to bytearray
        ## bytearray(list of integer): --> binary file
        ## list(binary file): --> list of integer
        # gg = np.where(X==256)[0].size
        X_bytearray= bytearray(list(np.array(X,dtype=int))) #convert integer to bytes; e.g.,[0,1] --> [b'\x00\x01]

        ## get index of perturbation.
        ## use try and exception to filter the corrupted input files (e.g. not a PE binary file)
        try:
            index_to_perturb, x_init = get_perturb_index_and_init_x(input=X_bytearray,
                                                                    preferable_extension_amount=opts['preferable_extension_amount'],
                                                                    preferable_shift_amount=opts['preferable_shift_amount'],
                                                                    max_input_size=opts['first_n_byte'])
        except:
            not_pe_file_count += 1
            continue

        ## convert initiated x (applied with adversarial attacks) from bytearray to list of integer
        ## cut the size of input to fixed number since the DOS/Shift attack increase length
        X_init = np.array(list(x_init))[:opts['first_n_byte']]

        if model_2:
            (adv_x_preds_1, adv_y_1), (adv_x_preds_2, adv_y_2), pert_size = adversary.perturbation(X_init,index_to_perturb=index_to_perturb)
            adv_labels.append((adv_y_1,adv_y_2))
        else:
            adv_x_preds_1, adv_y_1, pert_size = adversary.perturbation(X_init,index_to_perturb=index_to_perturb)
            adv_labels.append(adv_y_1)

        num_pert_bytes.append(pert_size)

    end = time.time()

    single_model = (False if model_2 else True)
    acc,FP,(acc1,FP1),(acc2,FP2) = get_acc_FP(adv_labels,single_model=single_model)
    print('------------------------------------------------')
    print(f'Two Models {not single_model}, Acc against adv_x (at least one detector predicted as malware): {acc}')
    print(f'False Positive: {FP}, which evade detectors successfully. ({len(adv_labels)} test samples in total)')
    print(f'Model1: {acc1} accuracy for detecting adversarial malware. FP: {FP1} number of adv_x evade successfully.')
    print(f'Model2: {acc2} accuracy for detecting adversarial malware. FP: {FP2} number of adv_x evade successfully.')
    print(f'average perturbation overhead: {len(num_pert_bytes)/num_samples} bytes')
    print(f'{len(adv_labels)} out of {i + 1} PE files used as testing files (to evade detectors)')
    print(f'{not_pe_file_count} files that can not processed/identified as PE binary. Filter out!')
    print(f'test time: {int((end - start) / 60)}m {int((end - start) % 60)}s.')

    if verbal:
        with open('../../result/evasion_results.txt', 'w') as f:
            print(f'Two Models {not single_model}, Acc against adv_x (at least one detector predicted as malware): {acc}',file=f)
            print(f'False Positive: {FP}, which evade detectors successfully. ({len(adv_labels)} test samples in total)',file=f)
            print(f'Model1: {acc1} accuracy for detecting adversarial malware. FP: {FP1} number of adv_x evade successfully.',file=f)
            print(f'Model2: {acc2} accuracy for detecting adversarial malware. FP: {FP2} number of adv_x evade successfully.',file=f)
            print(f'average perturbation overhead: {len(num_pert_bytes) / num_samples} bytes.',file=f)
            print(f'{len(adv_labels)} out of {i + 1} PE files used as testing files (to evade detectors).',file=f)
            print(f'{not_pe_file_count} files that can not processed/identified as PE binary. Filter out!',file=f)
            print(f'test time: {int((end - start) / 60)}m {int((end - start) % 60)}s.', file=f)


def main_test(opts):
    "gpu/cpu"
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('CUDA AVAIBABEL: ', torch.cuda.is_available())

    "load data"
    test_label_table = pd.read_csv(opts['test_label_path'], header=None, index_col=0)
    # test_label_table.index = test_label_table.index.str.upper()  # upper the string
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

    model_1 = MalConv_freezeEmbed(max_input_size=opts['first_n_byte'], window_size=opts['window_size']).to(device)
    model_2 = FireEye_freezeEmbed().to(device)

    model_1.load_state_dict(torch.load(opts['model_path_1'], map_location=device))
    model_2.load_state_dict(torch.load(opts['model_path_2'], map_location=device))

    model_1.eval()
    model_2.eval()


    "load adversary"
    if opts['adversary'] == 'FGSM':
        adversary = FGSM(model_1,model_2,eps=0.5,w_1=0.5,w_2=0.5,random_init=True,pert_init_with_benign=True)
    else:
        print('please specify an adversary')
        sys.exit()

    "get adv_y, adv_x"
    test_loop(model_1,test_data_loader,adversary,device,model_2=model_2)


if __name__ == '__main__':
    "fix random seed"
    torch.backends.cudnn.deterministic = True

    config_path = '../../config/config_two_models_adv.yaml'

    opts = yaml.load(open(config_path, 'r'))

    print(f'working dir is {config_path}')
    main_test(opts)
    print(f"adversary: {opts['adversary']}, working dir: {config_path}")


