"""
train malware detector DNN model from fireeye, (DNN fireeye integrated in secml package)
training on 720 benign and 720 malware dataset
validation data: 80 benign and 80 malware
"""
import os
os.sys.path.append('..')

import time
import yaml
import pandas as pd
from src.util import ExeDataset
from secml_malware.models.dnn_fireeye import DNN

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim




def binary_acc(preds,y):
    """
    Binary classfication: input pred is single value (after sigmoid function),
    """
    rounded_preds = torch.round(preds)
    correct = (rounded_preds==y).float()
    acc = correct.sum() / len(correct)

    return acc

def binary_acc_1(preds,y):
    "Binary_classfication: input preds are two values (two classes)"
    preds = torch.argmax(preds, 1)
    y = y.squeeze(1)
    correct = (preds == y)
    acc = float(correct.sum()) / len(correct)

    return acc




def evaluate_step(model, data_iterator,criterion,device):

    epoch_loss = 0
    epoch_acc = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in data_iterator:
            X,y = batch
            X,y = X.float().to(device), y.float().to(device)

            preds = model(X)
            preds,y = preds.squeeze(),y.squeeze()
            loss = criterion(preds,y)

            acc = binary_acc(preds,y)

            epoch_loss += loss.item()
            epoch_acc += acc
    avg_loss = epoch_loss/len(data_iterator)
    avg_acc = epoch_acc/len(data_iterator)

    return avg_loss, avg_acc


def train_step(model,data_iterator,optimizer,criterion,device):

    epoch_loss, epoch_acc = 0,0

    model.to(device)
    model.train()

    for batch in data_iterator:
        X, y = batch
        X, y = X.float().to(device), y.float().to(device)

        optimizer.zero_grad()

        preds = model(X)
        preds,y = preds.squeeze(),y.squeeze()
        loss = criterion(preds,y)
        acc = binary_acc(preds,y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    avg_loss = epoch_loss/len(data_iterator)
    avg_acc = epoch_acc/len(data_iterator)

    return avg_loss,avg_acc


def epoch_time(start,end):
    elapsed_time = end-start
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins*60))
    return elapsed_mins,elapsed_secs



def train_loop(epochs,model,train_data,val_data,lr,model_path,device):

    best_val_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # binary loss

    for epoch in range(1,epochs+1):

        start_time = time.time()

        train_loss,train_acc = train_step(model,train_data,optimizer,criterion,device)
        valid_loss,valid_acc = evaluate_step(model,val_data,criterion,device)

        end_time = time.time()

        epoch_mins,epoch_secs = epoch_time(start_time,end_time)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            print(f'Checkpoint saved at: {model_path}')

        print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc:.3f}')

        "update log file"
        # step_time = '%dm%ds' % (epoch_mins,epoch_secs)
        # print(log_msg.format(epoch, train_loss, train_loss, valid_loss, valid_acc, step_time),file=log, flush=True)



def main_train(opts):

    "gpu/cpu"
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('CUDA AVAIBABEL: ', torch.cuda.is_available())

    "log file"
    log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
    log_file = '%s%s.log' % (opts['log_dir'],opts['example'])
    log = open(log_file,'w')
    log.write('epoch,train_loss, train_acc, val_loss, val_acc, time\n')

    "load data"
    train_label_table = pd.read_csv(opts['train_label_path'], header=None, index_col=0)
    # train_label_table.index = train_label_table.index.str.upper()  # upper the string
    train_label_table = train_label_table.rename(columns={1: 'ground_truth'})
    val_label_table = pd.read_csv(opts['valid_label_path'], header=None, index_col=0)
    # val_label_table.index = val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

    # output the statistic of loading data
    print('Training Set:')
    print('\tTotal', len(train_label_table), 'files')
    print('\tMalware Count :', train_label_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:', train_label_table['ground_truth'].value_counts()[0])

    print('Validation Set:')
    print('\tTotal', len(val_label_table), 'files')
    print('\tMalware Count :', val_label_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:', val_label_table['ground_truth'].value_counts()[0])

    # Dataloader: parallelly generate batch data
    train_data_loader = DataLoader(ExeDataset(list(train_label_table.index), opts['train_data_path'],
                                              list(train_label_table.ground_truth),opts['first_n_byte']), batch_size=opts['batch_size'],
                                   shuffle=True, num_workers=opts['use_cpu'])
    valid_data_loader = DataLoader(ExeDataset(list(val_label_table.index), opts['valid_data_path'],
                                              list(val_label_table.ground_truth),opts['first_n_byte']), batch_size=opts['batch_size'],
                                   shuffle=False, num_workers=opts['use_cpu'])


    "load model"
    model = DNN(max_input_size=opts['first_n_byte'])

    model_save_path = '%s%s_model.pth' % (opts['checkpoint_dir'],opts['example'])

    "proceed to train loop"
    train_loop(opts['epochs'],model,train_data_loader,valid_data_loader,opts['lr'],model_save_path,device)



if __name__ == '__main__':

    config_path = '../config/config_secml_dnn.yaml'
    opts = yaml.load(open(config_path,'r'))
    main_train(opts)
