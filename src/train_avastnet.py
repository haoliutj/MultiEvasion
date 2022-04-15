import os
os.sys.path.append('..')

import time
import pandas as pd
from src.util import ExeDataset
from src.model import AvastNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



def binary_acc(preds,y):

    # round predictions to cloest integer; sigmoid -->[0,1]
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    acc = correct.sum() / len(correct)

    return acc

def binary_acc_1(preds,y):

    preds = torch.argmax(preds, 1)
    y = y.squeeze(1)
    correct = (preds == y)
    acc = float(correct.sum()) / len(correct)

    return acc

def testing_step(model,data_iterator,device):
    """
    get acc, comfusion matrix
    :param model:
    :param data_iterator:
    :param device:
    :return:
    """
    y_pred, y_test = [], []
    with torch.no_grad():
        for i, data in enumerate(data_iterator):
            test_x, test_y = data
            test_x, test_y = test_x.to(device), test_y.to(device)
            pred_lab = model(test_x)

            preds = torch.argmax(pred_lab, 1)   # binary for two values
            y_test += (test_y.cpu().detach().numpy().squeeze().tolist())
            y_pred += (preds.cpu().detach().numpy().tolist())

    print('--------> testing results <---------')
    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print(f"classification report: ".format(classification_report(y_test, y_pred, labels=[0, 1])))
    print(classification_report(y_test, y_pred, labels=[0, 1]))
    print(f'accuracy is {accuracy_score(y_test, y_pred)}')

    ## log
    with open(args.log_file_path,'a') as f:
        print('--------> testing results <---------',file=f)
        print('confusion matrix:',file=f)
        print(confusion_matrix(y_test, y_pred),file=f)
        print(f"classification report: ".format(classification_report(y_test, y_pred, labels=[0, 1])),file=f)
        print(classification_report(y_test, y_pred, labels=[0, 1]),file=f)
        print(f'accuracy is {accuracy_score(y_test, y_pred)}',file=f)


def evaluate_step(model, data_iterator,criterion,device):

    epoch_loss = 0
    epoch_acc = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in data_iterator:
            X,y = batch
            X,y = X.float().to(device), y.long().to(device)

            preds = model(X)
            loss = criterion(preds,y.squeeze(1))

            acc = binary_acc_1(preds,y)

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
        X, y = X.float().to(device), y.long().to(device)

        optimizer.zero_grad()

        preds = model(X)
        loss = criterion(preds,y.squeeze(1))
        acc = binary_acc_1(preds,y)

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
    criterion = nn.CrossEntropyLoss()

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



def main_train():

    "gpu/cpu"
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('CUDA AVAIBABEL: ', torch.cuda.is_available())

    "load data"
    train_label_table = pd.read_csv(args.train_label_path, header=None, index_col=0)
    train_label_table.index = train_label_table.index.str.upper()  # upper the string
    train_label_table = train_label_table.rename(columns={1: 'ground_truth'})
    test_label_table = pd.read_csv(args.test_label_path, header=None, index_col=0)
    test_label_table.index = test_label_table.index.str.upper()  # upper the string
    test_label_table = test_label_table.rename(columns={1: 'ground_truth'})
    val_label_table = pd.read_csv(args.val_label_path, header=None, index_col=0)
    val_label_table.index = val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

    # output the statistic of loading data
    print('Training Set:')
    print('\tTotal', len(train_label_table), 'files')
    print('\tMalware Count :', train_label_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:', train_label_table['ground_truth'].value_counts()[0])

    print('Testing Set:')
    print('\tTotal', len(test_label_table), 'files')
    print('\tMalware Count :', test_label_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:', test_label_table['ground_truth'].value_counts()[0])

    print('Validation Set:')
    print('\tTotal', len(val_label_table), 'files')
    print('\tMalware Count :', val_label_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:', val_label_table['ground_truth'].value_counts()[0])

    ## log
    with open(args.log_file_path,'a') as f:
        print('Training Set:',file=f)
        print('\tTotal', len(train_label_table), 'files',file=f)
        print('\tMalware Count :', train_label_table['ground_truth'].value_counts()[1],file=f)
        print('\tGoodware Count:', train_label_table['ground_truth'].value_counts()[0],file=f)

        print('Testing Set:',file=f)
        print('\tTotal', len(test_label_table), 'files',file=f)
        print('\tMalware Count :', test_label_table['ground_truth'].value_counts()[1],file=f)
        print('\tGoodware Count:', test_label_table['ground_truth'].value_counts()[0],file=f)

        print('Validation Set:',file=f)
        print('\tTotal', len(val_label_table), 'files',file=f)
        print('\tMalware Count :', val_label_table['ground_truth'].value_counts()[1],file=f)
        print('\tGoodware Count:', val_label_table['ground_truth'].value_counts()[0],file=f)
        print('\n',file=f)

    # Dataloader: parallelly generate batch data
    train_data_loader = DataLoader(ExeDataset(list(train_label_table.index),args.all_file_path,
                                              list(train_label_table.ground_truth),args.input_size),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    test_data_loader = DataLoader(ExeDataset(list(test_label_table.index), args.all_file_path,
                                              list(test_label_table.ground_truth), args.input_size),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers)
    valid_data_loader = DataLoader(ExeDataset(list(val_label_table.index), args.all_file_path,
                                              list(val_label_table.ground_truth),args.input_size),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers)


    "load model"
    model = AvastNet(vocab_size=256)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_save_path = '%s%s_model.pth' % (args.checkpoint_dir,args.model_name)

    "proceed to train loop"
    train_loop(args.epochs,model,train_data_loader,valid_data_loader,args.lr,model_save_path,device)

    ## test trained model
    trained_model = model
    trained_model.load_state_dict(torch.load(model_save_path,map_location=device))
    trained_model.eval()
    testing_step(trained_model,test_data_loader,device)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train malware detector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_dir',default='../checkpoint/',type=str,help='checkpoint path')
    parser.add_argument('--train_label_path',default='../data/train_data_label.csv',type=str,help='csv file of train file list')
    parser.add_argument('--test_label_path',default='../data/test_data_label.csv',type=str,help='csv file of test file list')
    parser.add_argument('--val_label_path',default='../data/val_data_label.csv',type=str,help='csv file of validation file list')
    parser.add_argument('--all_file_path',default='../data/all_file/',type=str,help='path stored all files')
    parser.add_argument('--model_name',default='AvastNet',type=str,help='model name')
    parser.add_argument('--input_size',default=102400,type=int,help='input size of model')
    parser.add_argument('--batch_size',default=32,type=int,help='batch size')
    parser.add_argument('--epochs',default=50,type=int,help='number of epoch')
    parser.add_argument('--lr',default=0.0001,type=float,help='learning rate')
    parser.add_argument('--num_workers',default=1,type=int,help='number of workers to load data')
    parser.add_argument('--log_file_path',default='../result/train_log.txt',type=str,help='file to record training details')

    args = parser.parse_args()
    print('\n',args,'\n')

    ## init log file
    log_file_split = args.log_file_path.split('/')
    log_file_folder = ('/').join(log_file_split[:-1]) + '/'
    log_file_name = ('_'+ args.model_name + '.').join(log_file_split[-1].split('.'))
    args.log_file_path = log_file_folder + log_file_name
    if not os.path.exists(log_file_folder):
        os.makedirs(log_file_folder)
    if os.path.exists(args.log_file_path):
        os.remove(args.log_file_path)
    with open(args.log_file_path,'w') as f:
        print('-'*20,'Input Parameters','-'*20,file=f)
        print(args,file=f)
        print('\n',file=f)

    start_time = time.time()
    main_train()
    mins,secs = epoch_time(start_time,time.time())
    print(f'runing time: {mins}m{secs}s')
    with open(args.log_file_path,'a') as f:
        print(f'runing time: {mins}m{secs}s',file=f)
