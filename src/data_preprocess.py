"""
rename files;
build file-id and label and output;
get the statistic of file length;

"""

import os
os.sys.path.append('..')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def file_rename(files,path,title,label,outputpath):
    """
    rename files, and create ground truth data as csv
    file1 label
    file2 label
    """
    file_names = []
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for i,f in enumerate(files):

        old_name = path + f

        new_title = title + '_' + str(i+1)
        new_name = path + new_title
        os.rename(old_name,new_name)

        # ground truth data name
        file_names.append(new_title)

    labels = [label]*(i+1)
    df_data = pd.DataFrame(file_names,columns=['name'])
    df_data['label'] = labels

    df_data.to_csv(outputpath,index=0)


def get_exe(files,path):
    "remove any files that are not end with exe"
    for f in files:
        if not f.endswith('exe'):
            os.remove(path + f)
    print('get exe files end!')


def concatenate_csv(data1, data2, out_path):
    """
    concatenate two data frame
    :param data1:
    :param data2:
    :param out_path:
    :return:
    """
    df = [data1, data2]
    df = pd.concat(df)
    df.to_csv(out_path, index=0)



def split_data(X,y,split_ratio=0.2):
    """
    split data into two
    :param X,y: list like, same length
    :return:
    """
    X_1,X_2,y_1,y_2 = train_test_split(X,y,test_size=split_ratio,random_state=42,shuffle=True,stratify=y)
    return X_1,X_2,y_1,y_2


def write2csv(X,y,out_path):
    df = pd.DataFrame(X,columns=['name'])
    df['label'] = y
    df.to_csv(out_path,index=0)


def get_fileLen_statistic(data_path,filenames):
    """
    the length of each pe file (raw byte)
    """
    file_length = []

    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')

    for i in range(len(filenames)):

        with open(data_path+filenames[i],'rb') as f:

            # directly get the length after read == the length of after convert to [0-255]
            f_len = len(f.read())
            if f_len < 100:
                print(f'{filenames[i]}: {f_len}')

            # file = [i+1 for i in f.read()] # 1-256

        file_length.append(f_len)

    return file_length


def plt_cdf(data,title,output,max_len=None):
    "plot cdf/pdf for data"

    cdf_func = sm.distributions.ECDF(data)
    if not max_len:
        x = np.linspace(min(data),max(data))
    else:
        x = np.linspace(min(data),max_len)
    y = cdf_func(x)

    plt.step(x,y)
    plt.title(title)

    " plot pdf"
    # fig,ax0 = plt.subplots(nrows=1,figsize=(6,6))
    # # second param to control the breadth of bar
    # ax0.hist(data, 10, density=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)

    plt.savefig(output)
    plt.show()


def write_bytes_to_csv():
    """
    convert byte to [1-256]
    """


# ********************************************

def main_rename(dir,label,path):
    "rename files and write label to file"
    # if dir == 'benign':
    #     path = '../../data/mal_self/%s/' % dir
    # elif dir == 'mal':
    #     path = '../data/%s/' % dir
    files = os.listdir(path)
    output_path = '../data/%s_phd_sample_label.csv' % dir
    file_rename(files, path, dir , label, output_path)


def main_get_exe(path):
    # path = '../data/pe/mal/'
    files = os.listdir(path)
    get_exe(files,path)



def main_split_data(data_path):
    "split data into train, val, test"
    data = pd.read_csv(data_path)
    X,y = data['name'],data['label']
    train_val_x,test_data,train_val_y,test_label = split_data(X,y,split_ratio=0.2)
    train_data,val_data,train_label,val_label = split_data(train_val_x,train_val_y,split_ratio=0.125)

    "train, test, val output path"
    train_data_path = '../data/train_data_label_phd_balanced.csv'
    test_data_path = '../data/test_data_label_phd_balanced.csv'
    val_data_path = '../data/val_data_label_phd_balanced.csv'

    write2csv(train_data,train_label,train_data_path)
    write2csv(test_data,test_label,test_data_path)
    write2csv(val_data,val_label,val_data_path)


def main_cdf(filename_path,file_path):
    """
    filename_path: the name of files
    file_path: the actual file's path
    """

    data = pd.read_csv(filename_path,header=None)
    file_names = list(data[0])

    mal_files, benign_files = [], []
    for f in file_names:
        if f.startswith('benign'):
            benign_files.append(f)
        elif f.startswith('mal'):
            mal_files.append(f)
        else:
            print('Abnormal file: ', f)

    # cdf and pdf for bengin
    ben_length = get_fileLen_statistic(file_path,benign_files)
    title = 'benign-distribution'
    output = '../figure/benign_distribution_scaled.eps'
    plt_cdf(ben_length,title,output,max_len=3000000)

    # cdf and pdf for mal
    mal_length = get_fileLen_statistic(file_path,mal_files)
    title = 'malware-distribution'
    output = '../figure/malware_distribution_scaled.eps'
    plt_cdf(mal_length,title,output,max_len=3000000)




if __name__ == '__main__':


    "remove non-exe (zip) files in benign"
    # path = '../../data/mal_self/benign/'
    # main_get_exe(path)

    "rename files: (dir=benign,label=0), (dir=mal, label=1)"
    # dir = 'mal'
    # label = 1
    # path = '../data/phd_dataset/all_malware/'
    # main_rename(dir,label,path)

    " concatenate benign and mal in csv as final dataset"
    # benign = pd.read_csv('../data/benign_phd_sample_label.csv')
    # mal = pd.read_csv('../data/mal_phd_sample_label.csv')
    # out_path = '../data/pe_phd_data_label_balanced.csv'
    # ###### random select same samples to keep benign and mal balance
    # if len(mal)<len(benign):
    #     benign = benign.sample(n=len(mal),random_state=42,axis=0)   # axis=0 -->row
    # elif len(mal) > len(benign):
    #     mal = mal.sample(n=len(benign),random_state=42,axis=0)
    #
    # concatenate_csv(benign,mal,out_path)

    "split data to train, val, test and write to csv"
    all_pe_data = '../data/pe_phd_data_label_balanced.csv'
    main_split_data(all_pe_data)

    "get file length distribution"
    # filename_path = '../data/pe_data_label.csv'
    # file_path = '../data/all_file/'
    # main_cdf(filename_path,file_path)


