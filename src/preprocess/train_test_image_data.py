"""
copy image into following file structure based on pre-defined train-val-test table
--image
    --train
        --benign
        --mal
    --test
        --benign
        --mal
    --val
        --benign
        --mal
"""

import pandas as pd
import os
import shutil



# csv_path = '../../data/test_data_label.csv'
# files = pd.read_csv(csv_path,header=None)
# ff = files.iloc[:,0].tolist()


def get_file_list(csv_path):
    df = pd.read_csv(csv_path, header=None)
    file_list = df.iloc[:,0].tolist()

    return file_list


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(path + 'mal'):
        os.makedirs(path + 'mal')

    if not os.path.exists(path + 'benign'):
        os.makedirs(path + 'benign')


def save_file2folders(path=None,file_list=None, output_path=None):
    """

    :param path: the original file path
    :param file_list:
    :param output_path: path to save copy file ('../../data/image/train/','../../data/image/test/','../../data/image/val/')
    :return:
    """
    ## create folders
    create_folder(output_path)

    for f in file_list:
        print(f)
        if f.startswith('mal'):
            shutil.copy(path + f + '.png', output_path+'mal')
        elif f.startswith('benign'):
            shutil.copy(path + f + '.png', output_path + 'benign')

    print('copy data finished!')


def main():
    all_file_path = '../../data/image/'
    train_file_path = '../../data/train_data_label.csv'
    test_file_path = '../../data/test_data_label.csv'
    val_file_path = '../../data/val_data_label.csv'

    train_file_list = get_file_list(train_file_path)
    save_file2folders(path=all_file_path,file_list=train_file_list,output_path='../../data/image_split/train/')

    test_file_list = get_file_list(test_file_path)
    save_file2folders(path=all_file_path, file_list=test_file_list, output_path='../../data/image_split/test/')

    val_file_list = get_file_list(val_file_path)
    save_file2folders(path=all_file_path, file_list=val_file_list, output_path='../../data/image_split/val/')

if __name__ == '__main__':
    main()