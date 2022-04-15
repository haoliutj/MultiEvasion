import os
os.sys.path.append('..')

from src import util
import numpy as np
import time


def converting(mal_file_path,all_file_path,file_type,url=False,csv=False,step=10):
    """

    :param mal_file_path: path for malware file names
    :param all_file_path: path for both benign and malware files
    :param file_type: 'mal' or 'ben'
    :param url: True or False. True: extract all features except url; False: only extract permission, intent, api call
    :param csv: True: save file as csv, otherwise save as npz
    :param step: To control the number of samples to convert each time
    :return:
    """
    start_time = time.time()

    if file_type == 'mal':
        filenames4Convert = util.get_mal_file(mal_file_path)
    else:
        mal_filenames = util.get_mal_file(mal_file_path)
        filenames4Convert = util.get_benign_file(mal_filenames, all_file_path)
    filenames_chunks = util.divide_list(filenames4Convert, step=step)
    "extract features based on parameter url=False/True"
    uniq_features_sorted = util.get_final_features_main(all_file_path, url=url)

    for i, file_chunk in enumerate(filenames_chunks):
        vecs = util.get_onehot_vecs(uniq_features_sorted, file_chunk, all_file_path)

        # build corresponding label 1/0
        if file_type == 'mal':
            y = np.ones(len(vecs), dtype=int)
        else:
            y = np.zeros(len(vecs),dtype=int)

        # save encoded vectors
        if csv:
            outpath = '../data/drebin/drebin_%s_vec_%d.csv' % (file_type,i)
            util.write2csv(vecs, y, outpath)
        else:
            outpath = '../data/drebin/drebin_%s_vec_%d.npz' % (file_type,i)
            util.save_sparse_data(vecs, y, outpath)
        print(f"encoded vector's shape is: {np.array(vecs).shape}")

    end_time = time.time()
    print(f"Running time is {(end_time - start_time) / 60.0 :.4f} minutes")




# def main_data_encoding(mal_file_path,all_file_path,url=False,csv=False):
#     """
#     convert data to binary vector based on frequency and
#     save to file
#     :param mal_file_path: path for malware file names
#     :param all_file_path: path for both benign and malware files
#     :param url: True: extract all features except url; False: only extract permission, intent, api call
#     :param csv: True: save file as csv, otherwise save as npz
#     :return: encoded binary vectors as csv file
#     """
    # #------------------------malware------------------------
    # mal_filenames = util.get_mal_file(mal_file_path)
    # mal_filenames_chunks = util.divide_list(mal_filenames,step=10000)
    #
    # start_time = time.time()
    # "extract features [api_call,intent,permission] by set url=False"
    # uniq_features_sorted = util.get_final_features_main(all_file_path,url=url)
    # for i,file_chunk in enumerate(mal_filenames_chunks):
    #     vecs = util.get_onehot_vecs(uniq_features_sorted,file_chunk,all_file_path)
    #
    #     # build corresponding label 1
    #     y = np.ones(len(vecs),dtype=int)
    #
    #     # save encoded vectors
    #     if csv:
    #         outpath = '../data/drebin/drebin_mal_vec_%d.csv' % i
    #         util.write2csv(vecs, y, outpath)
    #     else:
    #         outpath = '../data/drebin/drebin_mal_vec_%d.npz' % i
    #         util.save_sparse_data(vecs, y, outpath)
    #     print(f"encoded vector's shape is: {np.array(vecs).shape}")
    #
    #
    #
    # #-------------------------benign------------------------
    # benign_filenames = util.get_benign_file(mal_filenames,all_file_path)
    #
    # # reduced the size of file list in one loop since the limited memory
    # benign_filenames_chunks = util.divide_list(benign_filenames, step=10000)
    #
    # for i,file_chunk in enumerate(benign_filenames_chunks):
    #     vecs = util.get_onehot_vecs(file_chunk,all_file_path)
    #
    #     # build corresponding label 0
    #     y = np.zeros(len(vecs),dtype=int)
    #
    #     # save encoded vectors
    #     outpath = '../data/drebin/drebin_benign_vec_%d.csv' % i
    #     util.write2csv(vecs, y, outpath)
    #
    #
    # end_time = time.time()
    # print(f"Running time is {(end_time-start_time)/60.0 :.4f} minutes")



if __name__ == "__main__":
    all_file_path = '../data/source/feature_vectors'    # all files (benign and malware) that contain the features content
    mal_file_path = '../data/source/family_label_mal'   # files that contain the malware sample name

    file_type = 'ben'
    url= False  # extract features [api_call, intent, permission]
    csv = False # save file as npz (space efficient)

    ############# onehot encoding #############
    # main_data_encoding(mal_file_path, all_file_path,url=False,csv=True)

    converting(mal_file_path,all_file_path,file_type,url,csv)
