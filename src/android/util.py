import os, sys
os.sys.path.append('..')


from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import copy
import scipy.sparse
import time
from pympler import asizeof
from functools import reduce
import operator



def list_files(path):
    """
    list all files in a directory
    :param path:
    :return: a list of filenames
    """
    all_files = os.listdir(path)
    return all_files


def read_feature(file):
    "read files that contain features need special encoding"
    with open(file,'r',encoding='windows-1252') as f:
        # lines = f.readlines() # keep '\n'
        lines = f.read().splitlines() # able to remove newline "\n"
    return lines


def write2txt(list,outpath):
    "write a list to text file. each element is a line"
    with open(outpath,'w') as f:
        # f.write('\n'.join(list))
        f.write('\n'.join(str(x) for x in list))
    print(f"file has been written in {outpath}")


def read_txt(file):
    "read txt files into lines"
    with open(file,'r') as f:
        lines = f.readlines()
    return lines


def get_mal_file(path):
    """
    extract malware filenames from a format (filename,label)
    :param path:
    :return:
    """
    # file contain (filename,corresponding family label)
    filenames = read_txt(path)

    # only extract the name, not the label
    mal_files = []
    for filename in filenames:
        filename = filename.strip().split(',')
        mal_files.append(filename[0])
    return mal_files


def get_benign_file(mal_files,all_file_path):
    """
    get benigne filenames
    :param mal_files: malware file names
    :param all_file_path: path for both benign and malware files
    :return:
    """
    all_files = list_files(all_file_path)
    benign_files = copy.deepcopy(all_files)
    for file in mal_files:
        benign_files.remove(file)
    return benign_files



def get_all_features(all_file, path):
    """
    return a list that include all lines of all files
    :param all_file: filename(s) that needed to be read
    :param path: the path for the filename located
    :return:
    """
    all_features = []
    if isinstance(all_file, list):
        for file in all_file:
            if not file.endswith('.DS_Store'):
                all_features += read_feature('%s/%s' % (path, file))
    else:
        if not all_file.endswith('.DS_Store'):
            all_features += read_feature('%s/%s' % (path, all_file))

    # remove empty space and newline
    all_features = list(filter(lambda x: x not in ['','\n'],all_features))
    return all_features


def get_num_UniqFeatures(all_features):
    """
    input: all features in a list
    :return: number of all unique features
    """
    all_features_set = set(all_features)
    return len(all_features_set)


def features_statis(all_features,value):
    """
    create a dictionary based on all features
    obtain the statistic results for each feature
    output a sorted dictionary {f2:45,f1:12,f3:9}, sorted based on values
    :param all_features: all features (exist in all samples, may inlude many features that are exact same)
    :param value: for counting the number of keys that have this particular value
    :return: the sorted features dictionary, the sorted keys of the features dictionary, the number of keys with a certain
    value, all the keys that with a certain value
    """
    features_dic = {}
    for feature in all_features:
        if feature not in features_dic:
            features_dic[feature] = 1
        else:
            features_dic[feature] += 1

    # sort dictionary by values
    features_dic_sort = {k: v for k,v in sorted(features_dic.items(), key=lambda item: item[1],reverse=True)}
    #features_key_sort = [k for k,v in sorted(features_dic.items(), key=lambda item: item[1],reverse=True)]
    features_key_sort = list(features_dic_sort.keys())

    # get the number of keys (features) with particular value in dictionary
    # num_keys = sum(x == value for x in features_dic.values())
    keys_w_certain_value = []
    for key in features_dic:
        if features_dic[key] == value:
            keys_w_certain_value.append(key)

    return features_dic_sort,features_key_sort,keys_w_certain_value


def get_url_features(all_features):
    """
    get all url features
    :param all_features:
    :return:
    """
    feature_set = set(all_features)
    url_features = []
    for feature in feature_set:
        if feature.startswith("url::"):
            url_features.append(feature)
    return url_features



def get_filtered_features(all_features,url=False):
    """
    extract features include [permission, intent, API call] or [all feature except url]
    (note that many url contains above feature, therefore filtering url first, when url=False)
    :param all_features:
    :param url: True: extract all features except url; False: only extract permission, intent, api call
    :return: unique features after processed
    """
    feature_set = set(all_features)

    if not url:
        # get features [permission, intent, api call], but url
        api_call,call,intent,permission = [],[],[],[]
        for feature in feature_set:

            # if not feature.startswith(("url::" , 'service_receiver::' ,'provider::', 'activity::')):
            #     if "api_call::" in feature or "call::" in feature:
            #         api_call.append(feature)
            #     elif "intent" in feature:
            #         intent.append(feature)
            #     elif "permission" in feature:
            #         permission.append(feature)

            if feature.startswith('api_call::'):
                api_call.append((feature))
            elif feature.startswith('call::'):
                call.append(feature)
            elif feature.startswith('intent::'):
                intent.append(feature)
            elif feature.startswith('permission'):
                permission.append(feature)
        return [api_call,call,intent,permission]

        # cands_features = ['api','call','intent','permission']
        # for feature in feature_set:
        #     for cand in cands_features:
        #         if cand in feature and not feature.startswith('url::'):
        #             features.append(feature)
        #             break

    else:
        # get features except url
        features = []
        for feature in feature_set:
            if not feature.startswith("url::"):
                features.append(feature)
        return features


def get_final_features(all_features,filtered_uniq_features):
    """
    build sorted feature dictionary after filtering certain features
    :param all_features: all features without any filtering
    :param filtered_uniq_features: unique feature set after filtering certain features
    :return:
    """
    # get dictionary after features filtered
    features_dic = {}
    for feature in all_features:
        if feature not in features_dic:
            features_dic[feature] = 1
        else:
            features_dic[feature] += 1
    feature_filtered_dic = {}
    for feature in filtered_uniq_features:
        feature_filtered_dic[feature] = features_dic[feature]

    # initial sort dic by keys based on the string (fist letter) to keep the order that have same value
    # e.g., {'h':3,'d':1,'a':1,'w':1} -->{'a':1,'d':1,'h':3,'w':1}
    feature_filtered_dic = {k:v for k,v in sorted(feature_filtered_dic.items(), key=lambda item:item[0])}

    # then sort dic based on value in decending order --> {'h':3,'a':1,'d':1,'w':1}. the order 'a','d','w' is fixed
    feature_filtered_dic_sort = {k:v for k,v in sorted(feature_filtered_dic.items(), key=lambda  item: item[1],reverse=True)} # sort based on value
    feature_filtered_key_sort = list(feature_filtered_dic_sort.keys())

    return feature_filtered_dic_sort,feature_filtered_key_sort


def get_final_features_main(all_file_path,url=False):
    """
    get unique features that after features filtered and sorted
    :param all_file_path: path of all files
    :param url: True: extract features except url; False: extract features from [api call,intent,permission]
    :return:
    """
    # get unique sorted features
    all_files = list_files(all_file_path)
    all_features = get_all_features(all_files, all_file_path)

    uniq_features = get_filtered_features(all_features, url=url)  # extract features except url

    # flatten list when list as [[api_call],[call],[intent],[permission]] --> [api_call,call,intent,permission]
    if not url:
        uniq_features = reduce(operator.concat,uniq_features)

    _, uniq_features_sorted = get_final_features(all_features, uniq_features)
    # last = uniq_features_sorted[-1]
    # write sorted features
    write2txt(uniq_features_sorted, '../data/drebin/sorted_features.txt')

    return uniq_features_sorted



def get_stats4dataset(file_type,all_file_path,mal_file_path,n,value):
    """
    get the top n features of all features in terms of frequency in descending manner
    :param n: show n results of sorted features
    :param file_type: [benign,malware,all_data]
    :param all_file_path: the path for both of malware and benign samples
    :param mal_file_path: the file contain the malware samples' name
    :param value: for counting the number of keys that have this particular value
    :return: statistic results, top n sorted features in descending way
    """

    print('-' * 80)
    if file_type == 'malware':
        filenames = get_mal_file(mal_file_path)
    elif file_type == 'benign':
        mal_filenames = get_mal_file(mal_file_path)
        filenames = get_benign_file(mal_filenames, all_file_path)
    else:
        filenames = list_files(all_file_path)
    print(f'num of {file_type} samples: {len(filenames)}')

    # top n features of Malware in Drebin data
    all_features = get_all_features(filenames, all_file_path)
    num_features = get_num_UniqFeatures(all_features)
    print(f'num of unique features of all {file_type} samples: {num_features}', '\n')

    features_dic_sort, key_sort,key_w_certain_value = features_statis(all_features,value)
    num_keys = len(key_w_certain_value)
    top_features = pd.Series(features_dic_sort).head(n)
    print(f"feature stats of all {file_type} samples: ", '\n', top_features, '\n')
    # print(f"sorted key: {key_sort[:20]}", '\n')
    print(f"The number of keys that with value {value} is: {num_keys}")

    # get the number of url features
    url_features = get_url_features(all_features)

    # get processed features that only include api_call/call, intent, permission
    features_filtered = get_filtered_features(all_features,url=False)

    return key_w_certain_value,url_features,features_filtered



class onehot_encoder:
    """
    convert string/values features into one-hot verctor, the econding oder are unknow since its sklearn API.
    in other words, without consider the restrictions, which the the encoding oder should based on the frequency of the string/value,
    like the very first one in the encoding vector should the high frequency in the corpus fitted into encoder.
    e.g., [1,2,3,4] --> [0,1,1,0,1,1,0] (assume we have 7 vocb)
    """
    def __init__(self,X,all_X):
        """
        :param X: each input sample
        :param all_X: all strings/values extracted from all samples
        """
        self.X = X
        self.all_X = all_X

    def preprocess(self,data):
        """
        convert input to array if necessary, reshape data to [len(data),1] if necessary
        [1,2,3] --> array([[1],[2],[3]])
        :param data: a list/array of value/string, e.g., [1,2,3,4]
        :return: an array with shape [len(array),1]
        """
        # convert data to array
        if not isinstance(data,np.ndarray):
            data = data.np.array(data)

        # convert X shape, [len(x)] --> [len(x),1]
        if data.shape[1] !=1:
            data = data.reshape(len(data),1)
        return data

    def array_xor(self,data):
        """
        get final binary vector for each input
        e.g., [[0,1,0],[1,0,0],[1,1,0]] --> [1,1,0]
        :param data: an array of arrays. e.g.,[[0,1,0],[1,0,0],[1,1,0]]
        :return: an array of xor results. (final one hot encoding for a set of features)
        """
        enc_vec = data[0].astype(int)
        for i in range(1,len(data)):
            enc_vec = np.bitwise_xor(data[i].astype(int), enc_vec)
        return enc_vec

    def encoder(self):
        """
        get the fitted one hot encoder ready
        :return: fitted one hot encoder
        """
        onehot_enc = OneHotEncoder(sparse=False)
        onehot_enc_fit = onehot_enc.fit(self.all_X)
        return onehot_enc_fit

    def encoding(self,data):
        """
        get the final binary for the input
        e.g., [1,2,3,4] --> [0,1,1,0,1,1,0] (assume we have 7 vocb)
        :param data: an list/array of input consists of values/string
        :return: an array of vector
        """
        X = self.preprocess(data)
        onehot_enc_fit = self.encoder()
        X_enc = onehot_enc_fit.transform(X)
        X_enc = self.array_xor(X_enc)
        return X_enc


class onehot_encoder_v1:
    """
    convert string/values features into one-hot verctor, but add restrictioin here,
    which the encoding order is based on the strings/values frequency. e.g., v(i), the smaller i, the higher frequency indicate
    largest number of the string/value here. {he:5, is:1, in:2} --> [he,in,is] where, the encoding order should based on.
    :return:
    """
    def __init__(self, uniq_features_sorted):
        """
        :param uniq_features_sorted: a list, a sorted features based on its frequency in descending manner
        """
        self.all_features = uniq_features_sorted

    def encoding(self,X):
        """
        :param X: a list, each input sample that need to be encoded
        :return: encoded vector for X, with length len(uniq_feature_sorted)
        """
        # initiate a list with 0s with len(all_features) length
        X_enc = list(np.zeros(len(self.all_features),dtype=int))
        for x in X:
            try:
                index = self.all_features.index(x)
                X_enc[index] = 1
            except:
                pass
        return X_enc


def get_onehot_vecs(uniq_features_sorted,filenames,all_file_path):
    """
    encoding for a set of files, return a list of encoded lists
    :param uniq_features_sorted: unique features after fatures filtered and sorted
    :param filenames: a list of file names
    :param all_file_path: location for all files stored
    :return: a list of encoded vector lists
    """

    # encoding all files included in filenames
    vecs = []
    encoder = onehot_encoder_v1(uniq_features_sorted)
    for i, file in enumerate(filenames):
        print(f"{file}: {i}")
        X = get_all_features(file,all_file_path)
        X_enc = encoder.encoding(X)
        vecs.append(X_enc)
    return vecs


def write2csv(data,label,outpath):
    """
    write data to csv
    :param data: a list of lists, better be dataframe
    :return:
    """
    # preprocess the data format
    if isinstance(data,(list,np.ndarray)):
        data_df = pd.DataFrame(data)
    elif isinstance(data,pd.core.frame.DataFrame):
        pass
    else:
        sys.exit('input should format should be dataframe or list')

    data_df['label'] = label

    # # convert to SparseArray to be more memory-efficient
    # data_df = pd.get_dummies(data_df,sparse=True)

    data_df.to_csv(outpath,index=0)
    print(f"data saved in {outpath}.")



def save_sparse_data(data,label,outpath):
    """
    save sparse data in CSR format. memory efficient
    :param data: all input X samples
    :param label: labels samples
    :param outpath:
    :return:
    """

    # add label to data
    if not isinstance(data,(np.ndarray,list)):
        sys.exit('input data should be numpy.ndarray or list format')
    data = np.column_stack((data,label)) #  add elements to the end of each row

    # convert data to csr matrix, which is size efficient
    data_sparse = scipy.sparse.csr_matrix(data)

    scipy.sparse.save_npz(outpath,data_sparse)
    print(f"data saved in {outpath}.")
    print(f"data size before compressed: {asizeof.asizeof(data)/1000}M", '\n', f"size after compressed: {asizeof.asizeof(data_sparse)/1000}M")




def divide_list(lst,step=10000):
    """
    divide list into fixed size chunk
    :param lst:
    :param step:
    :return:
    """
    return [lst[i:i+step] for i in range(0,len(lst),step)]




# def main_data_encoding(mal_file_path,all_file_path,outpath):
#     """
#     convert data to binary vector based on frequency and
#     save to csv file
#     :param mal_file_path: path for malware file names
#     :param all_file_path: path for both benign and malware files
#     :return: encoded binary vectors as csv file
#     """
#
#     #------------------------malware------------------------
#     mal_filenames = get_mal_file(mal_file_path)
#     vecs_mal = get_onehot_vecs(mal_filenames,all_file_path)
#
#     # build corresponding label 1
#     y_mal = np.ones(len(mal_filenames),dtype=int)
#
#     # save to csv
#     write2csv(vecs_mal, y_mal, '../data/drebin/drebin_mal_vec.csv')
#
#     #-------------------------benign------------------------
#     benign_filenames = get_benign_file(mal_filenames,all_file_path)
#     vecs_benign = get_onehot_vecs(benign_filenames,all_file_path)
#
#     # build corresponding label 0
#     y_benign = np.zeros(len(benign_filenames),dtype=int)
#
#     #--------------concatenate benign and mal----------------
#     vecs = vecs_benign + vecs_mal
#     y = list(y_benign) + list(y_mal)
#
#     # save encoded vectors
#     write2csv(vecs,y,outpath)












