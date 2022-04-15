import os
os.sys.path.append('..')

from src import util
from functools import reduce
import operator


def main_stats(mal_file_path,all_file_path):
    """
    output some statistic results about the dataset
    :return:
    """
    # all_file_path = '../data/source/feature_vectors'  # all files (benign and malware) that contain the features content
    # mal_file_path = '../data/source/family_label_mal'  # files that contain the malware sample name

    n= 30
    value_1 = 1
    value_2 = 2

    # stats for both malware and benign
    util.get_stats4dataset('all_data',all_file_path,mal_file_path,n,value_1)
    keys_w_certain_value1,url_features,features_filtered = util.get_stats4dataset('all_data',all_file_path,mal_file_path,n,value_1)
    keys_w_certain_value2,_,_ = util.get_stats4dataset('all_data',all_file_path,mal_file_path,3,value_2)
    # # get stats in malwares
    keys_w_certain_value1_mal,url_features_mal,features_filtered_mal = util.get_stats4dataset('malware', all_file_path, mal_file_path,n,value_1)
    keys_w_certain_value2_mal,_,_ = util.get_stats4dataset('malware', all_file_path, mal_file_path,3,value_2)
    # # get stats for benign samples
    keys_w_certain_value1_ben,url_features_ben,features_filtered_ben = util.get_stats4dataset('benign', all_file_path, mal_file_path,n,value_1)
    keys_w_certain_value2_ben,_,_ = util.get_stats4dataset('benign', all_file_path, mal_file_path,3,value_2)

    # get the intersection between keys with certain value in malware and benign samples
    print('\n','*' * 30)
    keys_w1_intersection = set(keys_w_certain_value1_ben).intersection(keys_w_certain_value1_mal)
    print(f"There are {len(keys_w_certain_value1)} features in total that only have {value_1} occurence in all samples")
    print(f"There are {len(keys_w1_intersection)} features that with only {value_1} occurence in both malware and benign samples")
    print(f"There are {len(keys_w_certain_value1_mal)-len(keys_w1_intersection)} features with only {value_1} occurence only in malware samples")
    print(f"There are {len(keys_w_certain_value1_ben)-len(keys_w1_intersection)} features with only {value_1} occurence only in benign samples")
    print('*'*30)
    keys_w2_intersection = set(keys_w_certain_value2_ben).intersection(keys_w_certain_value2_mal)
    print(f"There are {len(keys_w_certain_value2)} features in total that only have {value_2} occurence in all samples")
    print(f"There are {len(keys_w2_intersection)} features that with only {value_2} occurence in both malware and benign samples")
    print(f"There are {len(keys_w_certain_value2_mal) - len(keys_w2_intersection)} features with only {value_2} occurence only in malware samples")
    print(f"There are {len(keys_w_certain_value2_ben) - len(keys_w2_intersection)} features with only {value_2} occurence only in benign samples")

    # get url features statisitcs
    print('*' * 30)
    url_features_intersection = set(url_features_ben).intersection(url_features_mal)    # only set has intersection function
    print(f"There are {len(url_features)} url features in total that in all samples")
    print(f"There are {len(url_features_intersection)} url intersection features in both malware and benign samples")
    print(f"There are {len(url_features_mal)} url features in malware samples")
    print(f"There are {len(url_features_ben)} url features in benign samples")
    print(f"There are {len(url_features_mal)-len(url_features_intersection)} url features only in malware samples")
    print(f"There are {len(url_features_ben)-len(url_features_intersection)} url features only in benign samples")

    # get intersection between url and features only with 1 occurence
    print('*' * 30)
    url_inter_one_occur = set(url_features).intersection(keys_w_certain_value1)
    print(f"There are {len(url_inter_one_occur)} url features that only has one occurence")

    # get stats of filtered features that only include api_call, intent and permission
    print("*"*30)
    print(f"There are {len(features_filtered[0])} API calls, {len(features_filtered[1])} calls, {len(features_filtered[2])} intents, {len(features_filtered[3])} permissions in all samples")
    print(f"There are {len(features_filtered_mal[0])} API calls,{len(features_filtered_mal[1])} calls, {len(features_filtered_mal[2])} intents, {len(features_filtered_mal[3])} permissions in malware samples")
    print(f"There are {len(features_filtered_ben[0])} API calls, {len(features_filtered_ben[1])} calls,{len(features_filtered_ben[2])} intents, {len(features_filtered_ben[3])} permissions in benign samples")
    # features_filtered = features_filtered[0] + features_filtered[1] + features_filtered[2] # concatenate all sub-list
    # features_filtered_mal = features_filtered_mal[0] + features_filtered_mal[1] + features_filtered_mal[2]
    # features_filtered_ben = features_filtered_ben[0] + features_filtered_ben[1] + features_filtered_ben[2]
    features_filtered = reduce(operator.concat, features_filtered)  # concatenate all sub-lists
    features_filtered_mal = reduce(operator.concat, features_filtered_mal)
    features_filtered_ben = reduce(operator.concat, features_filtered_ben)
    features_filtered_inter = set(features_filtered_ben).intersection(features_filtered_mal)
    print(f"There are {len(features_filtered)} features that include [api_call,intent,permission] in all samples")
    print(f"There are {len(features_filtered_mal)} features that include [api_call,intent,permission] in malware samples")
    print(f"There are {len(features_filtered_ben)} features that include [api_call,intent,permission] in benign samples")
    print(f"There are {len(features_filtered_inter)} features that include [api_call,intent,permission] in both benign and malware samples")





if __name__ == '__main__':

    all_file_path = '../data/source/feature_vectors'  # all files (benign and malware) that contain the features content
    mal_file_path = '../data/source/family_label_mal'  # files that contain the malware sample name
    main_stats(mal_file_path,all_file_path)