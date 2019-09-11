import utils
import pandas as pd
import csv
import numpy as np

def dict_print(dict):
    for key in dict:
        print(key, dict[key])


def key_print(dict):
    for key in dict:
        print(key)


def dict2str(geno_dict):
    dict_str = ''
    if geno_dict['activation']:
        for key in geno_dict:
            dict_str += str(key)
            dict_str += ':'
            dict_str += str(geno_dict[key])
            dict_str += ','
    else:
        dict_str = 'Not Activated'
    return dict_str


def save_geno(df, geno_dict, itr_idx):
    for key in geno_dict:
        candidate = geno_dict[key]
        global df_idx
        utils.DF_IDX += 1
        row = {'iteration':itr_idx, 'geno_type':candidate['name'], 'score': candidate['score'],
               'L0': dict2str(candidate['layers']['L0']), 'L1': dict2str(candidate['layers']['L1']),
               'L2': dict2str(candidate['layers']['L2']), 'L3': dict2str(candidate['layers']['L3']),
               'L4': dict2str(candidate['layers']['L4'])
               }
        df_row = pd.DataFrame(row, index = [utils.DF_IDX])
        # df = pd.concat([df, df_row], axis=0, keys=df.columns)
        df = df.append(df_row)
    return df


def create_csv(file_path):
    # file_path = 'C:/Users/Zhu/PycharmProjects/genetic-algorithm/test'
    with open(file_path, 'w+', newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ["iteration", "geno_type", "score", "advanced_score","num_layers", "#paras",
                    "L0-activation", "L0-neurons", "L0-function", "L0-bias", "L0-drop_out", "L0-drop_out_rate",
                    "L1-activation", "L1-neurons", "L1-function", "L1-bias", "L1-drop_out", "L1-drop_out_rate",
                    "L2-activation", "L2-neurons", "L2-function", "L2-bias", "L2-drop_out", "L2-drop_out_rate",
                    "L3-activation", "L3-neurons", "L3-function", "L3-bias", "L3-drop_out", "L3-drop_out_rate",
                    "L4-activation", "L4-neurons", "L4-function", "L4-bias", "L4-drop_out", "L4-drop_out_rate",
                    ]
        csv_write.writerow(csv_head)


def write_csv(file_path, candidate, ite_idx, num_paras):
    with open(file_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        data_row = [[ite_idx, candidate['name'], candidate['score'], candidate['ad_score'],count_num_layers(candidate, 5), num_paras,
                     # 0 th layer
                     candidate['layers']['L0']['activation'], candidate['layers']['L0']['neurons'],
                     candidate['layers']['L0']['activation_func'], candidate['layers']['L0']['use_bias'],
                     candidate['layers']['L0']['drop_out'], candidate['layers']['L0']['drop_out_prob'],
                     # first layer
                     candidate['layers']['L1']['activation'], candidate['layers']['L1']['neurons'],
                     candidate['layers']['L1']['activation_func'], candidate['layers']['L1']['use_bias'],
                     candidate['layers']['L1']['drop_out'], candidate['layers']['L1']['drop_out_prob'],
                     # second layer
                     candidate['layers']['L2']['activation'], candidate['layers']['L2']['neurons'],
                     candidate['layers']['L2']['activation_func'], candidate['layers']['L2']['use_bias'],
                     candidate['layers']['L2']['drop_out'], candidate['layers']['L2']['drop_out_prob'],
                     # third layer
                     candidate['layers']['L3']['activation'], candidate['layers']['L3']['neurons'],
                     candidate['layers']['L3']['activation_func'], candidate['layers']['L3']['use_bias'],
                     candidate['layers']['L3']['drop_out'], candidate['layers']['L3']['drop_out_prob'],
                     # fourth layer
                     candidate['layers']['L4']['activation'], candidate['layers']['L4']['neurons'],
                     candidate['layers']['L4']['activation_func'], candidate['layers']['L4']['use_bias'],
                     candidate['layers']['L4']['drop_out'], candidate['layers']['L4']['drop_out_prob'],
                     ]]
        csv_write.writerows(data_row)


def count_num_layers(candidate, max_num):
    num_layers = 0
    for i in range(max_num):
        layer_name = 'L' + str(i)
        if candidate['layers'][layer_name]['activation']:
            num_layers += 1

    return num_layers


def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train-mean)  # /(std+1e-7)
    x_test = (x_test-mean)  # /(std+1e-7)
    return x_train, x_test

def normalize_single(x_train, x_test):
    num_train = x_train.shape[3]
    num_test = x_test.shape[3]
    for i in range(num_train):
        mean = np.mean(x_train[i,:,:,:])
        std = np.std(x_train[i,:,:,:])
        x_train[i,:,:,:] = (x_train[i,:,:,:]-mean)/(std+1e-7)

    for j in range(num_test):
        mean = np.mean(x_test[i,:,:,:])
        std = np.std(x_test[i,:,:,:])
        x_test[i,:,:,:] = (x_test[i,:,:,:]-mean)/(std+1e-7)

    return x_train, x_test


