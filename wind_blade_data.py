import numpy as np
import os
import cv2
from tqdm import tqdm


def get_windblade_data(folder, num_train, num_test, ):
    x_raw_pos = []
    y_raw_pos = []
    x_raw_neg = []
    y_raw_neg = []
    x_raw_else = []
    y_raw_else = []

    for type in os.listdir(folder):
        if type in ['1']:
            label = 1
            for image_filename in tqdm(os.listdir(folder + type)):
                img_file = cv2.imread(folder + type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (224, 224), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    x_raw_pos.append(img_arr)
                    y_raw_pos.append(label)
        elif type in ['0']:
            label = 0
            for image_filename in tqdm(os.listdir(folder + type)):
                img_file = cv2.imread(folder + type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (224, 224), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    x_raw_neg.append(img_arr)
                    y_raw_neg.append(label)
        else:
            label = -1
            for image_filename in tqdm(os.listdir(folder + type)):
                img_file = cv2.imread(folder + type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (224, 224), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    x_raw_else.append(img_arr)
                    y_raw_else.append(label)

          # for image_filename in tqdm(os.listdir(folder+type)):
          #   img_file = cv2.imread(folder + type + '/' + image_filename)
          #   if img_file is not None:
          #       img_file = cv2.resize(img_file, (224, 224), interpolation=cv2.INTER_CUBIC)
          #       img_arr = np.asarray(img_file)
          #       x_raw.append(img_arr)
          #       y_raw.append(label)

    idx_pos = [i for i in range(len(y_raw_pos))]
    idx_neg = [i for i in range(len(y_raw_neg))]
    idx_else = [i for i in range(len(y_raw_else))]

    # np.random.shuffle(idx)
    x_raw_pos = np.asarray(x_raw_pos)
    y_raw_pos = np.asarray(y_raw_pos)
    x_raw_neg = np.asarray(x_raw_neg)
    y_raw_neg = np.asarray(y_raw_neg)
    x_raw_else = np.asarray(x_raw_else)
    y_raw_else = np.asarray(y_raw_else)

    x_raw_pos = x_raw_pos[idx_pos]
    y_raw_pos = y_raw_pos[idx_pos]
    x_raw_neg = x_raw_neg[idx_neg]
    y_raw_neg = y_raw_neg[idx_neg]


    x_raw_else = x_raw_else[idx_else]
    y_raw_else = y_raw_else[idx_else]


    x_train = np.concatenate((x_raw_pos[:int(num_train/2)], x_raw_neg[:int(num_train/2)]), axis=0)
    y_train = np.concatenate((y_raw_pos[:int(num_train/2)], y_raw_neg[:int(num_train/2)]), axis=0)
    x_test = np.concatenate((x_raw_pos[int(num_train/2):int((num_train+num_test)/2)], x_raw_neg[int(num_train/2):int((num_train+num_test)/2)]), axis=0)
    y_test = np.concatenate((y_raw_pos[int(num_train/2):int((num_train+num_test)/2)], y_raw_neg[int(num_train/2):int((num_train+num_test)/2)]), axis=0)

    print('====have a check on y train', np.sum(y_train), y_train.shape)

    return x_train, y_train, x_test, y_test



# data_path = 'C:/Users/zhu/PycharmProjects/GeneticAlgorithm/dataset_blades/'
# x_train, y_train, x_test, y_test = get_windblade_data(data_path, 2000, 400)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)