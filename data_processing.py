import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import keras

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def pad_whiteness(array, pad_width, iaxis, kwargs):
    array[:pad_width] = 0
    array[-pad_width:] = 0
    return array


def get_photo(pixel):
    assert len(pixel) == 3072
    # slice list and then reshape
    r = pixel[0:1024]
    r = np.reshape(r, [32, 32, 1])
    g = pixel[1024:2048]
    g = np.reshape(g, [32, 32, 1])
    b = pixel[2048:3072]
    b = np.reshape(b, [32, 32, 1])
    # concatenate r, g, b layers
    photo = np.concatenate([r, g, b], -1)
    photo_48 = np.zeros([48, 48, 3])
    for i in range(3):
        photo_48[:,:,i] = np.lib.pad(photo[:,:,i], (8,8), 'constant', constant_values=(0,0))
    return photo_48



def get_train_data(keyword, size=(32,32), normalized=False, filelist=[]):
    keyword = str.encode(keyword)
    print(type(filelist), len(filelist))

    assert keyword in [b'data', b'labels', b'batch_label', b'filenames']
    assert type(filelist).__name__ == 'list' and len(filelist) != 0
    assert type(normalized) is bool
    assert type(size).__name__  == 'tuple'

    files = []
    for i in filelist:
        if 1 <= i <= 5 and i not in files:
            files.append(i)

    if len(files) == 0:
        raise ValueError("no valid input files")

    if keyword == b'data':
        data = []
        for i in files:
            data.append(unpickle("C:/Users/Admin/PycharmProjects/genetic-algorithm/cifar-10-batches-py/data_batch_%d"%i)[keyword])
        data = np.concatenate(data, 0)
        if normalized == False:
            array = np.ndarray([len(data), size[0], size[1], 3], dtype=np.float32)
            for i in range(len(data)):
                img = cv2.resize(get_photo(data[i]), size)
                array[i] = cv2.resize(get_photo(data[i]), size)
            return array
        else:
            array = np.ndarray([len(data), size[0], size[1], 3], dtype=np.float32)
            for i in range(len(data)):
                array[i] = cv2.resize(get_photo(data[i]), size)/255
            return array
        pass
    elif keyword == b'labels':
        labels =[]
        for i in files:
            labels += unpickle("C:/Users/Admin/PycharmProjects/genetic-algorithm/cifar-10-batches-py/data_batch_%d"%i)[keyword]
        return labels
        pass
    elif keyword == b'batch_label':
        batch_label = []
        for i in files:
            batch_label.append(unpickle("C:/Users/Admin/PycharmProjects/genetic-algorithm/cifar-10-batches-py/data_batch_%d"%i)[keyword])
        return batch_label
        pass
    elif keyword == b'filenames':
        filenames = []
        for i in files:
            filenames += unpickle("C:/Users/Admin/PycharmProjects/genetic-algorithm/cifar-10-batches-py/data_batch_%d"%i)[keyword]
        return filenames
        pass


def get_data_cifar(list, num_samples):
    data = get_train_data('data', size=(48, 48), normalized=True, filelist=list)
    label = get_train_data('labels', size=(32, 32), normalized=True, filelist=list)
    data = np.asarray(data)
    label = np.asarray(label)

    selected_idx = np.random.choice(len(label), num_samples, replace=False)
    data = data[selected_idx, :]
    label = label[selected_idx]
    return data, label


def cifar_data(num_train, num_test):
    list = [1, 2, 3]
    num_samples = num_train + num_test
    data, label = get_data_cifar(list, num_samples)
    x_train = data[:num_train]
    y_train = label[:num_train]
    x_test = data[num_train: num_train+num_test]
    y_test = label[num_train: num_train+num_test]
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    list = [1,2,3,4,5]
    list_bool = (type(list).__name__ == 'list')
    print(list_bool, type(list).__name__)
    # data = get_train_data('data', size=(32,32), normalized=True, filelist=list)
    # label = get_train_data('labels', size=(32,32), normalized=True, filelist=list)
    # print(len(label))
    label_mat = np.zeros((5,10))
    list_0 = []
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []
    list_9 = []

    label_0 = []
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    label_5 = []
    label_6 = []
    label_7 = []
    label_8 = []
    label_9 = []

    for i in range(5):
        index = [i+1]
        data, label = get_data_cifar(index, 10000)
        for j in range(label.shape[0]):
            if label[j] == 0:
                list_0.append(data[j, :, :, :])

            elif label[j] == 1:
                list_1.append(data[j, :, :, :])

            elif label[j] == 2:
                list_2.append(data[j, :, :, :])

            elif label[j] == 3:
                list_3.append(data[j, :, :, :])

            elif label[j] == 4:
                list_4.append(data[j, :, :, :])

            elif label[j] == 5:
                list_5.append(data[j, :, :, :])

            elif label[j] == 6:
                list_6.append(data[j, :, :, :])

            elif label[j] == 7:
                list_7.append(data[j, :, :, :])

            elif label[j] == 8:
                list_8.append(data[j, :, :, :])

            elif label[j] == 9:
                list_9.append(data[j, :, :, :])

    data_overall = []
    data_overall.append(list_0)
    data_overall.append(list_1)
    data_overall.append(list_2)
    data_overall.append(list_3)
    data_overall.append(list_4)
    data_overall.append(list_5)
    data_overall.append(list_6)
    data_overall.append(list_7)
    data_overall.append(list_8)
    data_overall.append(list_9)

    data_overall = np.asarray(data_overall)
    print(data_overall.shape)

    idx = 156
    img_example = data[idx, :, :, :]
    print(label[idx])
    plt.figure()
    plt.imshow(img_example)
    plt.title(label[idx])
    plt.show()

