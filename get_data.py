import numpy as np
import os
import cv2
from tqdm import tqdm


def get_data(folder, num_samples):
    X_raw = []
    Y_raw = []
    print(os.listdir(folder))
    for wbc_type in os.listdir(folder):
        print(wbc_type)
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 0
            elif wbc_type in ['EOSINOPHIL']:
                label = 1
            elif wbc_type in ['MONOCYTE']:
                label = 2
            elif wbc_type in ['LYMPHOCYTE']:
                label = 3
            else:
                label = -1
            for image_filename in tqdm(os.listdir(folder+wbc_type)):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (96,124), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    X_raw.append(img_arr)
                    Y_raw.append(label)
    selected_idx = np.random.choice(len(Y_raw), num_samples, replace=False)
    print(selected_idx)
    X_raw = np.asarray(X_raw)
    Y_raw = np.asarray(Y_raw)
    X = X_raw[selected_idx, :]
    Y = Y_raw[selected_idx]
    return X, Y

# X_train, Y_train = get_data('C:/Users/zhu/PycharmProjects/GeneticAlgorithm/BloodCell/dataset2-master/images/TRAIN/', 400)
# X_test, Y_test = get_data('C:/Users/zhu/PycharmProjects/GeneticAlgorithm/BloodCell/dataset2-master/images/TEST/', 100)
# print(X_train.shape, Y_train.shape)
# print(Y_train[:20])
#
# idx = 20
# image = np.asarray(X_train[idx,:], dtype=np.uint8)
# cv2.imshow(str(Y_train[idx]), image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




