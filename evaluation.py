import numpy as np
from keras import losses, optimizers
from keras.utils import np_utils
from Geno2model import geno2model
from tools import dict_print, create_csv, write_csv, normalize, normalize_single
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.vgg16 import preprocess_input
import random


from wind_blade_data import get_windblade_data
from data_processing import cifar_data

def evaluate(genotype, ite_idx, num_train, num_test,file_path, save_model = False):
    acc = np.zeros((len(genotype), 1))
    i = 0
    if ite_idx == 0 and save_model:
        create_csv(file_path)

    for key in genotype:
        if genotype[key]['flag_change']:
            print("======= iteration idx is %d ======" % ite_idx)
            print("======= genotype idx is %d =======" % i)
            dict_print(genotype[key])
            model, model_test = geno2model(genotype[key])
            # x_train, y_train = get_data(
            #     'C:/Users/zhu/PycharmProjects/GeneticAlgorithm/BloodCell/dataset2-master/images/TRAIN/', 800)

            # x_train, y_train = get_data_cifar([2,3], 3000)
            # x_test, y_test = get_data_cifar([4], 800)

            es = EarlyStopping(monitor='val_loss', patience=15, verbose=0)
            rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
            callbacks = [es, rl]

            img_path = 'C:/Users/admin/PycharmProjects/genetic-algorithm/dataset_blades/'
            x_train, y_train, x_test, y_test = get_windblade_data(img_path, num_train, num_test)

            # convert y into one hot matrix
            y_train = np_utils.to_categorical(y_train, 2)
            y_test = np_utils.to_categorical(y_test, 2)
            print(x_train.shape)
            print(y_train.shape)

            model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])
            history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=0, shuffle=True,
                            validation_data=(x_test, y_test), callbacks=callbacks)
            print(history.history['val_acc'][-1])
            # y_test = model_test.predict(x = x_train[10])
            # print('the predicting value is',y_test, 'ground truth is', y_train[10])

            candidate = genotype[key]
            candidate['score'] = history.history['val_acc'][-1]
            candidate['flag_change'] = False
            i = i + 1

            if save_model:
                # count trainable parameters
                trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
                # write genotype into csv file
                write_csv(file_path, candidate, ite_idx, trainable_count)

            if K.backend() == 'tensorflow':
                K.clear_session()
        else:
            print('================= No changes to the model, so not necessary to re-train again ===================')
            if save_model:
                candidate = genotype[key]
                write_csv(file_path, candidate, ite_idx, num_paras=0)

    return genotype


def advanced_evaluate(geno_type, file_path, test_mode = False ,save_mode = False):

    if not test_mode:
        print("Now we are evaluating models from genetic algorithm.")
    else:
        print("Now we are evaluating test models")

    # read data from windblade data
    img_path = 'C:/Users/admin/PycharmProjects/genetic-algorithm/dataset_blades/'
    X, Y, x_test, y_test = get_windblade_data(img_path, 8024, 0)

    # read data from cifar-10
    # X, Y, X_test, Y_test = cifar_data(30000, 0)


    idx = [i for i in range(len(Y))]
    print('before shuffle index is ', idx)
    random.Random(0).shuffle(idx)
    print('have a look at index',idx)
    X = X[idx]
    Y = Y[idx]
    X = np.squeeze(X)
    Y = np.squeeze(Y)

    print('number of samples is:', X.shape, Y.shape)


    seed = 7
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    es = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    rl = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0)

    callbacks = [es, rl]
    for key in geno_type:
        if True:
            print('================================ start advanced evaluation ===============================')
            print('finishing model building part')
            cv_scores = []
            cv_accuracy = []
            print('start to train on k-fold validation ... ...')
            for train, test in k_fold.split(X, Y):
                model, model_test = geno2model(geno_type[key])


                #
                # print(type(y_train), type(y_test))
                # print('+++++++ have check at y train:', np.sum(y_train), y_train.size)
                # print('+++++++ have a check at y test', np.sum(y_test), y_test.size)

                print('have a look at Y distribution', Y[train][100:110], Y[test][23:33])
                print('the size of Y train is', Y[train].size)
                print('the sum of Y train is', np.sum(Y[train]))
                print('the size of Y test is', Y[test].size)
                print('the sum of Y test is', np.sum(Y[test]))

                if not test_mode:
                    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy,
                              metrics=['accuracy'])
                else:
                    model_test.compile(optimizer=optimizers.SGD(lr=0.001), loss=losses.mean_squared_error,
                              metrics=['accuracy'])

                print('model compile finished')

                x_train = X[train]
                # y_train = np_utils.to_categorical(Y[train])
                x_test = X[test]
                # y_test = np_utils.to_categorical(Y[test])

                # x_train = preprocess_input(x_train)
                # x_test = preprocess_input(x_test)

                x_train, x_test = normalize(x_train, x_test)

                print("=======================Have a look at input after normalization===============================")
                print(np.sum(x_train))

                y_train = Y[train]
                y_test = Y[test]

                print('data preparation finished')

                if not test_mode:
                    model.fit(x_train, y_train, validation_split=0.15, epochs=100, batch_size=32, verbose=1, callbacks=callbacks)
                else:
                    model_test.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=8, verbose=1, callbacks=callbacks)
                print('model training finished')

                if test_mode:
                    input("press enter to continue and get evaluating result")

                if not test_mode:
                    score, acc = model.evaluate(x_test, y_test, verbose=1)
                else:
                    score, acc = model_test.evaluate(x_test, y_test, verbose=1)
                cv_accuracy.append(acc)
                cv_scores.append(score)
                print('========== the score of current round is %f, accuracy is %f ====================' %(score, acc))

            input("press enter to continue")
            print('cross validation scores:', cv_scores, cv_accuracy)
            geno_type[key]['ad_score'] = [np.mean(cv_scores), np.mean(cv_accuracy), np.std(cv_accuracy)]
            geno_type[key]['flag_change_ad'] = False

            if K.backend() == 'tensorflow':
                K.clear_session()
            if save_mode:
                trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
                # write genotype into csv file
                write_csv(file_path, geno_type[key], ite_idx=0, num_paras=trainable_count)
        else:
            print('================= No changes to the model, so not necessary to re-train again ===================')

        if test_mode:
            continue

    return geno_type


