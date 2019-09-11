import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from keras.activations import relu, sigmoid, tanh
# recommend to use Keras with version under 2.1.5


def geno2model(geno, input_dims=25088):
    # input dims is 4608 for blood cell, 512 for cifar, 25088 for wind blade
    # input_dims is for dense layers not for VGG
    func_dict = {0: relu, 1: sigmoid, 2: tanh}
    layers_dict = geno["layers"]
    dense_length = len(layers_dict)
    dense_model = Sequential()
    layers_num = 0
    print('check number is', layers_num)
    for i in range(dense_length):
        layer_name = 'L' + str(i)
        if layers_dict[layer_name]['activation']:
            if layers_num == 0:
                dense_model.add(Dense(layers_dict[layer_name]['neurons'],
                                      activation=func_dict[layers_dict[layer_name]['activation_func']],
                                      use_bias=layers_dict[layer_name]['use_bias'],
                                      input_shape=(32, input_dims)))
            else:
                dense_model.add(Dense(layers_dict[layer_name]['neurons'],
                                      activation=func_dict[layers_dict[layer_name]['activation_func']],
                                      use_bias=layers_dict[layer_name]['use_bias']))
            if layers_dict[layer_name]['drop_out']:
                dense_model.add(Dropout(rate=layers_dict[layer_name]['drop_out_prob']))

            layers_num += 1

    dense_model.add(Dense(units=2, activation=tf.nn.softmax))

    # vgg model configuration for windblade dataset
    # vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=[224, 224, 3])

    # vgg model configuration for cifar-10
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=[224, 224, 3])
    x = Flatten()(vgg_model.output)
    vgg_layers = [l for l in vgg_model.layers]
    for layer in vgg_layers:
        layer.trainable = False
    y = dense_model(x)
    model = Model(input=vgg_layers[0].input, output=y)
    model.summary()

    test_dense = Sequential()
    # blood cell [4096 2048 2048 4], cifar [512, 256, 256, 10]
    # test_dense.add(Dense(units=512, activation=tf.nn.relu))
    # test_dense.add(Dropout(rate=0.5))
    # test_dense.add(Dense(units=256, activation=tf.nn.sigmoi))
    # test_dense.add(Dense(units=256, activation=tf.nn.relu))
    # test_dense.add(Dense(units=1, activation=tf.nn.softmax))


    test_dense.add(Dropout(rate=0.5))
    test_dense.add(Dense(units=256, activation=tf.nn.relu))
    # test_dense.add(Dense(units=128, activation=tf.nn.relu))
    # test_dense.add(Dense(units=64, activation=tf.nn.relu))
    test_dense.add(Dense(units=1, activation=tf.nn.sigmoid))

    y_test = test_dense(x)
    test_model = Model(input=vgg_layers[0].input, output=y_test)

    return model , test_model

