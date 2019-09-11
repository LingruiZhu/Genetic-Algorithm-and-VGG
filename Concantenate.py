from keras.applications.vgg16 import VGG16

model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
layers = [l for l in model.layers]

print(layers[len(layers)-3].name)
