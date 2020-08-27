import glob, os
import json
import matplotlib.pyplot as plt
import numpy as np
import pdb

root = 'dataset/'
r_dirs = os.listdir(root)
print(r_dirs)

#### TEST SETS
def load_data(folder):
    imgs = glob.glob(root+folder + 'inputs/' + '*.png')

    with open(root + folder+'labels.json') as json_file:
        labels = json.load(json_file)
    return imgs, labels

tr_X, tr_Y = load_data('train/')    
test1_X, test1_y = load_data('test_easy/')
print(len(test1_X), len(test1_y))
test2_X, test2_y = load_data('test_hard/')
print(len(test2_X), len(test2_y))

# DATA prep
def data_lists(imgs, labs):
    if len(imgs) != len(labs):
        print('imgs and labels not same length')
        return None
    X_data = []
    y_data = []
    for im in imgs:
        # read the img array
        X_data.append(plt.imread(im))
        # read the label
        y_data.append(labs[os.path.basename(im)])
    #print(y_data)
    X_data = np.array(X_data)
    return X_data, y_data

X_train, y_train = data_lists(tr_X, tr_Y)
# test easy for validation
X_val, y_val = data_lists(test1_X, test1_y)
# test hard for testing
X_test, y_test = data_lists(test2_X, test2_y)
print(X_train.shape)

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# from keras.optimizers import SGD


LR = 0.0001
loss = 'mae' #'mse'
loss = 'mse'
# loss = euclidean_distance_loss
def def_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(20, 20, 3)))
    # model.add(Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Conv2D(10, (1, 1), activation='relu', kernel_initializer='he_uniform'))
    # model.add(Flatten())
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(10, activation='softmax'))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    # compile model
    opt = keras.optimizers.SGD(lr=LR, momentum=0.9)
    # opt = keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

#     model.compile(optimizer=opt, loss=loss, metrics=[keras.metrics.MeanAbsoluteError()])
    model.compile(optimizer=opt, loss=loss)
    
    return model

model = def_model()
model.summary()

# FIT A MODEL
e = 100
callbacks = [EarlyStopping(monitor='val_loss', patience=15),
             ModelCheckpoint(filepath='new_model.h5', monitor='val_loss', save_best_only=True)]
batch = 16
history = model.fit(X_train, y_train, shuffle=True, callbacks=callbacks, epochs=e, batch_size=batch, validation_data=(X_val, y_val))

model = load_model('new_model.h5')

# test with random imgs
def pred_img(imgs, labs, idx = None):
    if idx == None:
        idx = np.random.randint(0,len(imgs))
    im = imgs[idx]
    inp_im = plt.imread(im).reshape(1,20,20,3)
    #print(inp_im.shape)
    im_key = os.path.basename(im)
    print('true label = ', labs[im_key])
    print('pred label = ', model.predict(inp_im))
#    print('pred label = ', np.int(model.predict(inp_im)))


# print('train data:')
# pred_img(tr_X, tr_Y)

# print('easy testing:')
# pred_img(test1_X, test1_y)
# print('hard testing:')
# pred_img(test2_X, test2_y)
def calc_stuff(x, y):
    pred = model.predict_proba(x)
    pred = [x[0] for x in pred]

    ER, MAE, CEP, CE95 = calc_metrics((y, pred))
    print('MAE= %.3f' % MAE, 'CEP= %.3f' % CEP, 'CE95= %.3f' % CE95)


from pretty_plot import *

print('metrics on train')
calc_stuff(X_train, y_train)
print('metrics on test_easy')
calc_stuff(X_val, y_val)
print('metrics on test_hard')
calc_stuff(X_test, y_test)
