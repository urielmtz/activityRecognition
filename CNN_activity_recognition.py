# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:48:53 2017

@author: Uriel Martinez
"""

# First try using data from walking activities generated in MATLAB

import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import numpy.ma as ma
from keras import regularizers

#np.random.seed(10)


def visualize_activity(activity_image):
    #activity = np.squeeze(activity_image, axis=0)
    activity = activity_image
    print(activity.shape)
    plt.imshow(activity)


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic



all_data = sio.loadmat('cnn_activity_gyro_accel_mag_ver1a_activity.mat');
raw_data = all_data['complete_data'];

# temporary manual selection of training and testing samples

trainingPosition = [i for i in range(724)];
testingPosition = [i+724 for i in range(300)];


#trainingPosition = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
#testingPosition = [18, 19, 20, 21, 22, 23];
#x_train = np.zeros((len(raw_data)*len(trainingPosition), 1, len(raw_data[0,0].T), len(raw_data[0,0])));
#x_test = np.zeros((len(raw_data)*len(testingPosition), 1, len(raw_data[0,0].T), len(raw_data[0,0])));

#x_train = np.random.random((48, 200, 18, 1));
#y_train = keras.utils.to_categorical(np.random.randint(3, size=(len(x_train), 1)), num_classes=3);

x_train = np.zeros((2172, 27, 200, 1));
x_test = np.zeros((900, 27, 200, 1));

y_train = np.zeros((2172,));
y_test = np.zeros((900,));

#y_train = keras.utils.to_categorical(np.random.randint(3, size=(len(x_train), 1)), num_classes=3);
#y_test = keras.utils.to_categorical(np.random.randint(3, size=(len(x_test), 1)), num_classes=3);

#x_test = np.random.random((24, 200, 18, 1));
#y_test = keras.utils.to_categorical(np.random.randint(3, size=(len(x_test), 1)), num_classes=3);


countPos = 0;
for i in range(0, len(raw_data)):
    for j in range(0, len(trainingPosition)):
        x_train[countPos,:,:,0] = raw_data[i,trainingPosition[j]];
#        y_train[countPos,0] = np.random.randint(3, size=(1,3));
        y_train[countPos,] = i;
        countPos = countPos + 1;


countPos = 0;
for i in range(0, len(raw_data)):
    for j in range(0, len(testingPosition)):
        x_test[countPos,:,:,0] = raw_data[i,testingPosition[j]];
#        y_test[countPos,0] = np.random.randint(3, size=(1,3));
        y_test[countPos,] = i;
        countPos = countPos + 1;       

        
#x_train = x_train + np.mean(x_train)
#x_test = x_test + np.mean(x_test)

y_train = keras.utils.to_categorical(y_train, 3);
y_test = keras.utils.to_categorical(y_test, 3);
        

model = Sequential();
model.add(Conv2D(32, (5, 5), activation='sigmoid', input_shape=(27, 200, 1)));
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.10))

model.add(Conv2D(16, (3, 3), activation='sigmoid'));
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.10))

model.add(Flatten())
model.add(Dense(200, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.10))
model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(x_train, y_train, batch_size=20, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=1)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
output = model.predict(x_test, batch_size=1)
outputData = np.argmax(output, axis=1)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


save_file = 0

if( save_file == 1 ):
    idFile = 'model_activity_12122017_0001';
    modelPath = idFile + '.h5';

    model.save(modelPath);

