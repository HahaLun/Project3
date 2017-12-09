
import os;
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 3. Import libraries and modules
import numpy as np
#np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# other imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import glob 
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input

from sklearn.metrics import classification_report


def tfInit() :
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def filePrefixWith(folder, filePrefix) :
    for filename in os.listdir(folder) : 
        if filename.startswith(filePrefix)  :
            return folder + "/" + filename;

    return "";


def train(epochs) :

    dataFolder = "./data"

    image_size = (224,224)
    # variables to hold features and labels
    features = []
    labels   = []

    class_count = 1000;
    X_test = []
    y_test = []
    name_test = []

    trainData = np.loadtxt("./train.txt", dtype="str", delimiter='  ' );
    for k in range(len(trainData)) :
        aLine = trainData[k];
        image_path = filePrefixWith(dataFolder, aLine[0]);
        label = int(aLine[1]);
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label] = 1;

        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        labels.append(ground_truth)
        features.append(x[0])



    trainData = np.loadtxt("./test.txt", dtype="str", delimiter='   ' );
    for k in range(len(trainData)) :
        aLine = trainData[k];
        image_path = filePrefixWith(dataFolder, aLine);
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_test.append(x[0])
        name_test.append(image_path)

    X_train = features
    y_train = labels;
    
    # 9. Fit model on training data
    X_train = np.array(X_train)
    Y_train = np.array(y_train)
    X_test = np.array(X_test)

    model = MobileNet(include_top=True,weights='imagenet', classes = class_count);

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
 
    model.fit(X_train, Y_train, batch_size=16,  epochs=epochs, verbose=1, validation_split=0.2  )

    Y_pred = model.predict(X_test)
    
    f = open('project3.txt', 'w')
    for k in range(len(name_test)) :
        thePrediction = Y_pred[k];
        nonzeroind = thePrediction.argmax(axis=0);
        f.write(str(nonzeroind) + '\n')  # python will convert \n to os.linesep

    f.close()  # you can omit in most cases as the destructor will call it
    del model


tfInit();
train(epochs=70);
