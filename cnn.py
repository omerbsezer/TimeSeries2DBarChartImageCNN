import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import math




def train_cnn(training_df, test_df, params):
    # Trains and evaluates CNN on the given train and test data, respectively.

    print("Training is starting ...")
    train_images = training_df.ix[:, :899].as_matrix()
    train_labels = training_df.ix[:, 900]
    train_prices = training_df.ix[: ,901]

    test_images = test_df.ix[:, :899].as_matrix()
    test_labels = test_df.ix[:, 900]
    test_prices = test_df.ix[:, 901]




    test_labels = keras.utils.to_categorical(test_labels, params["num_classes"])
    train_labels = keras.utils.to_categorical(train_labels, params["num_classes"])


    train_images = train_images.reshape(train_images.shape[0], params["input_w"], params["input_h"], 1)
    test_images = test_images.reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)



    # CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(params["input_w"], params["input_h"], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params["num_classes"], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy', 'mae', 'mse'])

    # metrics.accuracy_score, metrics.recall_score, metrics.average_precision_score, metrics.confusion_matrix
    train_data_size = train_images.shape[0]
    test_data_size = test_images.shape[0]


    print("model will be trained with {} and be tested with {} sample".format(train_data_size,test_data_size))
    # fit the model to the training data
    print("Fitting model to the training data...")
    print("")
    model.fit(train_images, train_labels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,validation_data=None)

    predictions = model.predict(test_images, batch_size=params["batch_size"], verbose=1)
    print("loss, accuracy, mean absolute error, mean squared error")
    print(model.evaluate(test_images, test_labels, batch_size=params["batch_size"], verbose=1))

    print("Train conf matrix: ", confusion_matrix(np.array(reverse_one_hot(train_labels)),
                                                  np.array(reverse_one_hot(model.predict(train_images, batch_size=params["batch_size"], verbose=1)))))

    print("Test conf matrix: ",  confusion_matrix(np.array(reverse_one_hot(test_labels)),
                                                  np.array(reverse_one_hot(predictions))))


    return predictions, test_labels, test_prices




# reverse one hot manually
def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return reversed_x




# read csv file and create df files
train_df = pd.read_csv("training.csv", header=None, index_col=None, delimiter=';')
test_df = pd.read_csv("test.csv", header=None, index_col=None, delimiter=';')


print("Before data imbalance")
l0_train = train_df.loc[train_df[900] == 0]
l1_train = train_df.loc[train_df[900] == 1]
l2_train = train_df.loc[train_df[900] == 2]
l0_train_size = l0_train.shape[0]
l1_train_size = l1_train.shape[0]
l2_train_size = l2_train.shape[0]
print('train_df => l0_size:',l0_train_size,'l1_size:',l1_train_size,'l2_size:',l2_train_size)
l0_test = test_df.loc[test_df[900] == 0]
l1_test = test_df.loc[test_df[900] == 1]
l2_test = test_df.loc[test_df[900] == 2]
l0_test_size = l0_test.shape[0]
l1_test_size = l1_test.shape[0]
l2_test_size = l2_test.shape[0]
print('test_df => l0_size:',l0_test_size,'l1_size:',l1_test_size,'l2_size:',l2_test_size)

# calculate the number of labels and find ratios
l0_l1_ratio = (l0_train_size//l1_train_size)
l0_l2_ratio = (l0_train_size//l2_train_size)
l1_l0_ratio = (l1_train_size//l0_train_size)
l1_l2_ratio = (l1_train_size//l2_train_size)
l2_l0_ratio = (l2_train_size//l0_train_size)
l2_l1_ratio = (l2_train_size//l1_train_size)
print("l0_l1_ratio:",l0_l1_ratio)
print("l0_l2_ratio:",l0_l2_ratio)
print("l1_l0_ratio:",l1_l0_ratio)
print("l1_l2_ratio:",l1_l2_ratio)
print("l2_l0_ratio:",l2_l0_ratio)
print("l2_l1_ratio:",l2_l1_ratio)
#if there is data imbalance, solution of data imbalance in training set
#data imbalance #0/#1>1

# to solve data imbalance
l0_new = pd.DataFrame()
l1_new = pd.DataFrame()
l2_new = pd.DataFrame()
for idx, row in train_df.iterrows():
     if row[900] == 0:
        for i in range(2):
             l0_new = l0_new.append(row)
     if row[900] == 1:
         for i in range(2):
             l1_new = l1_new.append(row)
     if row[900] == 2:
         for i in range(2):
             l2_new = l2_new.append(row)

train_df = train_df.append(l0_new)
train_df = train_df.append(l1_new)
train_df = train_df.append(l2_new)
#
# shuffle
train_df = shuffle(train_df)


print("After data imbalance")
l0_train = train_df.loc[train_df[900] == 0]
l1_train = train_df.loc[train_df[900] == 1]
l2_train = train_df.loc[train_df[900] == 2]
l0_train_size = l0_train.shape[0]
l1_train_size = l1_train.shape[0]
l2_train_size = l2_train.shape[0]
print('train_df => l0_size:',l0_train_size,'l1_size:',l1_train_size,'l2_size:',l2_train_size)
l0_test = test_df.loc[test_df[900] == 0]
l1_test = test_df.loc[test_df[900] == 1]
l2_test = test_df.loc[test_df[900] == 2]
l0_test_size = l0_test.shape[0]
l1_test_size = l1_test.shape[0]
l2_test_size = l2_test.shape[0]
print('test_df => l0_size:',l0_test_size,'l1_size:',l1_test_size,'l2_size:',l2_test_size)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print("train_df size: ", train_df.shape)

# parameters of the cnn
params = {"input_w": 30, "input_h": 30, "num_classes": 3, "batch_size": 1024, "epochs": 100}

#call cnn
predictions, test_labels, test_prices = train_cnn(train_df, test_df, params)

#results of the cnn
result_df = pd.DataFrame({"prediction": np.argmax(predictions, axis=1),
                           "test_label":np.argmax(test_labels, axis=1),
                           "test_price":test_prices})
result_df.to_csv("cnn_result.csv", sep=';', index=None)









