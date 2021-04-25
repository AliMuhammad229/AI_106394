# Leader Work ---> Ali Muhammad

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import np_utils

# Read the CSV files which we doenload from Kaggle
train = pd.read_csv('Kaggle Data/train.csv')
test = pd.read_csv("Kaggle Data/test.csv")
# Set our train data according to frames
df = pd.DataFrame(train)

# label cols i.e. y has only one column
y = train.iloc[:, :1]
# pixels cols i.e. x has 784 col of pixels
X = train.iloc[:, -784:]

# # No.of Rows & Columns
print(f'Number of Rows & Columns in Train.csv: {train.shape}')
print(f'Number of Rows & Columns in Train.csv: {test.shape}')


# label cols
train_y = train['label'].astype('float32')
# pixels cols
train_x = train.drop(['label'], axis=1).astype('int32')
# It has all the cols of test.csv which is 784
test_x = test.astype('float32')
print('\n')

# Shape our data
print(train_y .shape)
print(train_x.shape)
print(test_x.shape)
print('\n')

# Now we need to normalize according to rnae of pixels which is 0 - 255 so we divide 255
train_x = train_x/255.0
test_x = test_x/255.0

# Now we want to convert 784 pixels i.e. 28 * 28 because we want to used as a grey scale image
# This is the dimension of image
train_x = train_x.values.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.values.reshape(test_x.shape[0], 28, 28, 1)

# After the reshaping we can shape in again
print('\n')
print(train_x.shape)
print(test_x.shape)
print('\n')

# Mughees Work

# For Covolution in 2D we used this variables
noLables = 10
batchSize = 64
epochs = 40
# Grey scale image pixels
inputShape = (28,28,1)



# One hot encoding is that we have 0-9 labels so if the answer is 9
# except 9th index we have all the zero's
# Because we need to convert the labels into the 10 values
# Answer: 9 ---> 0 0 0 0 0 0 0 0 0 1, 4 ---> 0 0 0 1 0 0 0 0 0 0 This is one hot Encoding
train_y = OneHotEncoder(train_y, noLables)

# Print the labels of top 5 rows
print(train['label'].head())
# One hot Encoding of the same Label
# Print the encoding values which is binary bits
print(train_y)



# we will separte 20% test data from train.csv and remaining training data is 80%
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


train_x = x_train
train_y = y_train

# It will shape and counts the elements and gives col e.g: 150 --> elements and 4 ---> by 4
print('\n')
print(x_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('\n')

# Apply Sk Learn Categorical Naive Bayes on 20% of testing data
CNB = CategoricalNB()
CNB.fit(X_test, y_test)
print(y_test)
print('\n')
print(X_test)
print('\n')
# Predict on 20% test data of 784 cols variable
prediction = CNB.predict(X_test)
print(f'Predicted Digits & Labels: {prediction}')
print(f'Score: {CNB.score(X_test, y_test)}')
print('\n')
print(classification_report(y_test, prediction))




# Convolution/ Filtering
# Run model to convolve from this
model = Sequential()
# This is the covolution layer which is taken the input of 28, 28, 1
model.add(Conv2D(32, kernel_size = (3,3), input_shape= inputShape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
# Dropout is used as a regularizer
model.add(Dropout(0.2))
# Again using convolution layer of 32 of (3,3) filter
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D())
# For regularization part
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512, activation = 'relu'))

# Tanzeel Work

# sigmoid will give the output based on maximum probability
# 10 means direct 10 classes in this multiclass classification
model.add(Dense(noLables, activation = 'sigmoid'))

# Compile and model training with batch_Size  = 50, epochs = 20 and optimizer + adam
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# It gives the summary
model.summary()

# Verbose is just for boolean means 1 ---> True
# To fitting our model
model.fit(train_x, train_y, epochs = epochs, verbose = 1)

# To find the best accuracy on a specific epocs which is arround 0.990
# On this it loss 0.0967
# It will evaluate() between each iteration on x_test and y_test
# and store the loss in loss var and accuracy in accuracy
# But it will give better accuracy
loss , accuracy = model.evaluate(X_test, y_test, verbose = 0)
print("Loss : ",loss, "Accuracy : ", accuracy)

# To predict our model on test.csv
predicted_classes = model.predict(test_x)

# It creates a dataframes on Image Id col and Labels col which has the rows 28000
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)), "Label": predicted_classes})

# To create this submission it will turn it into comma separated values CSV
submissions.to_csv("submission.csv", index = False, header = True)
# Now we download this submission file
import files
files.download('submission.csv')
