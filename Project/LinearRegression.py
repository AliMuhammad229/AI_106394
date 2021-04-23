import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
#function to perform convolution
def convolve2D(image, filter):
  fX, fY = filter.shape # Get filter dimensions
  fNby2 = (fX//2)
  n = 28
  nn = n - (fNby2 *2) #new dimension of the reduced image size
  newImage = np.zeros((nn,nn)) #empty new 2D imange
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

#Read Data from CSV
train = pd.read_csv("Kaggle Data/train.csv")
X = train.drop('label',axis=1)
Y = train['label']




#Create Filter for convolution 5 x 5
# Same Dimension
filter = np.array([
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1]
])

# Different Dimensions
filter = np.array([
          [1,1,1,1,1],
          [1, 2, 2, 2, 1],
          [1, 2, 3, 2, 1],
          [1, 2, 2, 2, 1],
          [1,1,1,1,1]
])


# Apply only for 5 x 5 filters i.e. two different sizes
#convert from dataframe to numpy array
X = X.to_numpy()
print(f'Number of Rows & Columns: {X.shape}')

#new array with reduced number of features to store the small size images
sX = np.empty((0,576), int)


ss = 42000 #subset size for dry runs change to 42000 to run on whole data

#Perform convolve on all images
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,576))
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]
# print(sY)
print(sY.shape)
print(sX.shape)
print('\n')

# train and test model
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
print('\n')


linearRegressionModel = LinearRegression()
linearRegressionModel.fit(sXTrain, yTrain)


print(f'Score: {linearRegressionModel.score(sXTest, yTest)}')

print(sX.shape)

# To predict our model on test.csv
predictedClasses = linearRegressionModel.predict(sXTest)

# It creates a dataframes on Image Id col and Labels col which has the rows 28000
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictedClasses)+1)), "Label": predictedClasses})

# To create this submission it will turn it into comma separated values CSV
submissions.to_csv("submission.csv", index = False, header = True)
# Now we download this submission file

from google.colab import files
files.download('submission.csv')




#Create Filter for convolution 7 x 7
# Same Dimension
filter = np.array([
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1]
])

# # Different Dimension
filter = np.array([
          [1, 1, 1, 1, 1, 1, 1],
          [1, 2, 2, 2, 1, 1, 1],
          [1, 2, 3, 2, 1, 1, 1],
          [1, 2, 2, 2, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1]
])

# Apply only for 7 x 7 filters i.e. two different sizes
#convert from dataframe to numpy array
X = X.to_numpy()
print(f'Number of Rows & Columns: {X.shape}')

#new array with reduced number of features to store the small size images
sX = np.empty((0,484), int)


ss = 42000 #subset size for dry runs change to 42000 to run on whole data

#Perform convolve on all images
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,484))
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]
# print(sY)
print(sY.shape)
print(sX.shape)
print('\n')

# train and test model
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
print('\n')


linearRegressionModel = LinearRegression()
linearRegressionModel.fit(sXTrain, yTrain)


print(f'Score: {linearRegressionModel.score(sXTest, yTest)}')

print(sX.shape)





#Create Filter for convolution 9 x 9
# Same Dimension
filter = np.array([
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1]
])


# Different Dimension
filter = np.array([
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 2, 2, 2, 1, 1, 1, 1, 1],
          [1, 2, 3, 2, 1, 1, 1, 1, 1],
          [1, 2, 2, 2, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1]
])


# Apply only for 9 x 9 filters i.e. two different sizes
#convert from dataframe to numpy array
X = X.to_numpy()
print(f'Number of Rows & Columns: {X.shape}')

#new array with reduced number of features to store the small size images
sX = np.empty((0,400), int)


ss = 42000 #subset size for dry runs change to 42000 to run on whole data

#Perform convolve on all images
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,400))
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]
# print(sY)
print(sY.shape)
print(sX.shape)
print('\n')

# train and test model
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
print('\n')


linearRegressionModel = LinearRegression()
linearRegressionModel.fit(sXTrain, yTrain)


print(f'Score: {linearRegressionModel.score(sXTest, yTest)}')

print(sX.shape)