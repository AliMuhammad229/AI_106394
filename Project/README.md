# Group Members:

StdID | Name
------------ | -------------
**64413** | **Ali Muhammad (Lead)** 
62749 | Muhammad Mughees
60790 | Syed Muhammad Tanzeel


# Description: How we Achieved each Task?


In this project, we implement four classifications techniques


 1.  Multinomial Naïve Bayes
 2.  Linear Regression 
 3.  SVM 
 4.  KNN 

In each technique, we are using 3 convolution 5x5,7x7,9x9 on two different filters.
   1. We have 784 columns of pixels, it breaks into 28x28, then we create a new array of new dimensions/size (2D Array) and apply 5x5 convolution.
   2. After the implementation of 5x5 convolution, new array will became in the shape of (0, 576) because this 5x5 convolution will implemented on 42000                 rows of train.csv





# CV Score of Each Six Techniques: #

## 1. Multinomial Naive Bayes: ##

1. Through this, we can achieve a score on Multinomial Naive Buyer 5x5 convolution with same size filter (1, 1) is 0.76.
2. On 5x5 convolution, with different size filter (average filter) will be achieved 0.77.
3. Now apply 7x7 convolution (49 pixels) with same filter on Multinomial Naïve Bayer, it will get 0.71 score and it will not maintain the score as compare to 5x5      convolution.
4. On 7x7 convolution, with average filter (diff filter) achieved 0.72.
5. After 7x7 convolution, new array will become 0.484.
6. Now applying 9x9 convolution (81 pixels), with same filter on this model it will get 0.67 because when on the convolution, it is increasing and it will not        maintain accuracy.
7. On 9x9 convolution, with average filter (diff filter) we achieve 0.6835.
8. After the application of 9x9, convolution new array will become in the array of (0,400).



## 2. KNN: ##

1. In this model, it will find nearest neighbor on K-Value which is in the odd after get the sqrt on yTest (from CV).
2. By using this model on 5x5, convolution with same filter we achieved 0.93 score.
3. On 5x5 convolution, with different filter we achieved 0.94 score.
4. On 7x7 convolution, with different filter we achieved 0.922.
5. Now we applying 9x9 convolution, with same filter we achieve 0.879 score.
6. On 9x9 convolution, with different filter we achieved 0.897 because with 9x9 convolution this will reduce further on 784 it is 81 so it will not maintain the best score.



## 3. Linear Regression: ##

1. This model used to minimize the sum of square between  the observed and target in the data set and the target predicted by the linear approximation.
2. By using this model on 5x5, convolution with same filter we achieved 0.603 score.
3. On 5x5 convolution, with different filter we achieved 0.604 score
4. Now applying 7x7 convolution, with same filter we achieved 0.603 score.
5. On 7x7 convolution with different filter, we achieved 0.588 score.
6. Now applying 9x9 convolution, with same filter we achieved 0.5818 score.
7. On 9x9 convolution, with different filter we achieved 0.5841.



## 4. SVM ##

1. This model is different from other because it does not learn on the characteristics not like other models learn.
2. By using this model, on 5x5 convolution with same filter we achieve 0.88 score.
3. On 5x5 convolution, with different filter we achieve 0.87 score.
4. Now applying 7x7 convolution with same filter we achieve 0.89 score.
5. On 7x7 convolution, with different filter we achieved 0.87 score.
6. Now applying 9x9 convolution with same filter we achieve 0.76 score.
7. On 9x9 convolution, with different filter we achieve 0.77 score.




# Description: Important part of .py file: #


### Convolution Part: ###

1. In this part, we are applying 5x5,7x7,9x9 convolution to map on our 42000 data, It will help to predict and get the filtered image/label.
2. Explaining about its working, Firstly, we can break our 784 columns into 28x28 and create 2D Array and iterate on array filter will push into it.


### Models Part: ###

1. We implement four techniques and on these techniques, we are applying convolutions to get the best/good score.
2. But according to our views to work on this phase, we achieve best score on different filter(average filter) of convolution.



# Description of Classifier from Scikit Learn: #


### Multinomial Naive Bayes: ###

It is suitable discrete feature (counting, classification of tent)

#### Parameters: #### 
      1. It take the smoothing value which means 0 for no smoothing or 1 for Laplace smoothing.

### KNN: ###

It will find the nearest neighbors on K-value but this K-value is odd, after getting square root of yTest (from cross validation).

#### Parameters: #### 
      1. It takes K=7 (model) and also p value, if p=1 means euclidean distance and p=2 manhatten distance.

### Linear Regression: ###

It is used to minimize the sum of square. The observed target in the dataset and the target predicted by the linear approximation.

#### Parameters: #### 
      1. It takes max iter attribute to work on it.
      2. In this model we can define a range.


### SVM: ###

1. It will find the characteristics which matches the other classes.
2. In this model, we have advantage that we can’t note data points instead of note down the suppose vector.

#### Parameters: #### 
      1. It takes the 'C' value which is regularize value, greater the value of C causes more chances to works at its best.
      2. It takes gamma values.



# How We Achieved One Score with Other Techniques: #

1. By using these four techniques, KNN gives the best score among other three techniques on 5x5 convolution with different filter (0.94 or 94%)      so if we enhance this model by using 2x2 or 3x3 convolution then it will give 0.99 or 1 score.
2. Other techniques is CNN is categorical Naïve Bayer on cross fold to achieve 100% score.


