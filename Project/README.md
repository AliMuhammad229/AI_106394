# Group Members:

StdID | Name
------------ | -------------
**64413** | **Ali Muhammad (Lead)** 
62749 | Muhammad Mughees
60790 | Syed Muhammad Tanzeel


# Description: How we Achieved each Task?

In this project, we implement four classification techniques

1.  Multinomial Naïve Bayer
2.	Linear Regression 
3.	SUM
4.	KNN

In each technique, we are using 3 convolution 5x5,7x7,9x9 on two different filters.
  #^	We have 784 columns of pixels, it breaks into 28x28, then we create a new array of new dimensions/size (2D Array) and apply 5x5 convolution.
  #^	After the implementation of 5x5 convolution, new array will became in the shape of (0, 576) because this 5x5 convolution will implemented on 42000 rows of         train.csv




# What we learned from this Assignment?

1. We learned that, how to train and test our data on any classification problem and simply predict our digit/labels. 
2. After applying convolution and filter on test and training data, we see that it will get better accuracy and score.
3. Let’s suppose we got score of 0.95% after applying Categorial Naïve Baser and then fit our model on test data and then predict labels so we can achieve 0.990 accuracy.
4. We also learned how to implement hot-encoding on each number. For example: we have a digit 9 so:
                                    000000001
   on bit of 9th index remaining bits are off.  Through this one hot encoding we can evaluate of original training data and no. of labels (predicted Data), this one hot encoding we use/follow tensor flow documention.
5. We are applying Categorical naive Bayes, it is the classifier is suitable for classification with discrete features that are categorically distributed.
6. This model is expected to be called many times consecutively on different chunks of a dataset.
7. This model is useful when the whole dataset is too big to fit in memory at once.

